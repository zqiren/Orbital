# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for lifecycle message deferral during tool execution.

Verifies that lifecycle observer notifications (e.g., "[Sub-agent] claude-code started")
are buffered via Session.defer_message() and only drained AFTER the tool execution batch
completes, preventing injection between assistant→tool result sequences that would cause
LLM API 400 errors.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.session import Session


# ---------------------------------------------------------------------------
# Session-level deferral tests
# ---------------------------------------------------------------------------


class TestSessionDeferral:
    """Tests for Session.defer_message / pop_deferred_messages."""

    def _make_session(self) -> Session:
        """Create a fresh in-memory session backed by a temp file."""
        tmp = tempfile.mktemp(suffix=".jsonl")
        session = Session(tmp)
        # Create the file
        with open(tmp, "w") as f:
            pass
        return session

    def test_defer_message_not_in_session_immediately(self):
        """Deferred message should NOT appear in get_messages() until drained."""
        session = self._make_session()
        session.defer_message("[Sub-agent] claude-code started")

        # Not in the main message list
        msgs = session.get_messages()
        assert len(msgs) == 0

        # But IS in the internal deferred buffer
        assert len(session._deferred_messages) == 1
        assert session._deferred_messages[0]["content"] == "[Sub-agent] claude-code started"
        assert session._deferred_messages[0]["role"] == "system"
        assert session._deferred_messages[0]["source"] == "daemon"

    def test_pop_deferred_returns_messages_in_order(self):
        """pop_deferred_messages returns messages in FIFO order and clears the buffer."""
        session = self._make_session()
        session.defer_message("msg-1")
        session.defer_message("msg-2")
        session.defer_message("msg-3")

        popped = session.pop_deferred_messages()
        assert len(popped) == 3
        assert popped[0]["content"] == "msg-1"
        assert popped[1]["content"] == "msg-2"
        assert popped[2]["content"] == "msg-3"

        # Subsequent pop returns empty
        assert session.pop_deferred_messages() == []

    def test_deferred_drained_after_tool_batch(self):
        """Simulate: assistant with tool_calls → defer lifecycle → tool result → drain.

        Session order must be: assistant → tool → system (not assistant → system → tool).
        """
        session = self._make_session()

        # 1. Assistant message with a tool call
        session.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc_1", "name": "read", "arguments": {}}],
            "source": "management",
        })

        # 2. Lifecycle notification arrives while tool is executing
        session.defer_message("[Sub-agent] claude-code started")

        # 3. Tool result comes back
        session.append_tool_result("tc_1", "file contents here")

        # 4. Drain deferred (as the loop would do after the for-loop)
        for msg in session.pop_deferred_messages():
            session.append(msg)

        # Verify order: assistant → tool → system
        msgs = session.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "tool"
        assert msgs[2]["role"] == "system"
        assert msgs[2]["content"] == "[Sub-agent] claude-code started"

    def test_deferred_not_drained_between_tool_calls(self):
        """Multi-tool batch: lifecycle msg deferred mid-batch stays deferred until drain.

        assistant with [write:4, agent_message:5] → tool(write:4) → defer lifecycle →
        tool(agent_message:5) → drain deferred.
        Order: assistant → tool(write) → tool(agent_message) → system
        """
        session = self._make_session()

        # Assistant with two tool calls
        session.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "tc_write", "name": "write", "arguments": {}},
                {"id": "tc_agent", "name": "agent_message", "arguments": {}},
            ],
            "source": "management",
        })

        # First tool result
        session.append_tool_result("tc_write", "wrote file")

        # Lifecycle notification arrives between tool executions
        session.defer_message("[Sub-agent] claude-code started")

        # Second tool result
        session.append_tool_result("tc_agent", "message sent")

        # Drain deferred
        for msg in session.pop_deferred_messages():
            session.append(msg)

        # Verify order: assistant → tool(write) → tool(agent_message) → system
        msgs = session.get_messages()
        assert len(msgs) == 4
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "tool"
        assert msgs[1]["tool_call_id"] == "tc_write"
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "tc_agent"
        assert msgs[3]["role"] == "system"
        assert "[Sub-agent]" in msgs[3]["content"]

    def test_multi_tool_batch_with_lifecycle_events(self):
        """Two agent_message tool calls, each defers a lifecycle message.

        All tool results must come before all lifecycle messages.
        """
        session = self._make_session()

        # Assistant with two agent_message tool calls
        session.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "tc_am1", "name": "agent_message", "arguments": {}},
                {"id": "tc_am2", "name": "agent_message", "arguments": {}},
            ],
            "source": "management",
        })

        # Execute first tool, defer lifecycle
        session.append_tool_result("tc_am1", "sent to agent-1")
        session.defer_message("[Sub-agent] agent-1 started")

        # Execute second tool, defer another lifecycle
        session.append_tool_result("tc_am2", "sent to agent-2")
        session.defer_message("[Sub-agent] agent-2 started")

        # Drain deferred
        for msg in session.pop_deferred_messages():
            session.append(msg)

        msgs = session.get_messages()
        assert len(msgs) == 5  # assistant + 2 tools + 2 systems

        # All tool results before all lifecycle messages
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "tool"
        assert msgs[2]["role"] == "tool"
        assert msgs[3]["role"] == "system"
        assert msgs[4]["role"] == "system"
        assert "agent-1 started" in msgs[3]["content"]
        assert "agent-2 started" in msgs[4]["content"]


# ---------------------------------------------------------------------------
# AgentManager inject_system_message tests
# ---------------------------------------------------------------------------


class TestInjectSystemMessage:
    """Tests for AgentManager.inject_system_message deferral logic."""

    def _make_manager(self):
        from agent_os.daemon_v2.agent_manager import AgentManager

        project_store = MagicMock()
        ws = MagicMock()
        ws.broadcast = MagicMock()
        sub_agent_mgr = MagicMock()
        sub_agent_mgr.list_active = MagicMock(return_value=[])
        sub_agent_mgr.stop = AsyncMock()
        sub_agent_mgr.stop_all = AsyncMock()
        activity_translator = MagicMock()
        process_manager = MagicMock()
        process_manager.set_session = MagicMock()

        mgr = AgentManager(
            project_store=project_store,
            ws_manager=ws,
            sub_agent_manager=sub_agent_mgr,
            activity_translator=activity_translator,
            process_manager=process_manager,
        )
        return mgr, ws

    @pytest.mark.asyncio
    async def test_inject_system_message_idle_appends_directly(self):
        """When loop is idle (task done), message is appended to session immediately."""
        mgr, ws = self._make_manager()

        mock_session = MagicMock()
        mock_session.is_stopped.return_value = False
        mock_session._paused_for_approval = False

        mock_task = MagicMock()
        mock_task.done.return_value = True
        mock_task.exception.return_value = None

        handle = MagicMock(session=mock_session, task=mock_task, loop=MagicMock())
        mgr._handles["proj_1"] = handle

        with patch.object(mgr, "_start_loop", new_callable=AsyncMock) as mock_start:
            result = await mgr.inject_system_message("proj_1", "lifecycle event")

        assert result == "delivered"
        mock_session.append.assert_called_once()
        appended = mock_session.append.call_args[0][0]
        assert appended["role"] == "system"
        assert appended["content"] == "lifecycle event"
        assert appended["source"] == "daemon"

    @pytest.mark.asyncio
    async def test_inject_system_message_running_defers(self):
        """When loop is running (task not done), message is deferred, not appended."""
        mgr, ws = self._make_manager()

        mock_session = MagicMock()
        mock_task = MagicMock()
        mock_task.done.return_value = False  # Loop is running

        handle = MagicMock(session=mock_session, task=mock_task)
        mgr._handles["proj_1"] = handle

        result = await mgr.inject_system_message("proj_1", "[Sub-agent] claude-code started")

        assert result == "deferred"
        mock_session.defer_message.assert_called_once_with(
            "[Sub-agent] claude-code started", role="system", source="daemon"
        )
        # Should NOT have been appended directly
        mock_session.append.assert_not_called()


# ---------------------------------------------------------------------------
# Loop exit drain test
# ---------------------------------------------------------------------------


class TestLoopExitDrain:
    """Tests that deferred messages are drained on loop exit."""

    def test_loop_exit_drains_remaining_deferred(self):
        """Defer 2 messages, simulate loop exit drain, verify both appended."""
        tmp = tempfile.mktemp(suffix=".jsonl")
        session = Session(tmp)
        with open(tmp, "w") as f:
            pass

        # Simulate: two lifecycle messages were deferred during the loop
        session.defer_message("[Sub-agent] agent-1 completed")
        session.defer_message("[Sub-agent] agent-2 completed")

        assert len(session.get_messages()) == 0
        assert len(session._deferred_messages) == 2

        # Simulate the finally-block drain
        for msg in session.pop_deferred_messages():
            session.append(msg)

        msgs = session.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["content"] == "[Sub-agent] agent-1 completed"
        assert msgs[1]["content"] == "[Sub-agent] agent-2 completed"
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "system"

        # Buffer is now empty
        assert session.pop_deferred_messages() == []
