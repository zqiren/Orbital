# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for AgentMessageTool send-count limit and ToolRegistry.reset_run_state().

Verifies that:
- agent_message(send) is capped at max_sends_per_run
- on_run_start resets the counter
- start/stop/list/status actions are not counted
- ToolRegistry.reset_run_state() calls on_run_start on tools that have it
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools.agent_message import AgentMessageTool
from agent_os.agent.tools.base import Tool, ToolResult
from agent_os.agent.tools.registry import ToolRegistry


def _make_tool(max_sends=3):
    """Create an AgentMessageTool with a mock SubAgentManager."""
    mgr = MagicMock()
    mgr.send = AsyncMock(return_value="sent")
    mgr.start = AsyncMock(return_value="started")
    mgr.stop = AsyncMock(return_value="stopped")
    mgr.list_active = MagicMock(return_value=[])
    mgr.status = MagicMock(return_value="running")
    tool = AgentMessageTool(sub_agent_manager=mgr, project_id="proj-1",
                            max_sends_per_run=max_sends)
    return tool, mgr


class TestSendCountLimit:
    """agent_message(send) must be capped per run."""

    @pytest.mark.asyncio
    async def test_sends_succeed_up_to_limit(self):
        tool, mgr = _make_tool(max_sends=3)
        for i in range(3):
            result = await tool.execute(action="send", agent="a", message=f"msg-{i}")
            assert "sent" in result.content
        assert mgr.send.call_count == 3

    @pytest.mark.asyncio
    async def test_send_blocked_after_limit(self):
        tool, mgr = _make_tool(max_sends=3)
        for _ in range(3):
            await tool.execute(action="send", agent="a", message="msg")
        result = await tool.execute(action="send", agent="a", message="one-too-many")
        assert "send limit reached" in result.content
        assert "3" in result.content  # mentions the limit
        assert mgr.send.call_count == 3  # 4th was not forwarded

    @pytest.mark.asyncio
    async def test_send_blocked_message_instructs_summarize(self):
        tool, _ = _make_tool(max_sends=1)
        await tool.execute(action="send", agent="a", message="first")
        result = await tool.execute(action="send", agent="a", message="second")
        assert "Summarize" in result.content

    @pytest.mark.asyncio
    async def test_counter_increments_per_send(self):
        tool, _ = _make_tool(max_sends=10)
        assert tool._send_count == 0
        await tool.execute(action="send", agent="a", message="m")
        assert tool._send_count == 1
        await tool.execute(action="send", agent="a", message="m")
        assert tool._send_count == 2


class TestStartStopNotCounted:
    """start, stop, list, status must not increment the send counter."""

    @pytest.mark.asyncio
    async def test_start_not_counted(self):
        tool, _ = _make_tool(max_sends=1)
        await tool.execute(action="start", agent="a")
        assert tool._send_count == 0

    @pytest.mark.asyncio
    async def test_stop_not_counted(self):
        tool, _ = _make_tool(max_sends=1)
        await tool.execute(action="stop", agent="a")
        assert tool._send_count == 0

    @pytest.mark.asyncio
    async def test_list_not_counted(self):
        tool, _ = _make_tool(max_sends=1)
        await tool.execute(action="list", agent="")
        assert tool._send_count == 0

    @pytest.mark.asyncio
    async def test_status_not_counted(self):
        tool, _ = _make_tool(max_sends=1)
        await tool.execute(action="status", agent="a")
        assert tool._send_count == 0


class TestOnRunStart:
    """on_run_start must reset the send counter."""

    @pytest.mark.asyncio
    async def test_reset_clears_counter(self):
        tool, _ = _make_tool(max_sends=3)
        await tool.execute(action="send", agent="a", message="m1")
        await tool.execute(action="send", agent="a", message="m2")
        assert tool._send_count == 2
        tool.on_run_start()
        assert tool._send_count == 0

    @pytest.mark.asyncio
    async def test_sends_work_after_reset(self):
        tool, mgr = _make_tool(max_sends=2)
        await tool.execute(action="send", agent="a", message="m1")
        await tool.execute(action="send", agent="a", message="m2")
        # Now at limit
        result = await tool.execute(action="send", agent="a", message="m3")
        assert "send limit reached" in result.content
        # Reset and send again
        tool.on_run_start()
        result = await tool.execute(action="send", agent="a", message="m4")
        assert "sent" in result.content
        assert mgr.send.call_count == 3  # m1, m2, m4


class TestRegistryResetRunState:
    """ToolRegistry.reset_run_state() must call on_run_start on participating tools."""

    def test_calls_on_run_start(self):
        tool, _ = _make_tool(max_sends=5)
        tool._send_count = 4  # simulate accumulated sends
        registry = ToolRegistry()
        registry.register(tool)
        registry.reset_run_state()
        assert tool._send_count == 0

    def test_skips_tools_without_hook(self):
        """Tools without on_run_start should not cause errors."""

        class PlainTool(Tool):
            def __init__(self):
                self.name = "plain"
                self.description = "no hook"
                self.parameters = {"type": "object", "properties": {}}

            def execute(self, **kwargs):
                return ToolResult(content="ok")

        registry = ToolRegistry()
        registry.register(PlainTool())
        # Should not raise
        registry.reset_run_state()

    def test_calls_multiple_hooks(self):
        """All tools with on_run_start should be called."""
        tool1, _ = _make_tool(max_sends=5)
        tool1.name = "agent_message_1"
        tool1._send_count = 3

        tool2, _ = _make_tool(max_sends=5)
        tool2.name = "agent_message_2"
        tool2._send_count = 4

        registry = ToolRegistry()
        registry.register(tool1)
        registry.register(tool2)
        registry.reset_run_state()

        assert tool1._send_count == 0
        assert tool2._send_count == 0


class TestInjectDuringRun:
    """Verify counter behavior when messages are injected mid-run.

    When a user injects a message via REST while the loop is running,
    it's queued and drained in the same run() — the counter should NOT reset.
    """

    @pytest.mark.asyncio
    async def test_counter_accumulates_across_injected_messages(self):
        """Simulates: 3 sends, then a queued user message, then 2 more sends."""
        tool, mgr = _make_tool(max_sends=4)
        # First batch: 3 sends within the run
        for _ in range(3):
            await tool.execute(action="send", agent="a", message="batch1")
        assert tool._send_count == 3
        # Queued message drains (no reset — same run)
        # 4th send succeeds (at limit)
        result = await tool.execute(action="send", agent="a", message="batch2")
        assert "sent" in result.content
        # 5th send should be blocked
        result = await tool.execute(action="send", agent="a", message="batch2-over")
        assert "send limit reached" in result.content
        assert mgr.send.call_count == 4
