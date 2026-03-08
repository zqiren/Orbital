# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: nonce must flow through all inject_message code paths.

Bug: The first message in a session appears twice on mobile because the
nonce is dropped in the auto-start (Case 3) and queue (Case 1) paths.
The frontend dedup logic checks the nonce to skip messages it already
rendered optimistically -- without a nonce, it treats the broadcast as
a new message and renders a duplicate.

Fix: Thread nonce through start_agent -> loop.run -> session.append,
and store (content, nonce) tuples in the session queue.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.daemon_v2.activity_translator import ActivityTranslator


class TestAutoStartNonce:
    """Case 3: inject_message auto-starts agent -- nonce must reach broadcast."""

    @pytest.mark.asyncio
    async def test_auto_start_passes_nonce_to_start_agent(self):
        """inject_message Case 3 passes nonce to start_agent."""
        from agent_os.daemon_v2.agent_manager import AgentManager

        project_store = MagicMock()
        project_store.get_project.return_value = {
            "workspace": "/tmp/test",
            "name": "test",
            "model": "gpt-4",
            "api_key": "sk-test",
            "sdk": "openai",
        }

        mgr = AgentManager(
            project_store=project_store,
            ws_manager=MagicMock(),
            sub_agent_manager=MagicMock(),
            activity_translator=MagicMock(),
            process_manager=MagicMock(),
        )

        # Patch start_agent to capture the call
        with patch.object(mgr, "start_agent", new_callable=AsyncMock) as mock_start:
            await mgr.inject_message("proj_1", "hello", nonce="nonce-abc")

        mock_start.assert_called_once()
        _, kwargs = mock_start.call_args
        assert kwargs.get("initial_nonce") == "nonce-abc"

    @pytest.mark.asyncio
    async def test_loop_run_includes_nonce_in_initial_message(self):
        """AgentLoop.run() includes nonce in the appended initial message."""
        session = MagicMock()
        session._paused_for_approval = False
        session.resolve_pending_tool_calls = MagicMock()
        session.pop_queued_messages = MagicMock(return_value=[])
        session.is_stopped = MagicMock(return_value=False)
        session.is_paused = MagicMock(return_value=False)

        appended = []
        session.append = MagicMock(side_effect=lambda m: appended.append(m))

        provider = MagicMock()
        # Make stream return a text-only response to end the loop quickly
        from agent_os.agent.providers.types import LLMResponse, TokenUsage
        provider.stream = AsyncMock(return_value=iter([]))

        tool_registry = MagicMock()
        tool_registry.schemas = MagicMock(return_value=[])
        tool_registry.reset_run_state = MagicMock()

        context_manager = MagicMock()
        context_manager.prepare = MagicMock(return_value=[])
        context_manager.should_compact = MagicMock(return_value=False)

        loop = AgentLoop(session, provider, tool_registry, context_manager)

        # Mock _stream_response to return a text-only response
        text_response = LLMResponse(
            text="Hi there!",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "Hi there!"},
            has_tool_calls=False,
            finish_reason="stop",
            status_text=None,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )
        with patch.object(loop, "_stream_response_with", new_callable=AsyncMock,
                          return_value=text_response):
            await loop.run("hello", initial_nonce="nonce-xyz")

        # First appended message should be the initial user message with nonce
        assert len(appended) >= 1
        user_msg = appended[0]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "hello"
        assert user_msg["nonce"] == "nonce-xyz"

    @pytest.mark.asyncio
    async def test_loop_run_no_nonce_omits_field(self):
        """AgentLoop.run() without nonce does not add nonce field."""
        session = MagicMock()
        session._paused_for_approval = False
        session.resolve_pending_tool_calls = MagicMock()
        session.pop_queued_messages = MagicMock(return_value=[])
        session.is_stopped = MagicMock(return_value=False)
        session.is_paused = MagicMock(return_value=False)

        appended = []
        session.append = MagicMock(side_effect=lambda m: appended.append(m))

        provider = MagicMock()
        tool_registry = MagicMock()
        tool_registry.schemas = MagicMock(return_value=[])
        tool_registry.reset_run_state = MagicMock()

        context_manager = MagicMock()
        context_manager.prepare = MagicMock(return_value=[])
        context_manager.should_compact = MagicMock(return_value=False)

        loop = AgentLoop(session, provider, tool_registry, context_manager)

        from agent_os.agent.providers.types import LLMResponse, TokenUsage
        text_response = LLMResponse(
            text="Hi!",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "Hi!"},
            has_tool_calls=False,
            finish_reason="stop",
            status_text=None,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )
        with patch.object(loop, "_stream_response_with", new_callable=AsyncMock,
                          return_value=text_response):
            await loop.run("hello")

        user_msg = appended[0]
        assert user_msg["role"] == "user"
        assert "nonce" not in user_msg


class TestQueuedMessageNonce:
    """Case 1: message queued while loop running -- nonce preserved."""

    def test_queue_message_stores_nonce(self):
        """queue_message stores (content, nonce) tuple."""
        session = Session.__new__(Session)
        session._queue = []

        session.queue_message("hello", nonce="q-nonce-1")

        assert len(session._queue) == 1
        assert session._queue[0] == ("hello", "q-nonce-1")

    def test_queue_message_stores_none_nonce(self):
        """queue_message without nonce stores None."""
        session = Session.__new__(Session)
        session._queue = []

        session.queue_message("hello")

        assert session._queue[0] == ("hello", None)

    def test_pop_queued_messages_returns_tuples(self):
        """pop_queued_messages returns (content, nonce) tuples."""
        session = Session.__new__(Session)
        session._queue = []

        session.queue_message("msg1", nonce="n1")
        session.queue_message("msg2")

        result = session.pop_queued_messages()
        assert len(result) == 2
        assert result[0] == ("msg1", "n1")
        assert result[1] == ("msg2", None)
        assert session._queue == []

    @pytest.mark.asyncio
    async def test_inject_queued_message_preserves_nonce(self):
        """inject_message Case 1 (running) passes nonce to queue_message."""
        from agent_os.daemon_v2.agent_manager import AgentManager

        mgr = AgentManager(
            project_store=MagicMock(),
            ws_manager=MagicMock(),
            sub_agent_manager=MagicMock(),
            activity_translator=MagicMock(),
            process_manager=MagicMock(),
        )

        mock_session = MagicMock()
        mock_session._paused_for_approval = False
        queue_calls = []
        mock_session.queue_message = MagicMock(
            side_effect=lambda c, nonce=None: queue_calls.append((c, nonce))
        )

        mock_task = MagicMock()
        mock_task.done.return_value = False  # loop is running

        mock_handle = MagicMock()
        mock_handle.task = mock_task
        mock_handle.session = mock_session
        mock_handle.interceptor = None

        mgr._handles["proj_q"] = mock_handle

        result = await mgr.inject_message("proj_q", "queued msg", nonce="q-nonce")

        assert result == "queued"
        assert len(queue_calls) == 1
        assert queue_calls[0] == ("queued msg", "q-nonce")

    @pytest.mark.asyncio
    async def test_loop_drain_includes_nonce_from_queue(self):
        """Loop queue drain appends nonce from queued tuples."""
        session = MagicMock()
        session._paused_for_approval = False
        session.resolve_pending_tool_calls = MagicMock()
        session.is_stopped = MagicMock(return_value=True)  # stop after drain
        session.is_paused = MagicMock(return_value=False)

        # First call returns queued items, second returns empty
        call_count = [0]
        def pop_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return [("queued hello", "drain-nonce")]
            return []
        session.pop_queued_messages = MagicMock(side_effect=pop_side_effect)

        appended = []
        session.append = MagicMock(side_effect=lambda m: appended.append(m))

        provider = MagicMock()
        tool_registry = MagicMock()
        tool_registry.schemas = MagicMock(return_value=[])
        tool_registry.reset_run_state = MagicMock()

        context_manager = MagicMock()
        context_manager.prepare = MagicMock(return_value=[])

        loop = AgentLoop(session, provider, tool_registry, context_manager)

        await loop.run()

        # The queued message should be appended with nonce
        queued_msgs = [m for m in appended if m.get("role") == "user"]
        assert len(queued_msgs) == 1
        assert queued_msgs[0]["content"] == "queued hello"
        assert queued_msgs[0]["nonce"] == "drain-nonce"


class TestOnLoopDoneNonce:
    """_on_loop_done queue drain preserves nonce."""

    def test_on_loop_done_drain_includes_nonce(self):
        """_on_loop_done appends queued messages with nonce."""
        from agent_os.daemon_v2.agent_manager import AgentManager

        ws = MagicMock()
        sub_agent_manager = MagicMock()
        sub_agent_manager.list_active = MagicMock(return_value=[])

        mgr = AgentManager(
            project_store=MagicMock(),
            ws_manager=ws,
            sub_agent_manager=sub_agent_manager,
            activity_translator=MagicMock(),
            process_manager=MagicMock(),
        )

        session = MagicMock()
        session.is_stopped.return_value = False
        session._paused_for_approval = False
        session.pop_queued_messages.return_value = [("drain msg", "drain-n1")]

        appended = []
        session.append = MagicMock(side_effect=lambda m: appended.append(m))

        handle = MagicMock()
        handle.session = session
        task = MagicMock()
        task.exception.return_value = None
        handle.task = task

        mgr._handles["proj_d"] = handle

        callback = mgr._on_loop_done("proj_d")
        mock_future = MagicMock()
        with patch("asyncio.ensure_future", return_value=mock_future) as mock_ef:
            callback(task)
            if mock_ef.call_args:
                coro = mock_ef.call_args[0][0]
                coro.close()

        assert len(appended) == 1
        assert appended[0]["content"] == "drain msg"
        assert appended[0]["nonce"] == "drain-n1"


class TestBroadcastNonceEndToEnd:
    """End-to-end: nonce flows from session.append to WS broadcast."""

    def test_nonce_in_user_message_reaches_broadcast(self):
        """When session.append fires with nonce, WS event includes it."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        translator.on_message(
            {"role": "user", "content": "hello", "nonce": "e2e-nonce"},
            "proj_e2e",
        )

        payload = ws.broadcast.call_args[0][1]
        assert payload["type"] == "chat.user_message"
        assert payload["nonce"] == "e2e-nonce"

    def test_missing_nonce_sends_empty_string(self):
        """When nonce is missing from message, broadcast sends empty string."""
        ws = MagicMock()
        translator = ActivityTranslator(ws)

        translator.on_message(
            {"role": "user", "content": "no nonce"},
            "proj_e2e",
        )

        payload = ws.broadcast.call_args[0][1]
        assert payload["nonce"] == ""
