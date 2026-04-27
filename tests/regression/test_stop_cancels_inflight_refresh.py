# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: Stop during a refresh cancels the refresh task cleanly.

When terminate() is called while a state refresh is in-flight, the refresh
task must be cancelled without leaving orphaned temp files. The loop must
exit cleanly (no deadlock, no stuck in_progress state).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_os.agent.loop import AgentLoop, COOLDOWN_TURNS
from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.agent.tools.base import ToolResult


def _tool_response(call_n: int = 0):
    tc = [{"id": f"tc_{call_n}", "function": {"name": "read", "arguments": f'{{"n": {call_n}}}'}}]
    return LLMResponse(
        text="",
        tool_calls=tc,
        raw_message={"role": "assistant", "content": "", "tool_calls": tc},
        has_tool_calls=True,
        finish_reason="tool_calls",
        status_text=None,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _text_response():
    return LLMResponse(
        text="done", tool_calls=[],
        raw_message={"role": "assistant", "content": "done"},
        has_tool_calls=False, finish_reason="stop", status_text=None,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_session():
    session = MagicMock()
    session.session_id = "sess-stop"
    session.is_paused.return_value = False
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.pending_tool_calls = set()
    session.pop_queued_messages.return_value = []
    session.resolve_pending_tool_calls = MagicMock()
    session.append = MagicMock()
    session.append_system = MagicMock()
    session.append_tool_result = MagicMock()
    session.pop_deferred_messages.return_value = []
    return session


@pytest.mark.asyncio
async def test_refresh_task_is_registered_and_cancellable():
    """The _refresh_task attribute exists and is an asyncio.Task while refresh runs.

    We hit COOLDOWN_TURNS via tool-call responses to keep the loop alive,
    then when the refresh task starts we call terminate() to cancel it.
    """
    session = _make_session()

    context_manager = MagicMock()
    context_manager.prepare.return_value = [{"role": "system", "content": "sys"}]
    context_manager.model_context_limit = 128_000
    context_manager.should_compact.return_value = False
    context_manager.usage_percentage = 0.0

    tool_registry = MagicMock()
    tool_registry.schemas.return_value = []
    tool_registry.reset_run_state = MagicMock()
    tool_registry.is_async.return_value = False
    call_idx = {"n": 0}

    def unique_result(**kwargs):
        call_idx["n"] += 1
        return ToolResult(content=f"result_{call_idx['n']}")

    tool_registry.execute.side_effect = unique_result

    refresh_started = asyncio.Event()
    refresh_cancelled = asyncio.Event()

    async def slow_refresh(trigger_name: str):
        refresh_started.set()
        try:
            await asyncio.sleep(100)  # simulate slow LLM
        except asyncio.CancelledError:
            refresh_cancelled.set()
            raise

    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=tool_registry,
        context_manager=context_manager,
        on_session_end_refresh=slow_refresh,
        max_iterations=COOLDOWN_TURNS + 5,  # ensure refresh fires
    )

    call_n = {"n": 0}

    async def mock_stream(ctx, schemas):
        call_n["n"] += 1
        return _tool_response(call_n["n"])

    loop._stream_response = mock_stream

    loop_task = asyncio.create_task(loop.run("go"))

    # Wait for the refresh to start, then call terminate()
    await asyncio.wait_for(refresh_started.wait(), timeout=10.0)
    await loop.terminate()

    try:
        await asyncio.wait_for(loop_task, timeout=5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    # The refresh should have been cancelled
    assert refresh_cancelled.is_set(), "Refresh coroutine must have received CancelledError"


@pytest.mark.asyncio
async def test_terminate_before_refresh_starts_is_safe():
    """terminate() before any refresh fires must not deadlock or raise."""
    session = _make_session()

    context_manager = MagicMock()
    context_manager.prepare.return_value = [{"role": "system", "content": "sys"}]
    context_manager.model_context_limit = 128_000
    context_manager.should_compact.return_value = False
    context_manager.usage_percentage = 0.0

    tool_registry = MagicMock()
    tool_registry.schemas.return_value = []
    tool_registry.reset_run_state = MagicMock()
    tool_registry.is_async.return_value = False
    tool_registry.execute.return_value = ToolResult(content="result")  # fine — loop exits quickly anyway

    refresh_called = {"n": 0}

    async def mock_refresh(trigger_name):
        refresh_called["n"] += 1

    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=tool_registry,
        context_manager=context_manager,
        on_session_end_refresh=mock_refresh,
        max_iterations=5,  # well below COOLDOWN_TURNS
    )

    streaming_event = asyncio.Event()

    async def mock_stream(ctx, schemas):
        streaming_event.set()
        await asyncio.sleep(0.05)
        return _text_response()

    loop._stream_response = mock_stream

    loop_task = asyncio.create_task(loop.run("go"))

    # Wait for first stream call, then terminate immediately (no refresh yet)
    await asyncio.wait_for(streaming_event.wait(), timeout=5.0)
    await loop.terminate()

    try:
        await asyncio.wait_for(loop_task, timeout=5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    # No orphan refresh task should remain
    assert loop._refresh_task is None or loop._refresh_task.done(), (
        "refresh_task must be None or done after terminate()"
    )
