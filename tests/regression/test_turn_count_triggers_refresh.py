# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: there is NO turn-count memory trigger any more (spec 089).

The loop used to start a consolidation pass every 50 iterations regardless
of whether memory needed it. The memory editor now runs when a Layer-1 file
is over budget (reported by ContextManager.prepare()) or on an explicit
checkpoint_state. A long session with memory under budget must never start a
pass; the turn counter still runs, because the status line reports it.

We use varying tool call results to avoid repetition-detection kicks.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.session import persist_user_row


def _unique_tool_response(call_n: int):
    """Tool-call response with unique ID to prevent repetition detection."""
    tc = [{"id": f"tc_{call_n}", "function": {"name": "read", "arguments": f'{{"n": {call_n}}}'}}]
    return LLMResponse(
        text="",
        tool_calls=tc,
        raw_message={"role": "assistant", "content": "", "tool_calls": tc},
        has_tool_calls=True,
        finish_reason="tool_calls",
        status_text=None,
        usage=TokenUsage(input_tokens=50, output_tokens=10),
    )


def _text_response(text="Done."):
    return LLMResponse(
        text=text,
        tool_calls=[],
        raw_message={"role": "assistant", "content": text},
        has_tool_calls=False,
        finish_reason="stop",
        status_text=None,
        usage=TokenUsage(input_tokens=50, output_tokens=10),
    )


def _make_session():
    session = MagicMock()
    session.session_id = "sess-tc"
    session.session_uuid = "sess-tc"
    session.is_paused.return_value = False
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.pending_tool_calls = set()
    session.pop_queued_messages.return_value = []
    session.resolve_pending_tool_calls = MagicMock()
    session.append = MagicMock()
    session.append_system = MagicMock()
    session.append_tool_result = MagicMock()
    session.recent_activity.return_value = []
    session.get_messages.return_value = []
    session.pop_deferred_messages.return_value = []
    return session


def _make_context_manager():
    context_manager = MagicMock()
    context_manager.prepare.return_value = [{"role": "system", "content": "sys"}]
    context_manager.model_context_limit = 128_000
    context_manager.should_compact.return_value = False
    context_manager.usage_percentage = 0.0
    return context_manager


def _make_tool_registry():
    """Tool registry where execute returns unique result per call to avoid repetition lock."""
    call_idx = {"n": 0}

    def unique_result(**kwargs):
        call_idx["n"] += 1
        return ToolResult(content=f"result_{call_idx['n']}")

    registry = MagicMock()
    registry.schemas.return_value = []
    registry.reset_run_state = MagicMock()
    registry.is_async.return_value = False
    registry.execute.side_effect = unique_result
    return registry


@pytest.mark.asyncio
async def test_a_long_session_never_fires_a_pass_on_turn_count():
    session = _make_session()
    refresh_calls = []

    async def refresh(trigger_name):
        refresh_calls.append(trigger_name)

    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=_make_tool_registry(),
        context_manager=_make_context_manager(),
        on_session_end_refresh=refresh,
        max_iterations=120,
    )
    calls = {"n": 0}

    async def mock_stream(ctx, schemas):
        calls["n"] += 1
        if calls["n"] > 110:
            return _text_response()
        return _unique_tool_response(calls["n"])

    loop._stream_response = mock_stream
    persist_user_row(loop._session, "go")
    await asyncio.wait_for(loop.run(), timeout=30)
    await loop.drain_refresh()

    assert calls["n"] > 100
    assert refresh_calls == [], f"no pass should fire on turn count: {refresh_calls}"
    # The counter still runs for the status line.
    assert loop._turns_since_last_update > 100


@pytest.mark.asyncio
async def test_an_over_budget_report_fires_exactly_one_pass():
    session = _make_session()
    refresh_calls = []

    async def refresh(trigger_name):
        refresh_calls.append(trigger_name)

    cm = _make_context_manager()

    def _prepare():
        cm._on_memory_over_budget(["state"])
        return [{"role": "system", "content": "sys"}]

    cm.prepare.side_effect = _prepare
    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=_make_tool_registry(),
        context_manager=cm,
        on_session_end_refresh=refresh,
        max_iterations=30,
    )
    calls = {"n": 0}

    async def mock_stream(ctx, schemas):
        calls["n"] += 1
        if calls["n"] > 20:
            return _text_response()
        return _unique_tool_response(calls["n"])

    loop._stream_response = mock_stream
    persist_user_row(loop._session, "go")
    await asyncio.wait_for(loop.run(), timeout=30)
    await loop.drain_refresh()

    # Reported on every prepare(), but single-flight + debounce → one pass.
    assert refresh_calls == ["over_budget"]
