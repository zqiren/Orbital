# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: cooldown prevents back-to-back refreshes.

After an agent-decided refresh at turn N, turns_since_last_update resets to 0
and another full COOLDOWN_TURNS must elapse before the next turn-count refresh.

Over 2 * COOLDOWN_TURNS iterations we expect exactly 2 refreshes.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_os.agent.loop import AgentLoop, COOLDOWN_TURNS
from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.agent.tools.base import ToolResult


def _unique_tool_response(call_n: int):
    """Unique tool call to avoid repetition detection exit."""
    tc = [{"id": f"tc_{call_n}", "function": {"name": "read", "arguments": f'{{"n": {call_n}}}'}}]
    return LLMResponse(
        text="", tool_calls=tc,
        raw_message={"role": "assistant", "content": "", "tool_calls": tc},
        has_tool_calls=True, finish_reason="tool_calls", status_text=None,
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
    session.session_id = "sess-cooldown"
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


def _make_context_manager():
    context_manager = MagicMock()
    context_manager.prepare.return_value = [{"role": "system", "content": "sys"}]
    context_manager.model_context_limit = 128_000
    context_manager.should_compact.return_value = False
    context_manager.usage_percentage = 0.0
    return context_manager


def _make_tool_registry():
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
async def test_cooldown_resets_after_each_refresh():
    """turns_since_last_update resets to 0 after every refresh regardless of trigger."""
    session = _make_session()
    context_manager = _make_context_manager()
    tool_registry = _make_tool_registry()

    # Run exactly 2 * COOLDOWN_TURNS iterations; expect exactly 2 refreshes
    total_iters = 2 * COOLDOWN_TURNS
    refresh_count = {"n": 0}

    async def mock_refresh(trigger_name):
        refresh_count["n"] += 1

    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=tool_registry,
        context_manager=context_manager,
        on_session_end_refresh=mock_refresh,
        max_iterations=total_iters,
    )

    call_n = {"n": 0}

    async def mock_stream(context, tool_schemas):
        call_n["n"] += 1
        if call_n["n"] >= total_iters:
            return _text_response()
        return _unique_tool_response(call_n["n"])

    loop._stream_response = mock_stream
    await loop.run("go")

    # Two full COOLDOWN_TURNS windows should have produced exactly 2 refreshes
    assert refresh_count["n"] == 2, (
        f"Expected 2 refreshes over {total_iters} turns, got {refresh_count['n']}"
    )


@pytest.mark.asyncio
async def test_turns_since_last_update_not_cooled_down_when_below_threshold():
    """Before COOLDOWN_TURNS, no refresh fires."""
    session = _make_session()
    context_manager = _make_context_manager()
    tool_registry = _make_tool_registry()

    refresh_count = {"n": 0}

    async def mock_refresh(trigger_name):
        refresh_count["n"] += 1

    # Only run COOLDOWN_TURNS - 1 iterations → no refresh should fire
    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=tool_registry,
        context_manager=context_manager,
        on_session_end_refresh=mock_refresh,
        max_iterations=COOLDOWN_TURNS - 1,
    )

    call_n = {"n": 0}

    async def mock_stream(context, tool_schemas):
        call_n["n"] += 1
        if call_n["n"] >= COOLDOWN_TURNS - 1:
            return _text_response()
        return _unique_tool_response(call_n["n"])

    loop._stream_response = mock_stream
    await loop.run("go")

    assert refresh_count["n"] == 0, (
        f"Expected no refresh before COOLDOWN_TURNS ({COOLDOWN_TURNS}), "
        f"but got {refresh_count['n']} at {COOLDOWN_TURNS - 1} turns"
    )
