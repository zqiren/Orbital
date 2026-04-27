# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: turn-count trigger fires after COOLDOWN_TURNS without manual /new.

16 loop iterations with no manual /new call must result in the refresh
callback being invoked (turn-count trigger fires at turn 15). The check
is independent of agent_manager wiring — we test the loop directly.

We use varying tool call results to avoid repetition-detection kicks.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_os.agent.loop import AgentLoop, COOLDOWN_TURNS
from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.agent.tools.base import ToolResult


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
async def test_turn_count_fires_after_cooldown_turns():
    """After COOLDOWN_TURNS iterations the turn-count trigger fires."""
    session = _make_session()
    context_manager = _make_context_manager()
    tool_registry = _make_tool_registry()

    refresh_call_count = {"n": 0}

    async def mock_refresh(trigger_name: str):
        refresh_call_count["n"] += 1

    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=tool_registry,
        context_manager=context_manager,
        on_session_end_refresh=mock_refresh,
        max_iterations=COOLDOWN_TURNS + 1,  # 16 iterations
    )

    call_count = {"n": 0}

    async def mock_stream(context, tool_schemas):
        call_count["n"] += 1
        if call_count["n"] > COOLDOWN_TURNS:
            # End loop on iteration 16
            return _text_response("All done")
        # Use unique tool calls to keep the loop alive across iterations
        return _unique_tool_response(call_count["n"])

    loop._stream_response = mock_stream

    await loop.run("Start task")

    # Refresh must have fired at least once (turn-count trigger at turn 15)
    assert refresh_call_count["n"] >= 1, (
        f"Expected refresh to fire after {COOLDOWN_TURNS} turns, "
        f"but it fired {refresh_call_count['n']} times."
    )


@pytest.mark.asyncio
async def test_turns_since_last_update_resets_after_refresh():
    """After a turn-count refresh, turns_since_last_update resets to 0."""
    session = _make_session()
    context_manager = _make_context_manager()
    tool_registry = _make_tool_registry()

    async def mock_refresh(trigger_name: str):
        pass  # refresh fires, resetting the counter

    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=tool_registry,
        context_manager=context_manager,
        on_session_end_refresh=mock_refresh,
        max_iterations=COOLDOWN_TURNS + 1,
    )

    call_n = {"n": 0}

    async def mock_stream(context, tool_schemas):
        call_n["n"] += 1
        if call_n["n"] > COOLDOWN_TURNS:
            return _text_response("done")
        return _unique_tool_response(call_n["n"])

    loop._stream_response = mock_stream
    await loop.run("go")

    # After refresh at iteration 15, counter resets to 0, then increments by 1 at iter 16
    assert loop._turns_since_last_update <= COOLDOWN_TURNS, (
        f"Counter should reset after refresh, got {loop._turns_since_last_update}"
    )
