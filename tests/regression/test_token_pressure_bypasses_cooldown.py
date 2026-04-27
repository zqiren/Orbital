# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: token-pressure trigger fires before compaction even inside cooldown.

Even if a refresh just happened (turns_since_last_update = 0), the token-pressure
trigger must still fire before should_compact() is actioned, because data
preservation always beats redundancy at compaction boundaries.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.loop import AgentLoop, COOLDOWN_TURNS
from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.agent.tools.base import ToolResult


_tc_counter = {"n": 0}


def _tool_response(tool_name="read"):
    """Tool-call response — keeps the loop running past tool execution."""
    _tc_counter["n"] += 1
    tc = [{"id": f"tc_{tool_name}_{_tc_counter['n']}", "function": {"name": tool_name, "arguments": f'{{"n": {_tc_counter["n"]}}}'}}]
    return LLMResponse(
        text="",
        tool_calls=tc,
        raw_message={"role": "assistant", "content": "", "tool_calls": tc},
        has_tool_calls=True,
        finish_reason="tool_calls",
        status_text=None,
        usage=TokenUsage(input_tokens=50, output_tokens=10),
    )


def _text_response(text="ok"):
    return LLMResponse(
        text=text,
        tool_calls=[],
        raw_message={"role": "assistant", "content": text},
        has_tool_calls=False,
        finish_reason="stop",
        status_text=None,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_session():
    session = MagicMock()
    session.session_id = "sess-tp"
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
async def test_token_pressure_fires_before_compaction_within_cooldown():
    """Token-pressure refresh fires even when turns_since_last_update is 0."""
    session = _make_session()

    # should_compact returns True on first check, False thereafter
    compact_calls = {"n": 0}

    def should_compact_side():
        compact_calls["n"] += 1
        return compact_calls["n"] == 1

    context_manager = MagicMock()
    context_manager.prepare.return_value = [{"role": "system", "content": "sys"}]
    context_manager.model_context_limit = 128_000
    context_manager.should_compact.side_effect = should_compact_side
    context_manager.usage_percentage = 0.85

    call_idx = {"n": 0}

    def unique_result(**kwargs):
        call_idx["n"] += 1
        return ToolResult(content=f"result_{call_idx['n']}")

    tool_registry = MagicMock()
    tool_registry.schemas.return_value = []
    tool_registry.reset_run_state = MagicMock()
    tool_registry.is_async.return_value = False
    tool_registry.execute.side_effect = unique_result

    call_order = []

    async def mock_refresh(trigger_name: str):
        call_order.append(f"refresh:{trigger_name}")

    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=tool_registry,
        context_manager=context_manager,
        on_session_end_refresh=mock_refresh,
    )

    # Simulate that a refresh already happened — cooldown active (counter at 0)
    loop._turns_since_last_update = 0

    call_n = {"n": 0}

    async def mock_stream(context, tool_schemas):
        call_n["n"] += 1
        if call_n["n"] == 1:
            # First call: tool response (compaction check runs after tool execution)
            return _tool_response()
        # Second call (post-compaction): text response to end loop
        return _text_response("done")

    loop._stream_response = mock_stream

    async def mock_compact_run(sess, prov, utility_provider=None):
        call_order.append("compaction")

    with patch("agent_os.agent.compaction.run", new=mock_compact_run):
        with patch("agent_os.agent.compaction.inject_reorientation"):
            await loop.run("start")

    # Token-pressure refresh must have fired before compaction
    assert "refresh:token_pressure" in call_order, (
        f"Expected token-pressure refresh in call order: {call_order}"
    )
    assert "compaction" in call_order, (
        f"Expected compaction in call order: {call_order}"
    )
    tp_idx = call_order.index("refresh:token_pressure")
    comp_idx = call_order.index("compaction")
    assert tp_idx < comp_idx, (
        f"Token-pressure refresh must fire BEFORE compaction. "
        f"Order was: {call_order}"
    )


@pytest.mark.asyncio
async def test_token_pressure_fires_independently_of_turn_count():
    """Token-pressure fires regardless of turns_since_last_update count."""
    session = _make_session()

    fired_triggers = []

    compact_calls = {"n": 0}

    def should_compact():
        compact_calls["n"] += 1
        return compact_calls["n"] == 1

    context_manager = MagicMock()
    context_manager.prepare.return_value = [{"role": "system", "content": "sys"}]
    context_manager.model_context_limit = 128_000
    context_manager.should_compact.side_effect = should_compact
    context_manager.usage_percentage = 0.85

    call_idx2 = {"n": 0}

    def unique_result2(**kwargs):
        call_idx2["n"] += 1
        return ToolResult(content=f"result_{call_idx2['n']}")

    tool_registry = MagicMock()
    tool_registry.schemas.return_value = []
    tool_registry.reset_run_state = MagicMock()
    tool_registry.is_async.return_value = False
    tool_registry.execute.side_effect = unique_result2

    async def mock_refresh(trigger_name):
        fired_triggers.append(trigger_name)

    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=tool_registry,
        context_manager=context_manager,
        on_session_end_refresh=mock_refresh,
        max_iterations=3,
    )
    # turns_since_last_update starts at 0 → turn-count cannot fire at turn 1
    # but token-pressure must still fire after compaction check
    assert loop._turns_since_last_update == 0

    call_n = {"n": 0}

    async def mock_stream(ctx, schemas):
        call_n["n"] += 1
        if call_n["n"] == 1:
            return _tool_response()
        return _text_response("ok")

    loop._stream_response = mock_stream

    async def mock_compact_run(sess, prov, utility_provider=None):
        pass

    with patch("agent_os.agent.compaction.run", new=mock_compact_run):
        with patch("agent_os.agent.compaction.inject_reorientation"):
            await loop.run("go")

    assert "token_pressure" in fired_triggers, (
        f"Token-pressure trigger should fire regardless of turn count, "
        f"but fired triggers were: {fired_triggers}"
    )
    # Turn-count must NOT fire at turn 1 (below COOLDOWN_TURNS)
    # (it would fire at turn 1 only if turns_since_last_update >= COOLDOWN_TURNS)
    turn_count_fired = fired_triggers.count("turn_count")
    assert turn_count_fired == 0, (
        f"Turn-count trigger should NOT fire at turn 1 (below COOLDOWN_TURNS={COOLDOWN_TURNS}), "
        f"but fired triggers were: {fired_triggers}"
    )
