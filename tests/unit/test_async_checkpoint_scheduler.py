# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the AgentLoop async checkpoint scheduler (spec 013).

New invariant: agent_decided / turn_count refreshes NEVER block the loop.
One single-flight gate (_refresh_task) + dirty bit + monotonic debounce.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent_os.agent.loop import AgentLoop, REFRESH_DEBOUNCE_S


def _make_loop(refresh_calls, *, gate: asyncio.Event | None = None):
    """Minimal AgentLoop with a recording on_session_end_refresh callback.

    ``gate``: when given, the callback blocks until the event is set —
    lets a test hold a pass 'in flight' deterministically.
    """
    async def refresh_callback(trigger_name: str) -> None:
        refresh_calls.append(trigger_name)
        if gate is not None:
            await gate.wait()

    session = MagicMock()
    session.session_uuid = "sess-sched-test"
    ctx = MagicMock()
    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=MagicMock(),
        context_manager=ctx,
        on_session_end_refresh=refresh_callback,
    )
    return loop


@pytest.mark.asyncio
async def test_schedule_spawns_background_pass_and_returns_instantly():
    calls = []
    loop = _make_loop(calls)
    msg = loop.schedule_checkpoint("agent_decided")
    assert msg == "Consolidation scheduled in background."
    assert calls == []                       # returned before the pass ran
    await loop.drain_refresh()
    assert calls == ["agent_decided"]


@pytest.mark.asyncio
async def test_inflight_pass_coalesces_second_trigger():
    calls = []
    gate = asyncio.Event()
    loop = _make_loop(calls, gate=gate)
    loop.schedule_checkpoint("agent_decided")
    await asyncio.sleep(0)                   # let the pass start and block
    msg = loop.schedule_checkpoint("agent_decided")
    assert "coalesced" in msg
    assert loop._refresh_dirty is True
    gate.set()
    await loop.drain_refresh()
    assert calls == ["agent_decided"]        # exactly ONE pass ran


@pytest.mark.asyncio
async def test_debounce_defers_and_sets_dirty():
    calls = []
    loop = _make_loop(calls)
    loop.schedule_checkpoint("agent_decided")
    await loop.drain_refresh()
    msg = loop.schedule_checkpoint("agent_decided")   # within debounce window
    assert "deferred" in msg
    assert loop._refresh_dirty is True
    assert calls == ["agent_decided"]        # no second pass


@pytest.mark.asyncio
async def test_spawn_clears_dirty_and_resets_turn_counter():
    calls = []
    loop = _make_loop(calls)
    loop._refresh_dirty = True
    loop._turns_since_last_update = 99
    loop.schedule_checkpoint("turn_count")
    assert loop._refresh_dirty is False      # fresh pass covers prior triggers
    assert loop._turns_since_last_update == 0  # no refire churn while in flight
    await loop.drain_refresh()
    assert calls == ["turn_count"]


@pytest.mark.asyncio
async def test_maybe_consume_dirty_fires_after_debounce_elapses():
    calls = []
    loop = _make_loop(calls)
    loop.schedule_checkpoint("agent_decided")
    await loop.drain_refresh()
    loop._refresh_dirty = True
    loop._maybe_consume_dirty()              # debounce NOT elapsed → no-op
    assert loop._refresh_dirty is True       # unchanged — nothing fired
    assert calls == ["agent_decided"]
    loop._last_merge_at -= (REFRESH_DEBOUNCE_S + 1)   # simulate elapse
    loop._maybe_consume_dirty()
    assert loop._refresh_dirty is False
    await loop.drain_refresh()
    assert calls == ["agent_decided", "agent_decided_coalesced"]


@pytest.mark.asyncio
async def test_terminate_cancels_inflight_pass():
    calls = []
    gate = asyncio.Event()
    loop = _make_loop(calls, gate=gate)
    loop.schedule_checkpoint("agent_decided")
    await asyncio.sleep(0)
    task = loop._refresh_task
    assert task is not None and not task.done()
    await loop.terminate()
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_schedule_without_callback_is_safe():
    session = MagicMock()
    session.session_uuid = "sess-nocb"
    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=MagicMock(),
        context_manager=MagicMock(),
        on_session_end_refresh=None,
    )
    msg = loop.schedule_checkpoint("agent_decided")
    assert "unavailable" in msg
    await loop.drain_refresh()               # no-op, must not raise
