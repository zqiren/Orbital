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
    # Default: not stopped. session.stop() flips is_stopped() to True, same
    # as the real Session, so terminate()'s re-cancel guard can be exercised.
    session._stopped = False
    session.is_stopped.side_effect = lambda: session._stopped
    session.stop.side_effect = lambda: setattr(session, "_stopped", True)
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
    assert msg.startswith("Consolidation scheduled in background")
    assert calls == []                       # returned before the pass ran
    await loop.drain_refresh()
    assert calls == ["agent_decided"]


# ---------------------------------------------------------------------------
# Ack strings must set async expectations (orbital-marketing incident,
# 2026-07-09: the bare acks read as "done or failed" — the agent re-triggered,
# then hand-edited the file while the pass was still in flight).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scheduled_ack_sets_duration_and_persistence_expectations():
    loop = _make_loop([])
    msg = loop.schedule_checkpoint("agent_decided")
    assert "minute" in msg                   # can take a few minutes
    assert "may persist" in msg              # the hygiene flag will linger
    assert "do not re-trigger" in msg.lower() or "no further calls" in msg.lower()
    await loop.drain_refresh()


@pytest.mark.asyncio
async def test_coalesced_ack_says_no_further_calls_needed():
    calls = []
    gate = asyncio.Event()
    loop = _make_loop(calls, gate=gate)
    loop.schedule_checkpoint("agent_decided")
    await asyncio.sleep(0)
    msg = loop.schedule_checkpoint("agent_decided")
    assert "coalesced" in msg
    assert "no further calls" in msg.lower()
    gate.set()
    await loop.drain_refresh()


@pytest.mark.asyncio
async def test_deferred_ack_names_the_debounce_eta():
    calls = []
    loop = _make_loop(calls)
    loop.schedule_checkpoint("agent_decided")
    await loop.drain_refresh()
    msg = loop.schedule_checkpoint("agent_decided")   # within debounce window
    assert "deferred" in msg
    assert "automatically" in msg            # fires on its own — no re-call needed
    assert any(ch.isdigit() for ch in msg)   # a concrete ETA in seconds


# ---------------------------------------------------------------------------
# In-flight + outcome metadata pushed to the context manager, so the hygiene
# flag and status line can render scheduler state.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inflight_metadata_pushed_at_spawn_and_cleared_after():
    calls = []
    gate = asyncio.Event()
    loop = _make_loop(calls, gate=gate)
    ctx = loop._context_manager
    loop._current_iteration = 14

    loop.schedule_checkpoint("agent_decided")
    await asyncio.sleep(0)                   # pass started, blocked on gate
    assert ctx._refresh_in_flight is True
    assert ctx._refresh_in_flight_since_turn == 14

    gate.set()
    await loop.drain_refresh()
    assert ctx._refresh_in_flight is False
    assert ctx._refresh_in_flight_since_turn is None


@pytest.mark.asyncio
async def test_run_refresh_records_callback_outcome():
    outcomes = []

    async def refresh_callback(trigger_name: str) -> str:
        outcomes.append(trigger_name)
        return "backstop_only"

    session = MagicMock()
    session.session_uuid = "sess-outcome"
    session._stopped = False
    session.is_stopped.side_effect = lambda: session._stopped
    ctx = MagicMock()
    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=MagicMock(),
        context_manager=ctx,
        on_session_end_refresh=refresh_callback,
    )
    loop.schedule_checkpoint("agent_decided")
    await loop.drain_refresh()
    assert outcomes == ["agent_decided"]
    assert loop._last_refresh_outcome == "backstop_only"
    assert ctx._last_checkpoint_outcome == "backstop_only"


@pytest.mark.asyncio
async def test_run_refresh_outcome_failed_on_callback_exception():
    async def refresh_callback(trigger_name: str) -> str:
        raise RuntimeError("boom")

    session = MagicMock()
    session.session_uuid = "sess-outcome-fail"
    session._stopped = False
    session.is_stopped.side_effect = lambda: session._stopped
    ctx = MagicMock()
    loop = AgentLoop(
        session=session,
        provider=MagicMock(),
        tool_registry=MagicMock(),
        context_manager=ctx,
        on_session_end_refresh=refresh_callback,
    )
    loop.schedule_checkpoint("agent_decided")
    await loop.drain_refresh()
    assert loop._last_refresh_outcome == "failed"
    assert ctx._last_checkpoint_outcome == "failed"
    assert ctx._refresh_in_flight is False


@pytest.mark.asyncio
async def test_legacy_callback_returning_none_records_no_outcome():
    """Old-style callbacks return None — must not crash or fake an outcome."""
    calls = []
    loop = _make_loop(calls)                 # _make_loop's callback returns None
    loop.schedule_checkpoint("agent_decided")
    await loop.drain_refresh()
    assert loop._last_refresh_outcome is None


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
async def test_spawn_refresh_refuses_when_session_stopped():
    """Reviewer finding (final-review round): terminate()'s cancel_turn()
    await can wake the loop's own task into the turn boundary before
    session.stop() lands, letting it spawn a NEW background pass that
    nothing then cancels. Both seams must refuse once the session is
    stopped: schedule_checkpoint() (the entry used by turn_count /
    agent_decided) and _spawn_refresh() itself (the belt for the
    _maybe_consume_dirty path, which bypasses schedule_checkpoint)."""
    calls = []
    loop = _make_loop(calls)
    loop._session.stop()  # flips is_stopped() -> True, mirrors terminate()

    msg = loop.schedule_checkpoint("agent_decided")
    assert msg == "State refresh unavailable in this session."
    await loop.drain_refresh()
    assert calls == []
    assert loop._refresh_task is None

    # Belt: _spawn_refresh's own guard, exercised directly since
    # _maybe_consume_dirty calls it without going through schedule_checkpoint.
    loop._spawn_refresh("agent_decided_coalesced")
    assert calls == []
    assert loop._refresh_task is None


@pytest.mark.asyncio
async def test_terminate_recancel_closes_spawn_window():
    """Reviewer finding (final-review round): simulate the race where a pass
    gets spawned DURING terminate()'s `await cancel_turn()` — after the
    first `_refresh_task` cancel (which found nothing in flight) and before
    `session.stop()` lands. terminate()'s re-cancel after session.stop()
    must catch and cancel it; nothing should survive terminate()."""
    calls = []
    gate = asyncio.Event()
    loop = _make_loop(calls, gate=gate)
    original_cancel_turn = loop.cancel_turn
    spawned: dict[str, asyncio.Task] = {}

    async def fake_cancel_turn():
        # session.stop() hasn't run yet at this point, so this spawn is not
        # blocked by the is_stopped() guard — it exercises the terminate()
        # re-cancel fix specifically, not the spawn-time guard.
        loop._spawn_refresh("agent_decided_coalesced")
        spawned["task"] = loop._refresh_task
        await asyncio.sleep(0)  # let the pass start and block on the gate
        await original_cancel_turn()

    loop.cancel_turn = fake_cancel_turn

    await loop.terminate()
    await asyncio.sleep(0)

    task = spawned["task"]
    assert task is not None
    assert calls == ["agent_decided_coalesced"]  # it did start...
    assert task.cancelled() or task.done()       # ...but did not survive terminate()
    assert loop._refresh_task is None or loop._refresh_task.done()


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
