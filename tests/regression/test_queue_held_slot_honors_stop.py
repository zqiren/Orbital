# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression (mandatory interruptibility requirement): while the dispatcher
holds the slot for a continuation-pending session, it holds with NO live loop
task. The hold MUST remain responsive to stop / pause / shutdown, or the queue
becomes un-pausable and un-stoppable — a worse bug than the false-terminate it
fixes.

This guards two of the required interruptibility seams:
  * ``_stop_generation`` bump (user pause/stop via ``dispatcher.stop()``)
  * stored ``QueueRunState.PAUSED`` re-read mid-hold

In both cases the hold must break/yield control and the attempt must be
preserved (NOT closed BLOCKED, queue NOT advanced).
"""

import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState, QueueRunState


def _dispatcher_held(running_sub_agent=True):
    """A QueueDispatcher (via ``__new__``) parked in the slot-hold: a
    verdict-less ``yield_turn`` turn-end with a live sub-agent."""
    d = QueueDispatcher.__new__(QueueDispatcher)

    loop_obj = MagicMock()
    loop_obj.get_completion_state = MagicMock(return_value=("text", None, None))
    loop_obj._loop_exit_path = "yield_turn"
    loop_obj._exit_reason = "text"
    loop_obj._session = MagicMock()
    loop_obj._session.is_stopped = MagicMock(return_value=False)

    parked_task = MagicMock()
    parked_task.done = MagicMock(return_value=True)

    sub_mgr = MagicMock()
    sub_mgr.list_active = MagicMock(
        return_value=[{"status": "running"}] if running_sub_agent else [],
    )

    agent_manager = MagicMock()
    agent_manager.get_loop = MagicMock(return_value=loop_obj)
    agent_manager.get_loop_task = MagicMock(return_value=parked_task)
    agent_manager.get_sub_agent_manager = MagicMock(return_value=sub_mgr)
    agent_manager.inject_system_message = AsyncMock(return_value="delivered")

    d._agent_manager = agent_manager
    d._project_id = "proj_q"
    d._stop_generation = 0
    d._shutting_down = False
    d._max_runtime_seconds = 999
    d.HOLD_POLL_SEC = 0.01
    d._wait_for_loop_done = AsyncMock(return_value=None)

    store = MagicMock()
    store.auto_idle_if_empty.return_value = False
    store_state = MagicMock()
    store_state.state = QueueRunState.RUNNING
    store.load.return_value = store_state
    d._store = store

    d._idle_event = MagicMock()
    d._broadcast_advance = MagicMock()
    d._broadcast_state_changed = MagicMock()
    d._log_attempt_close = MagicMock()
    return d, agent_manager, store


def _assert_attempt_preserved(store, d):
    blocked_closes = [
        c for c in store.close_latest_attempt.call_args_list
        if c.kwargs.get("outcome") == AttemptOutcome.BLOCKED
    ]
    assert not blocked_closes, "paused hold must preserve the attempt (no BLOCKED close)"
    blocked_states = [
        c for c in store.set_item_state.call_args_list
        if ItemState.BLOCKED in c.args
    ]
    assert not blocked_states, "paused hold must not mark the item BLOCKED"
    advanced_blocked = [
        c for c in d._broadcast_advance.call_args_list if "blocked" in c.args
    ]
    assert not advanced_blocked, "paused hold must not advance the queue"


@pytest.mark.asyncio
async def test_held_slot_breaks_on_stop_generation_bump():
    d, agent_manager, store = _dispatcher_held()
    item = MagicMock()
    item.id = "item1"

    task = asyncio.create_task(d._await_and_handle(item, "sess"))
    await asyncio.sleep(0.05)
    assert not task.done(), "should be holding the slot before the stop"

    # User pause/stop bumps the generation counter (dispatcher.stop()).
    d._stop_generation += 1

    # The hold must observe the bump and return — preserving the attempt.
    await asyncio.wait_for(task, timeout=1.0)
    _assert_attempt_preserved(store, d)


@pytest.mark.asyncio
async def test_held_slot_breaks_on_stored_paused():
    d, agent_manager, store = _dispatcher_held()
    item = MagicMock()
    item.id = "item1"

    task = asyncio.create_task(d._await_and_handle(item, "sess"))
    await asyncio.sleep(0.05)
    assert not task.done(), "should be holding the slot before the pause"

    # Queue paused via the store (e.g. a budget/timed pause path) while held —
    # _run's pause check does not execute while control is inside the hold.
    store.load.return_value.state = QueueRunState.PAUSED

    await asyncio.wait_for(task, timeout=1.0)
    _assert_attempt_preserved(store, d)
