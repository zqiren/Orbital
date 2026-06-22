# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: queue verdict is session-level, not turn-level.

A turn that ends to await a dispatched sub-agent ("yield_turn") is NOT a
contract violation — the session has more turns to go (its next turn is
auto-triggered by the sub-agent completion push-back). The dispatcher must
keep the session ACTIVE in its slot, hold the slot, and NOT advance the queue
or mark the item BLOCKED.

Before the fix, ``_await_and_handle`` reads ``_exit_reason=="text"`` (a
``yield_turn`` sets only the diagnostic ``_loop_exit_path``, never
``_exit_reason``) and routes the verdict-less *turn*-end into the corrective
turn -> contract-violation BLOCKED path — terminating the whole *session*.

Direct-drive harness (``__new__``-bypass; stub ``_wait_for_loop_done``); the
fix parks in an indefinite hold, so we run ``_await_and_handle`` as a task and
cancel it after asserting side effects, to avoid hanging.
"""

import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState, QueueRunState


def _dispatcher_with_yield(running_sub_agent=True):
    """A QueueDispatcher (via ``__new__``) wired to drive ``_await_and_handle``
    into a verdict-less ``yield_turn`` turn-end with a live sub-agent."""
    d = QueueDispatcher.__new__(QueueDispatcher)

    loop_obj = MagicMock()
    # Verdict-less turn-end: text exit, but the diagnostic exit path records
    # that the turn parked on a sub-agent dispatch.
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


@pytest.mark.asyncio
async def test_yield_turn_with_live_subagent_keeps_session_active():
    d, agent_manager, store = _dispatcher_with_yield(running_sub_agent=True)
    item = MagicMock()
    item.id = "item1"

    task = asyncio.create_task(d._await_and_handle(item, "proj_q_sess"))
    # Let _await_and_handle classify and enter the slot-hold.
    await asyncio.sleep(0.05)

    try:
        # The session is HELD active — not terminated.
        assert not task.done(), "session should be held active, not terminated"

        # No corrective turn, no BLOCKED close/advance while continuation pending.
        agent_manager.inject_system_message.assert_not_called()

        blocked_closes = [
            c for c in store.close_latest_attempt.call_args_list
            if c.kwargs.get("outcome") == AttemptOutcome.BLOCKED
        ]
        assert not blocked_closes, "must not close attempt BLOCKED while held"

        blocked_states = [
            c for c in store.set_item_state.call_args_list
            if ItemState.BLOCKED in c.args
        ]
        assert not blocked_states, "item must not be marked BLOCKED while held"

        advanced_blocked = [
            c for c in d._broadcast_advance.call_args_list
            if "blocked" in c.args
        ]
        assert not advanced_blocked, "queue must not advance while held"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
