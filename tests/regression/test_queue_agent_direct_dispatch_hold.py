# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 079 §3.2 verification gate — the slot-hold on a no-first-turn dispatch.

A queue item that carries an ``agent`` is dispatched STRAIGHT to that worker
(``SubAgentManager.send(initiator="queue_item")``); the management loop never
takes a first turn. The dispatcher must still hold the slot until the worker's
terminal event wakes the manager, then classify the manager's verdict.

``_hold_slot_for_continuation`` was written for a DIFFERENT shape — "the manager
yielded its turn, the worker is live" — where a parked loop task always exists.
This suite is the gate the spec demands: it separates what the hold's BODY
already covers (it is agnostic about the parked task; probe tests below) from
the ENTRY condition, which keys on ``_loop_exit_path == "yield_turn"`` and
therefore does not fire when no manager turn ever ran.

Harness: ``__new__``-bypass direct drive, mirroring
``test_queue_yield_keeps_session_active.py``.
"""

import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState, QueueRunState


def _bare_dispatcher():
    """A ``QueueDispatcher`` with the fields the hold / handler paths read."""
    d = QueueDispatcher.__new__(QueueDispatcher)
    d._project_id = "proj_q"
    d._stop_generation = 0
    d._shutting_down = False
    d._max_runtime_seconds = 999
    d.HOLD_POLL_SEC = 0.01

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
    return d


def _no_turn_yet_manager(*, worker_running=True):
    """An AgentManager double for a session whose management loop NEVER ran.

    ``get_loop`` / ``get_loop_task`` both answer ``None`` — the state the direct
    worker dispatch leaves behind — while the sub-agent reports ``running``.
    """
    sub_mgr = MagicMock()
    sub_mgr.list_active = MagicMock(
        return_value=[{"status": "running"}] if worker_running else [],
    )
    agent_manager = MagicMock()
    agent_manager.get_loop = MagicMock(return_value=None)
    agent_manager.get_loop_task = MagicMock(return_value=None)
    agent_manager.get_sub_agent_manager = MagicMock(return_value=sub_mgr)
    agent_manager.inject_system_message = AsyncMock(return_value="delivered")
    return agent_manager, sub_mgr


# ---------------------------------------------------------------------------
# Probe: does the hold's BODY cope with a parked task that is None?
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hold_body_survives_a_loop_that_has_never_run():
    """The ``cancelled`` read must not explode when there is no loop object.

    ``_hold_slot_for_continuation`` reads ``loop_obj._exit_reason`` and
    ``loop_obj._session.is_stopped()``. On a no-first-turn dispatch ``get_loop``
    answers ``None`` for the whole hold, so the guard has to hold up.
    """
    d = _bare_dispatcher()
    d._agent_manager, _ = _no_turn_yet_manager()
    d._max_runtime_seconds = 0.05  # let the backstop end the hold

    outcome = await d._hold_slot_for_continuation(MagicMock(), "sess", 0)
    assert outcome == "timeout"


@pytest.mark.asyncio
async def test_hold_body_resumes_when_the_wake_installs_the_first_turn():
    """Next-turn detection compares identity against the parked task.

    Parked is ``None`` here (no first turn). The wake
    (``inject_system_message`` -> ``start_agent``) installs a task, and
    ``cur is not parked_task`` must read that as the session's next turn.
    """
    d = _bare_dispatcher()
    agent_manager, _ = _no_turn_yet_manager()
    d._agent_manager = agent_manager

    woken_task = MagicMock()
    calls = {"n": 0}

    def _task_after_wake(*_a, **_kw):
        calls["n"] += 1
        return woken_task if calls["n"] > 2 else None

    agent_manager.get_loop_task = MagicMock(side_effect=_task_after_wake)

    outcome = await d._hold_slot_for_continuation(MagicMock(), "sess", 0)
    assert outcome == "resume"


@pytest.mark.asyncio
async def test_hold_body_still_honours_pause_with_no_parked_task():
    d = _bare_dispatcher()
    d._agent_manager, _ = _no_turn_yet_manager()
    d._stop_generation = 7  # differs from gen_at_start below

    outcome = await d._hold_slot_for_continuation(MagicMock(), "sess", 0)
    assert outcome == "paused"


# ---------------------------------------------------------------------------
# Gate: does the ENTRY condition fire when no manager turn ever ran?
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_first_turn_dispatch_enters_the_hold_not_the_corrective_turn():
    """THE gate (spec 079 §3.2).

    ``_await_and_handle`` on a direct worker dispatch: no loop, no loop task, a
    live worker. It must hold the slot. Today it reads ``("text", None, None)``
    from ``_read_completion(None)``, finds no ``yield_turn`` exit path, and
    walks straight into the corrective turn -> contract-violation BLOCKED path,
    terminating an item whose worker is still working.
    """
    d = _bare_dispatcher()
    agent_manager, _ = _no_turn_yet_manager()
    d._agent_manager = agent_manager
    d._wait_for_loop_done = AsyncMock(return_value=None)

    item = MagicMock()
    item.id = "item1"
    item.agent = "codex"

    task = asyncio.create_task(
        d._await_and_handle(item, "proj_q_sess", awaiting_worker=True),
    )
    await asyncio.sleep(0.05)

    try:
        assert not task.done(), (
            "the slot must stay held while the worker runs, not terminate"
        )
        agent_manager.inject_system_message.assert_not_called()

        blocked_closes = [
            c for c in d._store.close_latest_attempt.call_args_list
            if c.kwargs.get("outcome") == AttemptOutcome.BLOCKED
        ]
        assert not blocked_closes, "must not close the attempt BLOCKED while held"

        blocked_states = [
            c for c in d._store.set_item_state.call_args_list
            if ItemState.BLOCKED in c.args
        ]
        assert not blocked_states, "item must not be marked BLOCKED while held"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_no_first_turn_hold_classifies_the_managers_wake_verdict():
    """After the wake, the dispatcher re-awaits and classifies as usual.

    The manager's wake turn calls ``mark_task_complete``; the item must reach
    DONE through the ordinary ``complete`` branch — the hold only defers the
    classification, it never replaces it.
    """
    d = _bare_dispatcher()
    agent_manager, _ = _no_turn_yet_manager()
    d._agent_manager = agent_manager
    d._wait_for_loop_done = AsyncMock(return_value=None)

    woken_loop = MagicMock()
    woken_loop.get_completion_state = MagicMock(
        return_value=("complete", "worker did it", None),
    )
    woken_task = MagicMock()
    woken_task.done = MagicMock(return_value=True)

    polls = {"n": 0}

    def _task_after_wake(*_a, **_kw):
        polls["n"] += 1
        return woken_task if polls["n"] > 2 else None

    def _loop_after_wake(*_a, **_kw):
        return woken_loop if polls["n"] > 2 else None

    agent_manager.get_loop_task = MagicMock(side_effect=_task_after_wake)
    agent_manager.get_loop = MagicMock(side_effect=_loop_after_wake)

    item = MagicMock()
    item.id = "item1"
    item.agent = "codex"

    await asyncio.wait_for(
        d._await_and_handle(item, "proj_q_sess", awaiting_worker=True),
        timeout=5,
    )

    d._store.close_latest_attempt.assert_called_once()
    assert (
        d._store.close_latest_attempt.call_args.kwargs["outcome"]
        == AttemptOutcome.COMPLETED
    )
    d._store.set_item_state.assert_called_with("item1", ItemState.DONE)


@pytest.mark.asyncio
async def test_no_first_turn_hold_blocks_the_item_when_the_wake_never_comes():
    """Backstop: a worker that dies without a terminal event must not pin the
    queue head forever — the hold expires and the item blocks visibly."""
    d = _bare_dispatcher()
    agent_manager, _ = _no_turn_yet_manager()
    d._agent_manager = agent_manager
    d._wait_for_loop_done = AsyncMock(return_value=None)
    d._max_runtime_seconds = 0.05

    item = MagicMock()
    item.id = "item1"
    item.agent = "codex"

    await asyncio.wait_for(
        d._await_and_handle(item, "proj_q_sess", awaiting_worker=True),
        timeout=5,
    )

    assert (
        d._store.close_latest_attempt.call_args.kwargs["block_reason_code"]
        == "hold_deadline"
    )
    d._store.set_item_state.assert_called_with("item1", ItemState.BLOCKED)
