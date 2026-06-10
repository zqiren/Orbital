# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Blocker B3: the dispatcher drain loop must survive a pause/resume cycle.

Observed live (2026-06-08 queue stress run, proj_82bad3b9443c): a 6-item queue
wedged after ONE pause/resume cycle. Items 1-2 drained; a /queue/stop (pause)
mid-item-2 followed by /queue/resume let item 2 finish — and then the queue
STOPPED ADVANCING (items 3-6 stuck queued, 0 attempts, queue.state=running,
loop dead). A manual resume kick did nothing, proving the dispatcher's main
``_run()`` task was DEAD, not merely idle.

Root cause: pause()'s ``terminate()`` does ``task.cancel()`` on the live
agent-loop task. ``_wait_for_loop_done`` awaits ``asyncio.shield(task)``, so the
cancellation surfaces there as ``CancelledError`` and is re-raised. It then
propagates UP through ``_await_and_handle``'s ``asyncio.wait_for(...)`` (which
only catches ``TimeoutError``), skipping the line-793 pause guard entirely, into
``_dispatch_one`` and finally ``_run`` — whose ``except CancelledError: return``
permanently kills the drain loop. resume() only spawns a SEPARATE task for the
parked item; it never restarts ``_run``, so once that item completes nothing
pulls the next queued item.

These tests lock in the two-layer fix:
  1. Root cause — a pause-induced CancelledError (loop-task cancelled by
     terminate(), _stop_generation bumped) must be treated like the existing
     pause guard: clean return from _await_and_handle, attempt preserved, and
     _run keeps running. Only a genuine dispatcher shutdown lets the
     CancelledError propagate to kill _run.
  2. Safety net — _ensure_run_task() idempotently recreates a dead _run task
     (and is a no-op on a live one); resume()/start() call it so a dead loop
     self-heals.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore


# ----------------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------------

class _FakeLoop:
    def __init__(self):
        self._exit_reason = "complete"
        self._exit_summary = "ok"
        self._exit_block_reason = None
        self._queue_state = "running"

    def get_completion_state(self):
        return (self._exit_reason, self._exit_summary, self._exit_block_reason)


class _PausingManager:
    """Fake AgentManager reproducing the live B3 sequence.

    The first dispatched item's loop task is CANCELLED out from under
    ``_wait_for_loop_done`` (exactly what ``terminate()`` does on pause). The
    test bumps ``_stop_generation`` first (as ``stop()`` does), so the
    dispatcher sees a pause-cancellation. Subsequent items complete normally —
    the assertion is that the drain loop survived and dispatched them.
    """

    def __init__(self):
        self._loop = _FakeLoop()
        self._sid = 0
        self._pending_sid: str | None = None
        self._task: asyncio.Task | None = None
        self.inject_count = 0
        # When True, the NEXT dispatched loop task is cancelled mid-flight to
        # simulate pause's terminate(); the dispatcher under test bumps
        # _stop_generation before this fires.
        self.cancel_next_loop = False
        self.dispatcher: QueueDispatcher | None = None

    def is_onboarding_complete(self, project_id):
        return True

    def get_loop(self, project_id, *, session_id=None):
        return self._loop

    def get_loop_task(self, project_id, *, session_id=None):
        return self._task

    def current_holder_session_id(self, project_id):
        return None

    def get_sub_agent_manager(self):
        return None

    async def new_session(self, project_id, *, session_id=None):
        self._sid += 1
        sid = f"sess_{self._sid}"
        self._pending_sid = sid
        return {"status": "ok", "session_id": sid, "session_uuid": f"proj_{self._sid:08d}"}

    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        self.inject_count += 1
        if self.cancel_next_loop:
            # Simulate pause: make the loop a long-running task, then (on a
            # later tick, AFTER _await_and_handle has snapshotted gen_at_start)
            # bump _stop_generation and cancel the task — exactly stop()'s
            # order: bump _stop_generation, then terminate() → task.cancel().
            self.cancel_next_loop = False

            async def _gets_cancelled():
                await asyncio.sleep(3600)

            self._task = asyncio.create_task(_gets_cancelled())

            async def _do_cancel():
                await asyncio.sleep(0.05)
                self.dispatcher._stop_generation += 1
                self._task.cancel()

            asyncio.create_task(_do_cancel())
        else:
            self._loop._exit_reason = "complete"
            self._loop._exit_summary = "ok"

            async def _instant():
                return None

            self._task = asyncio.create_task(_instant())
        return "delivered"


async def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.02):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


# ----------------------------------------------------------------------------
# B3 end-to-end reproduction at unit level
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_survives_pause_cancellation_and_keeps_draining(tmp_path):
    """The headline B3 repro: pausing (cancelling the live loop task) mid-item
    must NOT kill _run. A subsequently-queued item must still get dispatched."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("item that gets paused")
    store.add_item("item that must still drain")

    mgr = _PausingManager()
    dispatcher = QueueDispatcher(project_id="proj_b3", store=store, agent_manager=mgr)
    mgr.dispatcher = dispatcher
    dispatcher.IDLE_WAIT_TIMEOUT_SEC = 0.05
    dispatcher.LOOP_WAIT_POLL_SEC = 0.05

    # Arm the pause-cancellation on the FIRST dispatched item.
    mgr.cancel_next_loop = True

    await dispatcher.start()
    dispatcher.notify_new_item()

    # The first item's loop gets cancelled (pause). The dispatcher must take the
    # pause-guard early return — NOT propagate CancelledError into _run.
    # Wait for _run to have processed the pause and be idling, then assert it
    # is STILL ALIVE.
    await asyncio.sleep(0.3)
    assert dispatcher._task is not None and not dispatcher._task.done(), (
        "B3: _run() must survive a pause-induced loop cancellation — it was "
        "killed by the propagating CancelledError"
    )
    # First item's attempt was preserved (not closed) — it is still RUNNING.
    items = {it.content: it for it in store.load().items}
    paused_item = items["item that gets paused"]
    assert paused_item.state == ItemState.RUNNING, (
        "paused item's attempt must be preserved (state RUNNING, no close/advance)"
    )

    # Now simulate resume: the live agent loop "finishes" the parked item and
    # the queue should advance to drain the second item.
    mgr._loop._exit_reason = "complete"
    # Complete the parked attempt directly through the store (mirrors what a
    # resumed loop completing does), then flip back to running and kick.
    store.close_latest_attempt(paused_item.id, outcome=__import__(
        "agent_os.queue.models", fromlist=["AttemptOutcome"]).AttemptOutcome.COMPLETED, summary="done")
    store.set_item_state(paused_item.id, ItemState.DONE)
    store.set_queue_state(QueueRunState.RUNNING)
    dispatcher.notify_new_item()

    drained = await _wait_until(
        lambda: store.load().items[1].state == ItemState.DONE
    )
    await dispatcher.shutdown()

    assert drained, (
        "B3: after the pause/resume cycle the queue must CONTINUE to the next "
        "item; it stayed wedged — final: "
        + ", ".join(f"{it.content}={it.state.value}" for it in store.load().items)
    )


# ----------------------------------------------------------------------------
# Unit: _await_and_handle clean-returns on a pause-cancellation
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_await_and_handle_clean_return_on_pause_cancel(tmp_path):
    """_await_and_handle must swallow a CancelledError when _stop_generation
    was bumped (a pause), returning cleanly and preserving the attempt — no
    close, no advance, no rotation, and no exception propagated."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("paused mid-attempt")
    store.set_item_state(item.id, ItemState.RUNNING)
    from agent_os.queue.models import AttemptRecord
    store.append_attempt(item.id, AttemptRecord(session_id="sess_x"))

    mgr = _PausingManager()
    dispatcher = QueueDispatcher(project_id="proj_unit", store=store, agent_manager=mgr)
    dispatcher.LOOP_WAIT_POLL_SEC = 0.05

    # A loop task that gets cancelled, with _stop_generation bumped (pause).
    async def _gets_cancelled():
        await asyncio.sleep(3600)

    mgr._task = asyncio.create_task(_gets_cancelled())

    # Mirror stop()'s ordering: it bumps _stop_generation and THEN terminates
    # (cancels) the loop task — i.e. the bump happens AFTER _await_and_handle
    # has already snapshotted gen_at_start at the top of the attempt. Bumping
    # before the call would make gen_at_start == current and miss the guard.
    async def _cancel_soon():
        await asyncio.sleep(0.02)
        dispatcher._stop_generation += 1  # pause happened (mid-attempt)
        mgr._task.cancel()

    asyncio.create_task(_cancel_soon())

    # Must NOT raise; must return cleanly.
    await dispatcher._await_and_handle(item, "sess_x")

    reloaded = store.load().items[0]
    assert reloaded.state == ItemState.RUNNING, "attempt must be preserved (still RUNNING)"
    assert reloaded.attempts[-1].outcome is None, "attempt must NOT be closed"


# ----------------------------------------------------------------------------
# Unit: _ensure_run_task idempotency / self-heal
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_run_task_recreates_dead_task_and_noop_on_live(tmp_path):
    """_ensure_run_task recreates a done/None _task and is a no-op on a live one."""
    store = QueueStore(tmp_path / "queue.json")
    mgr = _PausingManager()
    dispatcher = QueueDispatcher(project_id="proj_ensure", store=store, agent_manager=mgr)
    dispatcher.IDLE_WAIT_TIMEOUT_SEC = 0.05

    # No task yet → ensure creates one.
    assert dispatcher._task is None
    dispatcher._ensure_run_task()
    first = dispatcher._task
    assert first is not None and not first.done(), "ensure must create a live _run task"

    # Live task → ensure is a no-op (same object).
    dispatcher._ensure_run_task()
    assert dispatcher._task is first, "ensure must NOT double-spawn a live loop"

    # Simulate a dead loop (the B3 wedge state): cancel the task, await it, then
    # ensure must recreate.
    first.cancel()
    try:
        await first
    except asyncio.CancelledError:
        pass
    assert dispatcher._task.done()
    dispatcher._ensure_run_task()
    assert dispatcher._task is not None and not dispatcher._task.done() and dispatcher._task is not first, (
        "ensure must recreate a dead _run task (self-heal)"
    )

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_shutdown_still_cancels_run_cleanly(tmp_path):
    """Sanity: the genuine shutdown path (set _shutting_down + cancel _task)
    still tears the loop down — the pause-cancel swallow must not break it."""
    store = QueueStore(tmp_path / "queue.json")
    mgr = _PausingManager()
    dispatcher = QueueDispatcher(project_id="proj_shutdown", store=store, agent_manager=mgr)
    dispatcher.IDLE_WAIT_TIMEOUT_SEC = 0.05
    await dispatcher.start()
    assert not dispatcher._task.done()
    await dispatcher.shutdown()
    assert dispatcher._task is None, "shutdown must tear down the dispatcher task"
