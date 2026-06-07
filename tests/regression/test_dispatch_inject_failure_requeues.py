# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: a queue item must never be left RUNNING with no live attempt.

Observed live on 2026-06-08 (item_cb75800b6b18, proj_82bad3b9443c): /queue/start
dispatched a queue item while the management loop slot was still held by a
long-running onboarding session. inject_message raised; _dispatch_one closed
the attempt as INTERRUPTED but left item.state == RUNNING. With the head item
stuck RUNNING and no attempt live, the dispatcher's main loop idles forever
(head-is-RUNNING guard) — the queue is wedged until daemon restart.

Two fixes, mirroring the existing reclaim_on_startup poison-pill convention:

  (1) On inject failure, _dispatch_one no longer leaves the item RUNNING. It
      increments interrupted_count and re-queues at head (priority) when under
      the cap, or marks BLOCKED once interruptions hit the cap. Attempt records
      are preserved.

  (2) Before minting a session + injecting, _dispatch_one checks whether
      another session already holds the management loop slot
      (current_holder_session_id). If so it backs off (no attempt minted) and
      lets the main loop re-poll — never burning an attempt against a busy slot.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, sid: str):
        self.session_id = sid


class _FakeLoop:
    def __init__(self):
        self._exit_reason = "text"
        self._exit_summary = None
        self._exit_block_reason = None
        self._queue_state = "chat"


class _InjectFailingAgentManager:
    """Mock manager whose inject_message raises a fixed number of times.

    Also models the management-loop slot via ``holder`` so the busy-slot guard
    can be exercised: current_holder_session_id returns ``holder``.
    """

    def __init__(self, *, fail_times: int, holder: Optional[str] = None):
        self._fail_remaining = fail_times
        self.holder = holder
        self._loop = _FakeLoop()
        self._session_counter = 0
        self._session = _FakeSession("sess_0")
        self._task = None
        self.new_session_calls = 0
        self.inject_attempts = 0

    def is_onboarding_complete(self, project_id):
        return True

    def current_holder_session_id(self, project_id):
        return self.holder

    def get_session(self, project_id):
        return self._session

    def get_loop(self, project_id, *, session_id=None):
        return self._loop

    def get_loop_task(self, project_id, *, session_id=None):
        return self._task

    def get_sub_agent_manager(self):
        return None

    async def new_session(self, project_id, *, session_id=None):
        self.new_session_calls += 1
        self._session_counter += 1
        sid = f"sess_{self._session_counter}"
        self._session = _FakeSession(sid)
        self._loop._exit_reason = "text"
        return {"status": "ok", "session_id": sid,
                "session_uuid": f"proj_{self._session_counter:08d}"}

    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        self.inject_attempts += 1
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise RuntimeError("inject failed (slot busy)")
        # Success path: script a completed loop so the item finishes.
        self._loop._exit_reason = "complete"
        self._loop._exit_summary = "done"

        async def _instant():
            return None

        self._task = asyncio.create_task(_instant())
        return "delivered"


async def _wait(predicate, timeout: float = 5.0, interval: float = 0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_inject_failure_requeues_item_not_running(tmp_path):
    """One inject failure → item back to QUEUED, interrupted_count==1, NOT
    RUNNING. The attempt record is preserved (closed INTERRUPTED).

    After the failure the slot is held busy so the re-queued state is
    observable (otherwise the dispatcher would immediately re-dispatch the
    re-queued item, which is itself correct behaviour but races the assert)."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("research the landscape")

    mgr = _InjectFailingAgentManager(fail_times=1)
    dispatcher = QueueDispatcher(
        project_id="proj_requeue",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    # Wrap inject_message so that immediately after it raises, a holder is
    # planted — pinning the re-queued item in QUEUED so the test can observe it
    # before the dispatcher's next poll would re-dispatch.
    orig_inject = mgr.inject_message

    async def _inject_then_hold(*args, **kwargs):
        try:
            return await orig_inject(*args, **kwargs)
        finally:
            mgr.holder = "now-busy"

    mgr.inject_message = _inject_then_hold

    await dispatcher.start()
    dispatcher.notify_new_item()

    # After the failure the item must be QUEUED with interrupted_count bumped.
    ok = await _wait(
        lambda: store.load().items[0].interrupted_count >= 1,
        timeout=5.0,
    )
    assert ok, "interrupted_count should have been incremented on inject failure"

    reloaded = store.load().items[0]
    # The cardinal invariant: NOT a zombie RUNNING item.
    assert reloaded.state != ItemState.RUNNING, (
        f"item must not stay RUNNING after inject failure; got {reloaded.state}"
    )
    assert reloaded.state == ItemState.QUEUED
    assert reloaded.interrupted_count == 1
    # Attempt record preserved and closed as INTERRUPTED.
    assert len(reloaded.attempts) == 1
    assert reloaded.attempts[0].outcome == AttemptOutcome.INTERRUPTED
    assert reloaded.attempts[0].block_reason == "inject failed"

    await dispatcher.shutdown()
    assert item.id == reloaded.id


@pytest.mark.asyncio
async def test_repeated_inject_failures_block_item(tmp_path):
    """Once interruptions hit the cap (>=2, matching reclaim_on_startup), the
    item is marked BLOCKED instead of re-queued forever."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("keeps failing to inject")

    mgr = _InjectFailingAgentManager(fail_times=10)
    dispatcher = QueueDispatcher(
        project_id="proj_block",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(
        lambda: store.load().items[0].state == ItemState.BLOCKED,
        timeout=8.0,
    )
    assert ok, (
        "repeated inject failures should land the item in BLOCKED, not loop "
        f"forever. state={store.load().items[0].state}, "
        f"interrupted_count={store.load().items[0].interrupted_count}"
    )

    reloaded = store.load().items[0]
    assert reloaded.state == ItemState.BLOCKED
    assert reloaded.interrupted_count >= 2
    # No zombie RUNNING ever leaked through.
    assert reloaded.state != ItemState.RUNNING

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_busy_slot_does_not_burn_attempt(tmp_path):
    """When another session holds the management loop slot, the dispatcher
    must NOT mint a session or an attempt — it backs off and waits. Once the
    slot frees, dispatch proceeds normally."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("waits for the slot")

    # Another session ("onboarding-research") holds the slot.
    mgr = _InjectFailingAgentManager(fail_times=0, holder="onboarding-research")
    dispatcher = QueueDispatcher(
        project_id="proj_busy",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    # Give the dispatcher several poll cycles. While the slot is held, no
    # session is minted, no attempt is created, item stays QUEUED.
    await asyncio.sleep(0.5)
    held = store.load().items[0]
    assert held.state == ItemState.QUEUED, (
        f"item should stay QUEUED while slot is busy; got {held.state}"
    )
    assert held.attempts == [], "no attempt should be minted against a busy slot"
    assert mgr.new_session_calls == 0, "no session should be minted while busy"
    assert mgr.inject_attempts == 0, "inject must not be attempted while busy"

    # Free the slot — dispatch should now proceed and complete.
    mgr.holder = None
    dispatcher.notify_new_item()

    ok = await _wait(
        lambda: store.load().items[0].state == ItemState.DONE,
        timeout=8.0,
    )
    assert ok, (
        "once the slot frees the item should dispatch and complete; "
        f"state={store.load().items[0].state}"
    )
    assert mgr.new_session_calls == 1
    assert mgr.inject_attempts == 1

    await dispatcher.shutdown()
