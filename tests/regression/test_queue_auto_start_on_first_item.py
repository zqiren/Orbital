# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the dispatcher auto-starts the agent and dispatches queue items
as *dedicated* runs — without folding them into a freshly-launched loop.

NEW CONTRACT (Steps 2+3 of the dispatcher session-lifecycle-manager work):

The queue dispatcher is the project's session-lifecycle manager. When it has
queueable work and no agent is running, it starts one itself (gated by
onboarding) and then dispatches the item. The previous design moved auto-start
out of the dispatcher into an explicit /queue/start endpoint specifically to
sidestep a race: the freshly-launched loop.run(None) was still in flight when
the item was injected, so inject_message took Case 1 (fold into session._queue)
and the loop's text-only exit was mis-attributed to the queued item as a
contract violation.

Steps 2+3 fix that race at the source instead of avoiding it: after
ensure_agent_started, the dispatcher waits for run-status to settle to 'idle'
before injecting, so every queue item hot-resumes a *dedicated* run (Case 2).

These tests assert the NEW invariant directly:
  1. Rapid-fire queue items from a no-agent state each produce a dedicated
     dispatched run; none is injected while a run is in flight (no folding).
  2. The onboarding gate prevents auto-start for a project with no captured
     state (no PROJECT_STATE.md).
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, sid: str):
        self.session_id = sid


class _FakeLoop:
    def __init__(self):
        self._exit_reason = "complete"
        self._exit_summary = "ok"
        self._exit_block_reason = None
        self._queue_state = "chat"


class _AutoStartManager:
    """Fake AgentManager modelling start → (briefly running) → idle.

    The load-bearing assertion lives in inject_message: if the dispatcher ever
    injects while the freshly-started loop is still 'in flight', that is the
    folding race — recorded in ``injected_in_flight``.
    """

    def __init__(self, *, onboarding: bool = True):
        self._onboarding = onboarding
        self._started = False
        self._run_in_flight = False
        self._loop = _FakeLoop()
        self._session: _FakeSession | None = None
        self._sid = 0
        self._task: asyncio.Task | None = None
        # observability
        self.ensure_calls = 0
        self.inject_count = 0
        self.injected_in_flight = False

    # --- observation accessors the dispatcher uses ---
    def is_onboarding_complete(self, project_id):
        return self._onboarding

    def has_handle(self, project_id):
        return self._started

    def get_session(self, project_id):
        return self._session

    def get_loop(self, project_id):
        return self._loop

    def get_loop_task(self, project_id):
        return self._task

    def get_run_status(self, project_id, *, session_id=None):
        # The launched loop.run(None) reports 'running' on the first poll after
        # start, then 'idle' once it settles. The dispatcher's idle-wait must
        # poll past the 'running' window before it injects.
        if self._run_in_flight:
            self._run_in_flight = False
            return "running"
        return "idle"

    # --- lifecycle actions (same methods the user endpoints call) ---
    async def ensure_agent_started(self, project_id):
        self.ensure_calls += 1
        self._started = True
        self._run_in_flight = True
        self._sid += 1
        self._session = _FakeSession(f"sess_{self._sid}")
        return True

    async def inject_message(self, project_id, content, *, nonce=None, session_id=None):
        if self._run_in_flight:
            # Injected while the freshly-started run is still in flight → the
            # Case-1 folding race the idle-wait is meant to prevent.
            self.injected_in_flight = True
        self.inject_count += 1
        self._loop._exit_reason = "complete"
        self._loop._exit_summary = "ok"

        async def _instant():
            return None

        self._task = asyncio.create_task(_instant())
        return "delivered"

    async def new_session(self, project_id):
        # Per-item rotation; the handle persists so the next item finds a
        # session and dispatches without re-starting.
        self._sid += 1
        self._session = _FakeSession(f"sess_{self._sid}")
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        return {"status": "new_session"}

    def get_sub_agent_manager(self):
        return None


async def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.02):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_rapid_fire_items_each_get_dedicated_run_no_folding(tmp_path):
    """§5.4 stress: five queue items added to a no-agent project all dispatch
    as dedicated runs; the dispatcher auto-starts once and the idle-wait keeps
    any item from being folded into the in-flight launch run."""
    store = QueueStore(tmp_path / "queue.json")
    for i in range(5):
        store.add_item(f"task {i}")

    mgr = _AutoStartManager(onboarding=True)
    dispatcher = QueueDispatcher(project_id="proj_autostart", store=store, agent_manager=mgr)
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait_until(
        lambda: all(it.state == ItemState.DONE for it in store.load().items)
    )
    await dispatcher.shutdown()

    assert ok, (
        "all five items should reach DONE; final: "
        + ", ".join(f"{it.id}={it.state.value}" for it in store.load().items)
    )
    assert mgr.ensure_calls >= 1, "dispatcher must auto-start the agent"
    assert mgr.inject_count == 5, "each item must get its own dedicated inject"
    assert not mgr.injected_in_flight, (
        "no item may be injected while the launch run is in flight (folding race)"
    )


@pytest.mark.asyncio
async def test_onboarding_gate_blocks_auto_start(tmp_path):
    """A project with no captured state (no PROJECT_STATE.md) must NOT be
    auto-started; the item stays queued."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("do work")

    mgr = _AutoStartManager(onboarding=False)
    dispatcher = QueueDispatcher(project_id="proj_no_onboard", store=store, agent_manager=mgr)
    # Keep the idle-poll snappy so the test doesn't sit on the 5s default.
    dispatcher.IDLE_WAIT_TIMEOUT_SEC = 0.05
    await dispatcher.start()
    dispatcher.notify_new_item()

    # Give the dispatcher several ticks; it should keep hitting the gate.
    await asyncio.sleep(0.3)
    await dispatcher.shutdown()

    assert mgr.ensure_calls == 0, "must not auto-start a pre-onboarding project"
    assert store.load().items[0].id == item.id
    assert store.load().items[0].state == ItemState.QUEUED, "item stays queued"
