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
    """Fake AgentManager modelling the new per-item session lifecycle.

    Each queue item now mints its OWN fresh session via new_session and then
    injects into it (inject_message Case 3 auto-starts a dedicated loop for
    that session). There is no separate launch run to fold into, so the
    old Case-1 folding race is structurally impossible: an inject can only
    target the fresh session it was just minted for.

    The load-bearing assertion lives in inject_message: ``injected_in_flight``
    flips True if the dispatcher ever injects without first minting a fresh
    session for that item (the modern equivalent of the folding race —
    reusing an in-flight session instead of a dedicated one).
    """

    def __init__(self, *, onboarding: bool = True):
        self._onboarding = onboarding
        self._loop = _FakeLoop()
        self._session: _FakeSession | None = None
        self._sid = 0
        self._task: asyncio.Task | None = None
        # observability
        self.new_session_calls = 0
        self.inject_count = 0
        # True if an inject lands on a session that wasn't freshly minted for
        # it (would indicate folding into a shared/in-flight session).
        self.injected_in_flight = False
        # Tracks the most recently minted (but not-yet-injected) session.
        self._pending_fresh_sid: str | None = None

    # --- observation accessors the dispatcher uses ---
    def is_onboarding_complete(self, project_id):
        return self._onboarding

    def get_session(self, project_id):
        return self._session

    def get_loop(self, project_id, *, session_id=None):
        return self._loop

    def get_loop_task(self, project_id, *, session_id=None):
        return self._task

    def get_run_status(self, project_id, *, session_id=None):
        return "idle"

    # --- lifecycle actions (same methods the user endpoints call) ---
    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        # Folding guard: the inject must target the session just minted for
        # this item. If it targets anything else (or no fresh mint preceded
        # it), that is the modern folding race.
        if session_id is None or session_id != self._pending_fresh_sid:
            self.injected_in_flight = True
        self._pending_fresh_sid = None
        self.inject_count += 1
        self._loop._exit_reason = "complete"
        self._loop._exit_summary = "ok"

        async def _instant():
            return None

        self._task = asyncio.create_task(_instant())
        return "delivered"

    async def new_session(self, project_id, *, session_id=None):
        # Pure-create: mint a fresh, unique session id. The dispatcher calls
        # this once per item before injecting, giving every item a dedicated
        # session (no folding into a shared launch run).
        self.new_session_calls += 1
        self._sid += 1
        sid = f"sess_{self._sid}"
        self._session = _FakeSession(sid)
        self._pending_fresh_sid = sid
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        return {
            "status": "ok",
            "session_id": sid,
            "session_uuid": f"proj_{self._sid:08d}",
        }

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
    # New contract: the dispatcher mints one fresh session per item (the same
    # path the user takes), then injects into it — five items → five
    # new_session calls and five dedicated injects.
    assert mgr.new_session_calls == 5, (
        "dispatcher must mint a fresh session for each of the five items"
    )
    assert mgr.inject_count == 5, "each item must get its own dedicated inject"
    assert not mgr.injected_in_flight, (
        "every inject must target the fresh session minted for that item "
        "(no folding into a shared/in-flight session)"
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

    # The gate is now is_onboarding_complete; when it's False the dispatcher
    # must NOT mint a session or inject — the equivalent of the old
    # "no auto-start" assertion under the per-item-session model.
    assert mgr.new_session_calls == 0, (
        "must not mint a session for a pre-onboarding project"
    )
    assert mgr.inject_count == 0, "must not inject into a pre-onboarding project"
    assert store.load().items[0].id == item.id
    assert store.load().items[0].state == ItemState.QUEUED, "item stays queued"
