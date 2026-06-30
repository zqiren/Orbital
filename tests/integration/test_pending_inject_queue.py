# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backend tests for the pending-input queue (Path A, spec 006).

Covers BACKLOG/specs/006-pending-input-queue.md §6 items 1-12:

  1.  Inject to B while A holds → 202 queued_pending_slot; registry has one
      entry; B's JSONL NOT written yet (persist-at-dispatch).
  2.  A idle → B auto-dispatched (inject_message called once = the sole
      JSONL writer), registry cleared, chat.pending_dispatched broadcast.
  3.  FIFO across sessions: B then C; A frees → B; B done → C.
  4.  In-order within a session: 3 messages to B delivered in order.
  5.  Run now from each holder state (running via _on_loop_done cancelled,
      pending_approval + waiting via cancel_message branches).
  6.  Error frees slot → B auto-dispatched.
  7.  stop_agent on waiting/pending_approval holder → B dispatched.
  8.  Sub-agent completion (waiting→idle via add_done_callback) → B dispatched
      (asserts it actually fires, not a no-op).
  9.  Stop waiting before dispatch → no dispatch; tombstone race → B not
      injected, not resurrected.
  10. Slot race vs dispatcher: pending re-enqueued at head on ValueError;
      dispatcher defer-check holds back while pending exists (no attempt
      minted / interrupted_count not bumped).
  11. Orphan cleanup: delete_session / delete_project / purge_pending.
  12. Unchanged: same-session-while-running still queues (no pending entry);
      idle inject still starts immediately (no pending entry).

The harness mirrors tests/integration/test_slot_enforcement.py: a real
AgentManager with mocked collaborators, fake ProjectHandles with live
asyncio.Tasks injected into _handles to model a held slot, and a FastAPI
TestClient for the HTTP-surface assertions.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState
from agent_os.queue.store import QueueStore


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _make_agent_manager():
    project_store = MagicMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()
    ws.broadcast_global = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.send = AsyncMock(return_value="ok")
    sub_agent_mgr.start = AsyncMock()
    sub_agent_mgr.get_transcript = MagicMock(return_value=None)
    sub_agent_mgr.stop_all = AsyncMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    activity_translator = MagicMock()
    process_manager = MagicMock()
    process_manager.set_session = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr, project_store, ws


def _mk_session(sid):
    session = MagicMock()
    session.session_id = sid
    session.session_uuid = sid
    session.is_stopped = MagicMock(return_value=False)
    session._paused_for_approval = False
    session.queue_message = MagicMock()
    session.append = MagicMock()
    session.resume = MagicMock()
    session.pop_deferred_messages = MagicMock(return_value=[])
    session.pop_queued_messages = MagicMock(return_value=[])
    return session


def _mk_handle(session):
    interceptor = MagicMock()
    interceptor.deactivate_bypass_all = MagicMock()
    interceptor._pending_approvals = {}
    return ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=interceptor,
        task=None,
    )


def _plant_running(mgr, project_id, sid, *, tasks=None, with_loop_done=False):
    """Plant a handle whose main loop task is alive (slot held)."""
    session = _mk_session(sid)
    handle = _mk_handle(session)

    async def _never_done():
        await asyncio.sleep(9999)

    task = asyncio.get_event_loop().create_task(_never_done())
    if with_loop_done:
        task.add_done_callback(mgr._on_loop_done(project_id, session_id=sid))
    handle.task = task
    mgr._handles[(project_id, sid)] = handle
    if tasks is not None:
        tasks.append(task)
    return handle, task


def _plant_waiting(mgr, project_id, sid, *, tasks=None):
    """Plant a handle in the ``waiting`` state: main task done, poll alive."""
    session = _mk_session(sid)
    handle = _mk_handle(session)

    async def _done():
        return None

    done_task = asyncio.get_event_loop().create_task(_done())
    handle.task = done_task
    mgr._handles[(project_id, sid)] = handle

    async def _never_done():
        await asyncio.sleep(9999)

    poll_task = asyncio.get_event_loop().create_task(_never_done())
    mgr._idle_poll_tasks[(project_id, sid)] = poll_task
    if tasks is not None:
        tasks.extend([done_task, poll_task])
    return handle, done_task, poll_task


def _plant_approval_paused(mgr, project_id, sid, *, tasks=None):
    """Plant a handle in ``pending_approval``: main task done, flag set."""
    session = _mk_session(sid)
    session._paused_for_approval = True
    handle = _mk_handle(session)

    async def _done():
        return None

    done_task = asyncio.get_event_loop().create_task(_done())
    handle.task = done_task
    mgr._handles[(project_id, sid)] = handle
    if tasks is not None:
        tasks.append(done_task)
    return handle, done_task


def _planting_inject(mgr, project_id, calls, tasks):
    """An inject_message stand-in that models start_agent taking the slot:
    the first inject for a session plants a running handle; later injects for
    that same (now-running) session simply 'queue' (Case 1)."""
    async def _inject(pid, content, *, nonce=None, session_id=None,
                      queue_state="chat"):
        calls.append((session_id, content, nonce))
        existing = mgr._handles.get((pid, session_id))
        if (existing is not None and existing.task is not None
                and not existing.task.done()):
            return "queued"
        _plant_running(mgr, pid, session_id, tasks=tasks)
        return "started"
    return _inject


def _build_client(mgr, project_store):
    saved = {
        k: getattr(agents_v2, k)
        for k in ("_project_store", "_agent_manager", "_ws_manager",
                  "_sub_agent_manager")
    }
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.send = AsyncMock(return_value="ok")
    agents_v2.configure(
        project_store=project_store,
        agent_manager=mgr,
        ws_manager=mgr._ws,            # share the manager's ws so all
        sub_agent_manager=sub_agent_mgr,  # broadcasts land on one mock
    )
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app), saved


def _restore(saved):
    for k, v in saved.items():
        setattr(agents_v2, k, v)


def _payloads(mgr):
    return [c.args[1] for c in mgr._ws.broadcast.call_args_list
            if len(c.args) >= 2]


async def _flush(n=12):
    for _ in range(n):
        await asyncio.sleep(0)


async def _cancel_tasks(tasks):
    for t in tasks:
        if not t.done():
            t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# §6.1 — enqueue on slot-held (HTTP), persist-at-dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inject_to_b_while_a_holds_enqueues_202():
    project_id = "proj-enqueue"
    mgr, project_store, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }
    tasks: list = []
    _plant_running(mgr, project_id, "sess-A", tasks=tasks)

    client, saved = _build_client(mgr, project_store)
    try:
        resp = client.post(
            f"/api/v2/agents/{project_id}/inject",
            json={"content": "hello from B", "session_id": "sess-B",
                  "nonce": "n-b"},
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["status"] == "queued_pending_slot"
        assert body["holding_session_id"] == "sess-A"
        assert body["queued_session_id"] == "sess-B"
        assert body["nonce"] == "n-b"
        assert body["position"] == 1

        # Registry has exactly one entry; persist-at-dispatch means NO handle
        # and NO JSONL write for B (no append on a B session anywhere).
        assert len(mgr._pending_inject[project_id]) == 1
        entry = mgr._pending_inject[project_id][0]
        assert entry.session_id == "sess-B"
        assert entry.content == "hello from B"
        assert entry.nonce == "n-b"
        assert (project_id, "sess-B") not in mgr._handles

        # chat.pending_enqueued broadcast carries text + nonce + holder.
        enq = [p for p in _payloads(mgr) if p.get("type") == "chat.pending_enqueued"]
        assert len(enq) == 1
        assert enq[0]["session_id"] == "sess-B"
        assert enq[0]["holder"] == "sess-A"
        assert enq[0]["nonce"] == "n-b"
        assert enq[0]["content"] == "hello from B"
        assert enq[0]["position"] == 1
    finally:
        await _cancel_tasks(tasks)
        client.close()
        _restore(saved)


# ---------------------------------------------------------------------------
# §6.2 — A idle → B auto-dispatched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_idle_then_b_auto_dispatched():
    project_id = "proj-autodispatch"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    _, a_task = _plant_running(mgr, project_id, "sess-A", tasks=tasks)
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B", nonce="n-b")

    inject = AsyncMock(return_value="started")
    mgr.inject_message = inject  # type: ignore[assignment]

    try:
        # Free the slot.
        a_task.cancel()
        try:
            await a_task
        except asyncio.CancelledError:
            pass
        assert mgr.current_holder_session_id(project_id) is None

        mgr._release_slot(project_id)
        await _flush()

        # inject_message (the sole JSONL writer) called exactly once for B.
        assert inject.await_count == 1
        _args, kwargs = inject.await_args
        assert _args[0] == project_id
        assert _args[1] == "run B"
        assert kwargs["session_id"] == "sess-B"
        assert kwargs["nonce"] == "n-b"

        # Registry cleared, dispatched broadcast emitted.
        assert project_id not in mgr._pending_inject
        disp = [p for p in _payloads(mgr)
                if p.get("type") == "chat.pending_dispatched"]
        assert len(disp) == 1
        assert disp[0]["session_id"] == "sess-B"
        assert disp[0]["nonce"] == "n-b"
    finally:
        await _cancel_tasks(tasks)


# ---------------------------------------------------------------------------
# §6.3 — FIFO across sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fifo_across_sessions_b_then_c():
    project_id = "proj-fifo"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    calls: list = []
    _, a_task = _plant_running(mgr, project_id, "sess-A", tasks=tasks)
    mgr.enqueue_pending_inject(project_id, "sess-B", "msg B")
    mgr.enqueue_pending_inject(project_id, "sess-C", "msg C")
    mgr.inject_message = _planting_inject(mgr, project_id, calls, tasks)  # type: ignore

    try:
        # A frees → B dispatches (FIFO head); C stays queued.
        a_task.cancel()
        try:
            await a_task
        except asyncio.CancelledError:
            pass
        mgr._release_slot(project_id)
        await _flush()
        assert [c[0] for c in calls] == ["sess-B"]
        assert mgr.current_holder_session_id(project_id) == "sess-B"
        assert [p.session_id for p in mgr._pending_inject[project_id]] == ["sess-C"]

        # B done → C dispatches.
        b_task = mgr._handles[(project_id, "sess-B")].task
        b_task.cancel()
        try:
            await b_task
        except asyncio.CancelledError:
            pass
        mgr._release_slot(project_id)
        await _flush()
        assert [c[0] for c in calls] == ["sess-B", "sess-C"]
        assert mgr.current_holder_session_id(project_id) == "sess-C"
        assert project_id not in mgr._pending_inject
    finally:
        await _cancel_tasks(tasks)


# ---------------------------------------------------------------------------
# §6.4 — in-order within a session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_order_within_session():
    project_id = "proj-inorder"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    calls: list = []
    _, a_task = _plant_running(mgr, project_id, "sess-A", tasks=tasks)
    for i in range(1, 4):
        mgr.enqueue_pending_inject(project_id, "sess-B", f"b{i}", nonce=f"n{i}")
    mgr.inject_message = _planting_inject(mgr, project_id, calls, tasks)  # type: ignore

    try:
        a_task.cancel()
        try:
            await a_task
        except asyncio.CancelledError:
            pass
        mgr._release_slot(project_id)
        await _flush()

        # All three delivered as one batch, in order, all to sess-B.
        assert calls == [
            ("sess-B", "b1", "n1"),
            ("sess-B", "b2", "n2"),
            ("sess-B", "b3", "n3"),
        ]
        assert project_id not in mgr._pending_inject
    finally:
        await _cancel_tasks(tasks)


# ---------------------------------------------------------------------------
# §6.5 — Run now from each holder state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_now_running_holder_dispatches_via_loop_done():
    """Cancelling a RUNNING holder frees the slot through _on_loop_done's
    cancelled branch → pending B dispatches (the headline B1 regression)."""
    project_id = "proj-runnow-running"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    calls: list = []
    _, a_task = _plant_running(mgr, project_id, "sess-A", tasks=tasks,
                               with_loop_done=True)
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B")
    mgr.inject_message = _planting_inject(mgr, project_id, calls, tasks)  # type: ignore

    try:
        a_task.cancel()        # what cancel_turn ultimately does to the loop
        try:
            await a_task
        except asyncio.CancelledError:
            pass
        await _flush()
        assert [c[0] for c in calls] == ["sess-B"]
    finally:
        await _cancel_tasks(tasks)


@pytest.mark.asyncio
async def test_run_now_pending_approval_holder_dispatches():
    project_id = "proj-runnow-approval"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    calls: list = []
    _, done_task = _plant_approval_paused(mgr, project_id, "sess-A", tasks=tasks)
    await done_task
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B")
    mgr.inject_message = _planting_inject(mgr, project_id, calls, tasks)  # type: ignore

    try:
        result = await mgr.cancel_message(project_id, session_id="sess-A")
        assert result["status"] == "cancelled"
        await _flush()
        assert [c[0] for c in calls] == ["sess-B"]
    finally:
        await _cancel_tasks(tasks)


@pytest.mark.asyncio
async def test_run_now_waiting_holder_dispatches():
    project_id = "proj-runnow-waiting"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    calls: list = []
    _, done_task, _poll = _plant_waiting(mgr, project_id, "sess-A", tasks=tasks)
    await done_task
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B")
    mgr.inject_message = _planting_inject(mgr, project_id, calls, tasks)  # type: ignore

    try:
        result = await mgr.cancel_message(project_id, session_id="sess-A")
        assert result["status"] == "cancelled"
        await _flush()
        assert [c[0] for c in calls] == ["sess-B"]
    finally:
        await _cancel_tasks(tasks)


# ---------------------------------------------------------------------------
# §6.6 — error frees the slot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_frees_slot_dispatches_b():
    project_id = "proj-error"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    calls: list = []
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B")
    mgr.inject_message = _planting_inject(mgr, project_id, calls, tasks)  # type: ignore

    session = _mk_session("sess-A")
    handle = _mk_handle(session)

    async def _raises():
        raise RuntimeError("boom")

    err_task = asyncio.get_event_loop().create_task(_raises())
    err_task.add_done_callback(mgr._on_loop_done(project_id, session_id="sess-A"))
    handle.task = err_task
    mgr._handles[(project_id, "sess-A")] = handle

    try:
        try:
            await err_task
        except RuntimeError:
            pass
        await _flush()
        assert [c[0] for c in calls] == ["sess-B"]
    finally:
        await _cancel_tasks(tasks)


# ---------------------------------------------------------------------------
# §6.7 — stop_agent on a waiting holder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_agent_on_waiting_holder_dispatches_b():
    project_id = "proj-stop-waiting"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    calls: list = []
    _, done_task, _poll = _plant_waiting(mgr, project_id, "sess-A", tasks=tasks)
    await done_task
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B")
    mgr.inject_message = _planting_inject(mgr, project_id, calls, tasks)  # type: ignore

    try:
        await mgr.stop_agent(project_id, session_id="sess-A")
        await _flush()
        assert [c[0] for c in calls] == ["sess-B"]
        assert (project_id, "sess-A") not in mgr._handles
    finally:
        await _cancel_tasks(tasks)


# ---------------------------------------------------------------------------
# §6.8 — sub-agent completion (poll add_done_callback) dispatches B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_completion_dispatches_b(monkeypatch):
    """The waiting→idle exit fires _release_slot via the poll task's
    done-callback (NOT a synchronous call, which would be a no-op). Asserts B
    is actually dispatched after the poll completes."""
    project_id = "proj-subagent-done"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    calls: list = []
    mgr.inject_message = _planting_inject(mgr, project_id, calls, tasks)  # type: ignore
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B")

    # Speed up the 2s poll cadence.
    real_sleep = asyncio.sleep

    async def _fast_sleep(_delay):
        await real_sleep(0)
    monkeypatch.setattr(
        "agent_os.daemon_v2.agent_manager.asyncio.sleep", _fast_sleep,
    )

    # list_active: busy during _on_loop_done, then empty during the poll.
    seen = {"n": 0}

    def _list_active(pid, *, session_id=None):
        seen["n"] += 1
        if seen["n"] == 1:
            return [{"status": "running", "handle": "sub"}]
        return []
    mgr._sub_agent_manager.list_active = _list_active

    session = _mk_session("sess-A")
    handle = _mk_handle(session)

    async def _done():
        return None

    main_task = asyncio.get_event_loop().create_task(_done())
    handle.task = main_task
    mgr._handles[(project_id, "sess-A")] = handle
    await main_task

    try:
        # Fire the loop-done callback → installs the waiting poll task with the
        # _release_slot done-callback.
        mgr._on_loop_done(project_id, session_id="sess-A")(main_task)
        poll_task = mgr._idle_poll_tasks[(project_id, "sess-A")]
        # Let the poll run to completion (sub-agents now idle) → done-callback.
        await poll_task
        await _flush()
        assert [c[0] for c in calls] == ["sess-B"], (
            "sub-agent completion must dispatch pending B via the poll task's "
            "add_done_callback (not a synchronous no-op)"
        )
    finally:
        await _cancel_tasks(tasks)


# ---------------------------------------------------------------------------
# §6.9 — stop waiting + tombstone race
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_pending_before_dispatch_no_dispatch():
    project_id = "proj-cancel-pending"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    _, a_task = _plant_running(mgr, project_id, "sess-A", tasks=tasks)
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B", nonce="n-b")
    inject = AsyncMock(return_value="started")
    mgr.inject_message = inject  # type: ignore[assignment]

    try:
        result = mgr.cancel_pending_inject(project_id, "sess-B", nonce="n-b")
        assert result["status"] == "cancelled"
        assert project_id not in mgr._pending_inject
        cancelled = [p for p in _payloads(mgr)
                     if p.get("type") == "chat.pending_cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0]["session_id"] == "sess-B"
        assert cancelled[0]["nonce"] == "n-b"

        # Free the slot — nothing pending → no dispatch.
        a_task.cancel()
        try:
            await a_task
        except asyncio.CancelledError:
            pass
        mgr._release_slot(project_id)
        await _flush()
        assert inject.await_count == 0
    finally:
        await _cancel_tasks(tasks)


@pytest.mark.asyncio
async def test_tombstone_race_cancel_after_fire_scheduled():
    """Cancel after _fire is scheduled (entry in-flight) → B not injected and
    not resurrected (B7)."""
    project_id = "proj-tombstone"
    mgr, _, _ = _make_agent_manager()
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B", nonce="n-b")
    inject = AsyncMock(return_value="started")
    mgr.inject_message = inject  # type: ignore[assignment]

    # No holder → _release_slot pops B into the in-flight batch and schedules
    # _fire (which has not run yet — no await happened).
    mgr._release_slot(project_id)
    assert mgr._pending_inflight.get(project_id), "batch should be in-flight"

    # Cancel before _fire runs → tombstones the in-flight id.
    mgr.cancel_pending_inject(project_id, "sess-B", nonce="n-b")
    await _flush()

    # _fire ran but skipped the tombstoned entry → no inject, no broadcast of
    # a dispatch, and not re-queued.
    assert inject.await_count == 0
    assert project_id not in mgr._pending_inject
    assert not mgr._pending_inflight.get(project_id)
    disp = [p for p in _payloads(mgr)
            if p.get("type") == "chat.pending_dispatched"]
    assert disp == []


# ---------------------------------------------------------------------------
# §6.10 — slot race vs dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_value_error_reenqueues_remainder_at_head():
    project_id = "proj-race"
    mgr, _, _ = _make_agent_manager()
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B", nonce="n-b")
    inject = AsyncMock(side_effect=ValueError("slot held"))
    mgr.inject_message = inject  # type: ignore[assignment]

    mgr._release_slot(project_id)
    await _flush()

    # Lost the slot race → B re-enqueued at head, nothing lost, in-flight clear.
    assert inject.await_count == 1
    assert [p.session_id for p in mgr._pending_inject[project_id]] == ["sess-B"]
    assert not mgr._pending_inflight.get(project_id)
    assert "n-b" == mgr._pending_inject[project_id][0].nonce


class _FakePendingMgr:
    """Lightweight dispatcher double exposing the slot + pending accessors."""

    def __init__(self, *, holder=None, has_pending=False):
        self._holder = holder
        self._has_pending = has_pending
        self.new_session_calls = 0

    def is_onboarding_complete(self, project_id):
        return True

    def current_holder_session_id(self, project_id):
        return self._holder

    def has_pending_inject(self, project_id):
        return self._has_pending

    async def new_session(self, project_id, *, session_id=None):
        self.new_session_calls += 1
        return {"status": "ok", "session_id": "sess_x",
                "session_uuid": "proj_00000001"}


@pytest.mark.asyncio
async def test_dispatcher_defers_when_pending_inject_present(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("autonomous task")
    mgr = _FakePendingMgr(holder=None, has_pending=True)
    dispatcher = QueueDispatcher(
        project_id="p", store=store, agent_manager=mgr, max_runtime_seconds=60,
    )
    dispatcher._wait_idle = AsyncMock()

    await dispatcher._dispatch_one(item)

    # Deferred WITHOUT minting a session/attempt; no spurious interrupted bump.
    dispatcher._wait_idle.assert_awaited()
    assert mgr.new_session_calls == 0
    reloaded = store.load().items[0]
    assert reloaded.state != ItemState.RUNNING
    assert reloaded.interrupted_count == 0
    assert reloaded.id == item.id


@pytest.mark.asyncio
async def test_dispatcher_proceeds_when_no_pending(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("autonomous task")
    mgr = _FakePendingMgr(holder=None, has_pending=False)
    # new_session raising aborts before injecting — enough to prove the
    # defer-check did NOT short-circuit (it minted a session).
    dispatcher = QueueDispatcher(
        project_id="p", store=store, agent_manager=mgr, max_runtime_seconds=60,
    )
    dispatcher._wait_idle = AsyncMock()

    await dispatcher._dispatch_one(store.load().items[0])
    assert mgr.new_session_calls == 1


# ---------------------------------------------------------------------------
# §6.11 — orphan cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_session_purges_pending(tmp_path):
    project_id = "proj-del-session"
    mgr, project_store, _ = _make_agent_manager()
    workspace = tmp_path / "ws"
    sessions_dir = workspace / "orbital" / "sessions"
    os.makedirs(sessions_dir, exist_ok=True)
    (sessions_dir / "sess-B.jsonl").write_text(
        '{"type":"session_start","session_id":"sess-B"}\n', encoding="utf-8",
    )
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": str(workspace), "name": "p",
    }
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B")
    mgr.enqueue_pending_inject(project_id, "sess-C", "run C")

    result = await mgr.delete_session(project_id, "sess-B")
    assert result["status"] == "deleted"
    # B purged, C survives.
    assert [p.session_id for p in mgr._pending_inject[project_id]] == ["sess-C"]


@pytest.mark.asyncio
async def test_purge_pending_on_project_teardown():
    project_id = "proj-del"
    mgr, _, _ = _make_agent_manager()
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B")
    mgr.enqueue_pending_inject(project_id, "sess-C", "run C")

    await mgr.shutdown_dispatcher(project_id)   # the project-deletion path
    assert project_id not in mgr._pending_inject

    # In-flight entries are tombstoned so a scheduled _fire is neutralized.
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B2")
    inject = AsyncMock(return_value="started")
    mgr.inject_message = inject  # type: ignore[assignment]
    mgr._release_slot(project_id)               # schedules _fire (B2 in-flight)
    mgr.purge_pending(project_id)               # delete arrives mid-flight
    await _flush()
    assert inject.await_count == 0


# ---------------------------------------------------------------------------
# §6.12 — unchanged behaviors (no spurious pending entries)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_session_while_running_still_queues_no_pending():
    project_id = "proj-same"
    mgr, project_store, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }
    tasks: list = []
    _plant_running(mgr, project_id, "sess-A", tasks=tasks)
    client, saved = _build_client(mgr, project_store)
    try:
        resp = client.post(
            f"/api/v2/agents/{project_id}/inject",
            json={"content": "more from A", "session_id": "sess-A"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "queued"}
        # Case-1 same-session queue → NO pending-inject entry created.
        assert project_id not in mgr._pending_inject
    finally:
        await _cancel_tasks(tasks)
        client.close()
        _restore(saved)


@pytest.mark.asyncio
async def test_idle_inject_starts_immediately_no_pending():
    project_id = "proj-idle"
    mgr, project_store, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }

    async def _fake_inject(pid, content, *, nonce=None, session_id=None,
                           queue_state="chat"):
        return "started"
    mgr.inject_message = _fake_inject  # type: ignore[assignment]

    client, saved = _build_client(mgr, project_store)
    try:
        resp = client.post(
            f"/api/v2/agents/{project_id}/inject",
            json={"content": "hello", "session_id": "sess-A"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "started"}
        assert project_id not in mgr._pending_inject
    finally:
        client.close()
        _restore(saved)


# ---------------------------------------------------------------------------
# REST recovery surfaces (GET /pending, run-status pending_count)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_pending_and_run_status_count():
    project_id = "proj-rest"
    mgr, project_store, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }
    tasks: list = []
    _plant_running(mgr, project_id, "sess-A", tasks=tasks)
    mgr.enqueue_pending_inject(project_id, "sess-B", "run B", nonce="n-b")

    client, saved = _build_client(mgr, project_store)
    try:
        resp = client.get(f"/api/v2/agents/{project_id}/pending")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["holder"] == "sess-A"
        assert len(body["pending"]) == 1
        assert body["pending"][0]["session_id"] == "sess-B"
        assert body["pending"][0]["nonce"] == "n-b"
        assert body["pending"][0]["position"] == 1
        assert "run B" in body["pending"][0]["content_preview"]

        rs = client.get(f"/api/v2/agents/{project_id}/run-status")
        assert rs.status_code == 200, rs.text
        assert rs.json()["pending_count"] == 1

        # POST /pending/cancel removes it + broadcasts.
        c = client.post(
            f"/api/v2/agents/{project_id}/pending/cancel",
            json={"session_id": "sess-B", "nonce": "n-b"},
        )
        assert c.status_code == 200, c.text
        assert c.json()["status"] == "cancelled"
        assert client.get(
            f"/api/v2/agents/{project_id}/run-status"
        ).json()["pending_count"] == 0
    finally:
        await _cancel_tasks(tasks)
        client.close()
        _restore(saved)
