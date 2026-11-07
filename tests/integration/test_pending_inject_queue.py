# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backend tests for queued messages (spec 006), after spec 155.

Spec 155 lifted the project's single active-loop slot: sessions of a project
run in parallel, so the cross-session pending FIFO (Path A: 202
``queued_pending_slot``, ``_pending_inject``, Run now, tombstones, owed
wakes, the dispatcher's pending defer-check, ``queued`` session rows) is gone.
What remains, and is covered here:

  - A message to B while A runs (or waits, or sits on an approval) starts B
    now: 200, nothing enqueued.
  - Same-session: a message to a session whose own turn is running still
    queues in that session (``queued_same_session``), lists through
    GET /pending (``kind: "same"``, full content), cancels/recalls through
    POST /pending/cancel, and its drain broadcasts chat.pending_dispatched
    per nonce.
  - Idle inject still starts immediately.

The harness: a real AgentManager with mocked collaborators, fake
ProjectHandles with live asyncio.Tasks injected into _handles, and a FastAPI
TestClient for the HTTP-surface assertions.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agent.session import Session
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


@pytest.mark.parametrize("a_state", ["running", "waiting", "pending_approval"])
@pytest.mark.asyncio
async def test_inject_to_b_while_a_is_busy_starts_b_now(a_state):
    """Spec 155: whatever A is doing, B's message starts B (200), and nothing
    is enqueued behind A."""
    project_id = f"proj-b-now-{a_state}"
    mgr, project_store, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }
    tasks: list = []
    {"running": _plant_running, "waiting": _plant_waiting,
     "pending_approval": _plant_approval_paused}[a_state](
        mgr, project_id, "sess-A", tasks=tasks)
    await _flush()   # let the planted "done" main task finish
    started: list = []

    async def _start(pid, config, content, nonce, *, session_id, queue_state="chat",
                     session=None):
        started.append((session_id, content))
        return "started"

    mgr._start_with_persisted_message = _start  # type: ignore[assignment]
    mgr._abuild_agent_config_from_project = AsyncMock(return_value=MagicMock())
    mgr._hydrate_session = AsyncMock(return_value=None)
    client, saved = _build_client(mgr, project_store)
    try:
        resp = client.post(
            f"/api/v2/agents/{project_id}/inject",
            json={"content": "run B", "session_id": "sess-B", "nonce": "n-b"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "started"}
        assert started == [("sess-B", "run B")]
        assert mgr.list_pending(project_id) == []
        assert mgr.get_run_status(project_id, session_id="sess-A") == a_state
    finally:
        await _cancel_tasks(tasks)
        client.close()
        _restore(saved)


@pytest.mark.asyncio
async def test_clear_same_session_pending_broadcasts_on_terminal():
    """A terminal turn-end (cancel/error/stop) with leftover same-session queued
    messages broadcasts chat.pending_dispatched per nonce so the FE 'Waiting…'
    line can't orphan (diff-review Finding 1, terminal paths). None-nonce
    entries are skipped (no FE bubble to clear)."""
    project_id = "proj-term"
    mgr, _, _ = _make_agent_manager()
    session = _mk_session("sess-A")
    session.list_queued = MagicMock(
        return_value=[("more from A", "n-1"), ("noNonce", None)]
    )
    mgr._handles[(project_id, "sess-A")] = _mk_handle(session)

    mgr._clear_same_session_pending(project_id, "sess-A")

    disp = [p for p in _payloads(mgr)
            if p.get("type") == "chat.pending_dispatched"]
    assert len(disp) == 1  # None-nonce skipped
    assert disp[0]["session_id"] == "sess-A"
    assert disp[0]["nonce"] == "n-1"


# ---------------------------------------------------------------------------
# §6.10 — slot race vs dispatcher
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
        assert resp.json() == {"status": "queued_same_session"}
        # Case-1 same-session queue: only ever the session's own entries.
        assert all(p["session_id"] == "sess-A" and p["kind"] == "same"
                   for p in mgr.list_pending(project_id))
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
        assert mgr.list_pending(project_id) == []
    finally:
        client.close()
        _restore(saved)


# ---------------------------------------------------------------------------
# REST recovery surfaces (GET /pending, run-status pending_count)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_pending_and_run_status_count(tmp_path):
    project_id = "proj-rest"
    mgr, project_store, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": str(tmp_path), "name": "p",
    }
    tasks: list = []
    _, session, _t = _plant_running_real_session(
        mgr, project_id, "sess-A", tmp_path, tasks=tasks,
    )
    session.queue_message("more for A", nonce="n-a")

    client, saved = _build_client(mgr, project_store)
    try:
        body = client.get(f"/api/v2/agents/{project_id}/pending").json()
        assert body["pending"] == [{
            "session_id": "sess-A", "nonce": "n-a", "content": "more for A",
            "position": 1, "kind": "same",
        }]
        rs = client.get(f"/api/v2/agents/{project_id}/run-status",
                        params={"session_id": "sess-A"})
        assert rs.json()["pending_count"] == 1
        assert client.get(f"/api/v2/agents/{project_id}/run-status",
                          params={"session_id": "sess-B"}).json()["pending_count"] == 0

        c = client.post(
            f"/api/v2/agents/{project_id}/pending/cancel",
            json={"session_id": "sess-A", "nonce": "n-a"},
        )
        assert c.status_code == 200, c.text
        assert c.json() == {"status": "cancelled", "removed": True}
        assert client.get(
            f"/api/v2/agents/{project_id}/run-status"
        ).json()["pending_count"] == 0
    finally:
        await _cancel_tasks(tasks)
        client.close()
        _restore(saved)


# ---------------------------------------------------------------------------
# v3 (§11d / §12) — same-session queued messages become first-class
# ---------------------------------------------------------------------------
#
# A "queued message" = a sent message whose turn hasn't started. v2 covered the
# CROSS-session kind (slot held by another session → _pending_inject). v3 makes
# the SAME-session kind (this session's turn mid-flight → session._queue)
# first-class: recallable (Session.remove_queued_message / list_queued),
# cancellable via the same endpoint (cancel_pending_inject removes from _queue,
# returns removed=bool), recoverable (list_pending tags kind=same), and its
# drain broadcasts chat.pending_dispatched per nonce.


def _plant_running_real_session(mgr, project_id, sid, workspace, *, tasks=None):
    """Plant a running handle (slot held) whose session is a REAL Session, so
    the same-session ``_queue`` / ``remove_queued_message`` / ``list_queued``
    behaviour is exercised end-to-end (not a mock)."""
    session = Session.new(sid, str(workspace), session_id=sid)
    handle = _mk_handle(session)

    async def _never_done():
        await asyncio.sleep(9999)

    task = asyncio.get_event_loop().create_task(_never_done())
    handle.task = task
    mgr._handles[(project_id, sid)] = handle
    if tasks is not None:
        tasks.append(task)
    return handle, session, task


# §11d.1 — Session.remove_queued_message (hit + miss) ------------------------


def test_session_remove_queued_message_hit_and_miss(tmp_path):
    session = Session.new("rmq", str(tmp_path))
    session.queue_message("first", nonce="n1")
    session.queue_message("second", nonce="n2")

    # Hit: returns content, removes only the matching tuple.
    assert session.remove_queued_message("n1") == "first"
    assert session.list_queued() == [("second", "n2")]

    # Miss: returns None, queue unchanged.
    assert session.remove_queued_message("nope") is None
    assert session.list_queued() == [("second", "n2")]


# §12 R3 — Session.list_queued is non-destructive ---------------------------


def test_session_list_queued_non_destructive(tmp_path):
    session = Session.new("lq", str(tmp_path))
    session.queue_message("a", nonce="na")
    session.queue_message("b", nonce=None)

    snapshot = session.list_queued()
    assert snapshot == [("a", "na"), ("b", None)]
    # Snapshot is a copy: mutating it must not touch the live queue, and a
    # second read still returns everything (unlike pop_queued_messages).
    snapshot.clear()
    assert session.list_queued() == [("a", "na"), ("b", None)]
    # pop still drains it (proves list_queued didn't consume).
    assert session.pop_queued_messages() == [("a", "na"), ("b", None)]
    assert session.list_queued() == []


# §11d.2 / §12 R2 — cancel removes a same-session queued msg → removed=True --


@pytest.mark.asyncio
async def test_cancel_removes_same_session_queued_returns_removed_true(tmp_path):
    project_id = "proj-cancel-same"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    _, session, _t = _plant_running_real_session(
        mgr, project_id, "sess-A", tmp_path, tasks=tasks,
    )
    session.queue_message("queued behind my own turn", nonce="q1")

    try:
        result = mgr.cancel_pending_inject(project_id, "sess-A", nonce="q1")
        assert result == {"status": "cancelled", "removed": True}
        # The same-session queue no longer holds it.
        assert session.list_queued() == []
        # chat.pending_cancelled broadcast carries the session + nonce.
        cancelled = [p for p in _payloads(mgr)
                     if p.get("type") == "chat.pending_cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0]["session_id"] == "sess-A"
        assert cancelled[0]["nonce"] == "q1"
    finally:
        await _cancel_tasks(tasks)


# §12 R2 — cancel of an unknown / already-dispatched nonce → removed=False ---


@pytest.mark.asyncio
async def test_cancel_unknown_nonce_returns_removed_false(tmp_path):
    project_id = "proj-cancel-unknown"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    _, session, _t = _plant_running_real_session(
        mgr, project_id, "sess-A", tmp_path, tasks=tasks,
    )
    # Nothing queued (or already drained/dispatched) for this nonce.
    try:
        result = mgr.cancel_pending_inject(project_id, "sess-A", nonce="gone")
        assert result == {"status": "cancelled", "removed": False}
        assert session.list_queued() == []
    finally:
        await _cancel_tasks(tasks)


# §12 R2 — cancel of an IN-FLIGHT (dispatching) entry → removed=False --------


@pytest.mark.asyncio
async def test_list_pending_includes_same_session_full_content(tmp_path):
    project_id = "proj-list-same"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []
    _, session, _t = _plant_running_real_session(
        mgr, project_id, "sess-A", tmp_path, tasks=tasks,
    )
    # A runs and has its own queued message (there is no cross kind any more).
    long_text = "x" * 500  # exceeds the old 200-char preview cap → full content
    session.queue_message(long_text, nonce="same-1")

    try:
        pending = mgr.list_pending(project_id)
        assert all(p["kind"] == "same" for p in pending)
        same = pending

        assert len(same) == 1
        assert same[0]["session_id"] == "sess-A"
        assert same[0]["nonce"] == "same-1"
        # FULL content, not a 200-char preview (§12 R3/R4).
        assert same[0]["content"] == long_text
        assert "content_preview" not in same[0]
    finally:
        await _cancel_tasks(tasks)


# §11d.3 — _on_loop_done same-session drain broadcasts chat.pending_dispatched


@pytest.mark.asyncio
async def test_on_loop_done_drain_broadcasts_pending_dispatched_per_nonce():
    project_id = "proj-drain-bcast"
    mgr, _, _ = _make_agent_manager()
    tasks: list = []

    session = _mk_session("sess-A")
    # Two nonced messages + one None-nonce message in the same-session queue.
    session.pop_queued_messages = MagicMock(
        return_value=[("c1", "n1"), ("c2", "n2"), ("c3", None)],
    )
    handle = _mk_handle(session)

    async def _done():
        return None

    main_task = asyncio.get_event_loop().create_task(_done())
    handle.task = main_task
    mgr._handles[(project_id, "sess-A")] = handle
    await main_task

    # The drain hot-resumes the loop via ensure_future(_start_loop); stub it.
    mgr._start_loop = AsyncMock()  # type: ignore[assignment]

    try:
        mgr._on_loop_done(project_id, session_id="sess-A")(main_task)
        await _flush()
        disp = [p for p in _payloads(mgr)
                if p.get("type") == "chat.pending_dispatched"]
        # One broadcast per drained NONCE; the None-nonce drain is skipped.
        assert [(p["session_id"], p["nonce"]) for p in disp] == [
            ("sess-A", "n1"), ("sess-A", "n2"),
        ]
        # The hot-resume was scheduled exactly once.
        assert mgr._start_loop.await_count == 1
    finally:
        await _cancel_tasks(tasks)


# ---------------------------------------------------------------------------
# Spec 081 — the session list carries the queued session through its whole
# life: enqueue → dispatch → cancel, over the HTTP surface.
# ---------------------------------------------------------------------------
