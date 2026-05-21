# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for Track J — slot enforcement MVP (Phase 1).

Per TASK/DISPATCH-2026-05-22.md §5 and TASK/INVESTIGATION-slot-enforcement-surface.md
§5 (F6 Phase 1). Verifies the active-loop slot guard at the inject route:

  1. Five sessions in one project attempting to start loops back-to-back:
     the first succeeds (200), the next four receive 202 with structured
     `slot_held` payload. When the first finishes, a retry for session 2
     succeeds.
  2. Backward compatibility: absent `session_id` skips the slot check
     entirely. Old clients continue to work unchanged.
  3. Inject from the holding session does NOT 202 — slot check is per-session,
     not per-loop-start.
  4. Sub-agent dispatches inside a turn do NOT consume a separate slot;
     they are not gated by the slot guard.

The tests drive the FastAPI route directly via TestClient. AgentManager
collaborators are stubbed so we exercise the inject-route guard logic
(`current_holder_session_id` + the 202 short-circuit) without needing a
real provider/loop. Test 1 simulates a held slot by injecting a fake
ProjectHandle with a live task into AgentManager._handles, then completes
it to free the slot.

Reference: TASK/DISPATCH-2026-05-22.md §5.5; F6 §5.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_manager():
    """Build a real AgentManager with mocked collaborators."""
    project_store = MagicMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()
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
    return mgr, project_store, ws, sub_agent_mgr


def _inject_running_handle(mgr: AgentManager, project_id: str,
                            holding_session_id: str):
    """Plant a fake ProjectHandle whose task is alive (slot held).

    Returns (handle, task). Caller must cancel/await the task to release
    the slot at end-of-test.
    """
    session = MagicMock()
    session.session_id = holding_session_id
    session.is_stopped = MagicMock(return_value=False)
    session._paused_for_approval = False
    # The inject path may also call session.queue_message; make that a no-op.
    session.queue_message = MagicMock()
    session.append = MagicMock()

    async def _never_done():
        await asyncio.sleep(9999)

    task = asyncio.get_event_loop().create_task(_never_done())

    interceptor = MagicMock()
    interceptor.deactivate_bypass_all = MagicMock()

    handle = ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=interceptor,
        task=task,
    )
    # ``_handles`` is keyed by SessionKey == (project_id, session_id) per
    # PR #22's multi-session foundation. Key the planted handle accordingly.
    mgr._handles[(project_id, holding_session_id)] = handle
    return handle, task


def _build_test_client(mgr, project_store, sub_agent_mgr=None):
    """Wire the test HTTP client with the AgentManager."""
    saved = {
        k: getattr(agents_v2, k)
        for k in ("_project_store", "_agent_manager", "_ws_manager",
                  "_sub_agent_manager")
    }
    if sub_agent_mgr is None:
        sub_agent_mgr = MagicMock()
        sub_agent_mgr.send = AsyncMock(return_value="ok")
        sub_agent_mgr.start = AsyncMock()
        sub_agent_mgr.get_transcript = MagicMock(return_value=None)
    ws_mock = MagicMock()
    ws_mock.broadcast = MagicMock()
    agents_v2.configure(
        project_store=project_store,
        agent_manager=mgr,
        ws_manager=ws_mock,
        sub_agent_manager=sub_agent_mgr,
    )
    app = FastAPI()
    app.include_router(agents_v2.router)
    client = TestClient(app)
    return client, saved


def _restore(saved):
    for k, v in saved.items():
        setattr(agents_v2, k, v)


# ---------------------------------------------------------------------------
# Test 1: five sessions serialize via the slot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_five_sessions_in_one_project_serialize_via_slot():
    """Five sessions, one project: first holds the slot, four get 202.

    Per DISPATCH-2026-05-22 §5.5 test 1.
    """
    project_id = "proj-five-sessions"
    mgr, project_store, _, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }

    # Session 1 holds the slot.
    handle, task = _inject_running_handle(mgr, project_id, "sess-1")

    client, saved = _build_test_client(mgr, project_store)
    try:
        # Session 1 can keep injecting (will go through normal queue path
        # in agent_manager.inject_message, which is mocked-out below).
        # We don't actually need to exercise session-1's happy path here;
        # the test focuses on the 4 rejected sessions.

        # Sessions 2..5: 202 with structured payload.
        for n in range(2, 6):
            resp = client.post(
                f"/api/v2/agents/{project_id}/inject",
                json={"content": f"hi from session {n}",
                      "session_id": f"sess-{n}"},
            )
            assert resp.status_code == 202, (
                f"session {n} expected 202, got {resp.status_code}: "
                f"{resp.text}"
            )
            body = resp.json()
            assert body["status"] == "slot_held"
            assert body["holding_session_id"] == "sess-1"
            assert "currently running" in body["message"]

        # Free the slot — task completes, session 1 releases.
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Now session 2's retry succeeds. After the holding-session task
        # is gone, current_holder_session_id() returns None, so the guard
        # does not fire. We don't want the inject to actually run an
        # agent — patch the manager's inject_message to short-circuit.
        async def _fake_inject(pid, content, *, nonce=None, session_id=None):
            return "queued"
        mgr.inject_message = _fake_inject  # type: ignore[assignment]

        assert mgr.current_holder_session_id(project_id) is None
        resp = client.post(
            f"/api/v2/agents/{project_id}/inject",
            json={"content": "retry from session 2",
                  "session_id": "sess-2"},
        )
        assert resp.status_code == 200, (
            f"session 2 retry expected 200, got {resp.status_code}: "
            f"{resp.text}"
        )
        assert resp.json() == {"status": "queued"}
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        client.close()
        _restore(saved)


# ---------------------------------------------------------------------------
# Test 2: backward compat — absent session_id skips the slot check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inject_with_absent_session_id_skips_slot_check():
    """Inject without `session_id` proceeds normally even when slot is held.

    Per DISPATCH-2026-05-22 §5.5 test 2 — backward-compat for old clients.
    """
    project_id = "proj-backcompat"
    mgr, project_store, _, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }
    handle, task = _inject_running_handle(mgr, project_id, "sess-A")

    # Patch the manager's inject_message so the request flows through
    # to the route's else-branch without spinning up a real loop.
    async def _fake_inject(pid, content, *, nonce=None, session_id=None):
        return "queued"
    mgr.inject_message = _fake_inject  # type: ignore[assignment]

    client, saved = _build_test_client(mgr, project_store)
    try:
        resp = client.post(
            f"/api/v2/agents/{project_id}/inject",
            json={"content": "anon client, no session_id"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "queued"}
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        client.close()
        _restore(saved)


# ---------------------------------------------------------------------------
# Test 3: inject from the holding session does NOT 202
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inject_to_holding_session_does_not_202():
    """Same session_id as the holder => no 202. Slot check is per-session.

    Per DISPATCH-2026-05-22 §5.5 test 3.
    """
    project_id = "proj-same-session"
    mgr, project_store, _, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }
    handle, task = _inject_running_handle(mgr, project_id, "sess-holder")

    async def _fake_inject(pid, content, *, nonce=None, session_id=None):
        return "queued"
    mgr.inject_message = _fake_inject  # type: ignore[assignment]

    client, saved = _build_test_client(mgr, project_store)
    try:
        resp = client.post(
            f"/api/v2/agents/{project_id}/inject",
            json={"content": "continuation from holder",
                  "session_id": "sess-holder"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "queued"}
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        client.close()
        _restore(saved)


# ---------------------------------------------------------------------------
# Test 4: sub-agent dispatch does NOT consume a separate slot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_agent_dispatch_does_not_consume_separate_slot():
    """Sub-agent dispatches inside a turn don't fire the slot guard.

    Per ACTIVE-session-and-queue-model.md §3 and DISPATCH-2026-05-22 §5.5
    test 4: sub-agent dispatches happen INSIDE the holding session's turn
    and do not consume a separate slot. The guard's contract is "different
    session_id + held slot = 202". When the dispatch is addressed to the
    same session_id (the holder), no 202.

    This test exercises the case where the holder dispatches to a
    sub-agent via /inject with target=@sub: that flow is "still inside
    the holder's turn", so it must NOT 202 even though the slot is
    held (by the holder itself).
    """
    project_id = "proj-subagent"
    mgr, project_store, _, _ = _make_agent_manager()
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }
    holding_sid = "sess-holder"
    handle, task = _inject_running_handle(mgr, project_id, holding_sid)

    sub_agent_mgr = MagicMock()
    sub_agent_mgr.send = AsyncMock(return_value="sub-agent reply")
    sub_agent_mgr.start = AsyncMock()
    sub_agent_mgr.get_transcript = MagicMock(return_value=None)

    client, saved = _build_test_client(mgr, project_store, sub_agent_mgr)
    try:
        # Sub-agent dispatch from the holder: target=@sub, session_id
        # matches the holder. Must NOT 202.
        resp = client.post(
            f"/api/v2/agents/{project_id}/inject",
            json={
                "content": "do the thing",
                "target": "researcher",
                "session_id": holding_sid,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "sub-agent reply"}
        sub_agent_mgr.send.assert_awaited()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        client.close()
        _restore(saved)


# ---------------------------------------------------------------------------
# Bonus coverage: current_holder_session_id accessor behaves correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_holder_session_id_returns_none_when_idle():
    """Accessor returns None when no handle exists or task is done."""
    mgr, _, _, _ = _make_agent_manager()
    # No handle at all.
    assert mgr.current_holder_session_id("nonexistent") is None

    # Handle with a done task.
    project_id = "proj-done"
    handle, task = _inject_running_handle(mgr, project_id, "sess-X")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Task is now done; accessor must return None.
    assert mgr.current_holder_session_id(project_id) is None


@pytest.mark.asyncio
async def test_current_holder_session_id_returns_holder_when_running():
    """Accessor returns the live session_id when the task is alive."""
    mgr, _, _, _ = _make_agent_manager()
    project_id = "proj-live"
    handle, task = _inject_running_handle(mgr, project_id, "sess-live")
    try:
        assert mgr.current_holder_session_id(project_id) == "sess-live"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
