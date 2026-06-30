# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: ``current_holder_session_id`` must report the slot as held
when a session is in ``waiting`` state (main loop done, sub-agents busy).

Before this fix: the slot predicate only checked
``handle.task is not None and not handle.task.done()``. Once the main loop
exited, the slot was reported as free even if sub-agents were still
writing to the workspace — letting other sessions dispatch concurrently
and violating the single-loop-per-project safety claim from
TASK/ACTIVE-session-and-queue-model.md §3.

After this fix: the predicate also returns the session as the slot holder
when ``_idle_poll_tasks[project_id]`` is alive (sub-agents being polled) or
when the session is paused for approval. Cross-session inject during
``waiting`` returns 202 ``slot_held``.

See: ``TASK/TASK-state-model-alignment-fixes.md`` §2.2.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


def _make_agent_manager():
    project_store = MagicMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.send = AsyncMock(return_value="ok")
    sub_agent_mgr.start = AsyncMock()
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
    return mgr, project_store


def _inject_waiting_handle(mgr, project_id, holding_session_id):
    """Plant a handle whose main-loop task is DONE but with a live
    ``_idle_poll_tasks`` entry — the ``waiting`` state."""
    session = MagicMock()
    session.session_id = holding_session_id
    session.is_stopped = MagicMock(return_value=False)
    session._paused_for_approval = False

    # Main loop task — done.
    async def _done_immediately():
        return None
    done_task = asyncio.get_event_loop().create_task(_done_immediately())

    handle = ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=done_task,
    )
    mgr._handles[(project_id, holding_session_id)] = handle

    # Sub-agent poll task — live. Keyed by the SessionKey tuple
    # (project_id, session_id) to match production: current_holder_session_id
    # and _stop_sub_agents look up _idle_poll_tasks via make_session_key, not a
    # bare project_id (the pre-multi-session relic this test predated).
    async def _never_done():
        await asyncio.sleep(9999)
    poll_task = asyncio.get_event_loop().create_task(_never_done())
    mgr._idle_poll_tasks[(project_id, holding_session_id)] = poll_task

    return handle, done_task, poll_task


def _inject_approval_paused_handle(mgr, project_id, holding_session_id):
    """Plant a handle in pending_approval — main-loop task done,
    ``_paused_for_approval`` True."""
    session = MagicMock()
    session.session_id = holding_session_id
    session.is_stopped = MagicMock(return_value=False)
    session._paused_for_approval = True

    async def _done_immediately():
        return None
    done_task = asyncio.get_event_loop().create_task(_done_immediately())

    handle = ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=done_task,
    )
    mgr._handles[(project_id, holding_session_id)] = handle
    return handle, done_task


def _build_test_client(mgr, project_store):
    saved = {
        k: getattr(agents_v2, k)
        for k in ("_project_store", "_agent_manager", "_ws_manager",
                  "_sub_agent_manager")
    }
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.send = AsyncMock(return_value="ok")
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
    return TestClient(app), saved


def _restore(saved):
    for k, v in saved.items():
        setattr(agents_v2, k, v)


@pytest.mark.asyncio
async def test_current_holder_returns_session_when_waiting():
    """Directly assert the accessor reports the waiting session as holder."""
    mgr, _ = _make_agent_manager()
    project_id = "proj_waiting_holds"
    handle, done_task, poll_task = _inject_waiting_handle(
        mgr, project_id, "sess-A",
    )

    try:
        # Wait for the done task to actually complete.
        await done_task
        assert handle.task.done(), "Sanity: main loop task must be done"
        assert not poll_task.done(), "Sanity: poll task must still be alive"

        holder = mgr.current_holder_session_id(project_id)
        assert holder == "sess-A", (
            f"Expected sess-A to still hold the slot during waiting, "
            f"got {holder!r}. The slot predicate didn't account for "
            f"_idle_poll_tasks."
        )
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_cross_session_inject_during_waiting_returns_202():
    """End-to-end: HTTP inject to a different session while session A is in
    ``waiting`` is ENQUEUED (Path A, spec 006) → 202 ``queued_pending_slot``
    with sess-A as the holding session.

    Rewritten from the old ``slot_held`` reject contract (spec 006 §5): a
    waiting holder still holds the slot, so B's message is queued for delivery
    when the slot frees rather than rejected."""
    mgr, project_store = _make_agent_manager()
    project_id = "proj_waiting_inject_202"
    project_store.get_project.return_value = {
        "project_id": project_id, "workspace": "/tmp/ws", "name": "p",
    }
    handle, done_task, poll_task = _inject_waiting_handle(
        mgr, project_id, "sess-A",
    )
    await done_task  # ensure waiting state is settled

    client, saved = _build_test_client(mgr, project_store)
    try:
        resp = client.post(
            f"/api/v2/agents/{project_id}/inject",
            json={"content": "hi from B", "session_id": "sess-B"},
        )
        assert resp.status_code == 202, (
            f"Expected 202 queued_pending_slot during waiting, got "
            f"{resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body["status"] == "queued_pending_slot"
        assert body["holding_session_id"] == "sess-A"
        assert body["queued_session_id"] == "sess-B"
        assert body["position"] == 1
        # Enqueued in the registry; nothing dispatched while A still waits.
        assert len(mgr._pending_inject[project_id]) == 1
    finally:
        client.close()
        _restore(saved)
        poll_task.cancel()
        try:
            await poll_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_current_holder_returns_session_when_pending_approval():
    """The slot predicate must also report the slot held when a session is
    paused for approval (the docstring's existing claim — previously the
    predicate failed to honor it because the task was done)."""
    mgr, _ = _make_agent_manager()
    project_id = "proj_approval_holds"
    handle, done_task = _inject_approval_paused_handle(
        mgr, project_id, "sess-A",
    )
    await done_task

    holder = mgr.current_holder_session_id(project_id)
    assert holder == "sess-A", (
        f"Expected sess-A to hold the slot while in pending_approval, "
        f"got {holder!r}. The slot predicate must honor "
        f"_paused_for_approval."
    )
