# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test: the user-facing ``/stop`` endpoint is gone.

Under the multi-session + idle-eviction model there are exactly two
user-facing "stop working" surfaces:

- ``/cancel`` — interrupt the current turn; the handle and session stay in
  memory, resumable immediately. This is what the UI Stop button calls.
- (nothing else) — resource teardown is automatic via idle eviction.

``stop_agent`` survives as an INTERNAL method (eviction sweep, daemon
shutdown, project deletion) but is no longer exposed as an HTTP route. A
``POST /api/v2/agents/{id}/stop`` must now 404 (route removed) — FastAPI
returns 404 for an unmatched path. ``/cancel`` must still 200 and leave the
session loaded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


# Explicit session id: "default" sentinel retired in 433912a (seam 3 / D1);
# the /cancel route forwards the body session_id to cancel_message.
SID = "sess-stoprm-1"


def _build_client():
    """AgentManager with one idle handle, wired behind a TestClient."""
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.stop_all = AsyncMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    mgr = AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=sub_agent_mgr,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )

    # An idle handle: no running task, not paused. /cancel on this returns
    # {"status": "idle"} (legitimate no-op) and leaves the handle in place.
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    handle = ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=None,
    )
    pid = "proj_stopgone"
    mgr._handles[(pid, SID)] = handle

    saved = {k: getattr(agents_v2, k)
             for k in ["_project_store", "_agent_manager", "_ws_manager", "_sub_agent_manager"]}
    agents_v2.configure(
        project_store=MagicMock(),
        agent_manager=mgr,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
    )
    app = FastAPI()
    app.include_router(agents_v2.router)
    client = TestClient(app)
    return client, mgr, pid, saved


def _restore(saved):
    for k, v in saved.items():
        setattr(agents_v2, k, v)


def test_stop_route_removed_returns_404():
    """POST /api/v2/agents/{id}/stop is no longer a route → 404 (or 405)."""
    client, _mgr, pid, saved = _build_client()
    try:
        resp = client.post(f"/api/v2/agents/{pid}/stop")
        assert resp.status_code in (404, 405), (
            f"/stop should be removed, got {resp.status_code}"
        )
    finally:
        client.close()
        _restore(saved)


def test_cancel_still_works_and_keeps_session():
    """POST /api/v2/agents/{id}/cancel still 200s and leaves the handle loaded."""
    client, mgr, pid, saved = _build_client()
    try:
        resp = client.post(f"/api/v2/agents/{pid}/cancel", json={"session_id": SID})
        assert resp.status_code == 200
        # idle handle → cancel is a no-op that reports the live status.
        assert resp.json()["status"] == "idle"
        # Session stays in memory, resumable immediately (NOT torn down).
        assert (pid, SID) in mgr._handles
    finally:
        client.close()
        _restore(saved)
