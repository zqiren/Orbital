# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression (Root C / seam 3): deleting a RUNNING project must stop the
holder loop, not orphan it.

``is_running(pid)`` is holder-aware (``_sid_read`` → the running uuid session)
but ``stop_agent(pid)`` is passthrough-None (``_resolve_session_id`` → None →
handle-miss → ``KeyError``). So the delete routes, which call ``stop_agent(pid)``
with no session id, raised ``KeyError`` (→ 500 for ``delete_project``; counted
"failed" for ``bulk_delete_projects``) and left the loop running against data the
user believes is gone. Fix is caller-side: forward the holder
(``current_holder_session_id(pid)``). The shared callee ``stop_agent``'s
None-policy is deliberately NOT changed.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import BulkDeleteRequest
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.models import make_session_key


def _running_handle():
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.task = MagicMock()
    h.task.done.return_value = False  # live loop → holds the slot
    return h


def _mgr_with_running_session(pid, sid):
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    mgr._ws.broadcast = MagicMock()
    mgr._handles[make_session_key(pid, sid)] = _running_handle()
    return mgr


@pytest.mark.asyncio
async def test_delete_running_project_stops_the_holder_loop(monkeypatch):
    pid, sid = "proj_x", "proj_x_sess1"
    mgr = _mgr_with_running_session(pid, sid)
    # The asymmetry that caused the bug: holder-aware read finds the uuid-keyed
    # running loop, even though stop_agent(pid) with no session id would miss it.
    assert mgr.is_running(pid) is True
    assert mgr.current_holder_session_id(pid) == sid

    stopped = []

    async def spy_stop(project_id, *, session_id=None):
        stopped.append(session_id)
        mgr._handles.pop(make_session_key(project_id, session_id), None)

    monkeypatch.setattr(mgr, "stop_agent", spy_stop)
    monkeypatch.setattr(mgr, "shutdown_dispatcher", AsyncMock())

    project_store = MagicMock()
    project_store.get_project.return_value = {"workspace": "", "project_id": pid}
    project_store.delete_project = MagicMock()
    agents_v2.configure(project_store, mgr, MagicMock(), MagicMock())
    monkeypatch.setattr(agents_v2, "_cleanup_project_files", lambda ws: None)

    result = await agents_v2.delete_project(pid)

    assert result == {"status": "deleted"}
    # The running holder loop was the one stopped (forwarded), not a (pid, None) miss.
    assert stopped == [sid]
    # No orphan: the loop is no longer running for this project.
    assert mgr.is_running(pid) is False


@pytest.mark.asyncio
async def test_bulk_delete_old_stops_running_project_holder(monkeypatch):
    pid, sid = "proj_old", "proj_old_sess1"
    mgr = _mgr_with_running_session(pid, sid)

    stopped = []

    async def spy_stop(project_id, *, session_id=None):
        stopped.append((project_id, session_id))
        mgr._handles.pop(make_session_key(project_id, session_id), None)

    monkeypatch.setattr(mgr, "stop_agent", spy_stop)

    project_store = MagicMock()
    project_store.list_projects.return_value = [
        {"project_id": pid, "workspace": "", "is_scratch": False},
    ]
    project_store.delete_project = MagicMock()
    agents_v2.configure(project_store, mgr, MagicMock(), MagicMock())
    monkeypatch.setattr(agents_v2, "_cleanup_project_files", lambda ws: None)

    result = await agents_v2.bulk_delete_projects(BulkDeleteRequest(project_ids=[pid]))

    assert result == {"deleted": 1, "failed": 0}
    # The running holder loop was stopped under its real session, not orphaned.
    assert stopped == [(pid, sid)]
    assert mgr.is_running(pid) is False
