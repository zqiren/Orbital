# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression LOCK on the EXISTING session-delete reject-running behavior.

This is NOT a RED-before test. The behavior already exists and is already
GREEN: ``AgentManager.delete_session`` (agent_manager.py:2063) raises
``RuntimeError`` when the targeted session is running, and the route
(agents_v2.py:971-972) translates that to HTTP 409. The companion
queue-item delete fix (DELETE IS IDLE-ONLY) mirrors this convention, so we lock
the session-delete side here to guard against an accidental regression that
would let the queue and session delete paths drift apart.

Mirror of ``tests/regression/test_session_deletion.py::
test_delete_running_session_returns_409`` kept as a focused, self-documenting
lock alongside the queue-item reject-running test.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.project_store import ProjectStore


def _make_store():
    tmpdir = tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({
        "name": "Reject Running Session Project",
        "workspace": tmpdir,
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    project = store.get_project(pid)
    sessions_dir = Path(ProjectPaths(project["workspace"]).sessions_dir)
    os.makedirs(sessions_dir, exist_ok=True)
    return store, pid, sessions_dir


def _make_manager(store):
    sub_agent_manager = MagicMock()
    sub_agent_manager.stop_all = AsyncMock()
    return AgentManager(
        project_store=store,
        ws_manager=MagicMock(),
        sub_agent_manager=sub_agent_manager,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        registry=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=None,
        credential_store=None,
    )


def _write_session(sessions_dir, uuid, session_id, ts="2026-06-08T08:00:00+00:00"):
    p = sessions_dir / f"{uuid}.jsonl"
    rows = [
        {"role": "meta", "event": "session_start", "session_id": session_id,
         "session_uuid": uuid, "timestamp": ts},
        {"role": "user", "content": "hi", "session_id": session_id, "timestamp": ts},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _make_app(store, manager):
    from agent_os.api.routes import agents_v2
    app = FastAPI()
    agents_v2.configure(
        project_store=store,
        agent_manager=manager,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=MagicMock(),
        credential_store=MagicMock(),
    )
    app.include_router(agents_v2.router)
    return TestClient(app)


def test_delete_running_session_returns_409_lock(tmp_path):
    """ALREADY-GREEN lock: deleting a running session is rejected (409) and the
    JSONL is left on disk. No RED-before phase — the behavior pre-exists."""
    store, pid, sessions_dir = _make_store()
    jsonl = _write_session(sessions_dir, "lockrun_aaaa", "sess_lockrun")
    mgr = _make_manager(store)

    # Install a live (running) handle: task not done, not paused, not stopped.
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.session_uuid = "lockrun_aaaa"
    session.session_id = "sess_lockrun"
    session._messages = [{"timestamp": "2026-06-08T09:00:00+00:00"}]
    task = MagicMock()
    task.done.return_value = False  # running
    mgr._handles[(pid, "sess_lockrun")] = ProjectHandle(
        session=session, loop=MagicMock(), provider=MagicMock(),
        registry=MagicMock(), context_manager=MagicMock(),
        interceptor=MagicMock(), task=task,
        config_snapshot={"workspace": store.get_project(pid)["workspace"]},
        started_at="2026-01-01T00:00:00+00:00",
    )

    client = _make_app(store, mgr)
    resp = client.request(
        "DELETE", f"/api/v2/agents/{pid}/sessions/sess_lockrun"
    )

    assert resp.status_code == 409, resp.text
    # JSONL must NOT have been deleted by the rejected request.
    assert jsonl.exists(), "running session's JSONL must survive a rejected delete"
