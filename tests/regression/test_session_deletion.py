# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for single-session deletion
(TASK-session-naming-and-deletion.md PART 2).

Deletion removes the session JSONL from disk. A running session cannot be
deleted (409). If the session holds a live (idle) handle, the handle is removed
(like eviction) and any idle-poll task cancelled before the file is unlinked.

Tests 6-9 from the spec, exercised through both the AgentManager.delete_session
method and the DELETE HTTP endpoint.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.project_store import ProjectStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store():
    tmpdir = tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({
        "name": "Del Project",
        "workspace": tmpdir,
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    project = store.get_project(pid)
    sessions_dir = Path(ProjectPaths(project["workspace"]).sessions_dir)
    os.makedirs(sessions_dir, exist_ok=True)
    return store, pid, sessions_dir


def _make_manager(store):
    # sub_agent_manager.stop_all is awaited by stop_agent (the idle-handle
    # teardown path), so it must be an AsyncMock.
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


def _write_session(sessions_dir, uuid, session_id, ts="2026-05-25T08:00:00+00:00"):
    p = sessions_dir / f"{uuid}.jsonl"
    rows = [
        {"role": "meta", "event": "session_start", "session_id": session_id,
         "session_uuid": uuid, "timestamp": ts},
        {"role": "user", "content": "hi", "session_id": session_id, "timestamp": ts},
        {"role": "assistant", "content": "yo", "session_id": session_id, "timestamp": ts},
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


# ---------------------------------------------------------------------------
# Test 6 — Delete idle session
# ---------------------------------------------------------------------------

def test_delete_idle_session_removes_file_and_listing(tmp_path):
    store, pid, sessions_dir = _make_store()
    jsonl = _write_session(sessions_dir, "del_aaaaaaaa", "sess_a")
    mgr = _make_manager(store)

    # Sanity: it shows up in the listing before deletion.
    before = {s["session_uuid"] for s in mgr.list_sessions(pid)}
    assert "del_aaaaaaaa" in before
    assert jsonl.exists()

    client = _make_app(store, mgr)
    resp = client.request("DELETE", f"/api/v2/agents/{pid}/sessions/del_aaaaaaaa")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "deleted"
    assert body["session_id"] == "del_aaaaaaaa"

    # File gone from disk.
    assert not jsonl.exists()
    # No longer in list_sessions.
    after = {s["session_uuid"] for s in mgr.list_sessions(pid)}
    assert "del_aaaaaaaa" not in after


# ---------------------------------------------------------------------------
# Test 7 — Delete running session returns 409
# ---------------------------------------------------------------------------

def test_delete_running_session_returns_409(tmp_path):
    store, pid, sessions_dir = _make_store()
    jsonl = _write_session(sessions_dir, "del_running1", "sess_run")
    mgr = _make_manager(store)

    # Install a live (running) handle for this session.
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.session_uuid = "del_running1"
    session.session_id = "sess_run"
    session._messages = [{"timestamp": "2026-05-25T09:00:00+00:00"}]
    task = MagicMock()
    task.done.return_value = False  # running
    mgr._handles[(pid, "sess_run")] = ProjectHandle(
        session=session, loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=MagicMock(), task=task,
        config_snapshot={"workspace": store.get_project(pid)["workspace"]},
        started_at="2026-01-01T00:00:00+00:00",
    )

    client = _make_app(store, mgr)
    resp = client.request("DELETE", f"/api/v2/agents/{pid}/sessions/sess_run")
    assert resp.status_code == 409, resp.text
    # File must NOT be deleted.
    assert jsonl.exists()


# ---------------------------------------------------------------------------
# Test 8 — Delete currently-viewed (idle) session that holds a handle
# ---------------------------------------------------------------------------

def test_delete_idle_handle_removes_handle_and_file(tmp_path):
    store, pid, sessions_dir = _make_store()
    jsonl = _write_session(sessions_dir, "del_idle111", "sess_idle")
    mgr = _make_manager(store)

    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.session_uuid = "del_idle111"
    session.session_id = "sess_idle"
    session._messages = [{"timestamp": "2026-05-25T09:00:00+00:00"}]
    task = MagicMock()
    task.done.return_value = True  # idle (loop finished)
    sk = (pid, "sess_idle")
    mgr._handles[sk] = ProjectHandle(
        session=session, loop=AsyncMock(), provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=MagicMock(), task=task,
        config_snapshot={"workspace": store.get_project(pid)["workspace"]},
        started_at="2026-01-01T00:00:00+00:00",
    )

    client = _make_app(store, mgr)
    resp = client.request("DELETE", f"/api/v2/agents/{pid}/sessions/sess_idle")
    assert resp.status_code == 200, resp.text

    # Handle removed from _handles.
    assert sk not in mgr._handles
    # File deleted.
    assert not jsonl.exists()


# ---------------------------------------------------------------------------
# Test 9 — Delete the only session
# ---------------------------------------------------------------------------

def test_delete_only_session_leaves_project_with_zero_sessions(tmp_path):
    store, pid, sessions_dir = _make_store()
    jsonl = _write_session(sessions_dir, "del_only1234", "sess_only")
    mgr = _make_manager(store)

    assert len(mgr.list_sessions(pid)) == 1

    client = _make_app(store, mgr)
    resp = client.request("DELETE", f"/api/v2/agents/{pid}/sessions/del_only1234")
    assert resp.status_code == 200, resp.text

    assert not jsonl.exists()
    assert mgr.list_sessions(pid) == []


def test_delete_nonexistent_session_returns_404(tmp_path):
    store, pid, sessions_dir = _make_store()
    mgr = _make_manager(store)
    client = _make_app(store, mgr)
    resp = client.request("DELETE", f"/api/v2/agents/{pid}/sessions/does_not_exist")
    assert resp.status_code == 404, resp.text
