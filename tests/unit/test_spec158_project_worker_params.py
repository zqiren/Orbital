# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Item 158 — a pinned worker's model / thinking, once picked, stays fixed
per project.

The pin-menu pick (spec 149, per conversation) is also stored on the project
(``sub_agent_params``) and used by every session there that has no pick of
its own, new sessions included, until it is changed again. A pick made inside
a session still wins inside that session. With nothing stored on the project
the Settings value (the global default) applies, as before.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


class _Store:
    def __init__(self, project):
        self.project = project

    def get_project(self, pid):
        return self.project if pid == self.project["project_id"] else None


def _mgr(project, sessions):
    mgr = SubAgentManager(process_manager=MagicMock(), project_store=_Store(project))
    mgr._session_resolver = lambda pid, sid: sessions.get(sid)
    return mgr


class _Sess:
    def __init__(self, params=None):
        self.sub_agent_params = params or {}

    def set_sub_agent_params(self, slug, params):
        if params:
            self.sub_agent_params[slug] = dict(params)
        else:
            self.sub_agent_params.pop(slug, None)


def test_new_session_uses_the_project_pick():
    project = {"project_id": "p", "sub_agent_params": {"codex": {"model": "gpt-5.6", "effort": "high"}}}
    mgr = _mgr(project, {"new": _Sess()})
    assert mgr.session_params("p", "new", "codex") == {"model": "gpt-5.6", "effort": "high"}
    assert mgr.session_params("p", "new") == {"codex": {"model": "gpt-5.6", "effort": "high"}}
    # A minted session with no file yet reads it too.
    assert mgr.session_params("p", "not-written-yet", "codex") == {"model": "gpt-5.6", "effort": "high"}


def test_session_pick_wins_inside_its_session():
    project = {"project_id": "p", "sub_agent_params": {"codex": {"model": "gpt-5.6"}}}
    mgr = _mgr(project, {"own": _Sess({"codex": {"effort": "low"}})})
    assert mgr.session_params("p", "own", "codex") == {"effort": "low"}
    assert mgr.session_params("p", "own") == {"codex": {"effort": "low"}}


def test_no_project_pick_follows_settings():
    mgr = _mgr({"project_id": "p"}, {"s": _Sess()})
    assert mgr.session_params("p", "s", "codex") == {}
    assert mgr.session_params("p", "s") == {}


# ── Through the routes (the new-session case) ─────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)
    from fastapi.testclient import TestClient
    from agent_os.api.app import create_app
    from agent_os.api.routes import settings as settings_routes
    app = create_app(data_dir=str(tmp_path / "data"))

    async def _no_live(*_a, **_k):
        return None
    monkeypatch.setattr(settings_routes, "_claude_live_models", _no_live)
    monkeypatch.setattr(settings_routes, "_codex_live_models", _no_live)
    return TestClient(app)


def _project(client, tmp_path):
    ws = str(tmp_path / "ws158")
    os.makedirs(ws, exist_ok=True)
    resp = client.post("/api/v2/projects", json={
        "name": "p158", "workspace": ws, "model": "gpt-4", "api_key": "test-key"})
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


def test_pick_in_one_session_carries_into_a_new_session(client, tmp_path):
    pid = _project(client, tmp_path)
    first = f"/api/v2/agents/{pid}/sessions/sess_a/sub-agent-params"
    r = client.patch(first, json={"slug": "claude-code", "model": "opus", "effort": "high"})
    assert r.status_code == 200, r.text
    # Stored on the project, additively.
    project = client.get(f"/api/v2/projects/{pid}").json()
    assert project["sub_agent_params"] == {"claude-code": {"model": "opus", "effort": "high"}}
    # A brand-new session in the project starts on the pick.
    minted = client.post(f"/api/v2/agents/{pid}/new-session", json={}).json()["session_id"]
    fresh = f"/api/v2/agents/{pid}/sessions/{minted}/sub-agent-params"
    assert client.get(fresh).json()["params"] == {"claude-code": {"model": "opus", "effort": "high"}}
    # A change inside the new session wins there, and becomes the project's.
    client.patch(fresh, json={"slug": "claude-code", "effort": "low"})
    assert client.get(fresh).json()["params"]["claude-code"] == {"model": "opus", "effort": "low"}
    assert client.get(f"/api/v2/projects/{pid}").json()["sub_agent_params"] == {
        "claude-code": {"model": "opus", "effort": "low"}}


def test_clearing_to_default_clears_the_project_pick(client, tmp_path):
    pid = _project(client, tmp_path)
    base = f"/api/v2/agents/{pid}/sessions/sess_b/sub-agent-params"
    client.patch(base, json={"slug": "codex", "effort": "low"})
    client.patch(base, json={"slug": "codex", "effort": None})
    assert client.get(f"/api/v2/projects/{pid}").json().get("sub_agent_params") in ({}, None)
    other = f"/api/v2/agents/{pid}/sessions/sess_c/sub-agent-params"
    assert client.get(other).json()["params"] == {}
