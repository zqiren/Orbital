# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""1125 — the pin menu's model switcher before the project has a session.

A fresh install lands in Quick Tasks with no session at all (no route
session id). The pin menu still offers each worker's model there; with no
conversation to hold the pick, it is the project's pick (item 158's
``sub_agent_params``), which the first send's new session starts on. These
routes read and write that pick without a session, validated exactly like
the per-conversation route.
"""

from __future__ import annotations

import os

import pytest


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
    ws = str(tmp_path / "ws1125")
    os.makedirs(ws, exist_ok=True)
    resp = client.post("/api/v2/projects", json={
        "name": "p1125", "workspace": ws, "model": "gpt-4", "api_key": "test-key"})
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


def test_pick_without_a_session_is_the_projects_and_the_first_session_starts_on_it(client, tmp_path):
    pid = _project(client, tmp_path)
    base = f"/api/v2/agents/{pid}/sub-agent-params"
    assert client.get(base).json()["params"] == {}

    r = client.patch(base, json={"slug": "claude-code", "model": "opus", "effort": "high"})
    assert r.status_code == 200, r.text
    assert r.json()["params"] == {"model": "opus", "effort": "high"}
    assert client.get(base).json()["params"] == {"claude-code": {"model": "opus", "effort": "high"}}
    assert client.get(f"/api/v2/projects/{pid}").json()["sub_agent_params"] == {
        "claude-code": {"model": "opus", "effort": "high"}}

    # One field at a time merges into the stored pick.
    client.patch(base, json={"slug": "claude-code", "effort": "low"})
    assert client.get(base).json()["params"]["claude-code"] == {"model": "opus", "effort": "low"}

    # The first session in the project starts on it.
    minted = client.post(f"/api/v2/agents/{pid}/new-session", json={}).json()["session_id"]
    fresh = f"/api/v2/agents/{pid}/sessions/{minted}/sub-agent-params"
    assert client.get(fresh).json()["params"] == {"claude-code": {"model": "opus", "effort": "low"}}


def test_default_clears_the_project_pick(client, tmp_path):
    pid = _project(client, tmp_path)
    base = f"/api/v2/agents/{pid}/sub-agent-params"
    client.patch(base, json={"slug": "codex", "effort": "low"})
    r = client.patch(base, json={"slug": "codex", "effort": None})
    assert r.status_code == 200, r.text
    assert r.json()["params"] == {}
    assert client.get(base).json()["params"] == {}
    assert client.get(f"/api/v2/projects/{pid}").json().get("sub_agent_params") in ({}, None)


def test_a_value_settings_would_refuse_is_refused_and_nothing_is_stored(client, tmp_path):
    pid = _project(client, tmp_path)
    base = f"/api/v2/agents/{pid}/sub-agent-params"
    r = client.patch(base, json={"slug": "claude-code", "effort": "not-a-level"})
    assert r.status_code == 400, r.text
    assert client.get(base).json()["params"] == {}


def test_unknown_project_is_404(client):
    assert client.get("/api/v2/agents/nope/sub-agent-params").status_code == 404
    r = client.patch("/api/v2/agents/nope/sub-agent-params", json={"slug": "codex", "effort": "low"})
    assert r.status_code == 404
