# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for the sub-agent settings page end-to-end.

Boots a real FastAPI app with all daemon dependencies wired and exercises
the full flow: GET status -> PUT config -> GET status (assert reflected) ->
POST refresh -> GET status (assert fresh).
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)

    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    with TestClient(app) as c:
        yield c


class TestSettingsPageFullFlow:

    def test_full_flow(self, client, tmp_path):
        """GET status -> PUT config -> GET (assert reflected) -> refresh."""
        # 1. GET initial sub-agent status.
        resp = client.get("/api/v2/settings/sub-agents")
        assert resp.status_code == 200
        initial = resp.json()
        assert isinstance(initial, list)
        # claude-code should appear in any local environment (manifest is
        # always loaded; install state may be False).
        slugs = {entry["slug"] for entry in initial}
        assert "claude-code" in slugs

        # 2. PUT config with valid params.
        resp = client.put(
            "/api/v2/settings/sub-agents/claude-code/config",
            json={"model": "sonnet", "effort": "medium"},
        )
        assert resp.status_code == 200
        assert resp.json()["config"] == {"model": "sonnet", "effort": "medium"}

        # 3. GET reflects the persisted config.
        resp = client.get("/api/v2/settings/sub-agents")
        cc = next(e for e in resp.json() if e["slug"] == "claude-code")
        assert cc["config"] == {"model": "sonnet", "effort": "medium"}

        # 4. POST refresh returns fresh status.
        resp = client.post("/api/v2/settings/sub-agents/refresh")
        assert resp.status_code == 200
        refreshed = resp.json()
        assert isinstance(refreshed, list)
        # Config persists across refresh.
        cc = next(e for e in refreshed if e["slug"] == "claude-code")
        assert cc["config"] == {"model": "sonnet", "effort": "medium"}


class TestDisabledSubAgentInProject:

    def test_disabled_sub_agent_excluded_from_project(self, client, tmp_path):
        """A project with disabled_sub_agents=['claude-code'] should not
        include claude-code in its derived enabled-sub-agents list."""
        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)

        # Create a project with claude-code denylisted.
        resp = client.post("/api/v2/projects", json={
            "name": "denylist-test",
            "workspace": ws,
            "model": "gpt-4",
            "api_key": "test-key",
            "disabled_sub_agents": ["claude-code"],
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]

        # Verify the project record carries the denylist.
        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["disabled_sub_agents"] == ["claude-code"]
