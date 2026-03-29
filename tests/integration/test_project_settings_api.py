# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for project settings API -- real user journeys."""
import os
import pytest
from fastapi.testclient import TestClient
from agent_os.api.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


class TestProjectSettingsJourney:
    """User journey: create project -> update settings -> verify persistence."""

    def test_create_project_with_agent_name(self, client, workspace):
        resp = client.post("/api/v2/projects", json={
            "name": "Auth Refactor",
            "workspace": workspace,
            "model": "gpt-4",
            "api_key": "sk-test",
            "agent_name": "AuthBot",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["agent_name"] == "AuthBot"
        assert data["is_scratch"] is False

    def test_create_project_defaults_agent_name(self, client, workspace):
        resp = client.post("/api/v2/projects", json={
            "name": "MyProject",
            "workspace": workspace,
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        assert resp.status_code == 201
        assert resp.json()["agent_name"] == "MyProject"

    def test_duplicate_agent_name_rejected(self, client, workspace, tmp_path):
        ws2 = tmp_path / "workspace2"
        ws2.mkdir()
        client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "k", "agent_name": "Bot",
        })
        resp = client.post("/api/v2/projects", json={
            "name": "P2", "workspace": str(ws2),
            "model": "m", "api_key": "k", "agent_name": "Bot",
        })
        assert resp.status_code == 409

    def test_update_agent_name(self, client, workspace):
        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "k",
        })
        pid = resp.json()["project_id"]
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "agent_name": "NewBot",
        })
        assert resp.status_code == 200
        assert resp.json()["agent_name"] == "NewBot"

    def test_get_project_includes_workspace_file_content(self, client, workspace):
        """User journey: save project goals via PUT, verify they appear in GET."""
        # Create project
        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "k",
        })
        pid = resp.json()["project_id"]

        # Write project goals via API
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "# Mission\nBuild auth system",
        })
        assert resp.status_code == 200

        # Read back via GET
        resp = client.get(f"/api/v2/projects/{pid}")
        data = resp.json()
        assert data["project_goals_content"] == "# Mission\nBuild auth system"

        # Verify file was created on disk
        goals_path = os.path.join(
            workspace, "orbital", "instructions", "project_goals.md"
        )
        assert os.path.exists(goals_path)
        with open(goals_path) as f:
            assert f.read() == "# Mission\nBuild auth system"

    def test_standing_rules_roundtrip(self, client, workspace):
        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "k",
        })
        pid = resp.json()["project_id"]

        client.put(f"/api/v2/projects/{pid}", json={
            "user_directives_content": "Always write tests\nUse PostgreSQL",
        })

        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.json()["user_directives_content"] == "Always write tests\nUse PostgreSQL"

    # --- New tests for settings sync fixes (GAP 1, 5) ---

    def test_put_response_includes_disk_content(self, client, workspace):
        """GAP 5: PUT must return project_goals_content and user_directives_content."""
        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "k",
        })
        pid = resp.json()["project_id"]

        # Save goals
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "test goals",
        })
        assert resp.status_code == 200
        assert resp.json()["project_goals_content"] == "test goals"

        # Save directives
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "user_directives_content": "test rules",
        })
        assert resp.status_code == 200
        assert resp.json()["user_directives_content"] == "test rules"
        # Goals should still be present from previous save
        assert resp.json()["project_goals_content"] == "test goals"

    def test_list_does_not_include_disk_content(self, client, workspace):
        """GAP 1: list endpoint stays lean — no disk content fields."""
        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "k",
        })
        pid = resp.json()["project_id"]

        # Write content via PUT
        client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "some goals",
            "user_directives_content": "some rules",
        })

        # List should NOT include disk content
        resp = client.get("/api/v2/projects")
        assert resp.status_code == 200
        proj = next(p for p in resp.json() if p["project_id"] == pid)
        assert "project_goals_content" not in proj
        assert "user_directives_content" not in proj

    def test_detail_endpoint_returns_disk_content(self, client, workspace):
        """GAP 1: detail endpoint returns disk content for frontend to populate textareas."""
        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "k",
        })
        pid = resp.json()["project_id"]

        client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "goals from api",
            "user_directives_content": "rules from api",
        })

        # Simulate what the frontend does: list first (no content), then detail
        list_resp = client.get("/api/v2/projects")
        list_proj = next(p for p in list_resp.json() if p["project_id"] == pid)
        assert "project_goals_content" not in list_proj

        detail_resp = client.get(f"/api/v2/projects/{pid}")
        detail = detail_resp.json()
        assert detail["project_goals_content"] == "goals from api"
        assert detail["user_directives_content"] == "rules from api"

    def test_scratch_project_cannot_be_deleted(self, client, workspace):
        resp = client.post("/api/v2/projects", json={
            "name": "Quick Tasks", "workspace": workspace,
            "model": "m", "api_key": "k", "is_scratch": True,
            "agent_name": "ScratchBot",
        })
        pid = resp.json()["project_id"]
        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 403
