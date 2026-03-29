# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Full user journey integration tests per CLAUDE.md requirements.

These tests verify real multi-step user scenarios end-to-end through the API,
using TestClient with real temp directories — no mocks.
"""
import os
import pytest
from fastapi.testclient import TestClient
from agent_os.api.app import create_app


@pytest.fixture
def setup(tmp_path):
    """Boot a fresh app with isolated data dir and workspace."""
    data_dir = str(tmp_path / "data")
    app = create_app(data_dir=data_dir)
    client = TestClient(app)
    ws = tmp_path / "workspace"
    ws.mkdir()
    return client, str(ws), tmp_path


class TestJourney_CreateProjectConfigureSettingsVerifyPrompt:
    """Journey: User creates project -> sets agent name -> writes goals and rules
    -> verifies all are persisted and returned correctly."""

    def test_full_project_setup_journey(self, setup):
        client, workspace, tmp_path = setup

        # 1. Create project with agent name
        resp = client.post("/api/v2/projects", json={
            "name": "Auth System",
            "workspace": workspace,
            "model": "gpt-4",
            "api_key": "sk-test-key-12345678",
            "agent_name": "AuthBot",
        })
        assert resp.status_code == 201
        data = resp.json()
        pid = data["project_id"]
        assert data["agent_name"] == "AuthBot"
        assert data["is_scratch"] is False

        # 2. Update project goals via PUT
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "# Mission\nBuild secure authentication system",
        })
        assert resp.status_code == 200

        # 3. Update standing rules via PUT
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "user_directives_content": "Always use bcrypt for hashing\nWrite tests first",
        })
        assert resp.status_code == 200

        # 4. Verify GET returns everything correctly
        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_name"] == "AuthBot"
        assert "Build secure authentication" in data["project_goals_content"]
        assert "bcrypt" in data["user_directives_content"]
        assert "Write tests first" in data["user_directives_content"]

        # 5. Verify files were created on disk
        goals_path = os.path.join(
            workspace, "orbital", "instructions", "project_goals.md"
        )
        rules_path = os.path.join(
            workspace, "orbital", "instructions", "user_directives.md"
        )
        assert os.path.exists(goals_path)
        assert os.path.exists(rules_path)
        with open(goals_path, encoding="utf-8") as f:
            assert "Build secure authentication" in f.read()
        with open(rules_path, encoding="utf-8") as f:
            content = f.read()
            assert "bcrypt" in content
            assert "Write tests first" in content

    def test_update_agent_name_reflects_in_get(self, setup):
        client, workspace, _ = setup

        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "sk-test-key-12345678",
        })
        pid = resp.json()["project_id"]

        resp = client.put(f"/api/v2/projects/{pid}", json={
            "agent_name": "RenamedBot",
        })
        assert resp.status_code == 200

        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.json()["agent_name"] == "RenamedBot"

    def test_goals_and_rules_start_empty(self, setup):
        """New project has empty goals and rules content."""
        client, workspace, _ = setup

        resp = client.post("/api/v2/projects", json={
            "name": "Fresh", "workspace": workspace,
            "model": "m", "api_key": "sk-test-key-12345678",
        })
        pid = resp.json()["project_id"]

        resp = client.get(f"/api/v2/projects/{pid}")
        data = resp.json()
        assert data["project_goals_content"] == ""
        assert data["user_directives_content"] == ""


class TestJourney_QuickTasksReadyOnFirstBoot:
    """Journey: Daemon starts -> Quick Tasks exists -> user can interact immediately."""

    def test_quick_tasks_ready_immediately(self, setup):
        client, _, _ = setup

        # 1. Check projects — Quick Tasks should exist from startup
        resp = client.get("/api/v2/projects")
        projects = resp.json()
        scratch = [p for p in projects if p.get("is_scratch")]
        assert len(scratch) == 1
        qt = scratch[0]
        assert qt["name"] == "Quick Tasks"
        assert qt["agent_name"] == "Assistant"

        # 2. Verify goals file exists (should skip onboarding)
        pid = qt["project_id"]
        resp = client.get(f"/api/v2/projects/{pid}")
        data = resp.json()
        assert "Quick Tasks Assistant" in data.get("project_goals_content", "")

    def test_quick_tasks_appears_first_in_project_list(self, setup):
        client, workspace, _ = setup

        # Create a regular project after boot
        client.post("/api/v2/projects", json={
            "name": "Regular Project", "workspace": workspace,
            "model": "m", "api_key": "sk-test-key-12345678",
        })

        resp = client.get("/api/v2/projects")
        projects = resp.json()
        assert len(projects) >= 2
        # Quick Tasks (scratch) should always be first
        assert projects[0].get("is_scratch") is True
        assert projects[0]["name"] == "Quick Tasks"

    def test_quick_tasks_cannot_be_deleted(self, setup):
        client, _, _ = setup

        resp = client.get("/api/v2/projects")
        scratch = [p for p in resp.json() if p.get("is_scratch")][0]

        resp = client.delete(f"/api/v2/projects/{scratch['project_id']}")
        assert resp.status_code == 403

    def test_quick_tasks_not_duplicated_on_second_app_creation(self, tmp_path):
        """Simulates daemon restart — scratch should not be created twice."""
        data_dir = str(tmp_path / "data")

        # First boot
        app1 = create_app(data_dir=data_dir)
        client1 = TestClient(app1)
        resp1 = client1.get("/api/v2/projects")
        scratch1 = [p for p in resp1.json() if p.get("is_scratch")]
        assert len(scratch1) == 1

        # Second boot (simulated restart)
        app2 = create_app(data_dir=data_dir)
        client2 = TestClient(app2)
        resp2 = client2.get("/api/v2/projects")
        scratch2 = [p for p in resp2.json() if p.get("is_scratch")]
        assert len(scratch2) == 1
        assert scratch2[0]["project_id"] == scratch1[0]["project_id"]


class TestJourney_GlobalPreferencesRoundTrip:
    """Journey: User sets global preferences -> they persist across requests."""

    def test_global_preferences_set_and_retrieved(self, setup):
        client, _, tmp_path = setup

        prefs_path = str(tmp_path / "user_preferences.md")
        resp = client.put("/api/v2/settings", json={
            "user_preferences_content": "I prefer TypeScript over JavaScript\nI'm a senior developer",
            "user_preferences_path": prefs_path,
        })
        assert resp.status_code == 200

        resp = client.get("/api/v2/settings")
        data = resp.json()
        assert "TypeScript" in data.get("user_preferences_content", "")
        assert "senior developer" in data.get("user_preferences_content", "")

    def test_scratch_workspace_persists(self, setup):
        client, _, tmp_path = setup

        scratch_dir = str(tmp_path / "my-scratch")
        resp = client.put("/api/v2/settings", json={
            "scratch_workspace": scratch_dir,
        })
        assert resp.status_code == 200

        resp = client.get("/api/v2/settings")
        assert resp.json().get("scratch_workspace") == scratch_dir


class TestJourney_AgentNameUniquenessAcrossProjects:
    """Journey: User tries to create two projects with same agent name -> rejected."""

    def test_cannot_reuse_agent_name(self, setup):
        client, workspace, tmp_path = setup

        ws2 = tmp_path / "ws2"
        ws2.mkdir()

        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "sk-test-key-12345678",
            "agent_name": "MyBot",
        })
        assert resp.status_code == 201

        resp = client.post("/api/v2/projects", json={
            "name": "P2", "workspace": str(ws2),
            "model": "m", "api_key": "sk-test-key-12345678",
            "agent_name": "MyBot",
        })
        assert resp.status_code == 409

    def test_can_rename_own_agent_name(self, setup):
        client, workspace, _ = setup

        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "sk-test-key-12345678",
            "agent_name": "OldName",
        })
        pid = resp.json()["project_id"]

        resp = client.put(f"/api/v2/projects/{pid}", json={
            "agent_name": "NewName",
        })
        assert resp.status_code == 200
        assert resp.json()["agent_name"] == "NewName"

    def test_cannot_steal_agent_name_via_update(self, setup):
        client, workspace, tmp_path = setup

        ws2 = tmp_path / "ws2"
        ws2.mkdir()

        client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "sk-test-key-12345678",
            "agent_name": "TakenName",
        })
        resp = client.post("/api/v2/projects", json={
            "name": "P2", "workspace": str(ws2),
            "model": "m", "api_key": "sk-test-key-12345678",
            "agent_name": "FreeName",
        })
        pid2 = resp.json()["project_id"]

        resp = client.put(f"/api/v2/projects/{pid2}", json={
            "agent_name": "TakenName",
        })
        assert resp.status_code == 409

    def test_cannot_collide_with_scratch_agent_name(self, setup):
        """Quick Tasks uses 'Assistant' — new projects can't claim it."""
        client, workspace, _ = setup

        resp = client.post("/api/v2/projects", json={
            "name": "P1", "workspace": workspace,
            "model": "m", "api_key": "sk-test-key-12345678",
            "agent_name": "Assistant",
        })
        assert resp.status_code == 409


class TestJourney_MultiStepProjectLifecycle:
    """Journey: Create project -> configure fully -> update -> verify consistency."""

    def test_full_lifecycle(self, setup):
        client, workspace, _ = setup

        # 1. Create
        resp = client.post("/api/v2/projects", json={
            "name": "Lifecycle Test",
            "workspace": workspace,
            "model": "gpt-4",
            "api_key": "sk-test-key-12345678",
            "agent_name": "LifeBot",
            "autonomy": "supervised",
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]

        # 2. Set goals
        client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "# Goal\nManage deployment pipeline",
        })

        # 3. Set rules
        client.put(f"/api/v2/projects/{pid}", json={
            "user_directives_content": "Never deploy on Fridays",
        })

        # 4. Change agent name
        client.put(f"/api/v2/projects/{pid}", json={
            "agent_name": "DeployBot",
        })

        # 5. Change autonomy
        client.put(f"/api/v2/projects/{pid}", json={
            "autonomy": "hands_off",
        })

        # 6. Verify everything in single GET
        resp = client.get(f"/api/v2/projects/{pid}")
        data = resp.json()
        assert data["agent_name"] == "DeployBot"
        assert data["autonomy"] == "hands_off"
        assert "deployment pipeline" in data["project_goals_content"]
        assert "Fridays" in data["user_directives_content"]

        # 7. Verify in list
        resp = client.get("/api/v2/projects")
        projects = resp.json()
        project_in_list = [p for p in projects if p["project_id"] == pid][0]
        assert project_in_list["agent_name"] == "DeployBot"

    def test_overwrite_goals_replaces_content(self, setup):
        client, workspace, _ = setup

        resp = client.post("/api/v2/projects", json={
            "name": "Overwrite Test", "workspace": workspace,
            "model": "m", "api_key": "sk-test-key-12345678",
        })
        pid = resp.json()["project_id"]

        # Write initial goals
        client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "Version 1 goals",
        })

        # Overwrite with new goals
        client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "Version 2 goals — completely different",
        })

        resp = client.get(f"/api/v2/projects/{pid}")
        content = resp.json()["project_goals_content"]
        assert "Version 2" in content
        assert "Version 1" not in content
