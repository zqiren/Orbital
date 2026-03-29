# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for scratch project auto-creation on daemon startup."""
import os
import pytest
from fastapi.testclient import TestClient
from agent_os.api.app import create_app


class TestScratchAutoCreation:
    """User journey: daemon boots -> scratch project exists -> user can chat immediately."""

    def test_scratch_project_created_on_startup(self, tmp_path):
        app = create_app(data_dir=str(tmp_path / "data"))
        client = TestClient(app)

        resp = client.get("/api/v2/projects")
        projects = resp.json()
        scratch_projects = [p for p in projects if p.get("is_scratch")]
        assert len(scratch_projects) == 1
        assert scratch_projects[0]["name"] == "Quick Tasks"
        assert scratch_projects[0]["agent_name"] == "Assistant"

    def test_scratch_project_has_goals_file(self, tmp_path):
        app = create_app(data_dir=str(tmp_path / "data"))
        client = TestClient(app)

        resp = client.get("/api/v2/projects")
        scratch = [p for p in resp.json() if p.get("is_scratch")][0]
        goals_path = os.path.join(
            scratch["workspace"], "orbital", "instructions", "project_goals.md"
        )
        assert os.path.exists(goals_path)
        with open(goals_path) as f:
            content = f.read()
        assert "Quick Tasks Assistant" in content

    def test_scratch_not_duplicated_on_restart(self, tmp_path):
        data_dir = str(tmp_path / "data")
        app1 = create_app(data_dir=data_dir)
        TestClient(app1)  # first boot

        app2 = create_app(data_dir=data_dir)
        client2 = TestClient(app2)  # second boot

        resp = client2.get("/api/v2/projects")
        scratch_projects = [p for p in resp.json() if p.get("is_scratch")]
        assert len(scratch_projects) == 1

    def test_scratch_project_appears_first_in_list(self, tmp_path):
        data_dir = str(tmp_path / "data")
        app = create_app(data_dir=data_dir)
        client = TestClient(app)

        # Create a regular project
        ws = tmp_path / "ws"
        ws.mkdir()
        client.post("/api/v2/projects", json={
            "name": "Regular", "workspace": str(ws),
            "model": "m", "api_key": "k",
        })

        resp = client.get("/api/v2/projects")
        projects = resp.json()
        # Scratch should be first
        assert projects[0].get("is_scratch") is True
