# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: Settings UI "instructions" field must sync to
instructions/project_goals.md on disk.

Bug: PUT /api/v2/projects/{id} stored the `instructions` field in projects.json
but never wrote it to disk. The prompt builder reads from
instructions/project_goals.md, so user-typed instructions never reached the
agent.

Fix: when `instructions` is in the request body and `project_goals_content` is
NOT, treat `instructions` as `project_goals_content` and write it to disk. If
both are present, `project_goals_content` wins (explicit field).
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app
from agent_os.daemon_v2.project_store import project_dir_name as _project_dir_name


@pytest.fixture
def client(tmp_path):
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


def _create_project(client, workspace, **overrides):
    payload = {
        "name": "SyncProj",
        "workspace": workspace,
        "model": "gpt-4",
        "api_key": "sk-test",
        **overrides,
    }
    resp = client.post("/api/v2/projects", json=payload)
    assert resp.status_code == 201
    return resp.json()["project_id"], resp.json()


def _goals_path(workspace: str, project_name: str, project_id: str) -> str:
    dir_name = _project_dir_name(project_name, project_id)
    return os.path.join(
        workspace, "orbital", dir_name, "instructions", "project_goals.md"
    )


class TestInstructionsSyncToDisk:

    def test_put_instructions_writes_project_goals_md(self, client, workspace):
        """PUT {'instructions': ...} must write to instructions/project_goals.md."""
        pid, proj = _create_project(client, workspace)

        resp = client.put(
            f"/api/v2/projects/{pid}",
            json={"instructions": "Always use TypeScript"},
        )
        assert resp.status_code == 200

        # File on disk must contain the instructions text
        goals_path = _goals_path(workspace, proj["name"], pid)
        assert os.path.isfile(goals_path), (
            f"Expected instructions/project_goals.md to exist at {goals_path}"
        )
        with open(goals_path, "r", encoding="utf-8") as f:
            assert f.read() == "Always use TypeScript"

    def test_get_returns_instructions_as_project_goals_content(
        self, client, workspace
    ):
        """After PUT with instructions, GET must return project_goals_content
        reflecting the synced content."""
        pid, _ = _create_project(client, workspace)

        client.put(
            f"/api/v2/projects/{pid}",
            json={"instructions": "Always use TypeScript"},
        )

        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_goals_content"] == "Always use TypeScript"

    def test_project_goals_content_wins_over_instructions(
        self, client, workspace
    ):
        """If both instructions and project_goals_content are present in the
        same PUT body, project_goals_content wins (it's the explicit field)."""
        pid, proj = _create_project(client, workspace)

        resp = client.put(
            f"/api/v2/projects/{pid}",
            json={
                "instructions": "from textarea",
                "project_goals_content": "from explicit field",
            },
        )
        assert resp.status_code == 200

        goals_path = _goals_path(workspace, proj["name"], pid)
        with open(goals_path, "r", encoding="utf-8") as f:
            assert f.read() == "from explicit field"

    def test_instructions_only_also_persists_in_projects_json(
        self, client, workspace
    ):
        """Backward compat: the `instructions` field should still be saved
        in projects.json (we don't remove the pre-existing persistence)."""
        pid, _ = _create_project(client, workspace)

        client.put(
            f"/api/v2/projects/{pid}",
            json={"instructions": "persisted rule"},
        )

        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        # The projects.json "instructions" field should still hold the value
        assert data.get("instructions") == "persisted rule"
