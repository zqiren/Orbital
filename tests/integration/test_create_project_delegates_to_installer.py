# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: POST /api/v2/projects delegates to the shared skills
reconciler. Since 1109 Task B no bundled skill lands on disk, and
``default_skills_reconciled=true`` is persisted in the project record."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app


FORMER_BUNDLED_SKILLS = {
    "efficient-execution",
    "learning-capture",
    "process-capture",
    "task-planning",
}


@pytest.fixture
def client(tmp_path):
    # Bypass the singleton daemon PID guard — a user may have a live daemon
    # running on the dev machine, which would otherwise make these tests
    # unrunnable locally.
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


def test_create_project_installs_no_bundled_skills_and_sets_flag(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    resp = client.post("/api/v2/projects", json={
        "name": "DelegatesProject",
        "workspace": str(workspace),
        "model": "gpt-4",
        "api_key": "sk-test-key-1234",
    })
    assert resp.status_code == 201, resp.text
    pid = resp.json()["project_id"]

    # None of the formerly bundled skills materialized on disk.
    skills_dir = workspace / "orbital" / "skills"
    subdirs = {d.name for d in skills_dir.iterdir()} if skills_dir.exists() else set()
    assert not subdirs & FORMER_BUNDLED_SKILLS

    # Reconciled flag surfaces through GET /projects/{pid}.
    get_resp = client.get(f"/api/v2/projects/{pid}")
    assert get_resp.status_code == 200
    assert get_resp.json().get("default_skills_reconciled") is True
