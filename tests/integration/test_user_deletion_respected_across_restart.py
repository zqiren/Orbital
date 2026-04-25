# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: user skill deletions survive daemon restarts.

1. Create a project — 4 skills are installed, ``default_skills_reconciled``
   flips to True.
2. Delete one skill via the API.
3. Simulate a daemon restart by constructing a fresh ProjectStore against
   the same data dir.
4. Run the installer again (as ``AgentManager.start_agent`` does).
5. Assert the deleted skill does NOT reappear and the remaining 3 still
   exist — the persistent flag short-circuits the reconcile, respecting the
   user deletion."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app
from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore


@pytest.fixture
def client_and_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Bypass the singleton daemon PID guard for local runs.
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=str(data_dir))
    return TestClient(app), data_dir


def test_deleted_skill_stays_deleted_across_restart(client_and_data, tmp_path):
    client, data_dir = client_and_data
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    resp = client.post("/api/v2/projects", json={
        "name": "RespectDeletion",
        "workspace": str(workspace),
        "model": "gpt-4",
        "api_key": "sk-test-key-1234",
    })
    assert resp.status_code == 201
    pid = resp.json()["project_id"]

    skills_dir = workspace / "skills"
    assert (skills_dir / "learning-capture").is_dir()

    # Delete one skill via the API. Use the actual skills delete endpoint.
    del_resp = client.delete(f"/api/v2/projects/{pid}/skills/learning-capture")
    # Accept either 200 or 204; the specific code is not the point of this test.
    assert del_resp.status_code in (200, 204), del_resp.text
    assert not (skills_dir / "learning-capture").exists()

    # Simulate daemon restart — fresh ProjectStore reads the same projects.json.
    fresh_store = ProjectStore(data_dir=str(data_dir))
    project = fresh_store.get_project(pid)
    # Flag should still be True — deletion doesn't touch the flag.
    assert project["default_skills_reconciled"] is True

    # Run the installer as AgentManager.start_agent would.
    result = install_default_skills(fresh_store, pid)
    assert result == {"status": "skipped_already_reconciled"}

    # Deleted skill does NOT reappear.
    assert not (skills_dir / "learning-capture").exists()
    # Remaining 3 are still there.
    for name in ("efficient-execution", "process-capture", "task-planning"):
        assert (skills_dir / name / "SKILL.md").is_file()
