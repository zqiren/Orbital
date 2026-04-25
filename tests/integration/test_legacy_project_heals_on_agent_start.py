# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: simulate the Orbital-marketing repair scenario.

A project exists in a pre-fix state — no ``default_skills_reconciled`` key
(or it is False) and the workspace skills directory has been removed (or
never existed because the pre-fix macOS build shipped without the bundled
default skills). The fixed daemon reconciles on the next agent start.

We simulate agent start by invoking the installer the way ``AgentManager``
does. Booting a full agent loop requires LLM credentials and a provider
round-trip, which is out of scope for a local TestClient test."""

import json
import shutil
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app
from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore


EXPECTED_SKILLS = {
    "efficient-execution",
    "learning-capture",
    "process-capture",
    "task-planning",
}


@pytest.fixture
def client_and_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Bypass the singleton daemon PID guard for local runs.
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=str(data_dir))
    return TestClient(app), data_dir


def test_legacy_project_without_flag_heals_on_reconcile_call(
    client_and_data, tmp_path,
):
    client, data_dir = client_and_data
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create a project via the API (which installs skills + sets flag).
    resp = client.post("/api/v2/projects", json={
        "name": "Orbital-marketing",
        "workspace": str(workspace),
        "model": "gpt-4",
        "api_key": "sk-test-key-1234",
    })
    assert resp.status_code == 201
    pid = resp.json()["project_id"]

    # Force a legacy pre-fix state:
    # 1. Wipe the on-disk skills/ dir (as if the macOS build never shipped it).
    skills_dir = workspace / "orbital" / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)

    # 2. Clear the reconciled flag directly in projects.json so the project
    #    record matches a pre-fix entry.
    projects_json = data_dir / "projects.json"
    with open(projects_json, "r", encoding="utf-8") as f:
        projects = json.load(f)
    projects[pid].pop("default_skills_reconciled", None)
    with open(projects_json, "w", encoding="utf-8") as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)

    # 3. Simulate daemon restart: new ProjectStore instance reads the edited
    #    projects.json. Legacy record is missing the key → surfaces as False.
    store2 = ProjectStore(data_dir=str(data_dir))
    project = store2.get_project(pid)
    assert project["default_skills_reconciled"] is False

    # Now trigger the reconcile the way AgentManager.start_agent does.
    result = install_default_skills(store2, pid)
    assert result["status"] == "ok"
    assert set(result["installed"]) == EXPECTED_SKILLS

    # Skills healed on disk.
    subdirs = {d.name for d in skills_dir.iterdir() if d.is_dir()}
    assert subdirs == EXPECTED_SKILLS

    # Flag is now True and persists.
    store3 = ProjectStore(data_dir=str(data_dir))
    assert store3.get_project(pid)["default_skills_reconciled"] is True
