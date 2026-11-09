# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: a legacy project without the reconciled flag.

A project exists in a pre-fix state — no ``default_skills_reconciled`` key
(or it is False) and no skills directory. It used to be healed with the
bundled default skills on the next agent start; since 1109 Task B Orbital
bundles none, so the reconcile only sets the flag and writes no skill.

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


FORMER_BUNDLED_SKILLS = {
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


def test_legacy_project_without_flag_reconciles_without_seeding(
    client_and_data, tmp_path,
):
    client, data_dir = client_and_data
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create a project via the API (which sets the flag).
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
    assert result["installed"] == []

    # No bundled skill written.
    subdirs = {d.name for d in skills_dir.iterdir()} if skills_dir.exists() else set()
    assert not subdirs & FORMER_BUNDLED_SKILLS

    # Flag is now True and persists.
    store3 = ProjectStore(data_dir=str(data_dir))
    assert store3.get_project(pid)["default_skills_reconciled"] is True
