# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: the settings "skills registry" heals a stranded legacy project.

Reproduces the exact user-visible symptom found on the live ``Orbital-marketing``
and ``qclaw-multi-agent`` projects: ``default_skills_reconciled`` is ``True`` but
the skills physically live at the pre-``orbital/`` path ``{workspace}/skills/``,
so ``GET /api/v2/projects/{id}/skills`` (what the settings UI calls) returns an
empty list. After the installer runs the way ``AgentManager.start_agent`` does,
the same endpoint surfaces every skill — including a user-authored one.
"""

import json
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
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=str(data_dir))
    return TestClient(app), data_dir


def test_skills_endpoint_empty_then_healed_for_stranded_legacy_project(
    client_and_data, tmp_path,
):
    client, data_dir = client_and_data
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    resp = client.post("/api/v2/projects", json={
        "name": "Orbital-marketing",
        "workspace": str(workspace),
        "model": "gpt-4",
        "api_key": "sk-test-key-1234",
    })
    assert resp.status_code == 201
    pid = resp.json()["project_id"]

    # Reproduce the live broken state: skills stranded at the legacy
    # {workspace}/skills/ path (a formerly bundled copy and a user-authored
    # one). The reconciled flag stays True.
    legacy = workspace / "skills"
    (legacy / "learning-capture").mkdir(parents=True)
    (legacy / "learning-capture" / "SKILL.md").write_text(
        "# Learning Capture\n\nCapture lessons.", encoding="utf-8"
    )
    user_skill = legacy / "humanizer"
    user_skill.mkdir()
    (user_skill / "SKILL.md").write_text(
        "# Humanizer\n\nRewrite text so it sounds human.", encoding="utf-8"
    )

    # Settings UI fetch — the registry is empty, the symptom the user reported.
    resp = client.get(f"/api/v2/projects/{pid}/skills")
    assert resp.status_code == 200
    assert resp.json() == []

    # Simulate the next agent start (new ProjectStore = daemon restart).
    with open(data_dir / "projects.json", "r", encoding="utf-8") as f:
        assert json.load(f)[pid]["default_skills_reconciled"] is True
    store2 = ProjectStore(data_dir=str(data_dir))
    install_default_skills(store2, pid)

    # Same endpoint now lists every skill, including the user-authored one.
    resp = client.get(f"/api/v2/projects/{pid}/skills")
    assert resp.status_code == 200
    names = {s["dir_name"] for s in resp.json()}
    assert names == {"learning-capture", "humanizer"}
    # And every reported path points at the current orbital/skills location.
    for s in resp.json():
        assert "/orbital/skills/" in s["path"]
