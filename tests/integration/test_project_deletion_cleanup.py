# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for project deletion file cleanup.

Verifies that DELETE /api/v2/projects/{id} removes {workspace}/orbital/ and
nothing else. User files in the workspace directory are preserved.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app


@pytest.fixture
def setup(tmp_path):
    """Boot a fresh app with isolated data dir and workspace.

    Patches the singleton PID guard so tests can run alongside a live daemon.
    """
    data_dir = str(tmp_path / "data")
    # Bypass the singleton daemon PID guard — a user may have a live daemon
    # running on the dev machine, which would otherwise make these tests
    # unrunnable locally.
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=data_dir)
    client = TestClient(app)
    ws = tmp_path / "workspace"
    ws.mkdir()
    return client, ws, tmp_path


def _create_project(client, workspace: str) -> str:
    resp = client.post("/api/v2/projects", json={
        "name": "Cleanup Test",
        "workspace": workspace,
        "model": "gpt-4",
        "api_key": "sk-test-key-12345678",
    })
    assert resp.status_code == 201
    return resp.json()["project_id"]


def _seed_new_layout(ws: Path) -> Path:
    """Create the new-layout file tree under {ws}/orbital/."""
    orbital = ws / "orbital"

    # Flat memory files
    (orbital).mkdir(parents=True, exist_ok=True)
    (orbital / "PROJECT_STATE.md").write_text("# State")
    (orbital / "DECISIONS.md").write_text("# Decisions")
    (orbital / "LESSONS.md").write_text("# Lessons")
    (orbital / "SESSION_LOG.md").write_text("# Log")
    (orbital / "CONTEXT.md").write_text("# Context")
    (orbital / "approval_history.jsonl").write_text('{"approved":true}')

    # Instructions sub-dir
    (orbital / "instructions").mkdir(parents=True, exist_ok=True)
    (orbital / "instructions" / "project_goals.md").write_text("# Goals")

    # Sessions
    (orbital / "sessions").mkdir(parents=True, exist_ok=True)
    (orbital / "sessions" / "sess_1.jsonl").write_text('{"role":"user"}')

    # Sub-agents
    (orbital / "sub_agents" / "claude-code").mkdir(parents=True, exist_ok=True)
    (orbital / "sub_agents" / "claude-code" / "tr_1.jsonl").write_text('{}')

    # Tool results
    (orbital / "tool-results" / "sess_1").mkdir(parents=True, exist_ok=True)
    (orbital / "tool-results" / "sess_1" / "turn_1_call_1.json").write_text('{}')

    # Output sub-dirs
    (orbital / "output" / "screenshots" / "sess_1").mkdir(parents=True, exist_ok=True)
    (orbital / "output" / "screenshots" / "sess_1" / "step_0001.png").write_bytes(b"PNG")
    (orbital / "output" / "pdfs").mkdir(parents=True, exist_ok=True)
    (orbital / "output" / "pdfs" / "doc.pdf").write_bytes(b"PDF")
    (orbital / "output" / "shell-output").mkdir(parents=True, exist_ok=True)
    (orbital / "output" / "shell-output" / "ls.txt").write_text("output")

    # Skills
    (orbital / "skills" / "my-skill").mkdir(parents=True, exist_ok=True)
    (orbital / "skills" / "my-skill" / "SKILL.md").write_text("# Skill")

    # Temp files
    (orbital / ".tmp").mkdir(parents=True, exist_ok=True)
    (orbital / ".tmp" / "cmd_x_stdout.txt").write_text("temp")

    # User files (must NOT be deleted)
    (ws / "my-code.py").write_text("# user code")
    (ws / "docs").mkdir(parents=True, exist_ok=True)
    (ws / "docs" / "report.md").write_text("# User report")

    return orbital


class TestDeleteRemovesOrbitalDir:
    """Project deletion removes {workspace}/orbital/ and nothing else."""

    def test_orbital_dir_removed(self, setup):
        client, ws, _ = setup
        pid = _create_project(client, str(ws))
        orbital = _seed_new_layout(ws)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200

        assert not orbital.exists(), "orbital/ should be completely removed"

    def test_user_file_preserved(self, setup):
        client, ws, _ = setup
        pid = _create_project(client, str(ws))
        _seed_new_layout(ws)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200

        assert (ws / "my-code.py").exists(), "user my-code.py must not be deleted"

    def test_user_docs_dir_preserved(self, setup):
        client, ws, _ = setup
        pid = _create_project(client, str(ws))
        _seed_new_layout(ws)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200

        assert (ws / "docs" / "report.md").exists(), "user docs/report.md must not be deleted"

    def test_workspace_dir_itself_preserved(self, setup):
        client, ws, _ = setup
        pid = _create_project(client, str(ws))
        _seed_new_layout(ws)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200

        assert ws.exists(), "workspace directory itself must not be removed"

    def test_orbital_output_sibling_not_created(self, setup):
        """Delete path must not create an orbital-output/ sibling tree."""
        client, ws, _ = setup
        pid = _create_project(client, str(ws))
        _seed_new_layout(ws)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200

        assert not (ws / "orbital-output").exists(), \
            "orbital-output/ must not be created by the delete path"

    def test_raw_project_id_dir_not_created(self, setup):
        """Delete path must not create a directory named after the raw project_id."""
        client, ws, _ = setup
        pid = _create_project(client, str(ws))
        _seed_new_layout(ws)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200

        # No directory matching the project_id should be created at workspace level
        assert not (ws / pid).exists(), \
            "raw project_id directory must not appear in workspace after delete"


class TestDeleteWithNoFiles:
    """Deletion of a project with no workspace files doesn't error."""

    def test_no_files_still_succeeds(self, setup):
        client, ws, _ = setup
        pid = _create_project(client, str(ws))

        # Don't seed any files — workspace is empty
        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Project should be gone from the list
        projects = client.get("/api/v2/projects").json()
        pids = [p["project_id"] for p in projects]
        assert pid not in pids
