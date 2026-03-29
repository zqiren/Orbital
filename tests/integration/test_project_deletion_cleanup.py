# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for project deletion file cleanup.

Verifies that DELETE /api/v2/projects/{id} removes project data files,
and that the clear_output flag controls agent_output/ deletion.
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


def _seed_project_files(ws, project_id: str):
    """Create the typical file tree a project accumulates."""
    agent_os = ws / "orbital"

    # Per-project session dir
    sessions = agent_os / project_id / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "session_abc.jsonl").write_text('{"role":"user"}')
    (agent_os / project_id / "PROJECT_STATE.md").write_text("# State")
    (agent_os / project_id / "LESSONS.md").write_text("# Lessons")

    # Browser screenshots (output path)
    output_dir = ws / "orbital-output" / project_id
    shots = output_dir / "screenshots"
    shots.mkdir(parents=True)
    (shots / "step_0001.png").write_bytes(b"PNG")

    # Browser PDFs (output path)
    pdfs = output_dir / "pdfs"
    pdfs.mkdir(parents=True)
    (pdfs / "report.pdf").write_bytes(b"PDF")

    # Shell output (output path)
    shell = output_dir / "shell-output"
    shell.mkdir(parents=True)
    (shell / "cmd_1.txt").write_text("output")

    # Instructions
    instr = agent_os / "instructions"
    instr.mkdir(parents=True)
    (instr / "project_goals.md").write_text("# Goals")

    # Approval history
    (agent_os / "approval_history.jsonl").write_text('{"approved":true}')

    # Temp files
    tmp = agent_os / ".tmp"
    tmp.mkdir(parents=True)
    (tmp / "scratch.txt").write_text("temp")

    # Agent output (user-facing artifacts)
    output = agent_os / "agent_output"
    output.mkdir(parents=True)
    (output / "report.md").write_text("# Report")

    return agent_os


class TestDeleteCleansInternalFiles:
    """Default deletion (no clear_output) removes internal data but keeps agent_output."""

    def test_internal_files_removed(self, setup):
        client, ws, _ = setup
        pid = _create_project(client, str(ws))
        agent_os = _seed_project_files(ws, pid)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200

        # Internal data should be gone
        assert not (agent_os / pid).exists(), "project session dir should be removed"
        output_dir = ws / "orbital-output" / pid
        assert not (output_dir / "screenshots").exists()
        assert not (output_dir / "pdfs").exists()
        assert not (output_dir / "shell-output").exists()
        assert not (agent_os / "instructions").exists()
        assert not (agent_os / "approval_history.jsonl").exists()
        assert not (agent_os / ".tmp").exists()

    def test_agent_output_preserved(self, setup):
        client, ws, _ = setup
        pid = _create_project(client, str(ws))
        agent_os = _seed_project_files(ws, pid)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200

        # Agent output should still be there
        assert (agent_os / "agent_output" / "report.md").exists(), \
            "agent_output should be preserved when clear_output is not set"


class TestDeleteWithClearOutput:
    """Deletion with clear_output=true also removes agent_output."""

    def test_agent_output_cleared(self, setup):
        client, ws, _ = setup
        pid = _create_project(client, str(ws))
        agent_os = _seed_project_files(ws, pid)

        resp = client.delete(f"/api/v2/projects/{pid}?clear_output=true")
        assert resp.status_code == 200

        # Everything including agent output should be gone
        assert not (agent_os / pid).exists()
        output_dir = ws / "orbital-output" / pid
        assert not (output_dir / "screenshots").exists()
        assert not (agent_os / "agent_output").exists(), \
            "agent_output should be removed when clear_output=true"


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
