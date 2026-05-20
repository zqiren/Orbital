# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: daemon startup removes orphaned session lock sentinels.

After the JSONL-lock migration, the legacy ``*.jsonl.lock`` sentinel files
no longer appear under normal use, but existing installs may still have
them on disk (from prior daemon runs or crashes). This test verifies
that ``create_app()`` sweeps them away during startup, while leaving
real session JSONL data files and unrelated files untouched.
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app


@pytest.fixture
def app_and_workspace(tmp_path):
    """Create a workspace + project, then build an app that registers it."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # First, seed a project via a transient app so projects.json records
    # the workspace. The cleanup sweep on the second create_app() call will
    # then pick it up.
    with patch("agent_os.api.app.acquire_pid_file"):
        seed_app = create_app(data_dir=str(data_dir))
    seed_client = TestClient(seed_app)
    resp = seed_client.post("/api/v2/projects", json={
        "name": "LockCleanup",
        "workspace": str(workspace),
        "model": "gpt-4",
        "api_key": "sk-test-key-1234",
    })
    assert resp.status_code == 201, resp.text
    return data_dir, workspace


def test_startup_removes_orphaned_session_lock_sentinels(app_and_workspace):
    data_dir, workspace = app_and_workspace
    sessions_dir = workspace / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Seed orphaned sentinels + a real JSONL + an unrelated file.
    orphan_a = sessions_dir / "sess_a.jsonl.lock"
    orphan_b = sessions_dir / "sess_b.jsonl.lock"
    real_jsonl = sessions_dir / "sess_real.jsonl"
    bystander = sessions_dir / "scratch.txt"

    orphan_a.write_text("")
    orphan_b.write_text("")
    real_jsonl.write_text('{"role":"meta","event":"session_start"}\n')
    bystander.write_text("not a sentinel")

    # Sanity: orphans are present before app starts.
    assert orphan_a.exists()
    assert orphan_b.exists()

    # Create a fresh app — its startup sweep should remove the orphans.
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=str(data_dir))
    # TestClient drives the startup events; instantiating create_app() alone
    # already ran the cleanup (it lives at import-time in the factory body),
    # but we use TestClient to exercise the full lifecycle for parity with
    # real daemon startup.
    with TestClient(app):
        pass

    # Sentinels gone.
    assert not orphan_a.exists(), "orphan_a should be swept by startup"
    assert not orphan_b.exists(), "orphan_b should be swept by startup"
    # Real JSONL and unrelated files preserved.
    assert real_jsonl.exists(), "real session JSONL must NOT be deleted"
    assert bystander.exists(), "non-sentinel file must NOT be deleted"


def test_startup_cleanup_handles_workspace_without_sessions_dir(tmp_path):
    """Workspace exists but has no orbital/sessions/ — cleanup is a no-op."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Note: no orbital/ subtree created.

    # Seed a project so the workspace ends up in projects.json.
    with patch("agent_os.api.app.acquire_pid_file"):
        seed_app = create_app(data_dir=str(data_dir))
    seed_client = TestClient(seed_app)
    resp = seed_client.post("/api/v2/projects", json={
        "name": "NoSessions",
        "workspace": str(workspace),
        "model": "gpt-4",
        "api_key": "sk-test-key-5678",
    })
    assert resp.status_code == 201, resp.text

    # Fresh app startup must not crash and must leave nothing behind.
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=str(data_dir))
    with TestClient(app):
        pass
    # The orbital subtree may have been created by project setup; sessions
    # dir, if present, must be empty of sentinels.
    sessions_dir = workspace / "orbital" / "sessions"
    if sessions_dir.exists():
        leftovers = [n for n in os.listdir(sessions_dir) if n.endswith(".jsonl.lock")]
        assert leftovers == []
