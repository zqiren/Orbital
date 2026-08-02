# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""`AGENTS.md` seeding at project creation (backlog #30).

Covers the seeder in isolation (seeds at the workspace *root*, never inside
``orbital/``, never clobbers, skips scratch, swallows write failures) plus the
route-level contract that a failing seed cannot fail ``POST /projects``.
"""

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app
from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.agent_md_seeder import (
    AGENT_MD_FILENAME,
    seed_project_agent_md,
)
from agent_os.daemon_v2.project_store import ProjectStore


@pytest.fixture
def store_and_project(tmp_path):
    """A non-scratch project over a fresh, empty workspace."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({
        "name": "Apollo",
        "workspace": str(workspace),
        "agent_name": "Apollo Agent",
    })
    return store, pid, workspace


@pytest.fixture
def client(tmp_path):
    # Bypass the singleton daemon PID guard for local runs.
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


def test_seeds_agent_md_at_workspace_root(store_and_project):
    store, pid, workspace = store_and_project

    result = seed_project_agent_md(store, pid)
    assert result["status"] == "ok"

    seeded = workspace / AGENT_MD_FILENAME
    assert AGENT_MD_FILENAME == "AGENTS.md"
    assert seeded.is_file()

    content = seeded.read_text(encoding="utf-8")
    assert "Apollo" in content
    assert "Apollo Agent" in content
    assert "orbital/PROJECT_STATE.md" in content


def test_does_not_write_inside_orbital_dir(store_and_project):
    """Guards against a refactor resurrecting the deleted orbital/AGENT.md."""
    store, pid, workspace = store_and_project

    seed_project_agent_md(store, pid)

    assert not (workspace / "orbital" / "AGENT.md").exists()
    assert not (workspace / "orbital" / AGENT_MD_FILENAME).exists()


def test_never_overwrites_existing(store_and_project):
    store, pid, workspace = store_and_project
    existing = workspace / AGENT_MD_FILENAME
    existing.write_text("hand-authored, do not touch\n", encoding="utf-8")

    result = seed_project_agent_md(store, pid)

    assert result["status"] == "skipped_exists"
    assert existing.read_text(encoding="utf-8") == "hand-authored, do not touch\n"


def test_idempotent_across_reruns(store_and_project):
    store, pid, workspace = store_and_project
    seeded = workspace / AGENT_MD_FILENAME

    assert seed_project_agent_md(store, pid)["status"] == "ok"
    first = seeded.read_bytes()

    assert seed_project_agent_md(store, pid)["status"] == "skipped_exists"
    assert seeded.read_bytes() == first


def test_skips_scratch_project(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "scratch-workspace"
    workspace.mkdir()

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({
        "name": "Quick Tasks",
        "workspace": str(workspace),
        "is_scratch": True,
    })

    result = seed_project_agent_md(store, pid)

    assert result == {"status": "skipped_scratch"}
    assert not (workspace / AGENT_MD_FILENAME).exists()


def test_write_failure_is_logged_and_does_not_raise(store_and_project, caplog):
    store, pid, workspace = store_and_project

    def exploding_open(*args, **kwargs):
        raise PermissionError("synthetic read-only workspace")

    logger_name = "agent_os.daemon_v2.agent_md_seeder"
    with patch("builtins.open", exploding_open):
        with caplog.at_level(logging.ERROR, logger=logger_name):
            result = seed_project_agent_md(store, pid)

    assert result["status"] == "write_failed"
    assert not (workspace / AGENT_MD_FILENAME).exists()
    assert any("failed to seed" in r.getMessage() for r in caplog.records)


def test_create_project_seeds_agent_md(client, tmp_path):
    workspace = tmp_path / "created-workspace"
    workspace.mkdir()

    resp = client.post("/api/v2/projects", json={
        "name": "SeededProject",
        "workspace": str(workspace),
        "model": "gpt-4",
        "api_key": "sk-test-key-1234",
    })

    assert resp.status_code == 201, resp.text
    seeded = workspace / AGENT_MD_FILENAME
    assert seeded.is_file()
    assert "SeededProject" in seeded.read_text(encoding="utf-8")


def test_create_project_succeeds_when_seed_fails(client, tmp_path, monkeypatch, caplog):
    workspace = tmp_path / "seed-fails-workspace"
    workspace.mkdir()

    def exploding_seeder(project_store, project_id):
        raise RuntimeError("synthetic seeder failure")

    monkeypatch.setattr(agents_v2, "seed_project_agent_md", exploding_seeder)

    logger_name = "agent_os.api.routes.agents_v2"
    with caplog.at_level(logging.ERROR, logger=logger_name):
        resp = client.post("/api/v2/projects", json={
            "name": "SeedFails",
            "workspace": str(workspace),
            "model": "gpt-4",
            "api_key": "sk-test-key-1234",
        })

    assert resp.status_code == 201, resp.text
    assert not (workspace / AGENT_MD_FILENAME).exists()

    error_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("AGENTS.md seeding failed" in m for m in error_msgs), (
        f"expected ERROR log about seeding failure, got: {error_msgs}"
    )
