# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: if the installer raises, the POST /projects route still
returns 201 and the project exists in the store. An ERROR is logged. The
reconciled flag stays False so agent start will retry reconciliation."""

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app
from agent_os.api.routes import agents_v2


@pytest.fixture
def client(tmp_path):
    # Bypass the singleton daemon PID guard for local runs.
    with patch("agent_os.api.app.acquire_pid_file"):
        app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


def test_create_project_succeeds_even_when_installer_raises(
    client, tmp_path, monkeypatch, caplog,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def exploding_installer(project_store, project_id):
        raise RuntimeError("synthetic installer failure")

    monkeypatch.setattr(agents_v2, "install_default_skills", exploding_installer)

    with caplog.at_level(logging.ERROR, logger="agent_os.api.routes.agents_v2"):
        resp = client.post("/api/v2/projects", json={
            "name": "InstallerFails",
            "workspace": str(workspace),
            "model": "gpt-4",
            "api_key": "sk-test-key-1234",
        })

    # 201 — creation succeeds despite installer failure.
    assert resp.status_code == 201, resp.text
    pid = resp.json()["project_id"]

    # Project is in the store.
    get_resp = client.get(f"/api/v2/projects/{pid}")
    assert get_resp.status_code == 200
    # Flag stays False — retry on next agent start.
    assert get_resp.json().get("default_skills_reconciled") is False

    # An ERROR was logged.
    error_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("default skills install failed" in m for m in error_msgs), (
        f"expected ERROR log about installer failure, got: {error_msgs}"
    )
