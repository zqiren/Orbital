# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 084 §3.3 — ``POST /projects`` honours ``auto_unique_name``.

Mounts the agents_v2 router over a real ``ProjectStore`` (same wiring as the
reorder route test) so the flag is exercised end to end through the request
model, the route and the store's uniquifier.
"""

import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes.agents_v2 import configure, router
from agent_os.daemon_v2.project_store import ProjectStore


@pytest.fixture
def client(tmp_path):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    app = FastAPI()
    configure(project_store=ProjectStore(data_dir), agent_manager=MagicMock(),
              ws_manager=MagicMock())
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws" / "new-chat"
    ws.mkdir(parents=True)
    return str(ws)


def _post(client, workspace, **extra):
    return client.post("/api/v2/projects",
                       json={"name": "new-chat", "workspace": workspace, **extra})


def test_twice_with_flag_is_201_201_with_distinct_names(client, workspace):
    r1 = _post(client, workspace, auto_unique_name=True)
    r2 = _post(client, workspace, auto_unique_name=True)
    assert (r1.status_code, r2.status_code) == (201, 201)
    assert r1.json()["name"] == "new-chat"
    assert r2.json()["name"] == "new-chat-2"
    assert r2.json()["agent_name"] == "new-chat-2"


def test_twice_without_flag_is_201_then_409(client, workspace):
    r1 = _post(client, workspace)
    r2 = _post(client, workspace)
    assert (r1.status_code, r2.status_code) == (201, 409)
    assert "already in use" in r2.json()["detail"]


def test_default_pinned_target_at_creation_is_kept_and_validated(client, workspace, monkeypatch):
    """Batch 1110 (dsh pass): POST /projects used to drop default_pinned_target
    silently (an unknown field); it is stored now, after the same check the
    PUT applies."""
    from agent_os.api.routes import agents_v2
    registry = MagicMock()
    registry.get = MagicMock(side_effect=lambda slug: object() if slug in ("dsh", "codex") else None)
    monkeypatch.setattr(agents_v2, "_setup_engine", MagicMock(_registry=registry))
    r = _post(client, workspace, default_pinned_target="dsh")
    assert r.status_code == 201, r.text
    pid = r.json()["project_id"]
    assert client.get(f"/api/v2/projects/{pid}").json()["default_pinned_target"] == "dsh"
    # Unknown slug: refused before anything is stored.
    r = _post(client, workspace, auto_unique_name=True, default_pinned_target="nope")
    assert r.status_code == 400
    assert "default_pinned_target" in r.text
