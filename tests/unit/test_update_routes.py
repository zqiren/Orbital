# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""In-app upgrade routes (agent_os/api/routes/update.py)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os import updater
from agent_os.api.routes import update as update_routes


@pytest.fixture
def client(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(update_routes.router)
    u = updater.configure(None, offered=lambda: {"update_available": True, "latest": "9.9.9"},
                          platform="darwin", cache_dir=tmp_path, frozen=False)
    yield TestClient(app), u
    updater._updater = None


def test_dev_daemon_reports_idle_and_cannot_install(client):
    c, _ = client
    snap = c.get("/api/v2/update/install").json()
    assert snap["state"] == "idle" and snap["can_install"] is False
    r = c.post("/api/v2/update/install", json={"region": "china"})
    assert r.status_code == 409 and r.json()["detail"] == {"code": "unsupported"}


def test_unknown_region_is_rejected(client):
    c, _ = client
    assert c.post("/api/v2/update/install", json={"region": "mars"}).status_code == 422


def test_dev_state_hook_needs_the_env_flag(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("AGENT_OS_UPDATE_DEV", raising=False)
    assert c.post("/api/v2/update/dev-state", json={"state": "downloading"}).status_code == 404
    monkeypatch.setenv("AGENT_OS_UPDATE_DEV", "1")
    r = c.post("/api/v2/update/dev-state",
               json={"state": "downloading", "version": "0.24.0", "progress": 37})
    assert r.json()["progress"] == 37
    assert c.get("/api/v2/update/install").json()["state"] == "downloading"
    assert c.post("/api/v2/update/dev-state", json={"clear": True}).json()["state"] == "idle"


def test_dev_state_hook_is_off_in_a_frozen_build(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("AGENT_OS_UPDATE_DEV", "1")
    monkeypatch.setattr(update_routes.sys, "frozen", True, raising=False)
    assert c.post("/api/v2/update/dev-state", json={"state": "ready"}).status_code == 404


def test_no_updater_configured():
    updater._updater = None
    app = FastAPI()
    app.include_router(update_routes.router)
    c = TestClient(app)
    assert c.get("/api/v2/update/install").json()["can_install"] is False
    assert c.post("/api/v2/update/install", json={}).status_code == 409


async def test_update_status_carries_the_install_state(client):
    from agent_os.api.routes import settings as settings_routes
    from agent_os import update_check

    _, u = client
    update_check._checker = None
    body = await settings_routes.get_update_status()
    assert body["install"]["state"] == "idle"
    assert body["update_available"] is False
