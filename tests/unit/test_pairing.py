# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for pairing routes (agent_os.api.routes.pairing)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import pairing


@pytest.fixture
def app_with_relay():
    """Create a test app with a mock relay client."""
    app = FastAPI()

    mock_relay = AsyncMock()
    mock_relay.relay_url = "http://localhost:3000"
    mock_relay.send_pairing_create = AsyncMock(return_value={
        "type": "pairing.code",
        "code": "123456",
    })
    mock_relay.send_pairing_revoke = AsyncMock()

    pairing.configure(mock_relay)
    app.include_router(pairing.router)

    return app, mock_relay


@pytest.fixture
def app_without_relay():
    """Create a test app with no relay configured."""
    app = FastAPI()
    pairing.configure(None)
    app.include_router(pairing.router)
    return app


@pytest.fixture
def client(app_with_relay):
    app, _ = app_with_relay
    return TestClient(app)


@pytest.fixture
def client_no_relay(app_without_relay):
    return TestClient(app_without_relay)


class TestStartPairing:
    def test_start_pairing_returns_code(self, client):
        resp = client.post("/api/v2/pairing/start")
        assert resp.status_code == 200
        assert resp.json()["code"] == "123456"

    def test_start_pairing_503_when_no_relay(self, client_no_relay):
        resp = client_no_relay.post("/api/v2/pairing/start")
        assert resp.status_code == 503


class TestListDevices:
    def test_list_empty(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(pairing, "_devices_file", lambda: tmp_path / "devices.json")
        resp = client.get("/api/v2/pairing/devices")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_with_devices(self, client, tmp_path, monkeypatch):
        devices = [{"phone_id": "ph_1", "name": "My Phone"}]
        f = tmp_path / "devices.json"
        f.write_text(json.dumps(devices))
        monkeypatch.setattr(pairing, "_devices_file", lambda: f)

        resp = client.get("/api/v2/pairing/devices")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["phone_id"] == "ph_1"


class TestRevokeDevice:
    def test_revoke_existing_device(self, client, tmp_path, monkeypatch):
        devices = [
            {"phone_id": "ph_1", "name": "Phone 1"},
            {"phone_id": "ph_2", "name": "Phone 2"},
        ]
        f = tmp_path / "devices.json"
        f.write_text(json.dumps(devices))
        monkeypatch.setattr(pairing, "_devices_file", lambda: f)

        resp = client.delete("/api/v2/pairing/devices/ph_1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "revoked"

        # Verify ph_1 is gone from file
        remaining = json.loads(f.read_text())
        assert len(remaining) == 1
        assert remaining[0]["phone_id"] == "ph_2"

    def test_revoke_nonexistent_device(self, client, tmp_path, monkeypatch):
        f = tmp_path / "devices.json"
        f.write_text("[]")
        monkeypatch.setattr(pairing, "_devices_file", lambda: f)

        resp = client.delete("/api/v2/pairing/devices/ph_missing")
        assert resp.status_code == 404


class TestAddPairedDevice:
    def test_add_new_device(self, tmp_path, monkeypatch):
        f = tmp_path / "devices.json"
        monkeypatch.setattr(pairing, "_devices_file", lambda: f)

        pairing.add_paired_device("ph_new")
        devices = json.loads(f.read_text())
        assert len(devices) == 1
        assert devices[0]["phone_id"] == "ph_new"
        assert "paired_at" in devices[0]

    def test_add_duplicate_is_noop(self, tmp_path, monkeypatch):
        f = tmp_path / "devices.json"
        f.write_text(json.dumps([{"phone_id": "ph_dup", "paired_at": "2025-01-01T00:00:00Z"}]))
        monkeypatch.setattr(pairing, "_devices_file", lambda: f)

        pairing.add_paired_device("ph_dup")
        devices = json.loads(f.read_text())
        assert len(devices) == 1  # not duplicated
