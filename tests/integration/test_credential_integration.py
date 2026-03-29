# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for credential store — full daemon user journey.

Boots the real FastAPI app via TestClient with mocked keyring.
"""

import json
import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(tmp_path):
    """Boot real FastAPI app with isolated data dir and mocked keyring."""
    data_dir = str(tmp_path / "orbital-data")
    os.makedirs(data_dir, exist_ok=True)

    mock_keyring = MagicMock()
    _keyring_storage = {}

    def _set(service, key, value):
        _keyring_storage[(service, key)] = value

    def _get(service, key):
        return _keyring_storage.get((service, key))

    def _delete(service, key):
        _keyring_storage.pop((service, key), None)

    mock_keyring.set_password = _set
    mock_keyring.get_password = _get
    mock_keyring.delete_password = _delete

    with patch("agent_os.daemon_v2.credential_store.keyring", mock_keyring), \
         patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True):
        from agent_os.api.app import create_app
        app = create_app(data_dir=data_dir)
        client = TestClient(app)
        yield client, _keyring_storage


# --- User Journey 1: Store + List + Use + Delete ---

def test_full_credential_lifecycle(app_client):
    client, keyring_storage = app_client

    # Step 1: Initially no credentials
    resp = client.get("/api/v2/credentials")
    assert resp.status_code == 200
    assert resp.json() == []

    # Step 2: Store a credential
    resp = client.post("/api/v2/credentials", json={
        "name": "twitter",
        "domain": "twitter.com",
        "fields": {"username": "johndoe", "password": "mypassword123"},
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "stored"

    # Step 3: Verify in keyring (values stored)
    assert keyring_storage[("agent-os-creds", "twitter.username")] == "johndoe"
    assert keyring_storage[("agent-os-creds", "twitter.password")] == "mypassword123"

    # Step 4: List credentials — metadata only, no values
    resp = client.get("/api/v2/credentials")
    assert resp.status_code == 200
    creds = resp.json()
    assert len(creds) == 1
    assert creds[0]["name"] == "twitter"
    assert creds[0]["domain"] == "twitter.com"
    raw = json.dumps(creds)
    assert "mypassword123" not in raw

    # Step 5: Delete credential
    resp = client.delete("/api/v2/credentials/twitter")
    assert resp.status_code == 200

    # Step 6: Verify gone
    resp = client.get("/api/v2/credentials")
    assert resp.status_code == 200
    assert resp.json() == []


# --- User Journey 2: Multiple credentials ---

def test_multiple_credentials(app_client):
    client, keyring_storage = app_client

    # Store two credentials
    client.post("/api/v2/credentials", json={
        "name": "github", "domain": "github.com",
        "fields": {"token": "ghp_xxxx"},
    })
    client.post("/api/v2/credentials", json={
        "name": "amazon", "domain": "amazon.com",
        "fields": {"email": "user@test.com", "password": "shop123"},
    })

    # List returns both
    resp = client.get("/api/v2/credentials")
    creds = resp.json()
    assert len(creds) == 2
    names = {c["name"] for c in creds}
    assert names == {"github", "amazon"}

    # Delete one, other remains
    client.delete("/api/v2/credentials/github")
    resp = client.get("/api/v2/credentials")
    creds = resp.json()
    assert len(creds) == 1
    assert creds[0]["name"] == "amazon"


# --- User Journey 3: Revoke (stub) ---

def test_revoke_returns_placeholder(app_client):
    client, _ = app_client
    client.post("/api/v2/credentials", json={
        "name": "twitter", "domain": "twitter.com",
        "fields": {"password": "x"},
    })
    resp = client.post("/api/v2/credentials/twitter/revoke")
    assert resp.status_code == 200
    assert resp.json()["status"] == "revoked"


# --- User Journey 4: Overwrite existing credential ---

def test_overwrite_existing_credential(app_client):
    client, keyring_storage = app_client

    client.post("/api/v2/credentials", json={
        "name": "twitter", "domain": "twitter.com",
        "fields": {"password": "old_password"},
    })
    assert keyring_storage[("agent-os-creds", "twitter.password")] == "old_password"

    # Store again with new value
    client.post("/api/v2/credentials", json={
        "name": "twitter", "domain": "twitter.com",
        "fields": {"password": "new_password"},
    })
    assert keyring_storage[("agent-os-creds", "twitter.password")] == "new_password"

    # Only one credential in list
    resp = client.get("/api/v2/credentials")
    assert len(resp.json()) == 1


# --- User Journey 5: Values never in API responses ---

def test_values_never_in_api_responses(app_client):
    client, _ = app_client

    client.post("/api/v2/credentials", json={
        "name": "bank", "domain": "mybank.com",
        "fields": {"password": "super_secret_password_123"},
    })

    # GET /credentials never returns values
    resp = client.get("/api/v2/credentials")
    raw = resp.text
    assert "super_secret_password_123" not in raw

    # DELETE response doesn't leak values
    resp = client.delete("/api/v2/credentials/bank")
    assert "super_secret_password_123" not in resp.text
