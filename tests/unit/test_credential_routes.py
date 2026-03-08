# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for credential API routes."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent_os.api.routes.credentials import router, configure


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.list_all.return_value = []
    store.get_metadata.return_value = None
    return store


@pytest.fixture
def mock_agent_manager():
    return MagicMock()


@pytest.fixture
def client(mock_store, mock_agent_manager):
    from fastapi import FastAPI
    app = FastAPI()
    configure(mock_store, agent_manager=mock_agent_manager)
    app.include_router(router)
    return TestClient(app)


def test_post_credentials_stores_and_returns(client, mock_store):
    resp = client.post("/api/v2/credentials", json={
        "name": "twitter",
        "domain": "twitter.com",
        "fields": {"username": "john", "password": "secret"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stored"
    assert data["name"] == "twitter"
    mock_store.store.assert_called_once_with(
        "twitter", "twitter.com", {"username": "john", "password": "secret"}
    )


def test_get_credentials_returns_list(client, mock_store):
    mock_store.list_all.return_value = [
        {"name": "twitter", "domain": "twitter.com", "fields": ["username", "password"]},
    ]
    resp = client.get("/api/v2/credentials")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "twitter"


def test_delete_credential(client, mock_store):
    resp = client.delete("/api/v2/credentials/twitter")
    assert resp.status_code == 200
    mock_store.delete.assert_called_once_with("twitter")


def test_revoke_credential_placeholder(client, mock_store):
    resp = client.post("/api/v2/credentials/twitter/revoke")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "revoked"
