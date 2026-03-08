# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for RelayRedactionMiddleware (agent_os.api.middleware)."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.middleware import RelayRedactionMiddleware


@pytest.fixture
def app():
    """Create a test app with the redaction middleware and a test endpoint."""
    app = FastAPI()
    app.add_middleware(RelayRedactionMiddleware)

    @app.get("/test/project")
    async def test_project():
        return {
            "name": "My Project",
            "api_key": "sk-abcdef1234567890",
            "model": "gpt-4",
        }

    @app.get("/test/projects")
    async def test_projects():
        return [
            {"name": "P1", "api_key": "key1", "model": "m1"},
            {"name": "P2", "api_key": "key2", "model": "m2"},
        ]

    @app.get("/test/no-key")
    async def test_no_key():
        return {"name": "safe", "value": 42}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestRedactionMiddleware:
    def test_relay_request_redacts_api_key(self, client):
        """Requests with X-Via-Relay: true have api_key replaced with ***."""
        resp = client.get("/test/project", headers={"X-Via-Relay": "true"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"] == "***"
        assert data["name"] == "My Project"
        assert data["model"] == "gpt-4"

    def test_non_relay_request_preserves_api_key(self, client):
        """Requests without X-Via-Relay header keep api_key intact."""
        resp = client.get("/test/project")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"] == "sk-abcdef1234567890"

    def test_relay_request_redacts_list_items(self, client):
        """api_key is redacted in list responses too."""
        resp = client.get("/test/projects", headers={"X-Via-Relay": "true"})
        data = resp.json()
        for item in data:
            assert item["api_key"] == "***"

    def test_non_relay_list_preserves_keys(self, client):
        resp = client.get("/test/projects")
        data = resp.json()
        assert data[0]["api_key"] == "key1"
        assert data[1]["api_key"] == "key2"

    def test_response_without_api_key_unchanged(self, client):
        """Responses without api_key are not modified."""
        resp = client.get("/test/no-key", headers={"X-Via-Relay": "true"})
        data = resp.json()
        assert data == {"name": "safe", "value": 42}
