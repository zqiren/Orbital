# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the /api/v2/connectors routes (Task B3).

Wires a real ConnectorManager (with fully local fakes) into the router and
drives it over an in-process ASGI transport — no sockets to Google, no browser."""

import threading
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import FastAPI

from agent_os.api.routes import connectors as connector_routes
from agent_os.connectors import load_catalog
from agent_os.connectors.manager import ConnectorManager
from agent_os.connectors.oauth import GOOGLE_ENDPOINTS, OAuthClientConfig

from tests.unit._mock_mcp_server import build_mock_server, in_memory_opener


class FakeCredentialStore:
    def __init__(self):
        self._values = {}
        self._meta = {}

    def store(self, name, domain, fields):
        for f, v in fields.items():
            self._values[(name, f)] = v
        self._meta[name] = {"domain": domain, "fields": sorted(fields)}

    def get_value(self, name, field):
        return self._values.get((name, field))

    def list_all(self):
        return [{"name": n, **m} for n, m in self._meta.items()]

    def delete(self, name):
        self._meta.pop(name, None)
        for k in [k for k in self._values if k[0] == name]:
            self._values.pop(k, None)


def _fake_browser():
    def _open(auth_url):
        q = parse_qs(urlparse(auth_url).query)
        redirect_uri = q["redirect_uri"][0]
        state = q["state"][0]

        def _hit():
            try:
                httpx.get(f"{redirect_uri}?code=c&state={state}", timeout=5)
            except Exception:
                pass
        threading.Thread(target=_hit, daemon=True).start()
    return _open


def _token_http_factory():
    def handler(request):
        return httpx.Response(200, json={
            "access_token": "at", "refresh_token": "rt", "expires_in": 3600,
            "scope": "https://www.googleapis.com/auth/calendar",
            "email": "user@example.com",
        })
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def client(tmp_path):
    server = build_mock_server()
    mgr = ConnectorManager(
        catalog=load_catalog(),
        credential_store=FakeCredentialStore(),
        data_dir=str(tmp_path),
        oauth_client_provider=lambda p: OAuthClientConfig("cid", "secret", GOOGLE_ENDPOINTS) if p == "google" else None,
        http_client_factory=_token_http_factory(),
        open_browser=_fake_browser(),
        session_opener=in_memory_opener(server),
    )
    app = FastAPI()
    connector_routes.configure(mgr)
    app.include_router(connector_routes.router)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_get_connectors_lists_catalog(client):
    async with client:
        r = await client.get("/api/v2/connectors")
        assert r.status_code == 200
        body = r.json()
        by_id = {c["id"]: c for c in body["connectors"]}
        assert {"google-calendar", "google-drive", "gmail"} <= set(by_id)
        assert by_id["google-calendar"]["connected"] is False
        assert by_id["gmail"]["status"] == "pending_verification"
        # manifest fields surface for the settings cards.
        assert by_id["google-calendar"]["name"] == "Google Calendar"
        assert by_id["google-calendar"]["icon"]


async def test_connect_then_reflects_connected(client):
    async with client:
        r = await client.post("/api/v2/connectors/google-calendar/connect")
        assert r.status_code == 200, r.text
        assert r.json() == {"connected": True, "account": "user@example.com"}

        r2 = await client.get("/api/v2/connectors")
        by_id = {c["id"]: c for c in r2.json()["connectors"]}
        assert by_id["google-calendar"]["connected"] is True
        assert by_id["google-calendar"]["account"] == "user@example.com"


async def test_connect_pending_verification_is_conflict(client):
    async with client:
        r = await client.post("/api/v2/connectors/gmail/connect")
        assert r.status_code == 409


async def test_connect_unknown_is_404(client):
    async with client:
        r = await client.post("/api/v2/connectors/nope/connect")
        assert r.status_code == 404


async def test_disconnect(client):
    async with client:
        await client.post("/api/v2/connectors/google-calendar/connect")
        r = await client.post("/api/v2/connectors/google-calendar/disconnect")
        assert r.status_code == 200
        by_id = {c["id"]: c for c in (await client.get("/api/v2/connectors")).json()["connectors"]}
        assert by_id["google-calendar"]["connected"] is False


async def test_add_and_remove_custom(client):
    async with client:
        r = await client.post("/api/v2/connectors/custom", json={
            "name": "Mock Server", "url": "https://mock/mcp", "auth_type": "none",
        })
        assert r.status_code == 200, r.text
        assert r.json()["id"] == "custom-mock-server"

        by_id = {c["id"]: c for c in (await client.get("/api/v2/connectors")).json()["connectors"]}
        assert "custom-mock-server" in by_id
        assert by_id["custom-mock-server"]["connected"] is True

        r2 = await client.delete("/api/v2/connectors/custom/custom-mock-server")
        assert r2.status_code == 200
        by_id2 = {c["id"]: c for c in (await client.get("/api/v2/connectors")).json()["connectors"]}
        assert "custom-mock-server" not in by_id2


async def test_add_custom_rejects_bad_auth_type(client):
    async with client:
        r = await client.post("/api/v2/connectors/custom", json={
            "name": "Bad", "url": "https://x/mcp", "auth_type": "local_native",
        })
        assert r.status_code == 422  # pydantic Literal rejects it
