# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for ConnectorManager lifecycle: connect/disconnect, custom servers,
lazy cached MCP sessions, token keying (Task B2)."""

import httpx
import pytest

from agent_os.connectors import load_catalog
from agent_os.connectors.manager import ConnectorError, Connected, ConnectorManager
from agent_os.connectors.oauth import GOOGLE_ENDPOINTS, OAuthClientConfig

from tests.unit._mock_mcp_server import build_mock_server, in_memory_opener


# ---- Fakes ----

class FakeCredentialStore:
    """In-memory stand-in with the UserCredentialStore method surface used by
    ConnectorManager (store / get_value / list_all / delete)."""

    def __init__(self):
        self._values: dict[tuple[str, str], str] = {}
        self._meta: dict[str, dict] = {}

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
        for key in [k for k in self._values if k[0] == name]:
            self._values.pop(key, None)


def _google_provider():
    def _provider(auth_provider):
        if auth_provider == "google":
            return OAuthClientConfig("cid", "secret", GOOGLE_ENDPOINTS)
        return None
    return _provider


def _token_http_factory(account="user@example.com"):
    def handler(request):
        return httpx.Response(200, json={
            "access_token": "at", "refresh_token": "rt",
            "expires_in": 3600,
            "scope": " ".join([
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/drive",
            ]),
            "email": account,
        })
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _fake_browser():
    import threading
    from urllib.parse import parse_qs, urlparse

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


def _make_manager(tmp_path, *, server=None, provider=None, http=None):
    server = server or build_mock_server()
    return ConnectorManager(
        catalog=load_catalog(),
        credential_store=FakeCredentialStore(),
        data_dir=str(tmp_path),
        oauth_client_provider=provider if provider is not None else _google_provider(),
        http_client_factory=http or _token_http_factory(),
        open_browser=_fake_browser(),
        session_opener=in_memory_opener(server),
    )


# ---- Tests ----

def test_list_connectors_starts_disconnected(tmp_path):
    mgr = _make_manager(tmp_path)
    entries = {e["manifest"].id: e for e in mgr.list_connectors()}
    assert {"google-calendar", "google-drive", "gmail"} <= set(entries)
    for cid in ("google-calendar", "google-drive", "gmail"):
        assert entries[cid]["connected"] is False
        assert entries[cid]["account"] is None


async def test_connect_stores_token_and_shares_across_google_connectors(tmp_path):
    mgr = _make_manager(tmp_path)
    result = await mgr.connect("google-calendar")
    assert isinstance(result, Connected)
    assert result.account == "user@example.com"

    entries = {e["manifest"].id: e for e in mgr.list_connectors()}
    # Shared google auth_provider -> connecting calendar connects drive + gmail.
    assert entries["google-calendar"]["connected"] is True
    assert entries["google-calendar"]["account"] == "user@example.com"
    assert entries["google-drive"]["connected"] is True
    assert entries["gmail"]["connected"] is True


async def test_token_is_keyed_by_provider_and_account(tmp_path):
    store = FakeCredentialStore()
    mgr = ConnectorManager(
        catalog=load_catalog(), credential_store=store, data_dir=str(tmp_path),
        oauth_client_provider=_google_provider(),
        http_client_factory=_token_http_factory(),
        open_browser=_fake_browser(),
        session_opener=in_memory_opener(build_mock_server()),
    )
    await mgr.connect("google-calendar")
    names = [m["name"] for m in store.list_all()]
    assert "connector:google:user@example.com" in names
    assert store.get_value("connector:google:user@example.com", "access_token") == "at"


async def test_connect_refuses_pending_verification(tmp_path):
    mgr = _make_manager(tmp_path)
    with pytest.raises(ConnectorError):
        await mgr.connect("gmail")


async def test_connect_errors_when_oauth_client_not_configured(tmp_path):
    mgr = _make_manager(tmp_path, provider=lambda p: None)
    with pytest.raises(Exception):
        await mgr.connect("google-calendar")


async def test_disconnect_removes_shared_google_token(tmp_path):
    mgr = _make_manager(tmp_path)
    await mgr.connect("google-calendar")
    await mgr.disconnect("google-calendar")
    entries = {e["manifest"].id: e for e in mgr.list_connectors()}
    assert entries["google-calendar"]["connected"] is False
    assert entries["google-drive"]["connected"] is False


async def test_add_custom_persists_and_lists_connected(tmp_path):
    server = build_mock_server()
    mgr = _make_manager(tmp_path, server=server)
    manifest = await mgr.add_custom("Mock Server", "https://mock/mcp", "none")
    assert manifest.id == "custom-mock-server"

    entries = {e["manifest"].id: e for e in mgr.list_connectors()}
    assert entries["custom-mock-server"]["connected"] is True  # auth_type none

    # Persisted to orbital-data/connectors.json -> survives a fresh manager.
    mgr2 = ConnectorManager(
        catalog=load_catalog(), credential_store=FakeCredentialStore(),
        data_dir=str(tmp_path), session_opener=in_memory_opener(server),
    )
    assert any(e["manifest"].id == "custom-mock-server" for e in mgr2.list_connectors())


async def test_custom_server_tools_reflect_through_session(tmp_path):
    server = build_mock_server()
    mgr = _make_manager(tmp_path, server=server)
    await mgr.add_custom("Mock Server", "https://mock/mcp", "none")

    tools = await mgr.list_tools("custom-mock-server")
    names = {t.name for t in tools}
    assert names == {"echo_read", "echo_write"}

    result = await mgr.call_tool("custom-mock-server", "echo_read", {"text": "hi"})
    assert result.content[0].text == "read:hi"
    assert result.isError is False


async def test_session_is_cached(tmp_path):
    server = build_mock_server()
    mgr = _make_manager(tmp_path, server=server)
    await mgr.add_custom("Mock Server", "https://mock/mcp", "none")
    s1 = await mgr.session("custom-mock-server")
    s2 = await mgr.session("custom-mock-server")
    assert s1 is s2
    await mgr.aclose()


async def test_remove_custom_drops_it(tmp_path):
    server = build_mock_server()
    mgr = _make_manager(tmp_path, server=server)
    await mgr.add_custom("Mock Server", "https://mock/mcp", "none")
    await mgr.remove_custom("custom-mock-server")
    assert not any(e["manifest"].id == "custom-mock-server" for e in mgr.list_connectors())
