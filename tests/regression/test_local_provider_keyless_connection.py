# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: local OpenAI-compatible models (LM Studio / llama.cpp / Ollama)
connect without an API key via the "Custom" provider.

The frontend's Test-Connection / Fetch-Models calls deliberately OMIT `provider`
(it sends `undefined` for the Custom provider) and OMIT `api_key` when the field
is empty — which is exactly the local-model case (these servers ignore the key).

The request models used to declare `provider` and `api_key` as required `str`
with no default, so FastAPI rejected the keyless custom body with a Pydantic
422 `type: missing` before the handler ever ran. Users saw a raw
`[{"type":"missing",...}]` blob on the Test-Connection button.

These tests pin the request contract (and the endpoint wiring) so a keyless
custom/local body is accepted.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2
from agent_os.api.routes.agents_v2 import FetchModelsRequest
# Aliased: a bare `TestConnectionRequest` import trips pytest's "looks like a
# test class" collection heuristic and emits a warning.
from agent_os.api.routes.agents_v2 import TestConnectionRequest as ConnectionRequest


# --- Request-contract level (directly targets the validation root cause) ------

def test_test_connection_request_accepts_keyless_custom_provider():
    """Body the frontend sends for a Custom/local provider: no provider, no key."""
    req = ConnectionRequest(
        model="local-model",
        base_url="http://localhost:1234/v1",
        sdk="openai",
    )
    assert req.provider == "custom"
    assert req.api_key == ""
    assert req.base_url == "http://localhost:1234/v1"


def test_fetch_models_request_accepts_keyless_custom_provider():
    """Model-listing for a local server: base_url only, no provider, no key."""
    req = FetchModelsRequest(base_url="http://localhost:1234/v1")
    assert req.provider == "custom"
    assert req.api_key is None


# --- Endpoint level (guards the FastAPI wiring through to the handler) ---------

@pytest.fixture
def provider_client(monkeypatch):
    """Minimal app with just the providers router. The test_connection handler
    tolerates a None registry (falls back to req.base_url), so no other wiring
    is needed. Both module globals are pinned via monkeypatch (auto-restored)
    so these tests can't leak state into later files."""
    app = FastAPI()
    app.include_router(agents_v2.router)
    monkeypatch.setattr(agents_v2, "_provider_registry", None)
    monkeypatch.setattr(agents_v2, "_credential_store", None)
    return TestClient(app, raise_server_exceptions=False)


def test_providers_models_keyless_omits_auth_header(provider_client, monkeypatch):
    """A keyless local server must be queried WITHOUT an Authorization header.
    Previously the handler sent `Authorization: Bearer ` (trailing space) when
    no key was given, which httpx rejects with 'Illegal header value' (502)."""
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "llama3.2:3b"}]}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    resp = provider_client.post(
        "/api/v2/providers/models",
        json={"base_url": "http://127.0.0.1:11434"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["models"] == ["llama3.2:3b"]
    # The crux: no trailing-space Bearer header when there is no key.
    assert "Authorization" not in captured["headers"], captured["headers"]


def test_providers_test_endpoint_accepts_keyless_custom_body(provider_client):
    """POST /providers/test with a keyless custom body must pass validation
    (not 422) and reach the handler, returning status: ok."""
    fake_provider = MagicMock()
    fake_provider.complete = AsyncMock(return_value=MagicMock())
    with patch(
        "agent_os.agent.providers.openai_compat.LLMProvider",
        return_value=fake_provider,
    ):
        resp = provider_client.post(
            "/api/v2/providers/test",
            json={
                "model": "local-model",
                "base_url": "http://localhost:1234/v1",
                "sdk": "openai",
            },
        )
    assert resp.status_code != 422, resp.text
    assert resp.json().get("status") == "ok"


# --- Stored-key fallback (Spec 47 follow-up) ----------------------------------
#
# The frontend clears the API-key field once a key is persisted (paste-and-save,
# or the TokenDance one-click flow), and its Test-Connection body omits the key
# when the field is empty. The handler used to construct the client with that
# empty string, so testing an already-saved key surfaced the raw SDK error
# "Missing credentials … set the OPENAI_API_KEY environment variable".


def _capture_provider_ctor(captured: dict):
    fake_provider = MagicMock()
    fake_provider.complete = AsyncMock(return_value=MagicMock())

    def ctor(model, api_key, base_url, **kw):
        captured["api_key"] = api_key
        return fake_provider

    return ctor


def test_providers_test_falls_back_to_stored_global_key(provider_client, monkeypatch):
    """Empty api_key + a saved global key = "test the stored key"."""
    store = MagicMock()
    store.get_api_key.return_value = "sk-stored-global-key"
    monkeypatch.setattr(agents_v2, "_credential_store", store)

    captured: dict = {}
    with patch(
        "agent_os.agent.providers.openai_compat.LLMProvider",
        side_effect=_capture_provider_ctor(captured),
    ):
        resp = provider_client.post(
            "/api/v2/providers/test",
            json={
                "provider": "tokendance",
                "model": "deepseek-v4-flash",
                "base_url": "https://tokendance.space/gateway/v1",
                "sdk": "openai",
            },
        )
    assert resp.status_code == 200, resp.text
    assert captured["api_key"] == "sk-stored-global-key"


def test_providers_test_typed_key_wins_over_stored_key(provider_client, monkeypatch):
    """A key typed into the field must be tested verbatim, never silently
    replaced by the stored one (that's the try-a-new-key-before-saving flow)."""
    store = MagicMock()
    store.get_api_key.return_value = "sk-stored-global-key"
    monkeypatch.setattr(agents_v2, "_credential_store", store)

    captured: dict = {}
    with patch(
        "agent_os.agent.providers.openai_compat.LLMProvider",
        side_effect=_capture_provider_ctor(captured),
    ):
        resp = provider_client.post(
            "/api/v2/providers/test",
            json={
                "provider": "tokendance",
                "model": "deepseek-v4-flash",
                "api_key": "sk-typed-fresh-key",
                "base_url": "https://tokendance.space/gateway/v1",
                "sdk": "openai",
            },
        )
    assert resp.status_code == 200, resp.text
    assert captured["api_key"] == "sk-typed-fresh-key"


def test_providers_test_keyless_stays_keyless_without_stored_key(provider_client):
    """Local/custom servers with no stored global key keep the pre-fix
    behavior: an empty key constructs a keyless client (these servers ignore
    auth entirely). The fixture pins _credential_store to None."""
    captured: dict = {}
    with patch(
        "agent_os.agent.providers.openai_compat.LLMProvider",
        side_effect=_capture_provider_ctor(captured),
    ):
        resp = provider_client.post(
            "/api/v2/providers/test",
            json={
                "model": "local-model",
                "base_url": "http://localhost:1234/v1",
                "sdk": "openai",
            },
        )
    assert resp.status_code == 200, resp.text
    assert captured["api_key"] == ""


# --- Model-list: stored-key fallback, request sdk, fail-fast timeout ----------
#
# Same bug class as the Test-Connection fallback above, on the sibling endpoint.
# The frontend clears the API-key field after a save (`doSave` -> setApiKey(''))
# and only puts `api_key` in the /providers/models body when the field is
# non-empty — but `apiKey` is in the fetch effect's dep array, so clearing it
# RE-FIRES the fetch with no key at all. 11 of the 14 registry providers answer
# /models with 401/403 when unauthenticated, so the live model list silently
# reverted to the static suggested list the moment the user pressed Save.


def _fake_httpx_client(captured: dict, payload: dict | None = None):
    """httpx.AsyncClient stand-in that records ctor kwargs + the GET it makes."""

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload if payload is not None else {"data": [{"id": "m-1"}]}

    class FakeClient:
        def __init__(self, *a, **kw):
            captured["client_kwargs"] = kw

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            return FakeResp()

    return FakeClient


def test_providers_models_falls_back_to_stored_global_key(provider_client, monkeypatch):
    """Empty api_key + a saved global key = list models with the stored key.

    This is the post-save steady state: the key field is empty, so the body
    carries no api_key, and the provider 401s without one.
    """
    import httpx

    store = MagicMock()
    store.get_api_key.return_value = "sk-stored-global-key"
    monkeypatch.setattr(agents_v2, "_credential_store", store)

    captured: dict = {}
    monkeypatch.setattr(httpx, "AsyncClient", _fake_httpx_client(captured))

    resp = provider_client.post(
        "/api/v2/providers/models",
        json={"provider": "deepseek", "base_url": "https://api.deepseek.com"},
    )
    assert resp.status_code == 200, resp.text
    assert captured["headers"].get("Authorization") == "Bearer sk-stored-global-key"


def test_providers_models_typed_key_wins_over_stored_key(provider_client, monkeypatch):
    """A key typed into the field lists models for THAT key — the
    try-before-you-save flow must not silently query the old stored key."""
    import httpx

    store = MagicMock()
    store.get_api_key.return_value = "sk-stored-global-key"
    monkeypatch.setattr(agents_v2, "_credential_store", store)

    captured: dict = {}
    monkeypatch.setattr(httpx, "AsyncClient", _fake_httpx_client(captured))

    resp = provider_client.post(
        "/api/v2/providers/models",
        json={
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "api_key": "sk-typed-fresh-key",
        },
    )
    assert resp.status_code == 200, resp.text
    assert captured["headers"].get("Authorization") == "Bearer sk-typed-fresh-key"


def test_providers_models_keyless_stays_keyless_without_stored_key(
    provider_client, monkeypatch
):
    """No typed key and no stored key = keyless request (LM Studio / Ollama).
    The fixture pins _credential_store to None."""
    import httpx

    captured: dict = {}
    monkeypatch.setattr(httpx, "AsyncClient", _fake_httpx_client(captured))

    resp = provider_client.post(
        "/api/v2/providers/models",
        json={"base_url": "http://localhost:1234/v1"},
    )
    assert resp.status_code == 200, resp.text
    assert "Authorization" not in captured["headers"], captured["headers"]


def test_providers_models_honors_request_sdk(provider_client, monkeypatch):
    """The Custom provider exposes an SDK dropdown, so an Anthropic-format
    endpoint must be listed via /v1/models + x-api-key. The handler used to
    read `sdk` off the registry entry only — and the registry's `custom` entry
    says "openai" — so custom+anthropic hit the wrong path with the wrong
    auth header. Also the path a per-model SDK override needs (OpenCode Go)."""
    import httpx

    captured: dict = {}
    monkeypatch.setattr(httpx, "AsyncClient", _fake_httpx_client(captured))

    resp = provider_client.post(
        "/api/v2/providers/models",
        json={
            "base_url": "https://router.example/anthropic",
            "api_key": "sk-k",
            "sdk": "anthropic",
        },
    )
    assert resp.status_code == 200, resp.text
    assert captured["url"] == "https://router.example/anthropic/v1/models"
    assert captured["headers"].get("x-api-key") == "sk-k"
    assert "Authorization" not in captured["headers"]


def test_providers_models_uses_fail_fast_timeout(provider_client, monkeypatch):
    """A blocked or slow /models endpoint must fail fast, not hold the picker
    spinning. 15s left the model dropdown loading long past the point a user
    reads it as broken (api.mistral.ai: ~10s to a TLS failure from CN)."""
    import httpx

    captured: dict = {}
    monkeypatch.setattr(httpx, "AsyncClient", _fake_httpx_client(captured))

    resp = provider_client.post(
        "/api/v2/providers/models",
        json={"base_url": "http://localhost:1234/v1"},
    )
    assert resp.status_code == 200, resp.text
    assert captured["client_kwargs"]["timeout"] <= 8


def test_providers_models_timeout_reports_a_non_empty_detail(provider_client, monkeypatch):
    """`str(httpx.ReadTimeout())` is the empty string, so a timed-out model
    fetch used to surface as `{"detail": ""}` — a blank error is unreadable in
    the daemon log and in any client that shows it. Fall back to the exception
    class name."""
    import httpx

    class BoomClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            raise httpx.ReadTimeout("")

    monkeypatch.setattr(httpx, "AsyncClient", BoomClient)

    resp = provider_client.post(
        "/api/v2/providers/models",
        json={"base_url": "http://127.0.0.1:9912/v1"},
    )
    assert resp.status_code == 502
    assert resp.json()["detail"].strip(), resp.text
