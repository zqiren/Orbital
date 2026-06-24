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
def provider_client():
    """Minimal app with just the providers router. The test_connection handler
    tolerates a None registry (falls back to req.base_url), so no other wiring
    is needed."""
    app = FastAPI()
    app.include_router(agents_v2.router)
    agents_v2._provider_registry = None
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
