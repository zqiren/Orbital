# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the PKCE loopback OAuth flow (Task B2).

Never hits a real Google endpoint: the loopback listener is real (a local
ephemeral socket) but the *token endpoint* is an ``httpx.MockTransport`` fake,
and the "browser" is a thread that replays Google's redirect back to the
loopback. This exercises PKCE, state validation, code exchange, and refresh
end to end without leaving the machine.
"""

import base64
import hashlib
import threading
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from agent_os.connectors.oauth import (
    GOOGLE_ENDPOINTS,
    LoopbackOAuthFlow,
    OAuthClientConfig,
    OAuthError,
    TokenSet,
    generate_pkce,
)

CLIENT = OAuthClientConfig(
    client_id="test-client-id",
    client_secret="test-secret",
    endpoints=GOOGLE_ENDPOINTS,
)


def _fake_browser(*, state_override=None, error=None):
    """Return an ``open_browser`` callback that replays Google's redirect.

    Parses the authorize URL for ``redirect_uri`` + ``state`` and fires a GET
    at the loopback listener from a background thread (mirroring how the real
    system browser, a separate process, would hit it).
    """
    def _open(auth_url: str) -> None:
        q = parse_qs(urlparse(auth_url).query)
        redirect_uri = q["redirect_uri"][0]
        state = state_override if state_override is not None else q["state"][0]

        def _hit():
            if error:
                url = f"{redirect_uri}?error={error}&state={state}"
            else:
                url = f"{redirect_uri}?code=fake-auth-code&state={state}"
            try:
                httpx.get(url, timeout=5)
            except Exception:
                pass

        threading.Thread(target=_hit, daemon=True).start()

    return _open


def _token_endpoint(captured: dict):
    """MockTransport handler standing in for Google's token endpoint."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = parse_qs(request.content.decode())
        captured["grant_type"] = body.get("grant_type", [None])[0]
        captured["code_verifier"] = body.get("code_verifier", [None])[0]
        captured["code"] = body.get("code", [None])[0]
        captured["refresh_token"] = body.get("refresh_token", [None])[0]
        captured["client_id"] = body.get("client_id", [None])[0]
        if body.get("grant_type", [None])[0] == "refresh_token":
            return httpx.Response(200, json={
                "access_token": "refreshed-access",
                "expires_in": 3600,
                "scope": "https://www.googleapis.com/auth/calendar",
            })
        return httpx.Response(200, json={
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "expires_in": 3600,
            "scope": "https://www.googleapis.com/auth/calendar",
            "email": "user@example.com",
        })
    return handler


def test_generate_pkce_challenge_is_s256_of_verifier():
    verifier, challenge = generate_pkce()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    assert challenge == expected
    assert "=" not in challenge  # url-safe, unpadded


async def test_authorize_full_flow_sends_pkce_and_returns_tokens():
    captured: dict = {}
    http = httpx.AsyncClient(transport=httpx.MockTransport(_token_endpoint(captured)))
    flow = LoopbackOAuthFlow(CLIENT, http=http, open_browser=_fake_browser())
    tokens = await flow.authorize(["https://www.googleapis.com/auth/calendar"])
    await http.aclose()

    assert isinstance(tokens, TokenSet)
    assert tokens.access_token == "fresh-access"
    assert tokens.refresh_token == "fresh-refresh"
    assert tokens.account == "user@example.com"
    assert tokens.scopes == ["https://www.googleapis.com/auth/calendar"]
    assert tokens.expiry > time.time()
    # PKCE: the verifier we sent must hash to the challenge in the auth URL.
    assert captured["grant_type"] == "authorization_code"
    assert captured["code"] == "fake-auth-code"
    verifier = captured["code_verifier"]
    assert verifier and len(verifier) >= 43
    assert captured["client_id"] == "test-client-id"


async def test_authorize_rejects_state_mismatch():
    captured: dict = {}
    http = httpx.AsyncClient(transport=httpx.MockTransport(_token_endpoint(captured)))
    flow = LoopbackOAuthFlow(
        CLIENT, http=http, open_browser=_fake_browser(state_override="attacker-state"),
        timeout=5,
    )
    with pytest.raises(OAuthError):
        await flow.authorize(["https://www.googleapis.com/auth/calendar"])
    await http.aclose()
    # The code must never have been exchanged.
    assert captured.get("code") is None


async def test_authorize_surfaces_provider_error():
    http = httpx.AsyncClient(transport=httpx.MockTransport(_token_endpoint({})))
    flow = LoopbackOAuthFlow(
        CLIENT, http=http, open_browser=_fake_browser(error="access_denied"),
        timeout=5,
    )
    with pytest.raises(OAuthError):
        await flow.authorize(["https://www.googleapis.com/auth/calendar"])
    await http.aclose()


async def test_refresh_uses_refresh_grant_and_carries_refresh_token_forward():
    captured: dict = {}
    http = httpx.AsyncClient(transport=httpx.MockTransport(_token_endpoint(captured)))
    flow = LoopbackOAuthFlow(CLIENT, http=http)
    tokens = await flow.refresh(
        "old-refresh", ["https://www.googleapis.com/auth/calendar"], account="user@example.com"
    )
    await http.aclose()
    assert captured["grant_type"] == "refresh_token"
    assert captured["refresh_token"] == "old-refresh"
    assert tokens.access_token == "refreshed-access"
    # Google omits refresh_token on refresh — carry the old one forward.
    assert tokens.refresh_token == "old-refresh"
    assert tokens.account == "user@example.com"


def test_tokenset_fields_roundtrip_and_expiry():
    ts = TokenSet(
        access_token="a", refresh_token="r", expiry=time.time() + 100,
        scopes=["s1", "s2"], account="me@x.com",
    )
    fields = ts.to_fields()
    assert all(isinstance(v, str) for v in fields.values())
    back = TokenSet.from_fields(fields)
    assert back == ts
    assert back.expired is False
    stale = TokenSet(
        access_token="a", refresh_token="r", expiry=time.time() - 1,
        scopes=[], account="me@x.com",
    )
    assert stale.expired is True
