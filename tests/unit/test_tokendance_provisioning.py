# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for TokenDance one-click key provisioning (Spec 47 Tier 2).

Never hits tokendance.space: the loopback listener is real (a local ephemeral
socket) but the exchange/balance endpoints are an ``httpx.MockTransport`` fake,
and the "browser" is a thread replaying their redirect back to the loopback —
mirroring ``test_connector_oauth.py``'s harness. TokenDance-specific contract
under test: the CSRF nonce rides inside ``callback_url``'s query (they have no
first-class ``state`` param — existing query params are preserved on redirect
and ``code`` is appended), the exchange is a JSON POST with no client secret,
and a minted key that fails balance validation is never returned.
"""

import base64
import hashlib
import json
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import HTTPException

from agent_os.api.routes import agents_v2
from agent_os.providers_auth import tokendance as td
from agent_os.providers_auth.tokendance import (
    TokenDanceProvisioningError,
    provision_api_key,
)

FRESH_KEY = "sk-td-fresh-key-9876"


def _fake_browser(*, state_override=None, error=None, no_code=False, capture=None):
    """Replay TokenDance's redirect: preserve the callback URL's existing
    query (which carries our state nonce) and append ``code``."""

    def _open(auth_url: str) -> None:
        if capture is not None:
            capture["auth_url"] = auth_url
        q = parse_qs(urlparse(auth_url).query)
        callback_url = q["callback_url"][0]
        cb_query = parse_qs(urlparse(callback_url).query)
        state = state_override if state_override is not None else cb_query["state"][0]
        base = callback_url.split("?")[0]

        def _hit():
            if error:
                url = f"{base}?state={state}&error={error}"
            elif no_code:
                url = f"{base}?state={state}"
            else:
                url = f"{base}?state={state}&code=fake-code-123"
            try:
                httpx.get(url, timeout=5)
            except Exception:
                pass

        threading.Thread(target=_hit, daemon=True).start()

    return _open


def _td_endpoints(captured: dict, *, exchange_status=200, balance_status=200):
    """MockTransport handler standing in for exchange + balance endpoints."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == td.EXCHANGE_URL and request.method == "POST":
            captured["exchange_body"] = json.loads(request.content.decode())
            if exchange_status != 200:
                return httpx.Response(exchange_status, json={"error": "forbidden"})
            return httpx.Response(200, json={"key": FRESH_KEY})
        if str(request.url) == td.BALANCE_URL and request.method == "GET":
            captured["balance_auth"] = request.headers.get("Authorization")
            if balance_status != 200:
                return httpx.Response(
                    balance_status,
                    json={"error": {"code": "unauthorized", "message": "bad key"}},
                )
            return httpx.Response(
                200,
                json={"balance": {"credits": 100, "credits_used": 0, "balance": 100}},
            )
        return httpx.Response(404)

    return handler


def _client(captured: dict, **kw) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(_td_endpoints(captured, **kw)))


async def test_happy_path_returns_validated_key():
    captured: dict = {}
    auth: dict = {}
    key = await provision_api_key(
        http=_client(captured), open_browser=_fake_browser(capture=auth)
    )
    assert key == FRESH_KEY

    # Authorize URL shape: their /auth page, loopback callback with the state
    # nonce embedded, S256 challenge, attribution + key label.
    parsed = urlparse(auth["auth_url"])
    assert auth["auth_url"].startswith(td.AUTHORIZE_URL + "?")
    q = parse_qs(parsed.query)
    assert q["callback_url"][0].startswith("http://127.0.0.1:")
    assert "state=" in q["callback_url"][0]
    assert q["code_challenge_method"] == ["S256"]
    assert q["app_url"] == [td.APP_URL]
    assert q["key_name"] == ["Orbital"]

    # Exchange body: JSON, no client secret, verifier binds to the challenge.
    body = captured["exchange_body"]
    assert body["code"] == "fake-code-123"
    assert body["code_challenge_method"] == "S256"
    assert set(body) == {"code", "code_verifier", "code_challenge_method"}
    expected_challenge = (
        base64.urlsafe_b64encode(
            hashlib.sha256(body["code_verifier"].encode()).digest()
        )
        .rstrip(b"=")
        .decode()
    )
    assert q["code_challenge"] == [expected_challenge]

    # The minted key was validated against the balance endpoint.
    assert captured["balance_auth"] == f"Bearer {FRESH_KEY}"


async def test_state_mismatch_aborts_before_exchange():
    captured: dict = {}
    with pytest.raises(TokenDanceProvisioningError, match="state mismatch"):
        await provision_api_key(
            http=_client(captured),
            open_browser=_fake_browser(state_override="evil-nonce"),
        )
    assert "exchange_body" not in captured


async def test_denied_consent_surfaces_error():
    captured: dict = {}
    with pytest.raises(TokenDanceProvisioningError, match="authorization denied"):
        await provision_api_key(
            http=_client(captured), open_browser=_fake_browser(error="access_denied")
        )
    assert "exchange_body" not in captured


async def test_missing_code_is_an_error():
    captured: dict = {}
    with pytest.raises(TokenDanceProvisioningError, match="no authorization code"):
        await provision_api_key(
            http=_client(captured), open_browser=_fake_browser(no_code=True)
        )


async def test_timeout_when_browser_never_returns():
    captured: dict = {}
    with pytest.raises(TokenDanceProvisioningError, match="timed out"):
        await provision_api_key(
            http=_client(captured), open_browser=lambda url: None, timeout=0.2
        )


async def test_exchange_403_fails_without_balance_check():
    captured: dict = {}
    with pytest.raises(TokenDanceProvisioningError, match="start the sign-in again"):
        await provision_api_key(
            http=_client(captured, exchange_status=403),
            open_browser=_fake_browser(),
        )
    assert "balance_auth" not in captured


async def test_invalid_minted_key_is_rejected():
    captured: dict = {}
    with pytest.raises(TokenDanceProvisioningError, match="failed validation"):
        await provision_api_key(
            http=_client(captured, balance_status=401),
            open_browser=_fake_browser(),
        )


def test_app_url_matches_registry_attribution_header():
    """Their app-attribution doc treats OAuth `app_url` (key dimension) and
    the `X-App-URL` request header (request dimension, wins when present) as
    the SAME identifier — the two must never drift, and both must byte-match
    the App URL registered in the partner backend. X-App-URL is the ONLY
    attribution header for new integrations (X-Site-URL is a legacy fallback;
    nothing else participates), so the entry sends exactly one header."""
    registry_path = (
        Path(td.__file__).resolve().parents[1] / "config" / "providers.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry["providers"]["tokendance"]
    assert entry["extra_headers"] == {"X-App-URL": td.APP_URL}


# ---- signin route ----


class _FakeCredentialStore:
    def __init__(self):
        self.stored: list[str] = []

    def set_api_key(self, key: str) -> dict:
        self.stored.append(key)
        return {"configured": True, "source": "keyring"}


def _signin_store(**kw):
    """Spec 082 §3.6: sign-in writes a CARD, not the one global key slot.

    The route moved off ``_credential_store`` entirely — it creates or
    refreshes exactly one card, never touches another card's key, and takes
    the default only when there is no default yet.
    """
    from tests.card_doubles import FakeCardStore

    return FakeCardStore(**kw)


@pytest.fixture
def _quiet_telemetry(monkeypatch):
    from agent_os import telemetry

    monkeypatch.setattr(telemetry, "emit", lambda *a, **k: None)
    monkeypatch.setattr(telemetry, "latch", lambda *a, **k: None)


async def test_signin_route_persists_key_on_a_card_and_returns_masks(
    monkeypatch, _quiet_telemetry
):
    store = _signin_store()
    monkeypatch.setattr(agents_v2, "_settings_store", store)
    monkeypatch.setattr(agents_v2, "_provider_registry", None)

    async def fake_provision(**kw):
        return FRESH_KEY

    async def fake_test(card_id):
        return {"ok": True, "status": 200, "code": None, "message": ""}

    monkeypatch.setattr(td, "provision_api_key", fake_provision)
    monkeypatch.setattr(agents_v2, "_test_and_record", fake_test)
    result = await agents_v2.tokendance_signin()

    # The key landed on exactly one new card, and is never returned raw.
    assert [c.provider for c in store.cards] == ["tokendance"]
    card_id = store.cards[0].id
    assert store.keys[card_id] == FRESH_KEY
    assert FRESH_KEY not in json.dumps(result, default=str)
    # Legacy fields kept for an older SPA reading the pre-cards shape.
    assert result["api_key_set"] is True
    assert result["api_key_masked"] == "sk-t...9876"
    # First card ever becomes the default.
    assert result["default_card_id"] == card_id


async def test_signin_route_does_not_steal_an_existing_default(
    monkeypatch, _quiet_telemetry
):
    """§3.6: it takes the default ONLY when there is no default yet.

    The pre-cards route overwrote the single global slot, evicting whichever
    provider's key was in it — the exact failure this spec exists to end.
    """
    store = _signin_store()
    kept = store.add(card_id="card_kept", provider="anthropic", model="claude-x",
                     key="sk-other", make_default=True)
    monkeypatch.setattr(agents_v2, "_settings_store", store)
    monkeypatch.setattr(agents_v2, "_provider_registry", None)

    async def fake_provision(**kw):
        return FRESH_KEY

    async def fake_test(card_id):
        return {"ok": True, "status": 200, "code": None, "message": ""}

    monkeypatch.setattr(td, "provision_api_key", fake_provision)
    monkeypatch.setattr(agents_v2, "_test_and_record", fake_test)
    result = await agents_v2.tokendance_signin()

    assert result["default_card_id"] == kept.id, "an existing default is kept"
    assert store.keys["card_kept"] == "sk-other", "another card's key is untouched"
    assert len(store.cards) == 2


async def test_signin_route_takes_a_default_whose_key_is_gone(
    monkeypatch, _quiet_telemetry
):
    """A stored default with no key is what puts the wizard on screen
    (``api_key_set`` is derived from it), so the minted key must end up on
    the default — otherwise the new card verifies green, ``api_key_set``
    stays false, and Next reports "that key was not saved" (2026-09-06)."""
    store = _signin_store()
    dead = store.add(card_id="card_dead", provider="tokendance",
                     model="deepseek-v4-flash", key="", make_default=True)
    monkeypatch.setattr(agents_v2, "_settings_store", store)
    monkeypatch.setattr(agents_v2, "_provider_registry", None)

    async def fake_provision(**kw):
        return FRESH_KEY

    async def fake_test(card_id):
        return {"ok": True, "status": 200, "code": None, "message": ""}

    monkeypatch.setattr(td, "provision_api_key", fake_provision)
    monkeypatch.setattr(agents_v2, "_test_and_record", fake_test)
    result = await agents_v2.tokendance_signin()

    new_id = next(c.id for c in store.cards if c.id != dead.id)
    assert result["default_card_id"] == new_id, "the keyless default is replaced"
    assert store.keys[new_id] == FRESH_KEY
    assert store.keys["card_dead"] == "", "the dead card is left alone, not rewritten"


async def test_signin_route_reconnect_promotes_when_default_is_keyless(
    monkeypatch, _quiet_telemetry
):
    """Re-connecting a NON-default TokenDance card by id while the stored
    default has no key: the refreshed card becomes the default, since the
    key it just received is the only usable one."""
    store = _signin_store()
    store.add(card_id="card_dead", provider="openai", model="gpt-4o",
              key="", make_default=True)
    mine = store.add(card_id="card_td", provider="tokendance",
                     model="deepseek-v4-flash", key="sk-old")
    monkeypatch.setattr(agents_v2, "_settings_store", store)
    monkeypatch.setattr(agents_v2, "_provider_registry", None)

    async def fake_provision(**kw):
        return FRESH_KEY

    async def fake_test(card_id):
        return {"ok": True, "status": 200, "code": None, "message": ""}

    monkeypatch.setattr(td, "provision_api_key", fake_provision)
    monkeypatch.setattr(agents_v2, "_test_and_record", fake_test)
    result = await agents_v2.tokendance_signin(
        agents_v2.TokenDanceSigninRequest(card_id=mine.id)
    )

    assert store.keys["card_td"] == FRESH_KEY
    assert result["default_card_id"] == "card_td"
    assert len(store.cards) == 2, "no third card was minted"


async def test_signin_route_501_without_settings_store(monkeypatch):
    monkeypatch.setattr(agents_v2, "_settings_store", None)
    with pytest.raises(HTTPException) as exc:
        await agents_v2.tokendance_signin()
    assert exc.value.status_code == 501


async def test_signin_route_flow_failure_is_502_and_persists_nothing(
    monkeypatch, _quiet_telemetry
):
    store = _signin_store()
    monkeypatch.setattr(agents_v2, "_settings_store", store)

    async def failing_provision(**kw):
        raise TokenDanceProvisioningError("sign-in timed out waiting for the browser redirect")

    monkeypatch.setattr(td, "provision_api_key", failing_provision)
    with pytest.raises(HTTPException) as exc:
        await agents_v2.tokendance_signin()
    assert exc.value.status_code == 502
    assert "timed out" in exc.value.detail
    assert store.cards == [], "a failed flow must not leave a card behind"
    # (card assertion above replaces the old global-slot check)
