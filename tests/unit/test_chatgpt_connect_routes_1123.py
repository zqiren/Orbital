# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 1123 routes: ChatGPT sign-in card, sign-out = delete card, the
connect screen's catalog and check, the model list shape, and upgrade safety
for existing cards. Real ``SettingsStore`` on a dict-backed key store; the
sign-in, the connection test and OpenAI's model list are faked.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi import HTTPException

from agent_os.api.routes import agents_v2, settings
from agent_os.config import provider_registry as pr
from agent_os.config.provider_registry import ProviderRegistry
from agent_os.daemon_v2.settings_store import CredentialCard, SettingsStore
from agent_os.providers_auth import chatgpt
from agent_os.providers_auth.token_cards import dump_bundle, parse_bundle


class DictKeys:
    def __init__(self):
        self.items: dict[str, str] = {}

    def get(self, cid):
        return self.items.get(cid)

    def set(self, cid, key):
        if not key:
            self.items.pop(cid, None)
            return {"source": "none"}
        self.items[cid] = key
        return {"source": "keychain"}

    def delete(self, cid):
        self.items.pop(cid, None)

    def source(self, cid):
        return "keychain" if self.items.get(cid) else "none"


def _bundle(**kw):
    b = {"provider": "chatgpt", "client_id": "oaiapp_x", "host_id": "urn:uuid:h",
         "sub": "user-1", "email": "person@example.com", "access_token": "at-1",
         "refresh_token": "rt-1", "id_token": "idt-1",
         "expires_at": time.time() + 3600, "scopes": chatgpt.SCOPE.split()}
    b.update(kw)
    return b


LIVE = [{"id": "gpt-6.2-sol", "display_name": "GPT-6.2 Sol", "context_window": 500000},
        {"id": "gpt-6-luna", "display_name": "GPT-6 Luna", "context_window": 272000}]


@pytest.fixture
def env(monkeypatch, tmp_path):
    from agent_os import telemetry

    monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)
    monkeypatch.setattr(telemetry, "emit", lambda *a, **k: None)
    monkeypatch.setattr(telemetry, "latch", lambda *a, **k: None)
    keys = DictKeys()
    store = SettingsStore(str(tmp_path), card_key_store=keys)
    monkeypatch.setattr(agents_v2, "_settings_store", store)
    monkeypatch.setattr(settings, "_settings_store", store)
    monkeypatch.setattr(settings, "_project_store", None)
    monkeypatch.setattr(agents_v2, "_credential_store", None)
    monkeypatch.setattr(agents_v2, "_provider_registry", ProviderRegistry())
    monkeypatch.setattr(agents_v2, "_RECENT_TESTS", {})
    monkeypatch.setattr(settings, "_refresh_agents_in_background", lambda *a: None)

    state = {"tests": [], "lists": [], "revoked": [], "signin_kw": None,
             "test_result": None, "revoke_error": None}

    async def fake_run(provider, model, api_key, base_url, sdk):
        state["tests"].append((provider, model, api_key, base_url, sdk))
        if state["test_result"]:
            return dict(state["test_result"])
        if api_key == "sk-bad":
            return {"ok": False, "status": 401, "code": "invalid_api_key",
                    "message": "Invalid API key"}
        return {"ok": True, "status": 200, "code": None,
                "message": f"Connected to {provider} using {model}"}

    async def fake_list(provider, api_key, base_url, sdk):
        state["lists"].append((provider, api_key, base_url, sdk))
        if provider == "chatgpt":
            for m in LIVE:
                pr.note_live_context_window(provider, m["id"], m["context_window"])
            return [dict(m) for m in LIVE]
        if api_key == "sk-bad":
            raise agents_v2._ModelListError(401, "Provider returned 401")
        return [{"id": "kimi-k2.6", "display_name": None, "context_window": None},
                {"id": "kimi-k2.5", "display_name": None, "context_window": None}]

    async def fake_sign_in(**kw):
        state["signin_kw"] = kw
        return _bundle()

    async def fake_revoke(bundle, **kw):
        state["revoked"].append(bundle["refresh_token"])
        if state["revoke_error"]:
            raise chatgpt.ChatGPTAuthError("revoke_failed", state["revoke_error"])

    monkeypatch.setattr(agents_v2, "run_connection_test", fake_run)
    monkeypatch.setattr(agents_v2, "_list_provider_models", fake_list)
    monkeypatch.setattr(chatgpt, "sign_in", fake_sign_in)
    monkeypatch.setattr(chatgpt, "revoke", fake_revoke)
    return store, keys, state


async def _signin(card_id=None):
    return await agents_v2.chatgpt_signin(agents_v2.ChatGPTSigninRequest(card_id=card_id))


# -- sign-in → card ------------------------------------------------------------

@pytest.mark.asyncio
async def test_signin_creates_a_tested_chatgpt_card_with_default_model(env):
    store, keys, state = env
    out = await _signin()
    card = out["card"]
    assert card["provider"] == "chatgpt" and card["model"] == "gpt-6.2-sol"
    assert card["auth"] == "signin" and card["account"] == {"email": "person@example.com"}
    assert card["key_masked"] == "" and card["needs_signin"] is False
    assert card["is_default"] and out["default_card_id"] == card["id"]
    assert out["test"]["ok"] and out["account"]["email"] == "person@example.com"
    # Stored as ONE secret; every key consumer gets the access token.
    assert parse_bundle(keys.items[card["id"]])["refresh_token"] == "rt-1"
    assert store.key_for(card["id"]) == "at-1"
    # The test ran on the ChatGPT route with the fresh access token.
    provider, model, key, base, sdk = state["tests"][-1]
    assert (provider, model, key) == ("chatgpt", "gpt-6.2-sol", "at-1")
    assert base == "https://api.openai.com/v1" and sdk == "openai-responses"
    # The route's context window, not the catalog's, drives the limits.
    assert ProviderRegistry().get_model_info("chatgpt", "gpt-6.2-sol").context_window == 500000
    # The response never carries a token.
    assert "rt-1" not in json.dumps(out) and "at-1" not in json.dumps(out)


@pytest.mark.asyncio
async def test_signin_again_on_a_card_reuses_its_registration(env):
    store, keys, state = env
    first = (await _signin())["card"]
    await agents_v2.chatgpt_signin(agents_v2.ChatGPTSigninRequest(card_id=first["id"]))
    kw = state["signin_kw"]
    assert kw["client_id"] == "oaiapp_x" and kw["id_token_hint"] == "idt-1"
    assert kw["expected_sub"] == "user-1" and kw["host_id"] == "urn:uuid:h"
    assert [c.id for c in store.stored_cards()] == [first["id"]]


@pytest.mark.asyncio
async def test_signin_failure_stores_nothing(env, monkeypatch):
    store, keys, state = env

    async def refused(**kw):
        raise chatgpt.ChatGPTAuthError("plan_usage_not_granted", "no plan usage",
                                       status=403)
    monkeypatch.setattr(chatgpt, "sign_in", refused)
    with pytest.raises(HTTPException) as e:
        await _signin()
    assert e.value.status_code == 403
    assert e.value.detail == {"code": "plan_usage_not_granted", "message": "no plan usage"}
    assert store.stored_cards() == [] and keys.items == {}


@pytest.mark.asyncio
async def test_a_chatgpt_card_cannot_be_made_or_rekeyed_by_hand(env):
    store, keys, state = env
    with pytest.raises(HTTPException) as e:
        await settings.create_card(settings.CreateCardRequest(
            provider="chatgpt", model="gpt-6.2-sol", api_key="sk-pasted"))
    assert e.value.detail["code"] == "signin_required"
    card = (await _signin())["card"]
    with pytest.raises(HTTPException) as e:
        await settings.update_card(card["id"], settings.UpdateCardRequest(api_key="sk-x"))
    assert e.value.detail["code"] == "card_key_managed"
    assert parse_bundle(keys.items[card["id"]]) is not None


@pytest.mark.asyncio
async def test_changing_the_model_tests_with_a_fresh_token_then_saves(env):
    store, keys, state = env
    card = (await _signin())["card"]
    # Make the stored token near expiry: the test must use a refreshed one.
    keys.items[card["id"]] = dump_bundle(_bundle(expires_at=0))

    async def refresher(b):
        return {**b, "access_token": "at-2", "refresh_token": "rt-2",
                "expires_at": time.time() + 3600}
    store.tokens._refresher = refresher
    out = await settings.update_card(card["id"], settings.UpdateCardRequest(model="gpt-6-luna"))
    assert out["card"]["model"] == "gpt-6-luna" and out["test"]["ok"]
    assert state["tests"][-1][:3] == ("chatgpt", "gpt-6-luna", "at-2")
    assert parse_bundle(keys.items[card["id"]])["refresh_token"] == "rt-2"


@pytest.mark.asyncio
async def test_a_dead_signin_marks_the_card_and_the_test_says_sign_in(env):
    store, keys, state = env
    card = (await _signin())["card"]
    keys.items[card["id"]] = dump_bundle(_bundle(expires_at=0))

    async def dead(b):
        raise chatgpt.ChatGPTAuthError("needs_signin", chatgpt.NEEDS_SIGNIN_MESSAGE,
                                       status=401, terminal=True)
    store.tokens._refresher = dead
    out = await settings.test_card(card["id"])
    assert out["test"]["code"] == "needs_signin" and not out["test"]["ok"]
    assert out["card"]["needs_signin"] is True
    assert out["card"]["last_error"]["code"] == "needs_signin"
    n = len(state["tests"])
    check = await agents_v2.provider_check(agents_v2.ProviderCheckRequest(card_id=card["id"]))
    assert check["ok"] is False and check["code"] == "needs_signin"
    assert len(state["tests"]) == n  # no provider call on a dead sign-in


# -- sign-out = delete ---------------------------------------------------------

@pytest.mark.asyncio
async def test_signout_revokes_then_removes_card_and_secret(env):
    store, keys, state = env
    card = (await _signin())["card"]
    out = await settings.delete_card(card["id"])
    assert out["deleted"] and out["revoked"] is True and out["revoke_error"] is None
    assert state["revoked"] == ["rt-1"]
    assert store.stored_cards() == [] and card["id"] not in keys.items


@pytest.mark.asyncio
async def test_signout_offline_still_removes_and_says_so(env):
    store, keys, state = env
    card = (await _signin())["card"]
    state["revoke_error"] = "Orbital could not reach OpenAI to revoke the sign-in"
    out = await settings.delete_card(card["id"])
    assert out["revoked"] is False and "could not reach" in out["revoke_error"]
    assert card["id"] not in keys.items


@pytest.mark.asyncio
async def test_deleting_an_api_key_card_does_not_call_revoke(env):
    store, keys, state = env
    c = store.create_card(provider="openai", model="gpt-6.2-sol", api_key="sk-good")
    out = await settings.delete_card(c.id)
    assert out["revoked"] is None and state["revoked"] == []


# -- catalog + check -----------------------------------------------------------

@pytest.mark.asyncio
async def test_catalog_lists_regions_consoles_and_signins(env):
    cat = {p["id"]: p for p in (await agents_v2.provider_catalog())["providers"]}
    moon = cat["moonshot"]
    assert [r["id"] for r in moon["regions"]] == ["global", "china"]
    assert moon["regions"][0]["console_url"] != moon["regions"][1]["console_url"]
    assert cat["openai"]["regions"] == [] and cat["openai"]["console_url"]
    assert cat["openai"]["signin"] is None and cat["openai"]["key_entry"]
    assert cat["chatgpt"]["signin"] == "chatgpt" and cat["chatgpt"]["auth"] == "signin"
    assert cat["chatgpt"]["key_entry"] is False
    assert cat["tokendance"]["signin"] == "tokendance" and cat["tokendance"]["key_entry"]
    assert cat["tokendance"]["regions"] == [] and cat["tokendance"]["default_region"] == "china"
    assert cat["custom"]["custom_endpoint"] and cat["custom"]["auth"] == "none"
    assert list(cat)[:2] == ["openai", "chatgpt"]


@pytest.mark.asyncio
async def test_check_passes_with_live_models_and_preselects(env):
    store, keys, state = env
    out = await agents_v2.provider_check(agents_v2.ProviderCheckRequest(
        provider="moonshot", region="china", api_key="sk-good"))
    assert out["ok"] and out["models"] == ["kimi-k2.6", "kimi-k2.5"]
    assert out["default_model"] == out["tested_model"]
    assert out["default_model"] in out["models"] and out["models_source"] == "live"
    assert state["lists"][-1][2] == "https://api.moonshot.cn/v1"  # region honoured
    # The check's verdict stands in for the Save's test (no second call).
    n = len(state["tests"])
    await settings.create_card(settings.CreateCardRequest(
        provider="moonshot", region="china", api_key="sk-good",
        model=out["default_model"]))
    assert len(state["tests"]) == n


@pytest.mark.asyncio
async def test_check_failure_carries_a_reason_and_no_models(env):
    out = await agents_v2.provider_check(agents_v2.ProviderCheckRequest(
        provider="moonshot", api_key="sk-bad"))
    assert out["ok"] is False and out["code"] == "invalid_api_key"
    assert out["message"] and out["models"] == [] and out["default_model"] is None


@pytest.mark.asyncio
async def test_check_refuses_a_pasted_chatgpt_key(env):
    with pytest.raises(HTTPException) as e:
        await agents_v2.provider_check(agents_v2.ProviderCheckRequest(
            provider="chatgpt", api_key="x"))
    assert e.value.detail["code"] == "signin_required"


# -- model list shape ----------------------------------------------------------

@pytest.mark.asyncio
async def test_models_route_reads_the_slug_shape(monkeypatch, env):
    import httpx

    store, keys, state = env
    monkeypatch.undo()  # real _list_provider_models; keep the registry
    monkeypatch.setattr(agents_v2, "_provider_registry", ProviderRegistry())
    real_client = httpx.AsyncClient

    def handler(request):
        assert request.headers["Authorization"] == "Bearer at-x"
        return httpx.Response(200, json={"models": [
            {"slug": "gpt-6.1-terra", "display_name": "GPT-6.1 Terra",
             "visibility": "list", "context_window": 480000},
            {"slug": "gpt-reserve", "visibility": "hide"},
            {"slug": "gpt-6-luna", "display_name": "GPT-6 Luna", "visibility": "list",
             "context_window": 272000}]})

    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler)))
    entries = await agents_v2._list_provider_models(
        "chatgpt", "at-x", "https://api.openai.com/v1", "openai-responses")
    assert [m["id"] for m in entries] == ["gpt-6.1-terra", "gpt-6-luna"]  # server order
    assert ProviderRegistry().get_model_info("chatgpt", "gpt-6.1-terra").context_window == 480000


# -- upgrade safety ------------------------------------------------------------

@pytest.mark.asyncio
async def test_existing_api_key_cards_are_untouched(env):
    store, keys, state = env
    c = store.create_card(provider="openai", model="gpt-6.2-sol", api_key="sk-live-1234")
    before = dict(keys.items)
    await _signin()
    assert keys.items[c.id] == before[c.id] == "sk-live-1234"
    assert store.key_for(c.id) == "sk-live-1234"
    m = store.masked_card(store.get_card(c.id))
    assert m["key_masked"] == "sk-l...1234" and m["auth"] == "api_key"
    assert m["account"] is None and m["needs_signin"] is False
    # The API-key card stays the default; the sign-in did not take it over.
    assert store.effective_default_card_id() == c.id


@pytest.mark.asyncio
async def test_chatgpt_card_on_disk_uses_only_fields_an_older_build_knows(env, tmp_path):
    store, keys, state = env
    card = (await _signin())["card"]
    raw = json.loads((tmp_path / "settings.json").read_text())
    row = next(r for r in raw["credential_cards"] if r["id"] == card["id"])
    # The pre-1123 CredentialCard fields (unchanged by this feature).
    old_fields = {"id", "name", "provider", "region", "base_url", "sdk", "model",
                  "created_at", "verified_at", "last_used_at", "last_error"}
    assert set(row) <= old_fields
    assert set(CredentialCard.model_fields) == old_fields
    # Nothing secret in settings.json.
    text = (tmp_path / "settings.json").read_text()
    assert "rt-1" not in text and "at-1" not in text and "person@" not in text


# -- NEEDS.md N1–N3: cancel / reopen / re-test a saved card on another region --

@pytest.mark.asyncio
async def test_cancel_and_reopen_a_pending_signin(env, monkeypatch):
    import asyncio
    import webbrowser

    store, keys, state = env
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))

    async def waiting(**kw):
        kw["on_url"]("https://auth.openai.com/api/accounts/authorize?x=1")
        await kw["cancel"].wait()
        raise chatgpt.ChatGPTAuthError("signin_cancelled", "cancelled", status=499)
    monkeypatch.setattr(chatgpt, "sign_in", waiting)
    task = asyncio.ensure_future(_signin())
    await asyncio.sleep(0.05)
    with pytest.raises(HTTPException) as e:
        await _signin()
    assert e.value.status_code == 409
    assert await agents_v2.signin_reopen("chatgpt") == {"reopened": True}
    assert opened == ["https://auth.openai.com/api/accounts/authorize?x=1"]
    assert await agents_v2.signin_cancel("chatgpt") == {"cancelled": True}
    with pytest.raises(HTTPException) as e:
        await task
    assert e.value.status_code == 499 and e.value.detail["code"] == "signin_cancelled"
    assert await agents_v2.signin_cancel("chatgpt") == {"cancelled": False}
    assert await agents_v2.signin_reopen("chatgpt") == {"reopened": False}
    assert store.stored_cards() == []


@pytest.mark.asyncio
async def test_tokendance_signin_can_be_cancelled(env, monkeypatch):
    import asyncio
    from agent_os.providers_auth import tokendance as td

    async def waiting(**kw):
        await kw["cancel"].wait()
        raise td.TokenDanceProvisioningError("sign-in was cancelled")
    monkeypatch.setattr(td, "provision_api_key", waiting)
    task = asyncio.ensure_future(agents_v2.tokendance_signin())
    await asyncio.sleep(0.05)
    assert await agents_v2.signin_cancel("tokendance") == {"cancelled": True}
    with pytest.raises(HTTPException) as e:
        await task
    assert e.value.status_code == 499


@pytest.mark.asyncio
async def test_check_a_saved_card_on_the_other_region_then_save_without_retest(env):
    store, keys, state = env
    c = store.create_card(provider="moonshot", model="kimi-k2.6", region="global",
                          api_key="sk-good")
    out = await agents_v2.provider_check(agents_v2.ProviderCheckRequest(
        card_id=c.id, region="china", model="kimi-k2.6"))
    assert out["ok"] and state["tests"][-1][3] == "https://api.moonshot.cn/v1"
    assert state["tests"][-1][2] == "sk-good"   # the stored key, never sent by the UI
    n = len(state["tests"])
    res = await settings.update_card(c.id, settings.UpdateCardRequest(
        region="china", model="kimi-k2.6"))
    assert res["card"]["region"] == "china" and len(state["tests"]) == n


@pytest.mark.asyncio
async def test_a_signed_in_default_counts_as_configured(env):
    store, keys, state = env
    await _signin()
    assert store.get_masked()["llm"]["api_key_set"] is True
