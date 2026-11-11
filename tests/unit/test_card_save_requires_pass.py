# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A provider key is saved only when its connection test PASSES.

Owner rule 2026-11-12 ("make it compulsory - you must pass the test to save
the key"), replacing spec 082 D9's "a failed test still saves, the card
carries the error". The daemon is the gate: every route that stores a
credential tests the card as it WOULD be stored first, and a failing verdict
of any kind refuses with ``connection_test_failed`` and writes nothing — no
card on an add, the stored card untouched on an edit, no key on the legacy
routes. Non-credential edits (a rename) still save without a test, and cards
stored earlier with a failing verdict are untouched: they list, re-test and
rename as before.
"""

from unittest.mock import MagicMock

import pytest

from agent_os.api.routes import agents_v2, settings
from agent_os.config.provider_registry import ProviderRegistry
from tests.card_doubles import FakeCardStore

GOOD = "sk-go-good"
MODEL = "deepseek-v4.1-flash"

# Every failure kind the connection test reports, keyed by the key that
# provokes it in the fake below.
FAILURES = {
    "sk-401": {"ok": False, "status": 401, "code": "invalid_api_key",
               "message": "Invalid API key"},
    "sk-402": {"ok": False, "status": 402, "code": "insufficient_credits",
               "message": "Insufficient balance"},
    "sk-429": {"ok": False, "status": 429, "code": "rate_limited",
               "message": "Rate limited — key works but slow down"},
    "sk-down": {"ok": False, "status": None, "code": "provider_unreachable",
                "message": "Connection error."},
}
BAD_MODEL = "no-such-model-xyz"
BAD_MODEL_RESULT = {"ok": False, "status": 400, "code": "provider_error",
                    "message": "Upstream request failed: Model is unavailable."}


@pytest.fixture
def env(monkeypatch):
    from agent_os import telemetry

    monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)
    emitted = []
    monkeypatch.setattr(telemetry, "emit", lambda *a, **k: emitted.append(a[0]))
    monkeypatch.setattr(telemetry, "latch", lambda *a, **k: None)
    store = FakeCardStore()
    monkeypatch.setattr(agents_v2, "_settings_store", store)
    monkeypatch.setattr(settings, "_settings_store", store)
    monkeypatch.setattr(agents_v2, "_credential_store", None)
    monkeypatch.setattr(agents_v2, "_provider_registry", ProviderRegistry())
    monkeypatch.setattr(agents_v2, "_RECENT_TESTS", {})
    monkeypatch.setattr(settings, "_refresh_agents_in_background", lambda *a: None)
    kicks = []
    monkeypatch.setattr(settings, "_import_scan_kick", kicks.append)

    calls = []

    async def fake_run(provider, model, api_key, base_url, sdk):
        calls.append((provider, model, api_key))
        if model == BAD_MODEL:
            return dict(BAD_MODEL_RESULT)
        if api_key in FAILURES:
            return dict(FAILURES[api_key])
        return {"ok": True, "status": 200, "code": None,
                "message": f"Connected to {provider} using {model}"}

    monkeypatch.setattr(agents_v2, "run_connection_test", fake_run)
    return store, calls, emitted, kicks


def _create(key=GOOD, model=MODEL):
    return settings.create_card(settings.CreateCardRequest(
        provider="opencode-go", model=model, api_key=key))


def _refusal(exc_info, expected):
    exc = exc_info.value
    assert exc.status_code == 400
    assert exc.detail["code"] == "connection_test_failed"
    assert exc.detail["message"] == expected["message"]
    assert exc.detail["test"]["code"] == expected["code"]


# -- add a card ---------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(FAILURES))
async def test_every_failure_kind_refuses_an_add_and_creates_nothing(env, key):
    store, _, emitted, kicks = env
    with pytest.raises(settings.HTTPException) as exc:
        await _create(key=key)
    _refusal(exc, FAILURES[key])
    assert store.cards == [] and store.keys == {} and store.health == []
    assert "key_set" not in emitted and kicks == []


async def test_an_unavailable_model_refuses_an_add(env):
    store, _, _, _ = env
    with pytest.raises(settings.HTTPException) as exc:
        await _create(model=BAD_MODEL)
    _refusal(exc, BAD_MODEL_RESULT)
    assert store.cards == []


async def test_a_passing_add_stores_the_card_verified(env):
    store, calls, emitted, _ = env
    res = await _create()
    assert res["test"]["ok"] is True
    assert [c.id for c in store.cards] == [res["card"]["id"]]
    assert store.keys[res["card"]["id"]] == GOOD
    assert store.health == [(res["card"]["id"], True, None)]
    assert len(calls) == 1 and "key_set" in emitted


# -- edit a card --------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(FAILURES))
async def test_a_failing_rekey_leaves_the_stored_card_exactly_as_it_was(env, key):
    store, _, _, _ = env
    card = store.add(card_id="c1", provider="opencode-go", model=MODEL, key=GOOD)
    before = card.model_dump()
    with pytest.raises(settings.HTTPException) as exc:
        await settings.update_card("c1", settings.UpdateCardRequest(
            api_key=key, name="renamed too"))
    _refusal(exc, FAILURES[key])
    assert store.keys["c1"] == GOOD
    assert store.resolve_card("c1").model_dump() == before
    assert store.health == []


async def test_a_failing_model_change_is_tested_with_the_stored_key_and_refused(env):
    store, calls, _, _ = env
    store.add(card_id="c1", provider="opencode-go", model=MODEL, key=GOOD)
    with pytest.raises(settings.HTTPException) as exc:
        await settings.update_card("c1", settings.UpdateCardRequest(model=BAD_MODEL))
    _refusal(exc, BAD_MODEL_RESULT)
    assert calls == [("opencode-go", BAD_MODEL, GOOD)]
    assert store.resolve_card("c1").model == MODEL


async def test_a_passing_rekey_stores_and_verifies(env):
    store, _, _, kicks = env
    store.add(card_id="c1", provider="opencode-go", model=MODEL, key="sk-401",
              make_default=True)
    res = await settings.update_card("c1", settings.UpdateCardRequest(api_key=GOOD))
    assert res["test"]["ok"] is True
    assert store.keys["c1"] == GOOD
    assert store.health == [("c1", True, None)]
    assert kicks == ["card_updated"]


async def test_a_rename_saves_without_a_test(env):
    store, calls, _, _ = env
    store.add(card_id="c1", provider="opencode-go", model=MODEL, key=GOOD)
    res = await settings.update_card("c1", settings.UpdateCardRequest(name="Work"))
    assert res["test"] is None and calls == []
    assert store.resolve_card("c1").name == "Work"


async def test_a_rename_with_the_forms_full_body_is_not_a_credential_change(env):
    """The edit form sends provider/region/model/sdk with every save. Their
    presence used to count as a credential change, so a rename spent a real
    test — and under the pass rule would be refused on a card whose key has
    since stopped working (live smoke 2026-11-12)."""
    store, calls, _, _ = env
    store.add(card_id="c1", provider="opencode-go", model=MODEL, key="sk-401")
    sdk = settings._test_target(store.resolve_card("c1"))[3]
    res = await settings.update_card("c1", settings.UpdateCardRequest(
        name="Work", provider="opencode-go", region="global", model=MODEL, sdk=sdk))
    assert res["test"] is None and calls == []
    assert store.resolve_card("c1").name == "Work"


async def test_changing_the_region_is_a_credential_change(env):
    store, calls, _, _ = env
    store.add(card_id="c1", provider="moonshot", model="kimi-k2", key=GOOD)
    await settings.update_card("c1", settings.UpdateCardRequest(region="china"))
    assert len(calls) == 1


# -- cards stored before the rule ---------------------------------------------


async def test_a_card_stored_with_a_failing_verdict_still_lists_retests_and_renames(env):
    store, calls, _, _ = env
    error = {"status": 401, "code": "invalid_api_key", "message": "Invalid API key",
             "at": "2026-11-01T00:00:00+00:00"}
    store.add(card_id="old", provider="opencode-go", model=MODEL, key="sk-401",
              last_error=error)
    store.masked_cards = lambda: [store.masked_card(c) for c in store.cards]

    listed = await settings.list_cards()
    assert listed["credential_cards"][0]["last_error"] == error

    res = await settings.test_card("old")      # 200 even when it fails
    assert res["test"]["ok"] is False
    assert store.health[-1][0] == "old" and store.health[-1][1] is False

    res = await settings.update_card("old", settings.UpdateCardRequest(name="Old"))
    assert res["test"] is None
    assert store.resolve_card("old").name == "Old"
    assert store.keys["old"] == "sk-401"


# -- legacy single-key routes -------------------------------------------------


async def test_legacy_put_settings_key_is_refused_when_its_test_fails(env):
    store, _, _, _ = env
    store.add(card_id="c1", provider="opencode-go", model=MODEL, key=GOOD,
              make_default=True)
    with pytest.raises(settings.HTTPException) as exc:
        await settings.update_settings(settings.UpdateSettingsRequest(
            llm_api_key="sk-401", telemetry_enabled=False))
    _refusal(exc, FAILURES["sk-401"])
    assert store.keys["c1"] == GOOD


async def test_legacy_put_settings_tests_the_merged_card(env):
    """Provider, model and key change together in one legacy PUT: the test
    runs on that combination, not on the stored card."""
    store, calls, _, _ = env
    store.add(card_id="c1", provider="opencode-go", model=MODEL, key=GOOD,
              make_default=True)
    await settings._legacy_key_gate(settings.UpdateSettingsRequest(
        llm_api_key="sk-new", llm_model="glm-5.3-flash"))
    assert calls == [("opencode-go", "glm-5.3-flash", "sk-new")]


async def test_legacy_key_routes_skip_the_test_when_the_env_key_rules(env, monkeypatch):
    """AGENT_OS_API_KEY set: the legacy routes persist nothing, so there is
    nothing to gate (the dev-daemon and test boot path)."""
    _, calls, _, _ = env
    monkeypatch.setenv("AGENT_OS_API_KEY", "sk-env")
    await settings._legacy_key_gate(settings.UpdateSettingsRequest(llm_api_key="sk-401"))
    creds = MagicMock()
    creds.set_api_key = MagicMock(return_value={"source": "environment"})
    monkeypatch.setattr(settings, "_credential_store", creds)
    res = await settings.set_api_key(settings.SetApiKeyRequest(api_key="sk-401"))
    assert res["source"] == "environment" and calls == []


async def test_legacy_put_api_key_is_refused_when_its_test_fails(env, monkeypatch):
    store, _, _, kicks = env
    store.add(card_id="c1", provider="opencode-go", model=MODEL, key=GOOD,
              make_default=True)
    creds = MagicMock()
    monkeypatch.setattr(settings, "_credential_store", creds)
    with pytest.raises(settings.HTTPException) as exc:
        await settings.set_api_key(settings.SetApiKeyRequest(api_key="sk-402"))
    _refusal(exc, FAILURES["sk-402"])
    creds.set_api_key.assert_not_called()
    assert kicks == []


async def test_legacy_put_api_key_stores_a_passing_key(env, monkeypatch):
    store, _, _, kicks = env
    store.add(card_id="c1", provider="opencode-go", model=MODEL, key="",
              make_default=True)
    creds = MagicMock()
    creds.set_api_key = MagicMock(return_value={"source": "keychain", "card_id": "c1"})
    monkeypatch.setattr(settings, "_credential_store", creds)
    res = await settings.set_api_key(settings.SetApiKeyRequest(api_key=GOOD))
    creds.set_api_key.assert_called_once_with(GOOD)
    assert res["card_id"] == "c1"
    assert store.health == [("c1", True, None)]
    assert kicks == ["legacy_key_saved"]
