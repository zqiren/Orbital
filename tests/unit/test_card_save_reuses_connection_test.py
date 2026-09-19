# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Card Save records the Test Connection verdict instead of re-testing.

Since v0.12.3 the provider form only lets a card be saved after Test
Connection ran on the exact inputs on screen. The save route still ran its own
connection test (spec 082 D9), so every Save cost a second completion on the
same provider/model/key/endpoint and made the user wait for it (user report
2026-09-19: "the save button still does the testing, which is unnecessary").

``/providers/test`` now remembers its verdict, keyed by a hash of the exact
effective inputs, for a short while; card create/update take that verdict
(single use) and only fall back to a real test when there is none — an older
client that never tested, a different key or model, or a stale entry. The
per-card "test" action always runs a real test: that is what it is for.
"""

import pytest

from agent_os.api.routes import agents_v2, settings
from agent_os.config.provider_registry import ProviderRegistry
from tests.card_doubles import FakeCardStore

KEY = "sk-go-same-key"


@pytest.fixture
def env(monkeypatch):
    from agent_os import telemetry

    monkeypatch.setattr(telemetry, "emit", lambda *a, **k: None)
    monkeypatch.setattr(telemetry, "latch", lambda *a, **k: None)
    store = FakeCardStore()
    monkeypatch.setattr(agents_v2, "_settings_store", store)
    monkeypatch.setattr(settings, "_settings_store", store)
    monkeypatch.setattr(agents_v2, "_credential_store", None)
    monkeypatch.setattr(agents_v2, "_provider_registry", ProviderRegistry())
    monkeypatch.setattr(agents_v2, "_RECENT_TESTS", {})

    calls = []

    async def fake_run(provider, model, api_key, base_url, sdk):
        calls.append((provider, model, api_key, base_url, sdk))
        if api_key == "sk-bad":
            return {"ok": False, "status": 401, "code": "invalid_api_key",
                    "message": "Invalid API key"}
        return {"ok": True, "status": 200, "code": None,
                "message": f"Connected to {provider} using {model}"}

    monkeypatch.setattr(agents_v2, "run_connection_test", fake_run)
    return store, calls


async def _form_test(model, key=KEY, provider="opencode-go"):
    req = agents_v2.TestConnectionRequest(provider=provider, model=model, api_key=key)
    try:
        return await agents_v2.test_connection(req)
    except agents_v2.HTTPException as exc:  # a failed verdict is a 4xx here
        return exc


async def _save(model, key=KEY, provider="opencode-go"):
    return await settings.create_card(settings.CreateCardRequest(
        provider=provider, model=model, api_key=key))


async def test_save_after_test_records_that_verdict_without_a_second_call(env):
    store, calls = env
    await _form_test("glm-5.3-flash")
    assert len(calls) == 1

    res = await _save("glm-5.3-flash")

    assert len(calls) == 1, "Save spent a second completion on tested inputs"
    assert res["test"]["ok"] is True
    assert res["test"]["message"] == "Connected to opencode-go using glm-5.3-flash"
    assert store.health == [(res["card"]["id"], True, None)]


async def test_a_failed_test_verdict_is_recorded_on_the_card_too(env):
    store, calls = env
    await _form_test("glm-5.3-flash", key="sk-bad")
    res = await _save("glm-5.3-flash", key="sk-bad")

    assert len(calls) == 1
    assert res["test"]["ok"] is False
    (card_id, verified, error) = store.health[0]
    assert card_id == res["card"]["id"] and verified is False
    assert error["code"] == "invalid_api_key"


async def test_different_inputs_still_get_a_real_test(env):
    _, calls = env
    await _form_test("glm-5.3-flash")
    await _save("deepseek-v4-flash")  # tested one model, saved another
    assert [c[1] for c in calls] == ["glm-5.3-flash", "deepseek-v4-flash"]


async def test_a_client_that_never_tested_still_gets_a_real_test(env):
    _, calls = env
    res = await _save("glm-5.3-flash")
    assert len(calls) == 1 and res["test"]["ok"] is True


async def test_a_verdict_is_used_once(env):
    _, calls = env
    await _form_test("glm-5.3-flash")
    await _save("glm-5.3-flash")
    await _save("glm-5.3-flash")  # a second card, no second test run
    assert len(calls) == 2


async def test_a_stale_verdict_is_not_reused(env, monkeypatch):
    _, calls = env
    clock = [1000.0]
    monkeypatch.setattr(agents_v2.time, "monotonic", lambda: clock[0])
    await _form_test("glm-5.3-flash")
    clock[0] += agents_v2._RECENT_TEST_TTL_S + 1
    await _save("glm-5.3-flash")
    assert len(calls) == 2


async def test_editing_a_card_reuses_the_test_run_with_its_own_key(env):
    """Edit form: the test goes out with card_id (the card's stored key) and
    the model on screen; the save PUTs just the model."""
    store, calls = env
    card = store.add(card_id="card_go", provider="opencode-go",
                     model="deepseek-v4-flash", key=KEY)
    req = agents_v2.TestConnectionRequest(card_id=card.id, model="glm-5.3-flash")
    await agents_v2.test_connection(req)

    res = await settings.update_card(
        card.id, settings.UpdateCardRequest(model="glm-5.3-flash"))

    assert len(calls) == 1
    assert res["test"]["ok"] is True


async def test_the_card_test_action_always_runs_a_real_test(env):
    store, calls = env
    card = store.add(card_id="card_go", provider="opencode-go",
                     model="glm-5.3-flash", key=KEY)
    await _form_test("glm-5.3-flash")
    await settings.test_card(card.id)
    assert len(calls) == 2


def test_the_remembered_verdict_never_holds_the_raw_key(env):
    agents_v2._remember_test(
        agents_v2._test_fingerprint("opencode-go", "m", KEY, None, None),
        {"ok": True, "status": 200, "code": None, "message": "ok"})
    assert KEY not in repr(agents_v2._RECENT_TESTS)
