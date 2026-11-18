# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A Custom card can speak the OpenAI Responses API (``sdk="openai-responses"``).

v0.24.0 added the Responses wire protocol for catalogued OpenAI / OpenCode
models only; a user-defined endpoint could pick Chat Completions or Anthropic
and nothing else, so a gateway that only serves ``/v1/responses`` was
unreachable. These tests pin the value through the whole path: the card
save and its compulsory connection test, the form's Test Connection (also
while editing a stored Custom card), the stored card, and the provider built
for a real run. Every call stays on Responses; none is coerced to chat.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.providers.responses_adapter import RESPONSES_SDK
from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.api.routes import agents_v2, settings
from agent_os.budget.normalize import ProviderSemantics
from agent_os.config.provider_registry import ProviderRegistry
from agent_os.daemon_v2.agent_manager import AgentManager
from tests.card_doubles import FakeCardStore

KEY = "sk-own-gateway"
BASE = "https://gateway.example/v1"
MODEL = "my-responses-model"


@pytest.fixture
def env(monkeypatch):
    from agent_os import telemetry

    monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)
    monkeypatch.setattr(telemetry, "emit", lambda *a, **k: None)
    monkeypatch.setattr(telemetry, "latch", lambda *a, **k: None)
    store = FakeCardStore()
    monkeypatch.setattr(agents_v2, "_settings_store", store)
    monkeypatch.setattr(settings, "_settings_store", store)
    monkeypatch.setattr(agents_v2, "_credential_store", None)
    monkeypatch.setattr(agents_v2, "_provider_registry", ProviderRegistry())
    monkeypatch.setattr(agents_v2, "_RECENT_TESTS", {})
    monkeypatch.setattr(settings, "_refresh_agents_in_background", lambda *a: None)

    calls = []

    async def fake_run(provider, model, api_key, base_url, sdk):
        calls.append({"provider": provider, "model": model, "api_key": api_key,
                      "base_url": base_url, "sdk": sdk})
        return {"ok": True, "status": 200, "code": None,
                "message": f"Connected to {provider} using {model}"}

    monkeypatch.setattr(agents_v2, "run_connection_test", fake_run)
    return store, calls


# -- saving a card --------------------------------------------------------------


async def test_add_custom_card_tests_and_stores_responses(env):
    store, calls = env
    res = await settings.create_card(settings.CreateCardRequest(
        provider="custom", model=MODEL, base_url=BASE, sdk=RESPONSES_SDK,
        api_key=KEY))
    assert calls == [{"provider": "custom", "model": MODEL, "api_key": KEY,
                      "base_url": BASE, "sdk": RESPONSES_SDK}]
    assert store.cards[0].sdk == RESPONSES_SDK
    assert res["card"]["sdk"] == RESPONSES_SDK


async def test_switching_a_stored_custom_card_to_responses_retests_on_it(env):
    store, calls = env
    store.add(card_id="card_c", provider="custom", model=MODEL, key=KEY,
              base_url=BASE, sdk="openai")
    await settings.update_card("card_c", settings.UpdateCardRequest(
        provider="custom", model=MODEL, base_url=BASE, sdk=RESPONSES_SDK))
    assert [c["sdk"] for c in calls] == [RESPONSES_SDK]
    assert store.resolve_card("card_c").sdk == RESPONSES_SDK


# -- the form's Test Connection ---------------------------------------------------


async def test_test_connection_sends_responses_for_a_new_custom_endpoint(env):
    _, calls = env
    await agents_v2.test_connection(agents_v2.TestConnectionRequest(
        model=MODEL, api_key=KEY, base_url=BASE, sdk=RESPONSES_SDK))
    assert calls[0]["sdk"] == RESPONSES_SDK
    assert calls[0]["base_url"] == BASE


async def test_test_connection_on_an_edited_custom_card_uses_the_protocol_on_screen(env):
    """Editing a stored Custom card without retyping its key sends card_id
    (for the key) plus the fields on screen. The endpoint the user just
    picked is what Test must call — it used to test the STORED protocol, so
    a switch to Responses reported the old endpoint's verdict."""
    store, calls = env
    store.add(card_id="card_c", provider="custom", model=MODEL, key=KEY,
              base_url=BASE, sdk="openai")
    await agents_v2.test_connection(agents_v2.TestConnectionRequest(
        model=MODEL, base_url=BASE, sdk=RESPONSES_SDK, card_id="card_c"))
    assert calls[0]["sdk"] == RESPONSES_SDK
    assert calls[0]["api_key"] == KEY


async def test_test_connection_on_a_custom_card_without_endpoint_fields_keeps_the_card(env):
    store, calls = env
    store.add(card_id="card_c", provider="custom", model=MODEL, key=KEY,
              base_url=BASE, sdk=RESPONSES_SDK)
    await agents_v2.test_connection(agents_v2.TestConnectionRequest(card_id="card_c"))
    assert calls[0]["sdk"] == RESPONSES_SDK
    assert calls[0]["base_url"] == BASE


async def test_registry_card_test_still_ignores_the_form_protocol(env):
    """Unchanged for registry cards: their endpoint is the registry's."""
    store, calls = env
    store.add(card_id="card_ds", provider="deepseek", model="deepseek-chat",
              key=KEY)
    await agents_v2.test_connection(agents_v2.TestConnectionRequest(
        model="deepseek-chat", sdk=RESPONSES_SDK, card_id="card_ds"))
    assert calls[0]["sdk"] == "openai"
    assert calls[0]["base_url"] == "https://api.deepseek.com"


async def test_run_connection_test_goes_out_over_responses(monkeypatch):
    """The real test function, one level down: a Custom endpoint on
    Responses calls the Responses path, never chat/completions."""
    responses = AsyncMock(return_value=LLMResponse(
        raw_message={}, text="hi", tool_calls=[], has_tool_calls=False,
        finish_reason="stop", status_text=None, usage=TokenUsage(1, 1)))
    chat = AsyncMock(side_effect=AssertionError("chat/completions was called"))
    monkeypatch.setattr(LLMProvider, "_complete_responses", responses)
    monkeypatch.setattr(LLMProvider, "_complete_openai", chat)
    monkeypatch.setattr(agents_v2, "_provider_registry", ProviderRegistry())
    result = await agents_v2.run_connection_test(
        "custom", MODEL, KEY, BASE, RESPONSES_SDK)
    assert result["ok"] is True
    responses.assert_awaited_once()


# -- the provider built for a real run --------------------------------------------


def _manager(store) -> AgentManager:
    project = {"project_id": "p1", "name": "P", "workspace": "/tmp/p1",
               "is_scratch": False, "card_id": "card_r"}
    project_store = MagicMock()
    project_store.get_project = MagicMock(side_effect=lambda pid: project if pid == "p1" else None)
    project_store.list_projects = MagicMock(return_value=[project])
    credential_store = MagicMock()
    credential_store.get_api_key = MagicMock(return_value=None)
    return AgentManager(
        project_store=project_store, ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), settings_store=store,
        credential_store=credential_store,
    )


def test_a_run_on_a_responses_custom_card_builds_a_responses_provider():
    store = FakeCardStore.with_default(
        card_id="card_r", provider="custom", model=MODEL, key=KEY,
        base_url=BASE, sdk=RESPONSES_SDK)
    mgr = _manager(store)
    cfg = mgr._build_agent_config_from_project("p1")
    assert cfg.sdk == RESPONSES_SDK
    assert cfg.base_url == BASE
    provider, _fallbacks, utility, _info = mgr._build_llm_providers(cfg)
    assert provider.sdk == RESPONSES_SDK
    assert provider.base_url == BASE
    assert utility.sdk == RESPONSES_SDK
    assert ProviderSemantics.from_sdk(provider.sdk) is ProviderSemantics.from_sdk("openai")


def test_a_responses_custom_card_as_fallback_keeps_its_protocol():
    store = FakeCardStore.with_default(
        card_id="card_r", provider="custom", model=MODEL, key=KEY,
        base_url=BASE, sdk="openai")
    store.add(card_id="card_fb", provider="custom", model="fb-model",
              key="sk-fb", base_url="https://fb.example/v1", sdk=RESPONSES_SDK)
    mgr = _manager(store)
    mgr._project_store.get_project("p1")["llm_fallback_models"] = [{"card_id": "card_fb"}]
    cfg = mgr._build_agent_config_from_project("p1")
    _p, fallbacks, _u, _i = mgr._build_llm_providers(cfg)
    assert [fb.sdk for fb in fallbacks] == [RESPONSES_SDK]
