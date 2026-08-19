# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-model endpoint overrides reach the LLM client construction sites.

An aggregator can serve different models over different wire protocols under
one account and one API key. On OpenCode Go, ``minimax-m3`` is Anthropic
``/messages`` while ``deepseek-v4-pro`` is OpenAI ``/chat/completions``; on the
Zen tier that same ``minimax-m3`` is ``/chat/completions``. So the protocol
belongs to (provider, model) — a provider-level ``sdk`` alone cannot express
it, and picking the wrong one sends the request to a path that does not exist
with an auth header the endpoint does not read.

``ModelInfo.sdk`` / ``.base_url`` carry the override; these tests pin that it
survives all the way to every LLMProvider construction in the agent manager —
main model, fallbacks, and the utility model, which resolve independently.
"""

import json
from unittest.mock import MagicMock

import pytest

from agent_os.config.provider_registry import ProviderRegistry
from agent_os.daemon_v2 import agent_manager as am_mod
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.models import AgentConfig, FallbackModelEntry


ROUTER_JSON = {
    "providers": {
        "router": {
            "display_name": "Mixed-protocol router",
            "base_url": "https://router.example/v1",
            "sdk": "openai",
            "suggested_models": ["oai-model"],
            "models": {
                "oai-model": {"context_window": 100, "max_output": 10},
                "ant-model": {
                    "context_window": 200,
                    "max_output": 20,
                    "sdk": "anthropic",
                    # One segment short: the Anthropic SDK appends /v1/messages.
                    "base_url": "https://router.example",
                },
                "_default": {"context_window": 50, "max_output": 5},
            },
        }
    },
    "defaults": {"unknown_model": {"context_window": 10}},
}


class RecordingProvider:
    """Stands in for LLMProvider. Accepts **kwargs deliberately: a factory
    pinned to today's exact signature goes stale the moment a keyword is
    added, which is how three sibling fixtures rotted on `extra_headers`."""

    calls: list = []

    def __init__(self, model, api_key=None, base_url=None, **kw):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.sdk = kw.get("sdk")
        self.provider = kw.get("provider")
        RecordingProvider.calls.append(self)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(ROUTER_JSON), encoding="utf-8")
    RecordingProvider.calls = []
    monkeypatch.setattr(am_mod, "LLMProvider", RecordingProvider)
    return AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        provider_registry=ProviderRegistry(str(path)),
    )


def _config(**kw):
    base = dict(
        workspace="/tmp/ws",
        model="oai-model",
        api_key="sk-test",
        base_url="https://router.example/v1",
        sdk="openai",
        provider="router",
    )
    base.update(kw)
    return AgentConfig(**base)


def test_model_without_override_keeps_the_provider_endpoint(manager):
    provider, _, _, _ = manager._build_llm_providers(_config())
    assert provider.sdk == "openai"
    assert provider.base_url == "https://router.example/v1"


def test_model_with_override_switches_protocol_and_endpoint(manager):
    """The crux: same provider, same key, different wire protocol."""
    provider, _, _, _ = manager._build_llm_providers(_config(model="ant-model"))
    assert provider.sdk == "anthropic"
    assert provider.base_url == "https://router.example"


def test_utility_model_resolves_its_own_endpoint(manager):
    """The utility model is a different model on the same provider, so it can
    land on the other protocol — it must not inherit the main model's."""
    _, _, utility, _ = manager._build_llm_providers(
        _config(model="oai-model", utility_model="ant-model")
    )
    assert utility.model == "ant-model"
    assert utility.sdk == "anthropic"
    assert utility.base_url == "https://router.example"


def test_fallback_model_resolves_its_own_endpoint(manager):
    """A fallback chain that crosses protocols is the realistic case: fall
    back from a rate-limited Anthropic-path model to an OpenAI-path one."""
    _, fallbacks, _, _ = manager._build_llm_providers(
        _config(
            model="oai-model",
            llm_fallback_models=[
                FallbackModelEntry(model="ant-model", provider="router")
            ],
        )
    )
    assert len(fallbacks) == 1
    assert fallbacks[0].sdk == "anthropic"
    assert fallbacks[0].base_url == "https://router.example"


def test_explicit_user_base_url_is_not_clobbered_for_plain_models(manager):
    """A user-typed endpoint still wins for models that declare no override —
    otherwise the Custom/self-hosted escape hatch stops working."""
    provider, _, _, _ = manager._build_llm_providers(
        _config(base_url="http://localhost:1234/v1")
    )
    assert provider.base_url == "http://localhost:1234/v1"


# --- /providers/test must resolve the same override -------------------------
#
# The frontend sends the PROVIDER-level sdk (it has no per-model knowledge), so
# without this the Test Connection button fails on a model that works fine in
# a real turn — worse than having no button.


@pytest.fixture
def route_client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from agent_os.api.routes import agents_v2

    path = tmp_path / "providers.json"
    path.write_text(json.dumps(ROUTER_JSON), encoding="utf-8")
    monkeypatch.setattr(agents_v2, "_provider_registry", ProviderRegistry(str(path)))
    monkeypatch.setattr(agents_v2, "_credential_store", None)
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app, raise_server_exceptions=False)


def _capture_test_connection(monkeypatch, captured):
    from unittest.mock import AsyncMock
    import agent_os.agent.providers.openai_compat as oc

    def ctor(model, api_key, base_url, **kw):
        captured["base_url"] = base_url
        captured["sdk"] = kw.get("sdk")
        inst = MagicMock()
        inst.complete = AsyncMock(return_value=MagicMock())
        return inst

    monkeypatch.setattr(oc, "LLMProvider", ctor)


def test_test_connection_uses_the_model_override(route_client, monkeypatch):
    captured: dict = {}
    _capture_test_connection(monkeypatch, captured)
    resp = route_client.post(
        "/api/v2/providers/test",
        json={
            "provider": "router",
            "model": "ant-model",
            "api_key": "sk-x",
            "base_url": "https://router.example/v1",
            "sdk": "openai",  # what the frontend knows: the provider default
        },
    )
    assert resp.status_code == 200, resp.text
    assert captured["sdk"] == "anthropic"
    assert captured["base_url"] == "https://router.example"


def test_test_connection_respects_an_explicit_sdk_without_an_override(
    route_client, monkeypatch
):
    """Custom / self-hosted has no model entry, so the user's SDK pick wins."""
    captured: dict = {}
    _capture_test_connection(monkeypatch, captured)
    resp = route_client.post(
        "/api/v2/providers/test",
        json={
            "model": "local-model",
            "base_url": "http://localhost:1234/v1",
            "sdk": "anthropic",
        },
    )
    assert resp.status_code == 200, resp.text
    assert captured["sdk"] == "anthropic"
    assert captured["base_url"] == "http://localhost:1234/v1"


# --- The shipped OpenCode entries -------------------------------------------
#
# Zen and Go are one account and one API key across two roots. The catalogs
# overlap by 16 models — and three of those are served over DIFFERENT wire
# protocols depending on tier. These pin the mapping, because it looks like an
# inconsistency someone would "fix".


@pytest.fixture(scope="module")
def real():
    return ProviderRegistry()


ZEN_ROOT = "https://opencode.ai/zen"
GO_ROOT = "https://opencode.ai/zen/go"


def test_minimax_protocol_differs_between_the_two_tiers(real):
    """Same model id, same vendor, same key — different protocol per tier.
    This is the whole reason per-model overrides exist."""
    for mid in ("minimax-m3", "minimax-m2.7", "minimax-m2.5"):
        go = real.get_model_info("opencode-go", mid)
        assert go.sdk == "anthropic", f"{mid} is /messages on Go"
        assert go.base_url == GO_ROOT

        zen = real.get_model_info("opencode-zen", mid)
        assert zen.sdk is None, f"{mid} is /chat/completions on Zen"
        assert zen.base_url is None


def test_claude_models_on_zen_use_the_anthropic_protocol(real):
    for mid in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"):
        info = real.get_model_info("opencode-zen", mid)
        assert info.sdk == "anthropic"
        assert info.base_url == ZEN_ROOT


def test_qwen_uses_the_anthropic_protocol_on_both_tiers(real):
    assert real.get_model_info("opencode-zen", "qwen3.7-max").sdk == "anthropic"
    assert real.get_model_info("opencode-go", "qwen3.8-max").sdk == "anthropic"


def test_open_models_stay_on_the_provider_default(real):
    for provider, mid in (
        ("opencode-zen", "deepseek-v4-pro"), ("opencode-go", "deepseek-v4-pro"),
        ("opencode-zen", "kimi-k3"), ("opencode-go", "glm-5.3"),
    ):
        info = real.get_model_info(provider, mid)
        assert info.sdk is None, f"{provider}/{mid} is /chat/completions"


def test_the_two_sdks_disagree_about_where_v1_lives(real):
    """The OpenAI client appends /chat/completions to base_url; the Anthropic
    client appends /v1/messages. So the provider base_url carries /v1 and the
    per-model Anthropic override must stop one segment short — otherwise the
    request goes to /zen/v1/v1/messages."""
    for key, root in (("opencode-zen", ZEN_ROOT), ("opencode-go", GO_ROOT)):
        provider_base = real.get_provider_data(key)["base_url"]
        assert provider_base == root + "/v1"
        overrides = {
            m: e["base_url"]
            for m, e in real.get_provider_data(key)["models"].items()
            if e.get("sdk") == "anthropic"
        }
        assert overrides, f"{key} must have Anthropic-protocol models"
        for mid, base in overrides.items():
            assert base == root, mid
            assert not base.endswith("/v1"), mid


def test_free_models_are_priced_at_zero(real):
    """Zen's free tier runs with no payment method on file (verified live), so
    a spend reading of $0 is correct, not a missing-price fallback."""
    from agent_os.agent.pricing import resolve_rates

    for mid in ("big-pickle", "deepseek-v4-flash-free", "hy3-free"):
        rates = resolve_rates("opencode-zen", mid)
        assert rates.input_per_1m == 0.0, mid
        assert rates.output_per_1m == 0.0, mid


def test_paid_model_on_zen_is_not_priced_as_free(real):
    """Zen's price table lists the free 'DeepSeek V4 Flash' row above the paid
    'DeepSeek V4 Flash (Off-Peak)' one; collapsing the qualifier collides them
    and would price the paid model at zero."""
    from agent_os.agent.pricing import resolve_rates

    rates = resolve_rates("opencode-zen", "deepseek-v4-flash")
    assert rates.input_per_1m == 0.22
    assert rates.output_per_1m == 0.66


# --- Billing errors must not read as auth errors -----------------------------


def test_credits_error_is_not_reported_as_an_invalid_key(route_client, monkeypatch):
    """OpenCode answers a paid model on a workspace with no payment method
    with HTTP 401 carrying a CreditsError body. Verified live: the key is
    valid — only the balance is missing. Mapping every 401 to "Invalid API
    key" sends the user off to re-issue a key that was never the problem."""
    import agent_os.agent.providers.openai_compat as oc
    from agent_os.agent.providers.types import LLMError
    from unittest.mock import AsyncMock

    def ctor(model, api_key, base_url, **kw):
        inst = MagicMock()
        inst.complete = AsyncMock(
            side_effect=LLMError(
                "No payment method. Add a payment method here: "
                "https://opencode.ai/workspace/wrk_x/billing",
                status_code=401,
            )
        )
        return inst

    monkeypatch.setattr(oc, "LLMProvider", ctor)
    resp = route_client.post(
        "/api/v2/providers/test",
        json={"provider": "router", "model": "oai-model", "api_key": "sk-valid"},
    )
    assert resp.status_code == 401
    detail = resp.json()["detail"]
    assert "Invalid API key" not in detail, detail
    assert "payment method" in detail.lower(), detail


def test_a_real_auth_failure_still_reads_as_an_invalid_key(route_client, monkeypatch):
    import agent_os.agent.providers.openai_compat as oc
    from agent_os.agent.providers.types import LLMError
    from unittest.mock import AsyncMock

    def ctor(model, api_key, base_url, **kw):
        inst = MagicMock()
        inst.complete = AsyncMock(
            side_effect=LLMError("Incorrect API key provided", status_code=401)
        )
        return inst

    monkeypatch.setattr(oc, "LLMProvider", ctor)
    resp = route_client.post(
        "/api/v2/providers/test",
        json={"provider": "router", "model": "oai-model", "api_key": "sk-bad"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid API key"


def test_every_opencode_model_carries_explicit_specs(real):
    """These catalogs span vendors, so flagship inheritance is actively wrong
    here: an unlisted free model would otherwise inherit Claude Sonnet's 1M
    window and vision flag. Anything we list, we spec — conservatively where
    the vendor publishes nothing."""
    for key in ("opencode-zen", "opencode-go"):
        for mid, entry in real.get_provider_data(key)["models"].items():
            assert "context_window" in entry, f"{key}/{mid}"
            assert "max_output" in entry, f"{key}/{mid}"
            assert "capabilities" in entry, f"{key}/{mid}"


def test_every_opencode_model_is_priced(real):
    """A missing price silently falls back to the provider _default, which
    would misreport spend on a per-token gateway."""
    from agent_os.agent.pricing import _load_pricing

    for key in ("opencode-zen", "opencode-go"):
        priced = set(_load_pricing().get(key, {}))
        listed = set(real.get_provider_data(key)["models"])
        assert not (listed - priced), f"{key}: unpriced {sorted(listed - priced)}"
