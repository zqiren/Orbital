# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for ProviderRegistry — model capability and metadata lookups."""

import json
import os
import tempfile

import pytest

from agent_os.config.provider_registry import (
    ProviderRegistry,
    ModelInfo,
    ModelCapabilities,
)


@pytest.fixture
def registry():
    """Registry backed by the real providers.json."""
    return ProviderRegistry()


@pytest.fixture
def custom_registry(tmp_path):
    """Registry backed by a minimal test providers.json."""
    data = {
        "providers": {
            "test_provider": {
                "display_name": "Test",
                "base_url": "https://api.test.com/v1",
                "sdk": "openai",
                "suggested_models": ["model-a", "model-b"],
                "models": {
                    "model-a": {
                        "display_name": "Model A",
                        "tier": "flagship",
                        "context_window": 200000,
                        "max_output": 16384,
                        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
                    },
                    "model-b": {
                        "display_name": "Model B",
                        "tier": "fast",
                        "context_window": 128000,
                        "max_output": 8192,
                        "capabilities": {"vision": False, "tool_use": True, "streaming": True},
                    },
                    "_default": {
                        "context_window": 64000,
                        "max_output": 4096,
                        "capabilities": {"vision": False, "tool_use": False, "streaming": True},
                    },
                },
            },
            "no_models_provider": {
                "display_name": "No Models",
                "sdk": "openai",
            },
            "no_suggested_provider": {
                "display_name": "No Suggested",
                "sdk": "openai",
                "models": {
                    "_default": {
                        "context_window": 64000,
                        "max_output": 4096,
                        "capabilities": {"vision": False, "tool_use": False, "streaming": True},
                    },
                },
            },
        },
        "defaults": {
            "unknown_model": {
                "context_window": 128000,
                "max_output": 8192,
                "capabilities": {"vision": False, "tool_use": True, "streaming": True},
            }
        },
    }
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(data))
    return ProviderRegistry(config_path=str(path))


# --- Exact match tests ---

class TestExactMatch:
    def test_exact_match_returns_correct_info(self, custom_registry):
        info = custom_registry.get_model_info("test_provider", "model-a")
        assert info.context_window == 200000
        assert info.max_output == 16384
        assert info.capabilities.vision is True
        assert info.capabilities.tool_use is True
        assert info.tier == "flagship"
        assert info.display_name == "Model A"

    def test_exact_match_no_vision(self, custom_registry):
        info = custom_registry.get_model_info("test_provider", "model-b")
        assert info.capabilities.vision is False
        assert info.max_output == 8192

    def test_convenience_methods(self, custom_registry):
        assert custom_registry.get_max_output("test_provider", "model-a") == 16384
        assert custom_registry.get_context_window("test_provider", "model-a") == 200000
        caps = custom_registry.get_capabilities("test_provider", "model-a")
        assert caps.vision is True


# --- Prefix match tests ---

class TestPrefixMatch:
    def test_prefix_match(self, custom_registry):
        info = custom_registry.get_model_info("test_provider", "model-a-20260301")
        assert info.context_window == 200000
        assert info.max_output == 16384

    def test_prefix_match_picks_longest(self, custom_registry):
        """model-b is longer prefix than model-a for 'model-b-turbo'."""
        info = custom_registry.get_model_info("test_provider", "model-b-turbo")
        assert info.max_output == 8192  # model-b, not model-a


# --- Fallback tests ---

class TestFallback:
    def test_unknown_model_inherits_latest_flagship_spec(self, custom_registry):
        """Stale-catalog fallback: an unknown model on a provider inherits the
        spec of suggested_models[0] (the provider's current flagship) rather
        than the conservative _default — a late catalog update almost always
        concerns a NEWER advanced model, so the newest entry is the best guess.
        """
        info = custom_registry.get_model_info("test_provider", "model-next-gen")
        assert info.context_window == 200000  # model-a's spec, not _default's
        assert info.max_output == 16384
        assert info.capabilities.vision is True

    def test_provider_default_used_when_no_suggested_models(self, custom_registry):
        info = custom_registry.get_model_info("no_suggested_provider", "unknown-model-xyz")
        assert info.context_window == 64000
        assert info.max_output == 4096
        assert info.capabilities.tool_use is False

    def test_unknown_provider_uses_global_default(self, custom_registry):
        info = custom_registry.get_model_info("nonexistent_provider", "any-model")
        assert info.context_window == 128000
        assert info.max_output == 8192
        assert info.capabilities.vision is False
        assert info.capabilities.tool_use is True

    def test_provider_without_models_uses_global_default(self, custom_registry):
        info = custom_registry.get_model_info("no_models_provider", "some-model")
        assert info.context_window == 128000
        assert info.max_output == 8192


# --- Real providers.json tests ---

class TestRealProviders:
    def test_anthropic_claude_opus(self, registry):
        info = registry.get_model_info("anthropic", "claude-opus-4-6")
        assert info.context_window == 1000000
        assert info.max_output == 128000
        assert info.capabilities.vision is True
        assert info.capabilities.tool_use is True

    def test_anthropic_haiku(self, registry):
        info = registry.get_model_info("anthropic", "claude-haiku-4-5")
        assert info.max_output == 8192
        assert info.capabilities.vision is True

    def test_anthropic_prefix_match_dated(self, registry):
        """claude-sonnet-4-5-20250929 matches claude-sonnet-4-5."""
        info = registry.get_model_info("anthropic", "claude-sonnet-4-5-20250929")
        assert info.context_window == 200000
        assert info.capabilities.vision is True

    def test_openai_gpt5(self, registry):
        info = registry.get_model_info("openai", "gpt-5.2")
        assert info.context_window == 400000
        assert info.max_output == 128000

    def test_deepseek_no_vision(self, registry):
        info = registry.get_model_info("deepseek", "deepseek-chat")
        assert info.capabilities.vision is False
        assert info.capabilities.tool_use is True

    def test_deepseek_reasoner_no_tool_use(self, registry):
        info = registry.get_model_info("deepseek", "deepseek-reasoner")
        assert info.capabilities.tool_use is False

    def test_moonshot_kimi_vision(self, registry):
        info = registry.get_model_info("moonshot", "kimi-k2.5")
        assert info.capabilities.vision is True
        assert info.context_window == 262144

    def test_google_gemini_large_context(self, registry):
        info = registry.get_model_info("google", "gemini-3-pro-preview")
        assert info.context_window == 1000000
        assert info.max_output == 65536

    def test_xai_grok_large_context(self, registry):
        info = registry.get_model_info("xai", "grok-4-1-fast-reasoning")
        assert info.context_window == 2000000

    def test_zhipu_glm5(self, registry):
        info = registry.get_model_info("zhipu", "glm-5")
        # GLM-5 is text-only per docs.z.ai; vision is served by separate glm-5v-turbo / glm-4.6v.
        assert info.capabilities.vision is False
        assert info.context_window == 200000

    def test_qwen35_max(self, registry):
        info = registry.get_model_info("qwen", "qwen3.5-max")
        assert info.capabilities.vision is True
        assert info.context_window == 262144

    def test_custom_uses_default(self, registry):
        info = registry.get_model_info("custom", "my-local-model")
        assert info.context_window == 128000
        assert info.capabilities.vision is False


# --- suggested_models tests ---

class TestSuggestedModels:
    def test_suggested_models_from_field(self, custom_registry):
        models = custom_registry.suggested_models("test_provider")
        assert models == ["model-a", "model-b"]

    def test_suggested_models_fallback_to_keys(self, custom_registry):
        """When no suggested_models field, return model keys minus _default."""
        models = custom_registry.suggested_models("no_models_provider")
        assert models == []


# --- all_providers tests ---

class TestAllProviders:
    def test_all_providers_returns_dict(self, registry):
        providers = registry.all_providers()
        assert isinstance(providers, dict)
        assert "anthropic" in providers
        assert "openai" in providers
        assert "deepseek" in providers

    def test_console_url_present_for_all_non_custom_providers(self, registry):
        """Spec 17: every provider a user can create a key for carries a console_url
        for the wizard's "Get your API key" link. Passed through raw (no dataclass
        strips it), so this also guards against a future strict response model
        dropping the field."""
        providers = registry.all_providers()
        non_custom = [k for k in providers if k not in ("custom", "unknown_model")]
        assert non_custom, "expected at least one non-custom provider"
        for key in non_custom:
            assert providers[key].get("console_url"), f"{key} missing console_url"

    def test_custom_provider_has_no_console_url(self, registry):
        providers = registry.all_providers()
        assert "console_url" not in providers["custom"]

    def test_no_china_endpoint_flag_only_on_openai_anthropic_google(self, registry):
        """Spec 17: the "no mainland-China endpoint" caption is driven by this
        flag, and only these three providers should carry it."""
        providers = registry.all_providers()
        flagged = {k for k, v in providers.items() if v.get("no_china_endpoint")}
        assert flagged == {"openai", "anthropic", "google"}

    def test_get_provider_data_includes_new_fields(self, registry):
        """get_provider_data is a raw passthrough — confirm it isn't stripped
        by any dataclass parsing on the provider-level (only model-level
        entries go through ModelInfo)."""
        openai_data = registry.get_provider_data("openai")
        assert openai_data["console_url"] == "https://platform.openai.com/api-keys"
        assert openai_data["no_china_endpoint"] is True

        deepseek_data = registry.get_provider_data("deepseek")
        assert deepseek_data["console_url"] == "https://platform.deepseek.com/api_keys"
        assert "no_china_endpoint" not in deepseek_data


# --- Edge cases ---

class TestEdgeCases:
    def test_missing_file_returns_defaults(self, tmp_path):
        reg = ProviderRegistry(config_path=str(tmp_path / "nonexistent.json"))
        info = reg.get_model_info("any", "any")
        assert info.context_window == 128000
        assert info.max_output == 8192

    def test_frozen_dataclasses(self, custom_registry):
        info = custom_registry.get_model_info("test_provider", "model-a")
        with pytest.raises(AttributeError):
            info.context_window = 999
        with pytest.raises(AttributeError):
            info.capabilities.vision = False


# --- July 2026 model additions + inherit-from-latest on the real catalog ---

class TestJuly2026Models:
    def test_gpt_56_family(self, registry):
        # Official specs: developers.openai.com/api/docs/models/gpt-5.6-{sol,terra,luna}
        for mid in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6"):
            info = registry.get_model_info("openai", mid)
            assert info.context_window == 1050000, mid
            assert info.max_output == 128000, mid
            assert info.capabilities.vision is True, mid

    def test_kimi_k3(self, registry):
        # Official specs: platform.kimi.ai/docs/guide/kimi-k3-quickstart
        info = registry.get_model_info("moonshot", "kimi-k3")
        assert info.context_window == 1048576
        assert info.max_output == 131072
        assert info.capabilities.vision is True
        # Reasoning is always-on with no toggle (like MiniMax-M3): locked-on
        # models make disable_reasoning a no-op, so utility calls budget long
        # single timeouts instead of retry ladders.
        assert info.reasoning.supported is True
        assert info.reasoning.enable == "model_only"
        assert info.reasoning.field == "reasoning_content"

    def test_claude_fable_5(self, registry):
        info = registry.get_model_info("anthropic", "claude-fable-5")
        assert info.context_window == 1000000
        assert info.max_output == 128000
        assert info.capabilities.vision is True
        # Thinking always on — cannot be disabled (explicit disabled → 400).
        assert info.reasoning.supported is True
        assert info.reasoning.enable == "model_only"

    def test_claude_opus_4_8_and_sonnet_5(self, registry):
        for mid in ("claude-opus-4-8", "claude-sonnet-5"):
            info = registry.get_model_info("anthropic", mid)
            assert info.context_window == 1000000, mid
            assert info.max_output == 128000, mid

    def test_unknown_future_model_inherits_flagship_not_generic_default(self, registry):
        # A model newer than the bundled catalog (no exact/prefix match) must
        # inherit the provider's current flagship spec, not the generic
        # 128000/8192 defaults.
        info = registry.get_model_info("openai", "gpt-7-hypothetical")
        assert info.context_window == 1050000
        assert info.max_output == 128000

    def test_custom_provider_unknown_model_keeps_conservative_defaults(self, registry):
        # `custom` (Ollama/vLLM/self-hosted) has no suggested models — a big
        # flagship guess would be wrong there; conservative defaults stay.
        info = registry.get_model_info("custom", "my-local-llama")
        assert info.context_window == 128000
        assert info.max_output == 8192
