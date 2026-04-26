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
    def test_provider_default(self, custom_registry):
        info = custom_registry.get_model_info("test_provider", "unknown-model-xyz")
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
