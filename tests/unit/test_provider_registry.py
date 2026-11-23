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
        # Real cap is 64K; held at 32K because the Anthropic path sends this
        # as max_tokens and prompt + max_tokens must fit the 200K window.
        assert info.max_output == 32000
        assert info.capabilities.vision is True

    def test_anthropic_prefix_match_dated(self, registry):
        """claude-sonnet-4-5-20250929 matches claude-sonnet-4-5."""
        info = registry.get_model_info("anthropic", "claude-sonnet-4-5-20250929")
        assert info.context_window == 200000
        assert info.capabilities.vision is True

    def test_openai_gpt5(self, registry):
        info = registry.get_model_info("openai", "gpt-5.2")
        # 400K total, 272K prompt cap; the window is the prompt cap because
        # compaction triggers at 80% of it.
        assert info.context_window == 272000
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
        # Sign-in-only providers (spec 1123, ``key_entry: false``) have no key
        # to get, so no "Get your API key" link.
        non_custom = [k for k in providers if k not in ("custom", "unknown_model")
                      and providers[k].get("key_entry", True)]
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
        # chatgpt (spec 1123) is OpenAI's own endpoint, so it has none either.
        assert flagged == {"openai", "anthropic", "google", "chatgpt"}

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
            # 1.05M total, 922K prompt cap (declared as the window).
            assert info.context_window == 922000, mid
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
        assert info.context_window == 922000  # gpt-6.2-sol's prompt cap
        assert info.max_output == 128000

    def test_custom_provider_unknown_model_keeps_conservative_defaults(self, registry):
        # `custom` (Ollama/vLLM/self-hosted) has no suggested models — a big
        # flagship guess would be wrong there; conservative defaults stay.
        info = registry.get_model_info("custom", "my-local-llama")
        assert info.context_window == 128000
        assert info.max_output == 8192


# ---------------------------------------------------------------------------
# Per-model endpoint overrides (OpenCode Zen / Go)
# ---------------------------------------------------------------------------
#
# `sdk` and `base_url` used to live only at provider level, which assumes one
# provider speaks one protocol at one address. Aggregators break that: on
# OpenCode Go, `minimax-m3` is served over the Anthropic /messages protocol
# while `deepseek-v4-pro` on the SAME account is OpenAI /chat/completions —
# and on the Zen tier that same `minimax-m3` is /chat/completions. The wire
# protocol is a property of (provider, model), not of either alone.


class TestPerModelEndpointOverride:
    @pytest.fixture
    def mixed_registry(self, tmp_path):
        data = {
            "providers": {
                "mixed": {
                    "display_name": "Mixed-protocol router",
                    "base_url": "https://router.example/v1",
                    "sdk": "openai",
                    "suggested_models": ["plain-model"],
                    "models": {
                        "plain-model": {"context_window": 100},
                        "anthropic-model": {
                            "context_window": 200,
                            # The Anthropic SDK appends /v1/messages itself, so
                            # its base_url stops one segment short.
                            "sdk": "anthropic",
                            "base_url": "https://router.example",
                        },
                        "_default": {"context_window": 50},
                    },
                }
            },
            "defaults": {"unknown_model": {"context_window": 10}},
        }
        path = tmp_path / "providers.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return ProviderRegistry(str(path))

    def test_model_without_override_inherits_provider(self, mixed_registry):
        info = mixed_registry.get_model_info("mixed", "plain-model")
        assert info.sdk is None
        assert info.base_url is None

    def test_model_with_override_reports_its_own_endpoint(self, mixed_registry):
        info = mixed_registry.get_model_info("mixed", "anthropic-model")
        assert info.sdk == "anthropic"
        assert info.base_url == "https://router.example"

    def test_override_survives_the_fallback_chain(self, mixed_registry):
        """An unknown model inherits suggested_models[0] — which has no
        override — so it must NOT pick up a sibling's protocol."""
        info = mixed_registry.get_model_info("mixed", "brand-new-model")
        assert info.sdk is None
        assert info.base_url is None

    def test_real_registry_models_default_to_no_override(self, registry):
        """Every provider that speaks one protocol keeps reporting None, so
        the resolution sites fall back to the provider-level config."""
        info = registry.get_model_info("deepseek", "deepseek-v4-pro")
        assert info.sdk is None
        assert info.base_url is None

    def test_default_entry_override_does_not_leak_to_unknown_models(self, tmp_path):
        """An override says "THIS model lives elsewhere". A model reached by
        the fallback chain is by definition one the registry does not know, so
        inheriting a sibling's endpoint is a guess that can only be wrong —
        a newly released model on a mixed-protocol aggregator would be sent to
        whichever protocol the flagship happens to use. Specs are inherited;
        the endpoint is not."""
        data = {
            "providers": {
                "mixed": {
                    "base_url": "https://router.example/v1",
                    "sdk": "openai",
                    # The flagship itself carries an override.
                    "suggested_models": ["ant-flagship"],
                    "models": {
                        "ant-flagship": {
                            "context_window": 400,
                            "sdk": "anthropic",
                            "base_url": "https://router.example",
                        },
                    },
                }
            },
            "defaults": {"unknown_model": {"context_window": 10}},
        }
        path = tmp_path / "providers.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        reg = ProviderRegistry(str(path))

        known = reg.get_model_info("mixed", "ant-flagship")
        assert known.sdk == "anthropic"

        unknown = reg.get_model_info("mixed", "brand-new-model")
        assert unknown.context_window == 400, "specs still inherit"
        assert unknown.sdk is None, "but never the endpoint"
        assert unknown.base_url is None


# --- Tencent Hunyuan (HY) via TokenHub ---

class TestHunyuan:
    """Tencent's HY models ride the OpenAI-compatible TokenHub gateway, which
    has one host per region with region-locked keys — the same two-region
    shape as moonshot/zhipu/qwen/minimax, so the settings UI's region toggle
    and the sandbox allowlist parity test cover it without new code."""

    def test_registered_with_two_regions_and_a_console(self, registry):
        data = registry.get_provider_data("hunyuan")
        assert data["sdk"] == "openai"
        assert data["base_url"] == "https://tokenhub-intl.tencentcloudmaas.com/v1"
        assert data["china_base_url"] == "https://tokenhub.tencentmaas.com/v1"
        assert data["console_url"].startswith("https://console.cloud.tencent.com/")
        assert data["currency"] == "CNY"
        assert "no_china_endpoint" not in data

    def test_flagship_first_in_suggestions(self, registry):
        assert registry.suggested_models("hunyuan") == ["hy4-preview", "hy3"]

    def test_model_specs(self, registry):
        hy4 = registry.get_model_info("hunyuan", "hy4-preview")
        assert hy4.context_window == 1_000_000
        assert hy4.max_output == 64_000
        assert hy4.reasoning.supported is True
        assert hy4.reasoning.field == "reasoning_content"
        hy3 = registry.get_model_info("hunyuan", "hy3")
        assert hy3.context_window == 192_000  # 256K total, 192K prompt cap
        assert hy3.max_output == 128_000

    def test_unknown_hy_model_inherits_the_flagship_spec(self, registry):
        # A just-released id (hy4, hy5-preview…) should get hy4-preview's
        # spec via the suggested_models[0] inheritance rule, never the
        # conservative unknown-model defaults.
        info = registry.get_model_info("hunyuan", "hy5-preview")
        assert info.context_window == 1_000_000


# --- Catalog windows refreshed against models.dev + vendor docs (2026-11-18) ---

class TestCatalogWindows1118:
    """The OpenCode GLM 5.2/5.3 entries were cloned from GLM-5.1 and kept its
    200K window, so a 1M model compacted at 160K. Direct Zhipu had no 5.2/5.3
    entries at all and prefix-matched ``glm-5`` (200K). Every id below must
    resolve to its real 1M window, by exact match, never via the prefix."""

    @pytest.mark.parametrize("provider,model", [
        ("opencode-go", "glm-5.2"),
        ("opencode-go", "glm-5.3"),
        ("opencode-go", "glm-5.3-flash"),
        ("opencode-zen", "glm-5.2"),
        ("opencode-zen", "glm-5.3"),
        ("opencode-zen", "glm-5.3-flash"),
        ("zhipu", "glm-5.2"),
        ("zhipu", "glm-5.3"),
        ("zhipu", "glm-5.3-flash"),
    ])
    def test_glm_5_2_and_5_3_have_the_1m_window(self, registry, provider, model):
        assert model in registry.get_provider_data(provider)["models"]
        assert registry.get_context_window(provider, model) >= 1_000_000
        assert registry.get_max_output(provider, model) == 131_072

    def test_glm_5_1_keeps_its_200k_window(self, registry):
        # GLM-5.1 really is 200K (docs.z.ai/guides/llm/glm-5.1); only the
        # newer ids were wrong.
        for provider in ("opencode-go", "opencode-zen", "zhipu"):
            assert registry.get_context_window(provider, "glm-5.1") == 200_000

    def test_zhipu_glm_5_3_flash_is_priced_as_flash(self):
        # Without its own pricing key it would prefix-match glm-5 ($1.0/$3.2).
        from agent_os.agent.pricing import get_cost_rates
        # get_cost_rates returns per-1K rates.
        assert get_cost_rates("glm-5.3-flash", "zhipu") == pytest.approx((0.15e-3, 0.5e-3))

    def test_zen_windows_confirmed_by_opencode_model_list(self, registry):
        # opencode.ai/zen/v1/models (and models.dev) cap these below the
        # upstream vendor's window.
        assert registry.get_context_window("opencode-zen", "minimax-m3") == 512_000
        assert registry.get_context_window("opencode-zen", "qwen3.5-plus") == 262_144
        assert registry.get_context_window("opencode-zen", "qwen3.6-plus") == 262_144

    def test_input_capped_models_declare_the_input_limit(self, registry):
        # OpenCode lists an input cap below the total context for these; the
        # window drives compaction at 80%, so it must be the input cap or the
        # prompt overflows before compaction triggers.
        assert registry.get_context_window("opencode-go", "hy3") == 192_000
        assert registry.get_context_window("opencode-zen", "big-pickle") == 160_000

    def test_anthropic_path_max_tokens_fit_the_window(self, registry):
        # On sdk=anthropic, max_output goes out as max_tokens. At the 80%
        # compaction point the prompt plus max_tokens must still fit.
        # Pre-existing dated snapshots that already break this (64K at 200K);
        # left for a separate decision rather than changed silently.
        known = {
            ("anthropic", "claude-haiku-4-5-20251001"),
            ("anthropic", "claude-sonnet-4-5-20250929"),
            ("anthropic", "claude-opus-4-5-20251101"),
            ("anthropic", "claude-sonnet-4-20250514"),
        }
        data = registry.all_providers()
        for provider, pdata in data.items():
            for model, entry in pdata.get("models", {}).items():
                if model.startswith("_") or (provider, model) in known:
                    continue
                if (entry.get("sdk") or pdata.get("sdk")) != "anthropic":
                    continue
                info = registry.get_model_info(provider, model)
                assert info.max_output is not None, (provider, model)
                assert 0.8 * info.context_window + info.max_output <= info.context_window, (
                    provider, model, info.context_window, info.max_output)
