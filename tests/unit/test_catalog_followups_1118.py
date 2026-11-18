# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Catalog follow-ups to the 2026-11-18 model-spec refresh.

1. Ids the version-boundary rule stopped prefix-matching (another version of
   a catalogued family) get their own entries where they are really served,
   so they do not silently lose their endpoint, vision, or price.
2. A JSON ``null`` for ``max_output`` / ``context_window`` falls back like a
   missing field (``max_tokens: null`` reached a provider).
3. Zhipu leads with GLM-5.3, then GLM-5.3-Flash.
4. GLM-5.3 / GLM-5.3-Flash reasoning is locked on (z.ai accepts only
   ``thinking.type=enabled``), so no off switch is ever sent.
5. GLM-5.3-Flash accepts images.
6. A model whose prompt cap is below its total window declares the prompt
   cap: compaction triggers at 80% of ``context_window``.
7. OpenCode protocol audit: no entry changes (``qwen3.8-flash`` on Go is
   listed as messages-only but works over Chat Completions live).
"""

from __future__ import annotations

import json

import pytest

from agent_os.agent.pricing import resolve_rates
from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.config.provider_registry import (
    ProviderRegistry,
    _FALLBACK_CONTEXT_WINDOW,
    _FALLBACK_MAX_OUTPUT,
)

NO_OVERRIDES = "/nonexistent/pricing-overrides.json"


@pytest.fixture(scope="module")
def reg():
    return ProviderRegistry()


# --- 1. entries for ids the new rule no longer prefix-matches --------------

# (provider, id, context_window, vision, sdk, base_url, input, output)
NEW_ENTRIES = [
    ("anthropic", "claude-fable-5-1", 1_000_000, True, None, None, 10.0, 50.0),
    ("anthropic", "claude-fable-5-2", 1_000_000, True, None, None, 8.0, 40.0),
    ("anthropic", "claude-sonnet-5-5", 1_000_000, True, None, None, 2.0, 10.0),
    ("anthropic", "claude-opus-4-1", 200_000, True, None, None, 15.0, 75.0),
    ("opencode-zen", "claude-fable-5-1", 1_000_000, True, "anthropic",
     "https://opencode.ai/zen", 10.0, 50.0),
    ("opencode-zen", "claude-fable-5-2", 1_000_000, True, "anthropic",
     "https://opencode.ai/zen", 8.0, 40.0),
    ("opencode-zen", "claude-opus-5-5", 1_000_000, True, "anthropic",
     "https://opencode.ai/zen", 4.0, 20.0),
    ("opencode-zen", "claude-sonnet-5-5", 1_000_000, True, "anthropic",
     "https://opencode.ai/zen", 2.0, 10.0),
    ("opencode-go", "kimi-k3.1-pro", 1_048_576, True, None, None, 2.0, 10.0),
    ("opencode-go", "minimax-m3.1-flash", 1_000_000, True, None, None, 0.15, 0.6),
    ("moonshot", "kimi-k3.1-pro", 1_048_576, True, None, None, 2.0, 10.0),
    ("moonshot", "kimi-k3.1-pro-highspeed", 1_048_576, True, None, None, 4.0, 20.0),
    ("zhipu", "glm-5.3-flashx", 1_000_000, True, None, None, 0.37, 1.25),
    ("zhipu", "glm-5v-turbo", 200_000, True, None, None, 1.2, 4.0),
    ("zhipu", "glm-4.6v", 128_000, True, None, None, 0.3, 0.9),
    ("zhipu", "glm-4.6v-flash", 128_000, True, None, None, 0.0, 0.0),
    # Not new versions, but uncatalogued: with GLM-5.3 as the flagship they
    # would inherit a 1M window (real: 128K / 64K).
    ("zhipu", "glm-4.5", 131_072, False, None, None, 0.6, 2.2),
    ("zhipu", "glm-4.5v", 64_000, True, None, None, 0.6, 1.8),
    ("xai", "grok-4.3", 1_000_000, True, None, None, 1.25, 2.5),
    ("xai", "grok-4.5", 500_000, True, None, None, 2.0, 6.0),
    ("xai", "grok-4.6", 500_000, True, None, None, 2.0, 6.0),
    ("xai", "grok-4.7", 500_000, True, None, None, 2.0, 6.0),
]


@pytest.mark.parametrize("provider,mid,ctx,vision,sdk,base_url,inp,out", NEW_ENTRIES)
def test_newer_version_has_its_own_entry(reg, provider, mid, ctx, vision, sdk,
                                         base_url, inp, out):
    pdata = reg.get_provider_data(provider)
    assert mid in pdata["models"] and mid in pdata["pricing"]
    info = reg.get_model_info(provider, mid)
    assert info.context_window == ctx
    assert info.capabilities.vision is vision
    assert (info.sdk, info.base_url) == (sdk, base_url)
    rates = resolve_rates(provider, mid, override_path=NO_OVERRIDES)
    assert (rates.input_per_1m, rates.output_per_1m) == pytest.approx((inp, out))


def test_zen_dated_claude_keeps_messages_endpoint(reg):
    # A dated snapshot of a new entry still inherits its endpoint.
    info = reg.get_model_info("opencode-zen", "claude-sonnet-5-5-20261101")
    assert info.sdk == "anthropic" and info.base_url == "https://opencode.ai/zen"


# --- 2. explicit nulls fall back ------------------------------------------

def test_null_limits_fall_back(tmp_path):
    data = {"providers": {"p": {"models": {
        "m": {"context_window": None, "max_output": None},
    }}}}
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(data))
    info = ProviderRegistry(config_path=str(path)).get_model_info("p", "m")
    assert info.context_window == _FALLBACK_CONTEXT_WINDOW
    assert info.max_output == _FALLBACK_MAX_OUTPUT


def test_every_catalog_entry_resolves_to_integer_limits(reg):
    for provider, pdata in reg.all_providers().items():
        for mid in pdata.get("models", {}):
            info = reg.get_model_info(provider, mid)
            assert isinstance(info.context_window, int), (provider, mid)
            assert isinstance(info.max_output, int), (provider, mid)


def test_catalog_null_max_output_is_now_the_fallback(reg):
    # kimi-k2.7-code carries "max_output": null in providers.json.
    assert reg.get_max_output("opencode-go", "kimi-k2.7-code") == _FALLBACK_MAX_OUTPUT


# --- 3. zhipu defaults ------------------------------------------------------

def test_zhipu_leads_with_glm_5_3_then_flash(reg):
    assert reg.suggested_models("zhipu") == [
        "glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1",
        "glm-5", "glm-4.7", "glm-4.6", "glm-4.5-air",
    ]


def test_unknown_zhipu_model_inherits_glm_5_3(reg):
    info = reg.get_model_info("zhipu", "glm-9-preview")
    assert info.context_window == reg.get_context_window("zhipu", "glm-5.3")


# --- 4 + 5. GLM-5.3 reasoning locked on; GLM-5.3-Flash vision -------------

GLM_53 = [(p, m) for p in ("zhipu", "opencode-go", "opencode-zen")
          for m in ("glm-5.3", "glm-5.3-flash")]


def _provider_for(reg, provider, mid):
    info = reg.get_model_info(provider, mid)
    return LLMProvider(mid, "sk-test", "https://example.invalid/v1",
                       provider=provider, sdk=info.sdk or "openai",
                       max_output=info.max_output, capabilities=info.capabilities,
                       reasoning=info.reasoning)


@pytest.mark.parametrize("provider,mid", GLM_53)
def test_glm_5_3_reasoning_is_locked_on(reg, provider, mid):
    info = reg.get_model_info(provider, mid)
    assert info.reasoning.supported is True
    assert info.reasoning.enable == "model_only"
    assert info.reasoning.field == "reasoning_content"
    assert _provider_for(reg, provider, mid).reasoning_locked_on is True


@pytest.mark.parametrize("provider", ("zhipu", "opencode-go", "opencode-zen"))
def test_glm_5_2_stays_hybrid(reg, provider):
    info = reg.get_model_info(provider, "glm-5.2")
    assert info.reasoning.enable == "param:thinking.type=enabled"
    assert _provider_for(reg, provider, "glm-5.2").reasoning_locked_on is False


@pytest.mark.parametrize("provider", ("zhipu", "opencode-go", "opencode-zen"))
def test_glm_5_3_flash_accepts_images_glm_5_3_does_not(reg, provider):
    assert reg.get_capabilities(provider, "glm-5.3-flash").vision is True
    assert reg.get_capabilities(provider, "glm-5.3").vision is False
    assert reg.get_capabilities(provider, "glm-5.2").vision is False


# --- 6. prompt caps -----------------------------------------------------------

OPENAI_272K = ["gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-mini", "gpt-5-nano",
               "gpt-5.3-codex", "gpt-5.4-mini", "gpt-5.4-nano"]
OPENAI_922K = ["gpt-5.4", "gpt-5.4-pro", "gpt-5.5", "gpt-5.5-pro", "gpt-5.6",
               "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-sol",
               "gpt-6-luna", "gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra",
               "gpt-6.2-sol", "gpt-6.2-astra", "gpt-6*"]


@pytest.mark.parametrize("mid", OPENAI_272K)
def test_openai_gpt5_declares_the_272k_prompt_cap(reg, mid):
    assert mid in reg.get_provider_data("openai")["models"]
    assert reg.get_context_window("openai", mid) == 272_000


@pytest.mark.parametrize("mid", OPENAI_922K)
def test_openai_1m_models_declare_the_922k_prompt_cap(reg, mid):
    assert mid in reg.get_provider_data("openai")["models"]
    assert reg.get_context_window("openai", mid) == 922_000


def test_future_gpt_6_gets_the_family_prompt_cap(reg):
    assert reg.get_context_window("openai", "gpt-6.3-sol") == 922_000


@pytest.mark.parametrize("provider,mid", [
    ("opencode-zen", "gpt-6-luna"), ("opencode-zen", "gpt-5.6-luna"),
    ("opencode-zen", "gpt-6.2-sol"),
    ("opencode-go", "gpt-6-luna"), ("opencode-go", "gpt-5.6-luna"),
])
def test_opencode_openai_models_declare_the_listed_prompt_cap(reg, provider, mid):
    assert reg.get_context_window(provider, mid) == 922_000


def test_hunyuan_hy3_declares_its_prompt_cap(reg):
    assert reg.get_context_window("hunyuan", "hy3") == 192_000


# --- 7. OpenCode protocol ---------------------------------------------------

def test_go_qwen_3_8_flash_stays_on_chat_completions(reg):
    # OpenCode's list names only Anthropic messages for it, but a live call
    # over Chat Completions answered 200 with reasoning (2026-11-19), so the
    # working protocol is kept.
    info = reg.get_model_info("opencode-go", "qwen3.8-flash")
    assert info.sdk is None and info.base_url is None
