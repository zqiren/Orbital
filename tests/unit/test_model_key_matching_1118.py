# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A catalog key never matches a DIFFERENT VERSION of its model family.

Bug (2026-11-18): ``glm-5.3-flash`` had no entry on some providers, prefix-
matched ``glm-5`` and silently got a 200K window and GLM-5's price. A plain
``startswith`` lets any key match every later version of its family. The rule
now: past the key, the id may only continue with a variant or snapshot suffix
(``-flash``, ``-free``, ``-20250929``, ``-0731``, ``@…``, ``:…``), never with a
version step (``.4``, ``-9``, ``1``). An id that fails falls through to the
next rung, exactly like an unknown id.

A key ending in ``*`` is an explicit FAMILY PATTERN that does match across
versions (``gpt-6*`` catches a future ``gpt-6.3-sol``); exact keys and longer
normal prefixes beat it.

Specs (``provider_registry``) and pricing (``agent.pricing``) share one helper,
so they cannot disagree about which entry an id belongs to.
"""

from __future__ import annotations

import json

import pytest

from agent_os.config.provider_registry import ProviderRegistry, match_model_key


# --- the rule itself, table-driven -----------------------------------------

REJECTED = [
    ("glm-5.4", "glm-5"),
    ("glm-5.4-flash", "glm-5"),
    ("glm-5.3-flash", "glm-5"),
    ("gpt-5.7", "gpt-5"),
    ("claude-opus-4-9", "claude-opus-4"),
    ("kimi-k3.1-pro", "kimi-k3"),
    ("grok-4-1-fast", "grok-4"),
    ("gpt-4o", "gpt-4"),              # mid-token continuation is another model
    ("o10", "o1"),
    ("openai/gpt-5.1", "openai/gpt-5"),
]

ACCEPTED = [
    ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
    ("gpt-5.6-sol-2026-07-09", "gpt-5.6-sol"),
    ("deepseek-v4-flash-0731", "deepseek-v4-flash"),
    ("glm-5.3-flash-free", "glm-5.3-flash"),
    ("gpt-5-codex", "gpt-5"),
    ("o3-mini-high", "o3-mini"),
    ("gemini-2.0-flash-001", "gemini-2.0-flash"),
    ("claude-sonnet-4-5@20250929", "claude-sonnet-4-5"),
    ("anthropic/claude-sonnet-4.5:beta", "anthropic/claude-sonnet-4.5"),
    ("deepseek-ai/DeepSeek-R1-0528", "deepseek-ai/DeepSeek-R1"),
    ("llama-3.1-8b-instant", "llama-3.1"),   # size token, not a version step
    ("qwen3-235b-a22b-instruct-2507", "qwen3-235b-a22b-instruct"),
]


@pytest.mark.parametrize("model,key", REJECTED)
def test_a_different_version_does_not_match(model, key):
    assert match_model_key(model, [key]) is None


@pytest.mark.parametrize("model,key", ACCEPTED)
def test_a_variant_or_snapshot_suffix_still_matches(model, key):
    assert match_model_key(model, [key]) == key


def test_exact_match_wins():
    assert match_model_key("glm-5", ["glm-5", "glm"]) == "glm-5"


def test_longest_valid_prefix_wins():
    keys = ["claude-sonnet-4", "claude-sonnet-4-5"]
    assert match_model_key("claude-sonnet-4-5-20250929", keys) == "claude-sonnet-4-5"


def test_a_rejected_longer_key_does_not_hide_a_valid_shorter_one():
    # "glm-5" is rejected for glm-5.4-flash; nothing else qualifies.
    assert match_model_key("glm-5.4-flash", ["glm-5", "glm-4.5-flash"]) is None


def test_underscore_keys_never_match():
    assert match_model_key("_default-x", ["_default"]) is None
    assert match_model_key("_x.1", ["_x*"]) is None


# --- family pattern ----------------------------------------------------------

@pytest.mark.parametrize("model", ["gpt-6.3-sol", "gpt-6.3", "gpt-6", "gpt-6-nova",
                                   "gpt-6.10-terra-2027-01-01"])
def test_family_pattern_matches_across_versions(model):
    assert match_model_key(model, ["gpt-6*"]) == "gpt-6*"


@pytest.mark.parametrize("model", ["gpt-60", "gpt-6o", "gpt-5.9"])
def test_family_pattern_stays_inside_its_family(model):
    assert match_model_key(model, ["gpt-6*"]) is None


def test_exact_entry_beats_family_pattern():
    assert match_model_key("gpt-6-sol", ["gpt-6*", "gpt-6-sol"]) == "gpt-6-sol"


def test_longer_normal_prefix_beats_family_pattern():
    keys = ["gpt-6*", "gpt-6.2-sol"]
    assert match_model_key("gpt-6.2-sol-2027-01-01", keys) == "gpt-6.2-sol"


def test_family_pattern_beats_shorter_normal_prefix():
    # "kimi" + "-k3" is a word suffix so "kimi" qualifies, but the family
    # stem is longer and more specific.
    assert match_model_key("kimi-k3.1-pro", ["kimi", "kimi-k*"]) == "kimi-k*"


def test_family_stem_ending_in_a_letter_takes_a_version_number():
    assert match_model_key("kimi-k3", ["kimi-k*"]) == "kimi-k*"
    assert match_model_key("kimi-kx", ["kimi-k*"]) is None


# --- the registry and pricing both use it ------------------------------------

def _write(tmp_path, data):
    p = tmp_path / "providers.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture
def mini(tmp_path):
    data = {"providers": {"p": {
        "sdk": "openai",
        "suggested_models": ["flag"],
        "models": {
            "flag": {"context_window": 1_000_000, "max_output": 64_000,
                     "sdk": "anthropic", "base_url": "https://x/anthropic"},
            "glm-5": {"context_window": 200_000, "max_output": 128_000},
            "fam*": {"context_window": 500_000, "max_output": 50_000,
                     "sdk": "openai-responses"},
            "_default": {"context_window": 64_000, "max_output": 4_096},
        },
    }}}
    return ProviderRegistry(config_path=_write(tmp_path, data))


def test_registry_rejected_id_falls_to_flagship_specs_without_endpoint(mini):
    info = mini.get_model_info("p", "glm-5.4-flash")
    assert info.context_window == 1_000_000       # flagship specs, not glm-5
    assert info.sdk is None and info.base_url is None


def test_registry_snapshot_still_matches(mini):
    assert mini.get_context_window("p", "glm-5-0731") == 200_000


def test_registry_family_pattern_carries_its_endpoint(mini):
    info = mini.get_model_info("p", "fam.2-pro")
    assert info.context_window == 500_000
    assert info.sdk == "openai-responses"


def test_real_catalog_glm_5_4_does_not_inherit_glm_5():
    reg = ProviderRegistry()
    for provider in ("zhipu", "opencode-zen"):
        flagship = reg.suggested_models(provider)[0]
        got = reg.get_model_info(provider, "glm-5.4-flash")
        want = reg.get_model_info(provider, flagship)
        assert got.context_window == want.context_window, provider
        assert got.max_output == want.max_output, provider
        assert got.sdk is None and got.base_url is None, provider


def test_real_catalog_future_gpt_6_routes_to_responses_via_family_pattern():
    reg = ProviderRegistry()
    assert "gpt-6*" in reg.get_provider_data("openai")["models"]
    assert "gpt-6" not in reg.get_provider_data("openai")["models"]
    info = reg.get_model_info("openai", "gpt-6.3-sol")
    assert info.sdk == "openai-responses"


def test_pricing_uses_the_same_rule(tmp_path):
    from agent_os.agent import pricing
    table = {"glm-5": {"input_per_1m": 1.0, "output_per_1m": 3.2},
             "fam*": {"input_per_1m": 7.0, "output_per_1m": 9.0},
             "_default": {"input_per_1m": 0.5, "output_per_1m": 0.6}}
    assert pricing._find_entry(table, "glm-5.4")["input_per_1m"] == 0.5
    assert pricing._find_entry(table, "glm-5-0731")["input_per_1m"] == 1.0
    assert pricing._find_entry(table, "fam.3")["input_per_1m"] == 7.0


def test_legacy_cost_rates_use_the_same_rule():
    from agent_os.agent.pricing import get_cost_rates, resolve_rates
    # gpt-6.3-sol must price through the family pattern on both interfaces.
    legacy_in, legacy_out = get_cost_rates("gpt-6.3-sol", "openai")
    r = resolve_rates("openai", "gpt-6.3-sol", override_path="/nonexistent/x.json")
    assert (legacy_in * 1000, legacy_out * 1000) == pytest.approx(
        (r.input_per_1m, r.output_per_1m))
    assert r.input_per_1m == pytest.approx(2.0)
