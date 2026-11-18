# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Catalog routing for the Responses API (sdk ``openai-responses``).

OpenAI: "Starting with GPT-5.4, Chat Completions does not support tool calling
with reasoning_effort values other than none" (migrate-to-responses guide), and
GPT-6 Astra / GPT-6.1 Sol / GPT-6.2 Sol refuse effort ``none`` outright. So
every OpenAI model from 5.4 on is routed to Responses, and so is an uncatalogued
``gpt-6.x`` id via the ``gpt-6*`` family-pattern entry: the registry honors
endpoint overrides only on exact, prefix or family-pattern matches (a plain
prefix no longer spans versions), and flagship inheritance alone would send a
new model to Chat Completions and the same 400.

OpenCode Go/Zen serve their GPT and Grok models ONLY over Responses (verified
on the wire 2026-10-10: 400 ModelProtocolUnsupported elsewhere).
"""

import pytest

from agent_os.agent.providers.responses_adapter import RESPONSES_SDK
from agent_os.config.provider_registry import ProviderRegistry


@pytest.fixture(scope="module")
def real():
    return ProviderRegistry()


NEW_OPENAI = ("gpt-6-sol", "gpt-6-luna", "gpt-6-astra", "gpt-6.1-sol",
              "gpt-6.1-terra", "gpt-6.2-sol", "gpt-6.2-astra")


@pytest.mark.parametrize("mid", NEW_OPENAI)
def test_gpt_6_family_is_catalogued_on_responses(real, mid):
    info = real.get_model_info("openai", mid)
    assert info.sdk == RESPONSES_SDK, mid
    assert info.base_url is None, "the OpenAI client appends /responses to /v1"
    # The 922K prompt cap, not the 1.05M total: compaction runs at 80% of it.
    assert info.context_window == 922_000
    assert info.max_output == 128_000
    assert info.capabilities.vision and info.capabilities.tool_use
    assert info.reasoning.supported is True
    assert info.reasoning.efforts, mid


@pytest.mark.parametrize("mid", ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro",
                                 "gpt-5.5", "gpt-5.5-pro", "gpt-5.6", "gpt-5.6-sol",
                                 "gpt-5.6-terra", "gpt-5.6-luna"))
def test_gpt_5_4_and_later_route_to_responses(real, mid):
    assert real.get_model_info("openai", mid).sdk == RESPONSES_SDK


@pytest.mark.parametrize("mid", ("gpt-4o", "gpt-4.1", "o3", "o4-mini", "gpt-5", "gpt-5.2"))
def test_older_openai_models_stay_on_chat_completions(real, mid):
    assert real.get_model_info("openai", mid).sdk is None


def test_effort_lists_drive_the_off_switch(real):
    """'none' is accepted by gpt-6-sol/-luna and 6.1-terra; refused by
    Astra, 6.1 Sol and 6.2 Sol (OpenAI model pages / models.dev snapshot)."""
    for mid in ("gpt-6-sol", "gpt-6-luna", "gpt-6.1-terra", "gpt-5.6-sol"):
        assert "none" in real.get_model_info("openai", mid).reasoning.efforts, mid
    for mid in ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.2-sol", "gpt-6.2-astra"):
        assert "none" not in real.get_model_info("openai", mid).reasoning.efforts, mid


def test_uncatalogued_gpt_6_model_still_routes_to_responses(real):
    """A future gpt-6.3-sol must not fall back to Chat Completions and hit
    the same 400; the ``gpt-6*`` family-pattern entry carries the endpoint,
    and its effort list omits 'none' so the off switch is never sent blind.
    Its window is the 922K prompt cap, like every catalogued GPT-6 model."""
    info = real.get_model_info("openai", "gpt-6.3-sol")
    assert info.sdk == RESPONSES_SDK
    assert info.context_window == 922_000
    assert "none" not in info.reasoning.efforts


def test_openai_suggests_current_models(real):
    suggested = real.suggested_models("openai")
    assert suggested[0] == "gpt-6.2-sol"
    for mid in suggested:
        assert mid in real.get_provider_data("openai")["models"], mid


def test_openai_provider_level_sdk_is_unchanged(real):
    """The protocol is per model; the provider (and every card) stays openai."""
    assert real.get_provider_data("openai")["sdk"] == "openai"


@pytest.mark.parametrize("mid", NEW_OPENAI)
def test_new_openai_models_are_priced(real, mid):
    from agent_os.agent.pricing import resolve_rates
    rates = resolve_rates("openai", mid)
    assert rates.input_per_1m > 0 and rates.output_per_1m > 0
    assert rates.cached_input_per_1m < rates.input_per_1m


@pytest.mark.parametrize("mid", ("grok-4.6", "grok-4.7", "gpt-5.6-luna", "gpt-6-luna"))
def test_opencode_go_responses_models(real, mid):
    info = real.get_model_info("opencode-go", mid)
    assert info.sdk == RESPONSES_SDK, mid
    assert info.base_url is None, "Go's /zen/go/v1 + /responses"


@pytest.mark.parametrize("mid", ("grok-4.6", "grok-4.7", "gpt-5.6-luna", "gpt-6-luna",
                                 "gpt-6.2-sol"))
def test_opencode_zen_responses_models(real, mid):
    info = real.get_model_info("opencode-zen", mid)
    assert info.sdk == RESPONSES_SDK, mid
    assert info.base_url is None


def test_opencode_notes_no_longer_say_responses_is_unspoken(real):
    for key in ("opencode-go", "opencode-zen"):
        notes = real.get_provider_data(key)["notes"]
        assert "Responses API, which Orbital does not speak" not in notes, key
        assert "Orbital speaks neither" not in notes, key


def test_test_connection_resolves_the_responses_endpoint():
    from agent_os.api.routes import agents_v2
    reg = ProviderRegistry()
    old = agents_v2._provider_registry
    agents_v2._provider_registry = reg
    try:
        _, base, sdk = agents_v2._resolve_test_endpoint("openai", "gpt-6.2-sol", None, "openai")
        assert sdk == RESPONSES_SDK
        assert base == "https://api.openai.com/v1"
    finally:
        agents_v2._provider_registry = old


def test_protocol_mismatch_text_no_longer_says_responses_is_unspoken():
    from agent_os.api.routes.agents_v2 import _describe_protocol_mismatch
    text = _describe_protocol_mismatch("grok-9", "openai")
    assert "cannot be used yet" not in text
    assert "Responses API" in text
    assert "OpenAI Responses" in _describe_protocol_mismatch("x", RESPONSES_SDK)
