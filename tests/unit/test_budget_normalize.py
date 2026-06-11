# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for agent_os.budget.normalize — token-ledger normalization.

TDD: these tests were written before the implementation.

Coverage:
  - NormalizedUsage dataclass fields
  - normalize_usage() with openai_compat semantics (cached_tokens ⊆ prompt_tokens)
  - normalize_usage() with anthropic semantics (additive / disjoint fields)
  - provider with NO cache fields at all
  - disjointness: sum of normalized fields == total billed tokens per semantic
  - adapter integration: Anthropic non-streaming path populates cache_write_tokens
  - adapter integration: Anthropic streaming (message_start) populates cache_write_tokens
  - OpenAI-compat path: cache_write_tokens is always 0
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_os.agent.providers.types import TokenUsage
from agent_os.budget.normalize import (
    NormalizedUsage,
    ProviderSemantics,
    normalize_usage,
)


# ---------------------------------------------------------------------------
# Helpers: fixture TokenUsage objects that mimic real provider shapes
# ---------------------------------------------------------------------------

def _make_openai_compat_usage(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> TokenUsage:
    """Mimic OpenAI-compat (Kimi/Moonshot/DeepSeek) usage.

    Semantics: cached_tokens ⊆ prompt_tokens (subset, not additive).
    The adapter stores: input_tokens=prompt_tokens, cache_read_tokens=cached_tokens,
    cache_write_tokens=0.
    """
    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cache_read_tokens=cached_tokens,
        cache_write_tokens=0,
    )


def _make_anthropic_usage(
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> TokenUsage:
    """Mimic Anthropic usage — additive/disjoint semantics.

    The adapter stores:
        input_tokens=input_tokens (already EXCLUDES cache fields),
        cache_read_tokens=cache_read_input_tokens,
        cache_write_tokens=cache_creation_input_tokens.
    """
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_input_tokens,
        cache_write_tokens=cache_creation_input_tokens,
    )


# ===========================================================================
# NormalizedUsage dataclass
# ===========================================================================

class TestNormalizedUsage:
    def test_fields_exist(self):
        n = NormalizedUsage(
            uncached_input=10,
            cache_read=5,
            cache_write=2,
            output=8,
        )
        assert n.uncached_input == 10
        assert n.cache_read == 5
        assert n.cache_write == 2
        assert n.output == 8

    def test_total_property(self):
        n = NormalizedUsage(uncached_input=10, cache_read=5, cache_write=2, output=8)
        assert n.total == 25


# ===========================================================================
# ProviderSemantics — from_sdk mapping and unknown-value rejection
# ===========================================================================

class TestProviderSemantics:
    """ProviderSemantics.from_sdk maps LLMProvider.sdk values explicitly."""

    def test_from_sdk_openai_maps_to_openai_compat(self):
        assert ProviderSemantics.from_sdk("openai") is ProviderSemantics.OPENAI_COMPAT

    def test_from_sdk_anthropic_maps_to_anthropic(self):
        assert ProviderSemantics.from_sdk("anthropic") is ProviderSemantics.ANTHROPIC

    def test_from_sdk_unknown_raises_value_error(self):
        with pytest.raises(ValueError):
            ProviderSemantics.from_sdk("gemini")

    def test_from_sdk_openai_compatible_alias_maps_to_openai_compat(self):
        """'openai-compatible' shipped as a one-off sdk value (mistral); a
        config still carrying it must ledger with subset semantics, never be
        silently skipped (which would report $0 cost forever)."""
        assert (
            ProviderSemantics.from_sdk("openai-compatible")
            is ProviderSemantics.OPENAI_COMPAT
        )

    def test_every_registry_sdk_value_maps_to_semantics(self):
        """Every sdk value in the shipped providers.json must resolve via
        from_sdk — a provider whose sdk doesn't map would silently never
        append ledger lines (warning + skip), breaking the 'every response
        ledgers' outcome. A new sdk value must be added to from_sdk first."""
        import json
        import os

        registry = os.path.join(
            os.path.dirname(__file__), "..", "..", "agent_os", "config",
            "providers.json",
        )
        with open(registry, encoding="utf-8") as f:
            providers = json.load(f)["providers"]
        sdks = {info["sdk"] for info in providers.values() if "sdk" in info}
        assert sdks, "expected at least one sdk value in providers.json"
        for sdk in sdks:
            ProviderSemantics.from_sdk(sdk)  # must not raise

    def test_normalize_accepts_enum_member(self):
        """normalize_usage takes the enum member directly (ledger entry path)."""
        usage = TokenUsage(input_tokens=100, output_tokens=20, cache_read_tokens=40)
        result = normalize_usage(usage, ProviderSemantics.OPENAI_COMPAT)
        assert result.uncached_input == 60
        assert result.cache_read == 40

    def test_normalize_accepts_plain_string(self):
        """str-enum: the plain string values still work."""
        usage = TokenUsage(input_tokens=100, output_tokens=20)
        result = normalize_usage(usage, "anthropic")
        assert result.uncached_input == 100

    def test_normalize_unknown_semantics_raises_value_error(self):
        usage = TokenUsage(input_tokens=100, output_tokens=20)
        with pytest.raises(ValueError):
            normalize_usage(usage, "gemini")


# ===========================================================================
# normalize_usage — openai_compat semantics
# ===========================================================================

class TestNormalizeOpenAICompat:
    """cached_tokens ⊆ prompt_tokens → uncached_input = prompt_tokens - cached_tokens."""

    def test_kimi_style_with_cache_hit(self):
        """Kimi/Moonshot: 1000 prompt, 300 cached → 700 uncached_input."""
        usage = _make_openai_compat_usage(
            prompt_tokens=1000,
            completion_tokens=200,
            cached_tokens=300,
        )
        result = normalize_usage(usage, semantics="openai_compat")
        assert result.uncached_input == 700
        assert result.cache_read == 300
        assert result.cache_write == 0
        assert result.output == 200

    def test_kimi_no_cache(self):
        """No cache hit → uncached_input == input_tokens."""
        usage = _make_openai_compat_usage(prompt_tokens=500, completion_tokens=50)
        result = normalize_usage(usage, semantics="openai_compat")
        assert result.uncached_input == 500
        assert result.cache_read == 0
        assert result.cache_write == 0
        assert result.output == 50

    def test_disjointness_openai_compat(self):
        """uncached_input + cache_read + cache_write + output == prompt_tokens + output_tokens."""
        usage = _make_openai_compat_usage(
            prompt_tokens=1000, completion_tokens=200, cached_tokens=300
        )
        result = normalize_usage(usage, semantics="openai_compat")
        total_billed = usage.input_tokens + usage.output_tokens  # 1000 + 200
        assert result.uncached_input + result.cache_read + result.cache_write + result.output == total_billed

    def test_full_cache_hit(self):
        """All input tokens are cached → uncached_input == 0."""
        usage = _make_openai_compat_usage(
            prompt_tokens=800, completion_tokens=100, cached_tokens=800
        )
        result = normalize_usage(usage, semantics="openai_compat")
        assert result.uncached_input == 0
        assert result.cache_read == 800

    def test_subset_violation_clamps_to_zero(self):
        """cache_read > input violates the subset contract — clamp, never negative."""
        usage = _make_openai_compat_usage(
            prompt_tokens=100, completion_tokens=10, cached_tokens=150
        )
        result = normalize_usage(usage, semantics="openai_compat")
        assert result.uncached_input == 0


# ===========================================================================
# normalize_usage — anthropic semantics
# ===========================================================================

class TestNormalizeAnthropic:
    """Anthropic: input_tokens already EXCLUDES cache fields (additive).

    uncached_input = input_tokens (unchanged);
    cache_read = cache_read_tokens;
    cache_write = cache_write_tokens.
    """

    def test_anthropic_with_cache_read_and_write(self):
        """Both cache_read and cache_write present."""
        usage = _make_anthropic_usage(
            input_tokens=500,
            output_tokens=100,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=50,
        )
        result = normalize_usage(usage, semantics="anthropic")
        assert result.uncached_input == 500   # unchanged
        assert result.cache_read == 200
        assert result.cache_write == 50
        assert result.output == 100

    def test_anthropic_cache_read_only(self):
        usage = _make_anthropic_usage(
            input_tokens=300,
            output_tokens=80,
            cache_read_input_tokens=150,
            cache_creation_input_tokens=0,
        )
        result = normalize_usage(usage, semantics="anthropic")
        assert result.uncached_input == 300
        assert result.cache_read == 150
        assert result.cache_write == 0
        assert result.output == 80

    def test_anthropic_no_cache(self):
        """No cache usage — everything is uncached input."""
        usage = _make_anthropic_usage(
            input_tokens=400, output_tokens=60
        )
        result = normalize_usage(usage, semantics="anthropic")
        assert result.uncached_input == 400
        assert result.cache_read == 0
        assert result.cache_write == 0
        assert result.output == 60

    def test_disjointness_anthropic(self):
        """uncached + cache_read + cache_write + output == input + cache_read + cache_write + output."""
        usage = _make_anthropic_usage(
            input_tokens=500,
            output_tokens=100,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=50,
        )
        result = normalize_usage(usage, semantics="anthropic")
        # Under Anthropic semantics, total billed = input + cache_read + cache_write + output
        total_billed = (
            usage.input_tokens
            + usage.cache_read_tokens
            + usage.cache_write_tokens
            + usage.output_tokens
        )
        assert result.uncached_input + result.cache_read + result.cache_write + result.output == total_billed


# ===========================================================================
# Provider with NO cache fields at all
# ===========================================================================

class TestNormalizeNoCache:
    """A provider that sets no cache fields: all tokens go to uncached_input."""

    def test_no_cache_openai_compat(self):
        usage = TokenUsage(input_tokens=300, output_tokens=75)
        result = normalize_usage(usage, semantics="openai_compat")
        assert result.uncached_input == 300
        assert result.cache_read == 0
        assert result.cache_write == 0
        assert result.output == 75

    def test_no_cache_anthropic(self):
        usage = TokenUsage(input_tokens=300, output_tokens=75)
        result = normalize_usage(usage, semantics="anthropic")
        assert result.uncached_input == 300
        assert result.cache_read == 0
        assert result.cache_write == 0
        assert result.output == 75

    def test_disjointness_no_cache_openai_compat(self):
        usage = TokenUsage(input_tokens=300, output_tokens=75)
        result = normalize_usage(usage, semantics="openai_compat")
        assert result.uncached_input + result.cache_read + result.cache_write + result.output == 375

    def test_disjointness_no_cache_anthropic(self):
        usage = TokenUsage(input_tokens=300, output_tokens=75)
        result = normalize_usage(usage, semantics="anthropic")
        assert result.uncached_input + result.cache_read + result.cache_write + result.output == 375


# ===========================================================================
# Adapter integration: Anthropic non-streaming path → cache_write_tokens
# ===========================================================================

class TestAnthropicAdapterCacheWrite:
    """Verify the Anthropic adapter captures cache_creation_input_tokens
    into TokenUsage.cache_write_tokens in BOTH non-streaming and streaming paths."""

    def test_non_streaming_cache_write_tokens(self):
        """translate_response_to_openai captures cache_creation_input_tokens."""
        from agent_os.agent.providers.anthropic_adapter import translate_response_to_openai

        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "hello"
        response.content = [text_block]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(
            input_tokens=100,
            output_tokens=30,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=15,
        )

        result = translate_response_to_openai(response)
        usage = result["usage"]
        assert usage.cache_write_tokens == 15
        assert usage.cache_read_tokens == 20
        assert usage.input_tokens == 100
        assert usage.output_tokens == 30

    def test_non_streaming_cache_write_tokens_zero_when_absent(self):
        """cache_creation_input_tokens missing/0 → cache_write_tokens == 0."""
        from agent_os.agent.providers.anthropic_adapter import translate_response_to_openai

        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "hello"
        response.content = [text_block]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(
            input_tokens=100,
            output_tokens=30,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )

        result = translate_response_to_openai(response)
        assert result["usage"].cache_write_tokens == 0

    def test_streaming_message_start_cache_write_tokens(self):
        """message_start event: cache_creation_input_tokens captured in StreamState."""
        from agent_os.agent.providers.anthropic_adapter import (
            translate_stream_event,
            StreamState,
        )

        state = StreamState()
        event = MagicMock(type="message_start")
        event.message = MagicMock()
        event.message.usage = MagicMock(
            input_tokens=80,
            output_tokens=0,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=25,
        )
        assert translate_stream_event(event, state) is None
        assert state.cache_write_tokens == 25
        assert state.cache_read_tokens == 10

    def test_streaming_message_stop_emits_cache_write_tokens(self):
        """message_stop emits final chunk with cache_write_tokens from state."""
        from agent_os.agent.providers.anthropic_adapter import (
            translate_stream_event,
            StreamState,
        )

        state = StreamState()
        state.input_tokens = 80
        state.output_tokens = 30
        state.cache_read_tokens = 10
        state.cache_write_tokens = 25

        event = MagicMock(type="message_stop")
        chunk = translate_stream_event(event, state)
        assert chunk is not None
        assert chunk.is_final is True
        assert chunk.usage.cache_write_tokens == 25
        assert chunk.usage.cache_read_tokens == 10


# ===========================================================================
# OpenAI-compat adapter: cache_write_tokens always 0
# ===========================================================================

class TestOpenAICompatCacheWriteTokens:
    """OpenAI-compat path must always set cache_write_tokens=0 (no such provider field)."""

    def test_make_token_usage_cache_write_is_zero(self):
        """_make_token_usage produces TokenUsage with cache_write_tokens=0."""
        from agent_os.agent.providers.openai_compat import _make_token_usage

        usage_obj = MagicMock()
        usage_obj.prompt_tokens = 500
        usage_obj.completion_tokens = 100
        # Simulate provider that has cached_tokens (Kimi/Moonshot)
        usage_obj.cache_read_input_tokens = None
        usage_obj.prompt_cache_hit_tokens = None
        usage_obj.cached_tokens = 300

        result = _make_token_usage(usage_obj)
        assert result.cache_write_tokens == 0
        assert result.cache_read_tokens == 300
        assert result.input_tokens == 500

    def test_make_token_usage_no_cache_write_is_zero(self):
        """Provider with no cache fields: cache_write_tokens == 0."""
        from agent_os.agent.providers.openai_compat import _make_token_usage

        usage_obj = MagicMock()
        usage_obj.prompt_tokens = 200
        usage_obj.completion_tokens = 50
        # No cache attributes
        del usage_obj.cache_read_input_tokens
        del usage_obj.prompt_cache_hit_tokens
        del usage_obj.cached_tokens

        result = _make_token_usage(usage_obj)
        assert result.cache_write_tokens == 0
