# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Token-usage normalization for the budget ledger.

Converts a provider TokenUsage into four DISJOINT fields so that ledger
events can be summed without double-counting tokens:

    uncached_input  — input tokens that were NOT served from cache
    cache_read      — tokens served from an existing cache entry (read hit)
    cache_write     — tokens written to create a new cache entry (write cost)
    output          — generated/completion tokens

Provider semantics differ in how they report cached tokens:

  OPENAI_COMPAT
    Providers such as Kimi/Moonshot, DeepSeek, and the standard OpenAI API
    use *subset* semantics: ``cache_read_tokens ⊆ input_tokens``.  The
    adapter stores ``input_tokens = prompt_tokens`` (the full prompt count
    including the cached portion).  To obtain disjoint fields:

        uncached_input = input_tokens - cache_read_tokens
        cache_read     = cache_read_tokens
        cache_write    = 0   (no write-cost field on these providers)
        output         = output_tokens

  ANTHROPIC
    Anthropic uses *additive* semantics: ``input_tokens`` already EXCLUDES
    the cache fields.  The adapter stores each field separately:

        uncached_input = input_tokens          (unchanged)
        cache_read     = cache_read_tokens
        cache_write    = cache_write_tokens
        output         = output_tokens

This module is the ONLY normalization entry point used by the ledger.
The ledger consumer should derive semantics from the provider's ``sdk``
attribute via ``ProviderSemantics.from_sdk(provider.sdk)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent_os.agent.providers.types import TokenUsage


class ProviderSemantics(str, Enum):
    """How a provider reports cached tokens relative to input_tokens.

    OPENAI_COMPAT — subset semantics: cache_read_tokens ⊆ input_tokens.
    ANTHROPIC     — additive semantics: input_tokens excludes cache fields.
    """

    OPENAI_COMPAT = "openai_compat"
    ANTHROPIC = "anthropic"

    @classmethod
    def from_sdk(cls, sdk: str) -> "ProviderSemantics":
        """Map ``LLMProvider.sdk`` ("openai" / "anthropic") to usage semantics.

        This is the documented entry point for the ledger consumer:
        ``ProviderSemantics.from_sdk(provider.sdk)``.
        """
        if sdk == "openai":
            return cls.OPENAI_COMPAT
        if sdk == "anthropic":
            return cls.ANTHROPIC
        raise ValueError(
            f"Unknown provider sdk {sdk!r}. Expected 'openai' or 'anthropic'."
        )


@dataclass(frozen=True)
class NormalizedUsage:
    """Four mutually-exclusive token categories; no token is counted twice."""

    uncached_input: int
    """Input tokens that were NOT served from cache."""

    cache_read: int
    """Input tokens served from an existing cache entry (billed at a lower rate)."""

    cache_write: int
    """Input tokens that created a new cache entry (billed at a higher rate)."""

    output: int
    """Generated / completion tokens."""

    @property
    def total(self) -> int:
        """Sum of all four fields — convenience for disjointness assertions."""
        return self.uncached_input + self.cache_read + self.cache_write + self.output


def normalize_usage(
    usage: TokenUsage, semantics: ProviderSemantics | str
) -> NormalizedUsage:
    """Convert a provider *TokenUsage* into disjoint *NormalizedUsage*.

    Args:
        usage:      Raw TokenUsage produced by the provider adapter.
        semantics:  A ProviderSemantics member, or its string value
                    (``"openai_compat"`` / ``"anthropic"``).

    Returns:
        NormalizedUsage with four mutually-exclusive token counts.

    Raises:
        ValueError: if *semantics* is not a recognised value.
    """
    semantics = ProviderSemantics(semantics)  # raises ValueError on unknown

    if semantics is ProviderSemantics.OPENAI_COMPAT:
        # cached_tokens ⊆ prompt_tokens → subtract to get uncached portion.
        # Clamped at 0: a negative delta means the provider violated the
        # subset contract (cache_read > input); never emit negative tokens.
        return NormalizedUsage(
            uncached_input=max(0, usage.input_tokens - usage.cache_read_tokens),
            cache_read=usage.cache_read_tokens,
            cache_write=0,  # OpenAI-compat providers have no cache-write field
            output=usage.output_tokens,
        )

    # ProviderSemantics.ANTHROPIC
    # input_tokens already excludes cache fields → copy through as-is.
    return NormalizedUsage(
        uncached_input=usage.input_tokens,
        cache_read=usage.cache_read_tokens,
        cache_write=usage.cache_write_tokens,
        output=usage.output_tokens,
    )
