# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model pricing lookup from providers.json."""

import json
import os

_PROVIDERS_JSON = os.path.join(os.path.dirname(__file__), "..", "config", "providers.json")

# Global fallback when provider has no pricing data at all
_GLOBAL_DEFAULT_INPUT_PER_1M = 3.0
_GLOBAL_DEFAULT_OUTPUT_PER_1M = 15.0

# Loaded once on first call
_pricing_cache: dict | None = None


def _load_pricing() -> dict:
    """Load pricing section from providers.json. Returns {provider: {model: {input_per_1m, output_per_1m}}}."""
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache
    try:
        with open(_PROVIDERS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        _pricing_cache = {}
        for provider_key, provider_info in data.get("providers", {}).items():
            pricing = provider_info.get("pricing", {})
            if pricing:
                _pricing_cache[provider_key] = pricing
    except (OSError, json.JSONDecodeError):
        _pricing_cache = {}
    return _pricing_cache


def get_cost_rates(model: str, provider: str = "custom") -> tuple[float, float]:
    """Return (cost_per_1k_input, cost_per_1k_output) for a model.

    Lookup order:
    1. Exact match in provider's pricing dict
    2. Prefix match (e.g. "claude-opus-4-6-20260301" matches "claude-opus-4-6")
    3. Provider "_default" entry
    4. Global fallback ($3.00/$15.00 per 1M tokens)

    Returns per-1K rates (per-1M / 1000) for compatibility with existing code.
    """
    pricing = _load_pricing()
    provider_pricing = pricing.get(provider, {})

    rates = None

    # 1. Exact match
    if model in provider_pricing:
        rates = provider_pricing[model]
    else:
        # 2. Prefix match: find the longest key that is a prefix of the model
        best_match = ""
        for key in provider_pricing:
            if key.startswith("_"):
                continue
            if model.startswith(key) and len(key) > len(best_match):
                best_match = key
        if best_match:
            rates = provider_pricing[best_match]

    # 3. Provider default
    if rates is None:
        rates = provider_pricing.get("_default")

    # 4. Global fallback
    if rates is None:
        input_per_1m = _GLOBAL_DEFAULT_INPUT_PER_1M
        output_per_1m = _GLOBAL_DEFAULT_OUTPUT_PER_1M
    else:
        input_per_1m = rates.get("input_per_1m", _GLOBAL_DEFAULT_INPUT_PER_1M)
        output_per_1m = rates.get("output_per_1m", _GLOBAL_DEFAULT_OUTPUT_PER_1M)

    # Convert per-1M to per-1K
    return input_per_1m / 1000, output_per_1m / 1000


# Default safety-net token budget when no dollar budget is configured
_SAFETY_NET_TOKEN_BUDGET = 100_000_000  # 100M tokens

# Assumed ratio of input to output tokens in agentic workloads.
# Input dominates because the growing conversation context is re-sent each turn.
_INPUT_TOKEN_RATIO = 0.85


def budget_usd_to_token_budget(
    budget_usd: float | None,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
    input_ratio: float = _INPUT_TOKEN_RATIO,
) -> int:
    """Convert a dollar budget to an approximate cumulative token budget.

    Uses a blended cost rate weighted by input_ratio (default 85% input,
    15% output) to account for agentic workloads where input tokens dominate
    due to context re-sending.

    Returns _SAFETY_NET_TOKEN_BUDGET when budget_usd is None (no cap set).
    """
    if budget_usd is None:
        return _SAFETY_NET_TOKEN_BUDGET

    if budget_usd <= 0:
        return 0

    blended_per_1k = input_ratio * cost_per_1k_input + (1 - input_ratio) * cost_per_1k_output
    if blended_per_1k <= 0:
        return _SAFETY_NET_TOKEN_BUDGET

    return int(budget_usd / blended_per_1k * 1000)
