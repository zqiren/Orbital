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
