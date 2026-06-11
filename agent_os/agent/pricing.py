# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model pricing lookup from providers.json.

Two public interfaces live here:

1. **Legacy interface** (``get_cost_rates`` only — the per-1K rate lookup):
   - ``get_cost_rates(model, provider)`` → ``(cost_per_1k_input, cost_per_1k_output)``
   (``budget_usd_to_token_budget`` was removed in Budget Piece 2; see the note
   at its former call site below.)

2. **Tiered-rates interface** (new, consumed by the cost-ledger module):
   - ``resolve_rates(provider, model)`` → ``ResolvedRates``

   ``ResolvedRates`` is a frozen dataclass with five fields:
   ``input_per_1m``, ``cached_input_per_1m``, ``cache_write_per_1m``,
   ``output_per_1m``, ``currency``.

Override file
-------------
Users may place a sparse JSON file at the *override path* to override rates
per provider/model.  The schema mirrors the ``pricing`` blocks in
``providers.json``: a dict keyed by provider, then model (or ``_default``),
with any subset of the four per-1M rate fields.  A ``_currency`` key at
the provider level overrides the default currency for that provider.

Example::

    {
      "anthropic": {
        "_currency": "USD",
        "claude-sonnet-4-6": { "output_per_1m": 12.0 }
      }
    }

Resolution order (per field)
-----------------------------
1. Override file: exact model match → prefix match → ``_default``
2. Defaults (providers.json): exact match → prefix match → ``_default``
3. Global fallback (3.0/15.0 USD, no cache discount)

Missing ``cached_input_per_1m`` / ``cache_write_per_1m`` fall back to
``input_per_1m`` (conservative — no cache discount).

Merge semantics
---------------
The exact → prefix → ``_default`` chain runs **independently on each layer**,
and the two resolved entries are then merged per-field (override entry wins
where a field is present, defaults fill the rest).  Consequences:

- An override ``_default`` entry acts as a **blanket per-provider override**:
  its fields apply to every model of that provider — including models that
  resolve via an exact or prefix match in the defaults — because the override
  chain resolves to ``_default`` for any model without a more specific
  override key.  ``{"anthropic": {"_default": {"output_per_1m": 9.0}}}``
  therefore changes ``output_per_1m`` for ALL anthropic models while their
  other fields still come from each model's own defaults entry.
- This per-layer design preserves the per-field guarantee regardless of key
  granularity: an override exact key carrying a single field (e.g. a pinned
  snapshot name not present in the defaults) still gets its remaining fields
  from whatever the defaults chain resolves for that model.  (Deep-merging
  the two tables before running the chain would instead let such a key
  shadow the defaults' prefix entry and lose its other fields.)
- ``_default`` participates in the chain on both layers.  The only
  override-file key excluded from the chain is the ``_currency`` meta-key.

Cache invalidation
------------------
The override file is checked on every ``resolve_rates()`` call via an mtime
comparison.  If the file's mtime has changed since the last load, the override
cache is invalidated and the file is re-read.  This means *editing
pricing-overrides.json changes reported cost without a daemon restart*.

The providers.json defaults cache (``_pricing_cache``) is loaded once and
never invalidated (it is a checked-in file that only changes via a code
deploy / daemon restart).

Path resolution
---------------
The default override path is ``{AGENT_OS_DATA_DIR}/pricing-overrides.json``
where ``AGENT_OS_DATA_DIR`` comes from the ``AGENT_OS_DATA_DIR`` environment
variable (set by the desktop app) or falls back to the platform-specific
application data directory.  The path can be injected via the ``override_path``
parameter of ``resolve_rates()`` for testing.

Nothing outside this module reads either file directly.
"""

from __future__ import annotations

import json
import os
import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PROVIDERS_JSON = os.path.join(os.path.dirname(__file__), "..", "config", "providers.json")

# Global fallback when provider has no pricing data at all
_GLOBAL_DEFAULT_INPUT_PER_1M = 3.0
_GLOBAL_DEFAULT_OUTPUT_PER_1M = 15.0
_GLOBAL_DEFAULT_CURRENCY = "USD"

# ---------------------------------------------------------------------------
# Legacy cache (providers.json defaults, loaded once)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Full providers cache (includes currency, all rate fields)
# ---------------------------------------------------------------------------

_full_providers_cache: dict | None = None


def _load_full_providers() -> dict:
    """Load full providers dict from providers.json (includes currency + cache rates).

    Returns dict keyed by provider_key → {currency, pricing: {model: {rates...}}}.
    Loaded once and never invalidated (checked-in file).
    """
    global _full_providers_cache
    if _full_providers_cache is not None:
        return _full_providers_cache
    try:
        with open(_PROVIDERS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        _full_providers_cache = {}
        for provider_key, provider_info in data.get("providers", {}).items():
            _full_providers_cache[provider_key] = {
                "currency": provider_info.get("currency", _GLOBAL_DEFAULT_CURRENCY),
                "pricing": provider_info.get("pricing", {}),
            }
    except (OSError, json.JSONDecodeError):
        _full_providers_cache = {}
    return _full_providers_cache


# ---------------------------------------------------------------------------
# Override file cache (mtime-invalidated)
# ---------------------------------------------------------------------------

_override_cache: dict | None = None
_override_mtime: float | None = None


def _default_override_path() -> str:
    """Return the default path for the user override file.

    Uses AGENT_OS_DATA_DIR env var (set by the desktop app) or falls back
    to the platform-appropriate application data directory.
    """
    data_dir = os.environ.get("AGENT_OS_DATA_DIR")
    if not data_dir:
        if sys.platform == "win32":
            data_dir = os.path.join(
                os.environ.get("APPDATA", os.path.expanduser("~")), "Orbital"
            )
        elif sys.platform == "darwin":
            data_dir = os.path.join(
                os.path.expanduser("~"), "Library", "Application Support", "Orbital"
            )
        else:
            data_dir = os.path.join(os.path.expanduser("~"), ".orbital")
    return os.path.join(data_dir, "pricing-overrides.json")


# Module-level override path (injectable for tests via override_path param or
# by setting this variable directly)
_OVERRIDE_FILE: str = _default_override_path()


def _load_override(path: str) -> dict:
    """Load override file, using mtime-based cache invalidation.

    Returns {} if the file doesn't exist or is unparseable.
    """
    global _override_cache, _override_mtime

    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        # File doesn't exist — return empty without caching so we re-check next call
        _override_cache = {}
        _override_mtime = None
        return {}

    if _override_cache is not None and _override_mtime == mtime:
        return _override_cache

    # File exists and mtime changed (or first load)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError) as e:
        # User-authored file: surface the problem instead of silently billing
        # at default rates.
        logger.warning("pricing overrides file unparseable, ignoring: %s (%s)", path, e)
        data = {}

    _override_cache = data
    _override_mtime = mtime
    return _override_cache


# ---------------------------------------------------------------------------
# ResolvedRates dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedRates:
    """Per-model rates resolved from defaults + override layer.

    All per-1M rates are in the provider's native currency (see ``currency``).
    Missing cache rates conservatively fall back to ``input_per_1m``
    (no discount assumed).
    """
    input_per_1m: float
    cached_input_per_1m: float
    cache_write_per_1m: float
    output_per_1m: float
    currency: str  # ISO 4217 (e.g. "USD", "CNY")


# ---------------------------------------------------------------------------
# Internal: model entry lookup (exact → prefix → _default)
# ---------------------------------------------------------------------------

def _find_entry(pricing_dict: dict, model: str) -> dict | None:
    """Resolve a model key in a pricing dict using exact → prefix → _default."""
    if not pricing_dict:
        return None

    # 1. Exact match
    if model in pricing_dict:
        return dict(pricing_dict[model])

    # 2. Prefix match: longest key that is a prefix of model (skip _ keys)
    best_match = ""
    for key in pricing_dict:
        if key.startswith("_"):
            continue
        if model.startswith(key) and len(key) > len(best_match):
            best_match = key
    if best_match:
        return dict(pricing_dict[best_match])

    # 3. Provider _default
    default = pricing_dict.get("_default")
    if default is not None:
        return dict(default)

    return None


# ---------------------------------------------------------------------------
# Main resolution function
# ---------------------------------------------------------------------------

def resolve_rates(
    provider: str,
    model: str,
    override_path: str | None = None,
) -> ResolvedRates:
    """Resolve all four per-1M rates + currency for a (provider, model) pair.

    Resolution per field:
    1. Override file (override_path or module-level _OVERRIDE_FILE):
       exact → prefix → _default chain
    2. Defaults from providers.json:
       exact → prefix → _default chain
    3. Global fallback (3.0/15.0 USD, no cache discount)

    The chain runs independently on each layer; the two resolved entries are
    merged per-field with the override winning where present.  An override
    ``_default`` therefore acts as a blanket per-provider override — its
    fields beat even exact-match defaults entries (see "Merge semantics" in
    the module docstring).

    Missing ``cached_input_per_1m`` / ``cache_write_per_1m`` fall back to
    ``input_per_1m`` (conservative: no cache discount assumed).

    Currency: taken from providers.json ``currency`` field, overridable via
    ``_currency`` key in the override file's provider block.

    Args:
        provider: Provider key (e.g. "anthropic", "openai")
        model: Model key (e.g. "claude-sonnet-4-6")
        override_path: Path to the override file. Defaults to
            ``_OVERRIDE_FILE`` (module-level, set from AGENT_OS_DATA_DIR).
    """
    path = override_path if override_path is not None else _OVERRIDE_FILE

    # Load full defaults (providers.json)
    full = _load_full_providers()
    provider_data = full.get(provider, {})
    default_pricing = provider_data.get("pricing", {})
    default_currency = provider_data.get("currency", _GLOBAL_DEFAULT_CURRENCY)

    # Load override. Only the _currency meta-key is excluded from the model
    # chain — _default MUST survive so the override layer's exact → prefix →
    # _default chain works (see "Merge semantics" in the module docstring).
    overrides = _load_override(path)
    override_provider = overrides.get(provider, {})
    override_pricing = {k: v for k, v in override_provider.items() if k != "_currency"}
    override_currency = override_provider.get("_currency")

    # Resolve currency
    currency = override_currency if override_currency else default_currency

    # Resolve base entry from defaults
    default_entry = _find_entry(default_pricing, model) or {}

    # Resolve override entry
    override_entry = _find_entry(override_pricing, model) if override_pricing else {}
    if override_entry is None:
        override_entry = {}

    # Merge: override wins per-field where present
    merged = {**default_entry, **override_entry}

    # Extract rates with fallback chain
    input_per_1m = merged.get(
        "input_per_1m",
        _GLOBAL_DEFAULT_INPUT_PER_1M,
    )
    output_per_1m = merged.get(
        "output_per_1m",
        _GLOBAL_DEFAULT_OUTPUT_PER_1M,
    )
    # Cache rates: fall back to input_per_1m if not specified (conservative)
    cached_input_per_1m = merged.get("cached_input_per_1m", input_per_1m)
    cache_write_per_1m = merged.get("cache_write_per_1m", input_per_1m)

    return ResolvedRates(
        input_per_1m=input_per_1m,
        cached_input_per_1m=cached_input_per_1m,
        cache_write_per_1m=cache_write_per_1m,
        output_per_1m=output_per_1m,
        currency=currency,
    )


# ---------------------------------------------------------------------------
# Per-field origin (additive — shares ALL internals with resolve_rates)
#
# Consumed by the pricing-table API (GET /api/v2/pricing) so the editor UI can
# render which fields are user overrides vs. checked-in defaults, and offer a
# per-field revert. This does NOT reimplement the merge: it calls the same
# _load_full_providers / _load_override / _find_entry internals and derives the
# origin from which layer supplied each field. The numeric value it reports for
# every field is byte-identical to what resolve_rates() returns.
# ---------------------------------------------------------------------------

# The four per-1M rate field names, in editor-grid order.
RATE_FIELDS = (
    "input_per_1m",
    "cached_input_per_1m",
    "cache_write_per_1m",
    "output_per_1m",
)


def resolve_rates_with_origin(
    provider: str,
    model: str,
    override_path: str | None = None,
) -> tuple[ResolvedRates, dict]:
    """Resolve rates AND report the per-field origin (default vs override).

    Returns ``(rates, origin)`` where ``rates`` is byte-identical to
    ``resolve_rates(provider, model, override_path)`` and ``origin`` is::

        {
          "input_per_1m": "default" | "override",
          "cached_input_per_1m": ...,
          "cache_write_per_1m": ...,
          "output_per_1m": ...,
          "currency": "default" | "override",
        }

    A field's origin is ``"override"`` iff the override layer's resolved entry
    (exact → prefix → ``_default`` chain) supplies that exact field — i.e. the
    field key is present in the override entry that wins the per-field merge in
    ``resolve_rates``. Because the two layers resolve independently and merge
    per-field, an override ``_default`` carrying a single field marks ONLY that
    field as ``"override"`` for every model of the provider (blanket per-field
    override); the remaining fields keep ``"default"`` origin. The cache-rate
    fallback (``cached_input_per_1m`` / ``cache_write_per_1m`` → ``input_per_1m``
    when absent in both layers) is treated as ``"default"`` origin, since the
    value derives from the defaults layer's ``input_per_1m``, not a user edit.

    ``currency`` origin is ``"override"`` iff the provider's ``_currency``
    meta-key is set in the override file.
    """
    path = override_path if override_path is not None else _OVERRIDE_FILE

    full = _load_full_providers()
    provider_data = full.get(provider, {})
    default_pricing = provider_data.get("pricing", {})
    default_currency = provider_data.get("currency", _GLOBAL_DEFAULT_CURRENCY)

    overrides = _load_override(path)
    override_provider = overrides.get(provider, {})
    override_pricing = {k: v for k, v in override_provider.items() if k != "_currency"}
    override_currency = override_provider.get("_currency")

    currency = override_currency if override_currency else default_currency

    default_entry = _find_entry(default_pricing, model) or {}
    override_entry = _find_entry(override_pricing, model) if override_pricing else {}
    if override_entry is None:
        override_entry = {}

    merged = {**default_entry, **override_entry}

    input_per_1m = merged.get("input_per_1m", _GLOBAL_DEFAULT_INPUT_PER_1M)
    output_per_1m = merged.get("output_per_1m", _GLOBAL_DEFAULT_OUTPUT_PER_1M)
    cached_input_per_1m = merged.get("cached_input_per_1m", input_per_1m)
    cache_write_per_1m = merged.get("cache_write_per_1m", input_per_1m)

    rates = ResolvedRates(
        input_per_1m=input_per_1m,
        cached_input_per_1m=cached_input_per_1m,
        cache_write_per_1m=cache_write_per_1m,
        output_per_1m=output_per_1m,
        currency=currency,
    )

    origin = {
        field: ("override" if field in override_entry else "default")
        for field in RATE_FIELDS
    }
    origin["currency"] = "override" if override_currency else "default"
    return rates, origin


def pricing_table_with_origin(override_path: str | None = None) -> dict:
    """Build the resolved pricing table with per-field origin for every model.

    Covers EVERY provider/model that has pricing in providers.json PLUS any
    extra provider/model keys present only in the override file (forward-compat
    — a user may pin a snapshot model name not yet in the checked-in defaults).
    Resolution for each (provider, model) goes through
    ``resolve_rates_with_origin``, so values agree exactly with
    ``resolve_rates``.

    Shape (codes only, designed for the editor consumer)::

        {
          "providers": {
            "anthropic": {
              "currency": "USD",
              "currency_origin": "default" | "override",
              "models": {
                "claude-sonnet-4-6": {
                  "input_per_1m":        {"value": 3.0,  "origin": "default"},
                  "cached_input_per_1m": {"value": 0.3,  "origin": "default"},
                  "cache_write_per_1m":  {"value": 3.75, "origin": "default"},
                  "output_per_1m":       {"value": 15.0, "origin": "default"}
                },
                "_default": { ...same 4-field grid... }
              }
            }
          }
        }

    The ``_default`` pseudo-model is included verbatim as a model key so the
    editor can present (and override) the provider-wide fallback row. Model
    keys are sorted with ``_default`` last for stable rendering.
    """
    full = _load_full_providers()
    path = override_path if override_path is not None else _OVERRIDE_FILE
    overrides = _load_override(path)

    # Union of provider keys: defaults ∪ override file.
    provider_keys = set(full.keys()) | set(
        k for k in overrides.keys() if isinstance(overrides.get(k), dict)
    )

    out: dict = {"providers": {}}
    for provider in sorted(provider_keys):
        default_pricing = full.get(provider, {}).get("pricing", {})
        override_provider = overrides.get(provider, {})
        if not isinstance(override_provider, dict):
            override_provider = {}
        override_pricing = {
            k: v for k, v in override_provider.items() if k != "_currency"
        }

        # Union of model keys: defaults ∪ override (forward-compat for
        # override-only model names, including a pinned snapshot).
        model_keys = set(default_pricing.keys()) | set(override_pricing.keys())

        models_out: dict = {}
        for model in _sorted_model_keys(model_keys):
            rates, origin = resolve_rates_with_origin(
                provider, model, override_path=path
            )
            models_out[model] = {
                "input_per_1m": {
                    "value": rates.input_per_1m,
                    "origin": origin["input_per_1m"],
                },
                "cached_input_per_1m": {
                    "value": rates.cached_input_per_1m,
                    "origin": origin["cached_input_per_1m"],
                },
                "cache_write_per_1m": {
                    "value": rates.cache_write_per_1m,
                    "origin": origin["cache_write_per_1m"],
                },
                "output_per_1m": {
                    "value": rates.output_per_1m,
                    "origin": origin["output_per_1m"],
                },
            }

        # Currency: resolve once at provider level (model-independent).
        _, prov_origin = resolve_rates_with_origin(
            provider, "_default", override_path=path
        )
        default_currency = full.get(provider, {}).get(
            "currency", _GLOBAL_DEFAULT_CURRENCY
        )
        override_currency = override_provider.get("_currency")
        out["providers"][provider] = {
            "currency": override_currency if override_currency else default_currency,
            "currency_origin": prov_origin["currency"],
            "models": models_out,
        }
    return out


def _sorted_model_keys(model_keys) -> list[str]:
    """Sort model keys alphabetically with ``_default`` pinned last."""
    real = sorted(k for k in model_keys if k != "_default")
    if "_default" in model_keys:
        real.append("_default")
    return real


def override_file_path() -> str:
    """Return the resolved override-file path currently in effect.

    Exposes the module-level ``_OVERRIDE_FILE`` (set from ``AGENT_OS_DATA_DIR``
    at import, injectable by tests). The pricing API surfaces this as a
    read-only debug field so operators can see which file a write targets.
    """
    return _OVERRIDE_FILE


# ---------------------------------------------------------------------------
# Legacy interface (unchanged)
# ---------------------------------------------------------------------------

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


# NOTE: ``budget_usd_to_token_budget`` was DELETED in Budget Piece 2. It
# converted a dollar budget into a cumulative token cap, feeding the loop's
# token gate as a SECOND budget-blocking path. Piece 2 mandates exactly ONE
# path that can block on budget (the pre-flight guard in ``budget/guard.py``
# reading the ledger), so the dollar→token derivation and this helper are gone.
# The loop's token gate remains a PURE non-budget safety-net cap fed by
# ``AgentConfig.token_budget`` (default 100M).
