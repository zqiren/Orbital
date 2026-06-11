# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the tiered rates table (Task 3).

Tests cover:
(a) Per-field override: override only one field → other three still from defaults
(b) Missing cache fields fall back to input_per_1m (conservative)
(c) Exact → prefix → _default chain still applies after merge
(d) Currency resolution (moonshot → CNY, anthropic → USD)
(e) Mtime invalidation: modifying override file changes rates without restart
(f) Legacy functions unchanged (pinned outputs for a couple of providers/models)
"""

import json
import os
import time

import pytest

import agent_os.agent.pricing as pricing_mod
from agent_os.agent.pricing import (
    ResolvedRates,
    resolve_rates,
    get_cost_rates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_override(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_pricing_caches(tmp_path):
    """Reset ALL pricing caches before and after each test."""
    pricing_mod._pricing_cache = None
    pricing_mod._full_providers_cache = None
    pricing_mod._override_cache = None
    pricing_mod._override_mtime = None
    yield
    pricing_mod._pricing_cache = None
    pricing_mod._full_providers_cache = None
    pricing_mod._override_cache = None
    pricing_mod._override_mtime = None


@pytest.fixture
def override_file(tmp_path):
    """Return path to a temp override file and inject it into the module."""
    path = str(tmp_path / "pricing-overrides.json")
    pricing_mod._OVERRIDE_FILE = path
    yield path
    # Restore default so other tests are unaffected
    pricing_mod._OVERRIDE_FILE = pricing_mod._default_override_path()


# ---------------------------------------------------------------------------
# (a) Per-field override: only override one field, others come from defaults
# ---------------------------------------------------------------------------

class TestPerFieldOverride:
    def test_override_only_cached_input(self, override_file):
        """Override only cached_input_per_1m; other fields come from defaults."""
        # anthropic claude-sonnet-4-6: input=3.0, output=15.0 per 1M
        # Anthropic cache: cached_input = 0.1 * input = 0.3, cache_write = 1.25 * input = 3.75
        # We override only cached_input_per_1m to 0.5
        _write_override(override_file, {
            "anthropic": {
                "claude-sonnet-4-6": {
                    "cached_input_per_1m": 0.5
                }
            }
        })

        rates = resolve_rates("anthropic", "claude-sonnet-4-6",
                              override_path=override_file)
        # Overridden field
        assert rates.cached_input_per_1m == pytest.approx(0.5)
        # Non-overridden fields from defaults
        assert rates.input_per_1m == pytest.approx(3.0)
        assert rates.output_per_1m == pytest.approx(15.0)
        # cache_write comes from default (1.25 * 3.0 = 3.75 for anthropic)
        assert rates.cache_write_per_1m == pytest.approx(3.75)

    def test_override_only_output(self, override_file):
        """Override only output_per_1m; input and cache fields come from defaults."""
        _write_override(override_file, {
            "anthropic": {
                "claude-haiku-4-5": {
                    "output_per_1m": 99.0
                }
            }
        })

        rates = resolve_rates("anthropic", "claude-haiku-4-5",
                              override_path=override_file)
        assert rates.output_per_1m == pytest.approx(99.0)
        # input_per_1m from defaults (1.0)
        assert rates.input_per_1m == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# (b) Missing cache fields fall back to input_per_1m
# ---------------------------------------------------------------------------

class TestCacheFallback:
    def test_provider_without_cache_rates_uses_input_fallback(self, override_file):
        """Provider with no cache rates uses input_per_1m as conservative fallback."""
        # openai has no cached_input_per_1m or cache_write_per_1m in defaults
        # gpt-5.2: input=1.75, output=14.0 per 1M, no cache rates
        _write_override(override_file, {})  # empty override

        rates = resolve_rates("openai", "gpt-5.2", override_path=override_file)
        # No cache rates in providers.json → fallback to input_per_1m
        assert rates.cached_input_per_1m == pytest.approx(1.75)
        assert rates.cache_write_per_1m == pytest.approx(1.75)

    def test_provider_with_partial_cache_rates(self, override_file):
        """When only cached_input_per_1m is present, cache_write falls back to input."""
        _write_override(override_file, {
            "openai": {
                "gpt-5.2": {
                    "cached_input_per_1m": 0.5
                }
            }
        })

        rates = resolve_rates("openai", "gpt-5.2", override_path=override_file)
        assert rates.cached_input_per_1m == pytest.approx(0.5)
        # cache_write still from base (no cache_write in providers.json → fallback to input)
        assert rates.cache_write_per_1m == pytest.approx(1.75)

    def test_anthropic_has_cache_rates_from_defaults(self, override_file):
        """Anthropic default rates include cache pricing (0.1x read, 1.25x write)."""
        _write_override(override_file, {})  # empty override

        # claude-sonnet-4-6: input=3.0, output=15.0
        rates = resolve_rates("anthropic", "claude-sonnet-4-6",
                              override_path=override_file)
        assert rates.input_per_1m == pytest.approx(3.0)
        assert rates.output_per_1m == pytest.approx(15.0)
        assert rates.cached_input_per_1m == pytest.approx(0.3)   # 0.1 * 3.0
        assert rates.cache_write_per_1m == pytest.approx(3.75)   # 1.25 * 3.0


# ---------------------------------------------------------------------------
# (c) Exact → prefix → _default chain still applies after merge
# ---------------------------------------------------------------------------

class TestResolutionChain:
    def test_exact_match_beats_prefix(self, override_file):
        """Exact match is used over prefix match."""
        _write_override(override_file, {})

        # claude-sonnet-4-6 is an exact match in anthropic pricing
        rates = resolve_rates("anthropic", "claude-sonnet-4-6",
                              override_path=override_file)
        assert rates.input_per_1m == pytest.approx(3.0)

    def test_prefix_match_used_for_versioned_model(self, override_file):
        """Model with date suffix matches via prefix."""
        _write_override(override_file, {})

        # claude-opus-4-6-20260301 should prefix-match claude-opus-4-6 (input=5.0)
        rates = resolve_rates("anthropic", "claude-opus-4-6-20260301",
                              override_path=override_file)
        assert rates.input_per_1m == pytest.approx(5.0)
        assert rates.output_per_1m == pytest.approx(25.0)

    def test_default_fallback_for_unknown_model(self, override_file):
        """Unknown model uses provider _default."""
        _write_override(override_file, {})

        # openai _default: input=2.5, output=15.0
        rates = resolve_rates("openai", "completely-unknown-model-xyz",
                              override_path=override_file)
        assert rates.input_per_1m == pytest.approx(2.5)
        assert rates.output_per_1m == pytest.approx(15.0)

    def test_override_applied_on_prefix_resolution(self, override_file):
        """Override is applied after prefix resolution, not just on exact matches."""
        _write_override(override_file, {
            "anthropic": {
                "claude-opus-4-6": {
                    "output_per_1m": 99.0
                }
            }
        })

        # claude-opus-4-6-versioned should prefix-match claude-opus-4-6
        # and then have the override applied
        rates = resolve_rates("anthropic", "claude-opus-4-6-20260301",
                              override_path=override_file)
        assert rates.output_per_1m == pytest.approx(99.0)
        assert rates.input_per_1m == pytest.approx(5.0)

    def test_global_fallback_unknown_provider(self, override_file):
        """Unknown provider falls back to global default rates (3.0/15.0 USD)."""
        _write_override(override_file, {})

        rates = resolve_rates("nonexistent_provider", "some-model",
                              override_path=override_file)
        assert rates.input_per_1m == pytest.approx(3.0)
        assert rates.output_per_1m == pytest.approx(15.0)
        assert rates.currency == "USD"


# ---------------------------------------------------------------------------
# (c2) _default in the override file participates in the chain
# ---------------------------------------------------------------------------

class TestOverrideDefaultEntry:
    """Regression: _default entries in the override file must NOT be silently
    dropped (the original filter stripped every key starting with '_', which
    was meant for the _currency meta-key only)."""

    def test_override_default_affects_unknown_model(self, override_file):
        """Override _default-only applies to a model with no defaults entry
        beyond the provider _default (chain rung 3 on both layers)."""
        _write_override(override_file, {
            "anthropic": {
                "_default": {
                    "output_per_1m": 999.0
                }
            }
        })

        # unknown-model-xyz resolves to _default on both layers.
        # Override _default field wins; other fields from defaults _default.
        rates = resolve_rates("anthropic", "unknown-model-xyz",
                              override_path=override_file)
        assert rates.output_per_1m == pytest.approx(999.0)
        # input from defaults _default (3.0), unchanged
        assert rates.input_per_1m == pytest.approx(3.0)
        # cache rates from defaults _default (anthropic ships them)
        assert rates.cached_input_per_1m == pytest.approx(0.3)
        assert rates.cache_write_per_1m == pytest.approx(3.75)

    def test_override_default_is_blanket_beats_defaults_exact(self, override_file):
        """PINNED SEMANTICS: the chain runs per-layer, so an override _default
        is a blanket per-provider override — its fields win per-field even
        over an exact-match entry in the defaults."""
        _write_override(override_file, {
            "anthropic": {
                "_default": {
                    "output_per_1m": 999.0
                }
            }
        })

        # claude-sonnet-4-6 is an EXACT match in defaults (3.0/15.0).
        # The override layer resolves to its _default → output field wins.
        rates = resolve_rates("anthropic", "claude-sonnet-4-6",
                              override_path=override_file)
        assert rates.output_per_1m == pytest.approx(999.0)
        # All other fields still from the defaults exact entry.
        assert rates.input_per_1m == pytest.approx(3.0)
        assert rates.cached_input_per_1m == pytest.approx(0.3)
        assert rates.cache_write_per_1m == pytest.approx(3.75)

    def test_override_exact_beats_override_default(self, override_file):
        """Within the override layer the chain still prefers exact over
        _default."""
        _write_override(override_file, {
            "anthropic": {
                "_default": {
                    "output_per_1m": 999.0
                },
                "claude-sonnet-4-6": {
                    "output_per_1m": 111.0
                }
            }
        })

        rates = resolve_rates("anthropic", "claude-sonnet-4-6",
                              override_path=override_file)
        assert rates.output_per_1m == pytest.approx(111.0)
        # A different model falls to the override _default
        rates2 = resolve_rates("anthropic", "claude-haiku-4-5",
                               override_path=override_file)
        assert rates2.output_per_1m == pytest.approx(999.0)
        assert rates2.input_per_1m == pytest.approx(1.0)  # defaults exact

    def test_currency_meta_key_excluded_from_chain(self, override_file):
        """_currency is a meta-key: it sets currency but never participates
        in model-entry resolution (rates unaffected by its presence)."""
        _write_override(override_file, {
            "anthropic": {
                "_currency": "EUR"
            }
        })

        rates = resolve_rates("anthropic", "claude-sonnet-4-6",
                              override_path=override_file)
        # Currency applied
        assert rates.currency == "EUR"
        # All rates untouched (defaults exact entry)
        assert rates.input_per_1m == pytest.approx(3.0)
        assert rates.output_per_1m == pytest.approx(15.0)
        assert rates.cached_input_per_1m == pytest.approx(0.3)
        assert rates.cache_write_per_1m == pytest.approx(3.75)
        # Unknown model: _currency must not act as a _default-style rate entry
        rates2 = resolve_rates("anthropic", "unknown-model-xyz",
                               override_path=override_file)
        assert rates2.input_per_1m == pytest.approx(3.0)
        assert rates2.output_per_1m == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# (d) Currency resolution
# ---------------------------------------------------------------------------

class TestCurrencyResolution:
    def test_moonshot_currency_is_cny(self, override_file):
        """Moonshot provider returns CNY."""
        _write_override(override_file, {})

        rates = resolve_rates("moonshot", "kimi-k2.5", override_path=override_file)
        assert rates.currency == "CNY"

    def test_anthropic_currency_is_usd(self, override_file):
        """Anthropic provider returns USD."""
        _write_override(override_file, {})

        rates = resolve_rates("anthropic", "claude-sonnet-4-6",
                              override_path=override_file)
        assert rates.currency == "USD"

    def test_openai_currency_is_usd(self, override_file):
        """OpenAI provider returns USD."""
        _write_override(override_file, {})

        rates = resolve_rates("openai", "gpt-5.2", override_path=override_file)
        assert rates.currency == "USD"

    def test_deepseek_currency_is_usd(self, override_file):
        """DeepSeek provider returns USD."""
        _write_override(override_file, {})

        rates = resolve_rates("deepseek", "deepseek-v4-pro",
                              override_path=override_file)
        assert rates.currency == "USD"

    def test_currency_override_in_override_file(self, override_file):
        """Override file can change currency."""
        _write_override(override_file, {
            "openai": {
                "_currency": "EUR"
            }
        })

        rates = resolve_rates("openai", "gpt-5.2", override_path=override_file)
        assert rates.currency == "EUR"


# ---------------------------------------------------------------------------
# (e) Mtime invalidation
# ---------------------------------------------------------------------------

class TestMtimeInvalidation:
    def test_new_rates_after_file_modification(self, override_file):
        """After modifying override file, resolve_rates returns new values."""
        # First write: no override
        _write_override(override_file, {})
        rates1 = resolve_rates("openai", "gpt-5.2", override_path=override_file)
        assert rates1.input_per_1m == pytest.approx(1.75)

        # Bump mtime by 2 seconds to ensure cache invalidation
        stat = os.stat(override_file)
        new_mtime = stat.st_mtime + 2.0
        os.utime(override_file, (new_mtime, new_mtime))

        # Write new override with different rate
        _write_override(override_file, {
            "openai": {
                "gpt-5.2": {
                    "input_per_1m": 99.0
                }
            }
        })
        # Set mtime ahead again so cache sees the change
        stat2 = os.stat(override_file)
        newer_mtime = stat2.st_mtime + 2.0
        os.utime(override_file, (newer_mtime, newer_mtime))

        rates2 = resolve_rates("openai", "gpt-5.2", override_path=override_file)
        assert rates2.input_per_1m == pytest.approx(99.0)

    def test_same_mtime_uses_cache(self, override_file):
        """Same mtime returns cached result without re-reading file."""
        _write_override(override_file, {
            "openai": {
                "gpt-5.2": {
                    "input_per_1m": 42.0
                }
            }
        })
        rates1 = resolve_rates("openai", "gpt-5.2", override_path=override_file)
        assert rates1.input_per_1m == pytest.approx(42.0)

        # Overwrite with different content but keep the same mtime
        stat = os.stat(override_file)
        _write_override(override_file, {
            "openai": {
                "gpt-5.2": {
                    "input_per_1m": 999.0
                }
            }
        })
        os.utime(override_file, (stat.st_mtime, stat.st_mtime))

        # Should still return old cached value
        rates2 = resolve_rates("openai", "gpt-5.2", override_path=override_file)
        assert rates2.input_per_1m == pytest.approx(42.0)

    def test_nonexistent_override_file_uses_defaults(self, tmp_path):
        """If override file doesn't exist, defaults are used without error."""
        nonexistent = str(tmp_path / "no-such-file.json")

        rates = resolve_rates("openai", "gpt-5.2", override_path=nonexistent)
        assert rates.input_per_1m == pytest.approx(1.75)
        assert rates.currency == "USD"


# ---------------------------------------------------------------------------
# (f) Legacy functions unchanged
# ---------------------------------------------------------------------------

class TestLegacyFunctionsUnchanged:
    """Verify get_cost_rates produces byte-identical outputs to before.

    (``budget_usd_to_token_budget`` was deleted in Budget Piece 2 — the
    dollar→token derivation was a second budget-blocking path; the byte-identity
    tests for it were removed with the function.)"""

    def test_get_cost_rates_anthropic_sonnet(self):
        """Anthropic sonnet pricing unchanged."""
        ci, co = get_cost_rates("claude-sonnet-4-6", "anthropic")
        assert ci == pytest.approx(3.0 / 1000)   # 0.003
        assert co == pytest.approx(15.0 / 1000)  # 0.015

    def test_get_cost_rates_openai_gpt4o(self):
        """OpenAI gpt-4o pricing unchanged."""
        ci, co = get_cost_rates("gpt-4o", "openai")
        assert ci == pytest.approx(2.5 / 1000)
        assert co == pytest.approx(10.0 / 1000)

    def test_get_cost_rates_deepseek_v4_pro(self):
        """DeepSeek v4-pro pricing unchanged."""
        ci, co = get_cost_rates("deepseek-v4-pro", "deepseek")
        assert ci == pytest.approx(0.435 / 1000)
        assert co == pytest.approx(0.87 / 1000)

    def test_get_cost_rates_prefix_match_unchanged(self):
        """Prefix match still works for legacy function."""
        ci, co = get_cost_rates("claude-opus-4-6-20260301", "anthropic")
        assert ci == pytest.approx(5.0 / 1000)
        assert co == pytest.approx(25.0 / 1000)

    def test_get_cost_rates_global_fallback_unchanged(self):
        """Global fallback unchanged for unknown providers."""
        ci, co = get_cost_rates("some-model", "nonexistent_provider")
        assert ci == pytest.approx(3.0 / 1000)
        assert co == pytest.approx(15.0 / 1000)


# ---------------------------------------------------------------------------
# ResolvedRates dataclass
# ---------------------------------------------------------------------------

class TestResolvedRates:
    def test_is_frozen_dataclass(self, override_file):
        """ResolvedRates is frozen (immutable)."""
        _write_override(override_file, {})
        rates = resolve_rates("anthropic", "claude-sonnet-4-6",
                              override_path=override_file)
        with pytest.raises((AttributeError, TypeError)):
            rates.input_per_1m = 999.0  # type: ignore

    def test_has_all_five_fields(self, override_file):
        """ResolvedRates has all five required fields."""
        _write_override(override_file, {})
        rates = resolve_rates("openai", "gpt-5.2", override_path=override_file)
        # Verify all fields exist
        assert hasattr(rates, "input_per_1m")
        assert hasattr(rates, "cached_input_per_1m")
        assert hasattr(rates, "cache_write_per_1m")
        assert hasattr(rates, "output_per_1m")
        assert hasattr(rates, "currency")
