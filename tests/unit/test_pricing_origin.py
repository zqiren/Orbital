# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the per-field origin helpers (Task P3-D).

Covers ``resolve_rates_with_origin`` and ``pricing_table_with_origin``:

(a) default-only model → every field origin "default", value == resolve_rates
(b) per-field override → only the touched field is "override"; others "default"
(c) override-only model (not in providers.json defaults) appears in the table
(d) override ``_default`` is a blanket per-field override (origin semantics
    reuse the merge documentation in pricing.py's docstring)
(e) ``_currency`` sets currency_origin "override" without touching rate origins
(f) the reported value is byte-identical to resolve_rates for every field
"""

import json

import pytest

import agent_os.agent.pricing as pricing_mod
from agent_os.agent.pricing import (
    resolve_rates,
    resolve_rates_with_origin,
    pricing_table_with_origin,
)


def _write_override(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@pytest.fixture(autouse=True)
def _reset_pricing_caches():
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
    path = str(tmp_path / "pricing-overrides.json")
    pricing_mod._OVERRIDE_FILE = path
    yield path
    pricing_mod._OVERRIDE_FILE = pricing_mod._default_override_path()


# ---------------------------------------------------------------------------
# (a) default-only model
# ---------------------------------------------------------------------------

class TestDefaultOnlyModel:
    def test_all_fields_default_origin(self, override_file):
        _write_override(override_file, {})
        rates, origin = resolve_rates_with_origin(
            "anthropic", "claude-sonnet-4-6", override_path=override_file
        )
        assert origin["input_per_1m"] == "default"
        assert origin["cached_input_per_1m"] == "default"
        assert origin["cache_write_per_1m"] == "default"
        assert origin["output_per_1m"] == "default"
        assert origin["currency"] == "default"

    def test_value_matches_resolve_rates(self, override_file):
        _write_override(override_file, {})
        rates, _ = resolve_rates_with_origin(
            "anthropic", "claude-sonnet-4-6", override_path=override_file
        )
        ref = resolve_rates(
            "anthropic", "claude-sonnet-4-6", override_path=override_file
        )
        assert rates == ref

    def test_cache_fallback_field_is_default_origin(self, override_file):
        """openai has no cache fields → cached_input falls back to input but
        origin stays 'default' (value derives from the defaults layer)."""
        _write_override(override_file, {})
        rates, origin = resolve_rates_with_origin(
            "openai", "gpt-5.2", override_path=override_file
        )
        assert rates.cached_input_per_1m == pytest.approx(rates.input_per_1m)
        assert origin["cached_input_per_1m"] == "default"
        assert origin["cache_write_per_1m"] == "default"


# ---------------------------------------------------------------------------
# (b) per-field override
# ---------------------------------------------------------------------------

class TestPerFieldOrigin:
    def test_only_overridden_field_is_override(self, override_file):
        _write_override(override_file, {
            "anthropic": {"claude-sonnet-4-6": {"output_per_1m": 99.0}}
        })
        rates, origin = resolve_rates_with_origin(
            "anthropic", "claude-sonnet-4-6", override_path=override_file
        )
        assert origin["output_per_1m"] == "override"
        assert rates.output_per_1m == pytest.approx(99.0)
        # Untouched fields stay default.
        assert origin["input_per_1m"] == "default"
        assert origin["cached_input_per_1m"] == "default"
        assert origin["cache_write_per_1m"] == "default"
        # And their values still match resolve_rates.
        ref = resolve_rates(
            "anthropic", "claude-sonnet-4-6", override_path=override_file
        )
        assert rates == ref

    def test_override_on_prefix_resolution(self, override_file):
        """Override applied via prefix match marks that field 'override' for a
        versioned model name too."""
        _write_override(override_file, {
            "anthropic": {"claude-opus-4-6": {"output_per_1m": 77.0}}
        })
        rates, origin = resolve_rates_with_origin(
            "anthropic", "claude-opus-4-6-20260301", override_path=override_file
        )
        assert origin["output_per_1m"] == "override"
        assert origin["input_per_1m"] == "default"
        assert rates.output_per_1m == pytest.approx(77.0)


# ---------------------------------------------------------------------------
# (c) override-only model appears in the table
# ---------------------------------------------------------------------------

class TestOverrideOnlyModelInTable:
    def test_pinned_snapshot_model_appears(self, override_file):
        """A model name present ONLY in the override file (e.g. a pinned
        snapshot not in providers.json) shows up in the resolved table."""
        _write_override(override_file, {
            "anthropic": {
                "claude-sonnet-4-6-99991231": {"input_per_1m": 2.0}
            }
        })
        table = pricing_table_with_origin(override_path=override_file)
        models = table["providers"]["anthropic"]["models"]
        assert "claude-sonnet-4-6-99991231" in models
        cell = models["claude-sonnet-4-6-99991231"]
        assert cell["input_per_1m"]["value"] == pytest.approx(2.0)
        assert cell["input_per_1m"]["origin"] == "override"
        # Its other fields fill from the defaults prefix entry (claude-sonnet-4-6).
        assert cell["output_per_1m"]["value"] == pytest.approx(15.0)
        assert cell["output_per_1m"]["origin"] == "default"

    def test_override_only_provider_appears(self, override_file):
        """A provider present ONLY in the override file appears too."""
        _write_override(override_file, {
            "brand-new-provider": {
                "some-model": {"input_per_1m": 1.0, "output_per_1m": 2.0}
            }
        })
        table = pricing_table_with_origin(override_path=override_file)
        assert "brand-new-provider" in table["providers"]
        models = table["providers"]["brand-new-provider"]["models"]
        assert "some-model" in models
        assert models["some-model"]["input_per_1m"]["origin"] == "override"

    def test_table_covers_every_default_provider_and_model(self, override_file):
        """The table must include every provider/model that has pricing in
        providers.json (no override needed)."""
        _write_override(override_file, {})
        table = pricing_table_with_origin(override_path=override_file)
        full = pricing_mod._load_full_providers()
        for provider, data in full.items():
            assert provider in table["providers"], provider
            out_models = table["providers"][provider]["models"]
            for model in data.get("pricing", {}):
                assert model in out_models, f"{provider}/{model}"

    def test_default_pseudo_model_present_and_last(self, override_file):
        _write_override(override_file, {})
        table = pricing_table_with_origin(override_path=override_file)
        keys = list(table["providers"]["anthropic"]["models"].keys())
        assert "_default" in keys
        assert keys[-1] == "_default"


# ---------------------------------------------------------------------------
# (d) override _default blanket semantics
# ---------------------------------------------------------------------------

class TestOverrideDefaultBlanketOrigin:
    def test_blanket_marks_only_that_field_override(self, override_file):
        """An override _default carrying one field marks ONLY that field
        'override' for EVERY model — even an exact-match defaults model."""
        _write_override(override_file, {
            "anthropic": {"_default": {"output_per_1m": 999.0}}
        })
        # Exact-match model in defaults still gets the blanket field.
        rates, origin = resolve_rates_with_origin(
            "anthropic", "claude-sonnet-4-6", override_path=override_file
        )
        assert origin["output_per_1m"] == "override"
        assert rates.output_per_1m == pytest.approx(999.0)
        # Other fields untouched.
        assert origin["input_per_1m"] == "default"
        assert origin["cached_input_per_1m"] == "default"
        assert origin["cache_write_per_1m"] == "default"
        # Byte-identical to resolve_rates.
        assert rates == resolve_rates(
            "anthropic", "claude-sonnet-4-6", override_path=override_file
        )

    def test_blanket_applies_across_models_in_table(self, override_file):
        _write_override(override_file, {
            "anthropic": {"_default": {"output_per_1m": 999.0}}
        })
        table = pricing_table_with_origin(override_path=override_file)
        models = table["providers"]["anthropic"]["models"]
        for name, cell in models.items():
            assert cell["output_per_1m"]["origin"] == "override", name
            assert cell["output_per_1m"]["value"] == pytest.approx(999.0), name


# ---------------------------------------------------------------------------
# (e) currency origin
# ---------------------------------------------------------------------------

class TestCurrencyOrigin:
    def test_currency_override_sets_currency_origin(self, override_file):
        _write_override(override_file, {"openai": {"_currency": "EUR"}})
        rates, origin = resolve_rates_with_origin(
            "openai", "gpt-5.2", override_path=override_file
        )
        assert rates.currency == "EUR"
        assert origin["currency"] == "override"
        # Rate fields untouched.
        assert origin["input_per_1m"] == "default"

    def test_currency_origin_in_table(self, override_file):
        _write_override(override_file, {"openai": {"_currency": "EUR"}})
        table = pricing_table_with_origin(override_path=override_file)
        prov = table["providers"]["openai"]
        assert prov["currency"] == "EUR"
        assert prov["currency_origin"] == "override"
        # Another provider keeps default currency origin.
        assert table["providers"]["anthropic"]["currency_origin"] == "default"


# ---------------------------------------------------------------------------
# (f) value parity with resolve_rates across the whole table
# ---------------------------------------------------------------------------

class TestTableValueParity:
    def test_every_table_cell_matches_resolve_rates(self, override_file):
        _write_override(override_file, {
            "anthropic": {
                "_currency": "USD",
                "_default": {"output_per_1m": 42.0},
                "claude-sonnet-4-6": {"input_per_1m": 1.5},
            },
            "openai": {"gpt-5.2": {"cached_input_per_1m": 0.1}},
        })
        table = pricing_table_with_origin(override_path=override_file)
        for provider, pdata in table["providers"].items():
            for model, cell in pdata["models"].items():
                ref = resolve_rates(provider, model, override_path=override_file)
                assert cell["input_per_1m"]["value"] == pytest.approx(
                    ref.input_per_1m), f"{provider}/{model} input"
                assert cell["cached_input_per_1m"]["value"] == pytest.approx(
                    ref.cached_input_per_1m), f"{provider}/{model} cached"
                assert cell["cache_write_per_1m"]["value"] == pytest.approx(
                    ref.cache_write_per_1m), f"{provider}/{model} cwrite"
                assert cell["output_per_1m"]["value"] == pytest.approx(
                    ref.output_per_1m), f"{provider}/{model} output"
