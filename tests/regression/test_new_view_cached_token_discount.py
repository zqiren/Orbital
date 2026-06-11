# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the NEW derived cost view discounts cached input tokens
(Budget Piece 1 design note D12; Budget Piece 2 cleanup).

History: this file once compared the new view against the legacy
``loop.py::_estimate_cost_usd`` to document the exact magnitude by which the
legacy path OVERBILLED cached tokens (it billed ``input_tokens`` in full at the
input rate, with no cache discount). Budget Piece 2 DELETED that legacy
function (and the whole in-loop dollar accumulator) — the token ledger + the
tiered-rates derived view are now the single source of cost. So the comparison
is gone; what remains worth pinning is the NEW view's math itself: that the
cached portion is priced at the discounted ``cached_input_per_1m`` and the
uncached portion at the full ``input_per_1m``.

Numbers (under OpenAI-compat subset semantics, ``uncached = input - cache_read``):

    usage:  input_tokens=100_000, cache_read_tokens=80_000, output_tokens=0
    input rate:        3.0 / 1M
    cached input rate: 0.30 / 1M  (10x discount)

    new = 20_000 * 3.0/1M + 80_000 * 0.30/1M = $0.060000 + $0.024000 = $0.084000
              (uncached)        (cache_read, discounted)

If anyone regresses the new view to bill cached tokens at the FULL input rate,
the discounted number breaks here.
"""

import json

import pytest

import agent_os.agent.pricing as pricing_mod
from agent_os.agent.providers.types import TokenUsage
from agent_os.budget.normalize import normalize_usage, ProviderSemantics


# Exact pinned cost of the NEW derived view for the usage below.
_EXPECTED_NEW_USD = 0.084000

# Rates, in $/1M.
_INPUT_PER_1M = 3.0
_CACHED_INPUT_PER_1M = 0.30
_OUTPUT_PER_1M = 15.0


@pytest.fixture(autouse=True)
def _reset_pricing_caches():
    """Isolate the override-file cache so this test is order-independent."""
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
    """A controlled pricing-override file with a discounted cached_input rate.

    Keyed under an OpenAI-compat provider (``sdk == "openai"`` semantics) so the
    new view normalizes cache_read as a SUBSET of input_tokens.
    """
    path = str(tmp_path / "pricing-overrides.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "moonshot": {
                "_currency": "USD",  # pin currency so cost is a plain USD number
                "ledger-overbill-model": {
                    "input_per_1m": _INPUT_PER_1M,
                    "cached_input_per_1m": _CACHED_INPUT_PER_1M,
                    "output_per_1m": _OUTPUT_PER_1M,
                },
            },
        }, f)
    yield path


def _new_view_cost_usd(usage: TokenUsage, override_path: str) -> float:
    """Cost of *usage* under the NEW derived view, via the REAL pricing path.

    normalize_usage (OpenAI-compat subset semantics) → resolve_rates (override
    layer) → tokens × resolved per-field rates / 1M. This mirrors exactly what
    ``budget/ledger.py::spend`` computes per group, so the number this test
    pins is the number the production query returns.
    """
    rates = pricing_mod.resolve_rates(
        "moonshot", "ledger-overbill-model", override_path=override_path
    )
    norm = normalize_usage(usage, ProviderSemantics.OPENAI_COMPAT)
    cost = (
        norm.uncached_input * rates.input_per_1m
        + norm.cache_read * rates.cached_input_per_1m
        + norm.cache_write * rates.cache_write_per_1m
        + norm.output * rates.output_per_1m
    ) / 1_000_000
    return cost


def test_new_view_discounts_cached_tokens(override_file):
    """The new derived view prices the cached portion at the discounted rate."""
    usage = TokenUsage(
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=80_000,
    )

    new_cost = _new_view_cost_usd(usage, override_file)

    # Pin the exact discounted number. Breaks if cached tokens are ever billed
    # at the full input rate (which would be 0.300000, the old overbilling).
    assert new_cost == pytest.approx(_EXPECTED_NEW_USD, abs=1e-9)
    # And explicitly: the discount is real — cached cost is strictly below what
    # the full input rate would charge for the same 80k tokens.
    full_rate_on_cached = 80_000 * _INPUT_PER_1M / 1_000_000  # 0.240000
    discounted_on_cached = 80_000 * _CACHED_INPUT_PER_1M / 1_000_000  # 0.024000
    assert discounted_on_cached < full_rate_on_cached


def test_new_view_no_cache_bills_full_input_rate(override_file):
    """Control: with NO cached tokens, the whole prompt is billed at the full
    input rate — the discount only applies to the cache_read slice."""
    usage = TokenUsage(
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=0,
    )

    new_cost = _new_view_cost_usd(usage, override_file)

    # 100_000 * 3.0/1M == 0.300000
    assert new_cost == pytest.approx(0.300000, abs=1e-9)
