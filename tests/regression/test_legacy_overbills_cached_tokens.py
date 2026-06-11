# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the LEGACY ``_estimate_cost_usd`` overbills cached tokens vs. the
NEW derived cost view, given IDENTICAL usage (Budget Piece 1, design note D12).

The legacy budget path (``agent_os/agent/loop.py::_estimate_cost_usd``) bills
``input_tokens`` IN FULL at the input rate — cached tokens get NO discount. The
new derived view normalizes usage into disjoint fields (under OpenAI-compat
subset semantics, ``uncached_input = input_tokens - cache_read_tokens``) and
prices the cached portion at the discounted ``cached_input_per_1m`` resolved
from the tiered-rates table.

This test pins BOTH numbers so it documents the exact overbilling magnitude and
FAILS if anyone "fixes" the new view to match legacy (or regresses legacy to
match the new view). It calls the REAL legacy function (imported from the loop
module) and prices the new view through the REAL ``resolve_rates`` path with a
controlled override file — no reimplementation of either side's math.

Numbers (held constant on both sides so the delta is PURELY the cache discount):

    usage:  input_tokens=100_000, cache_read_tokens=80_000, output_tokens=0
    input rate:        3.0 / 1M  (== 0.003 / 1k, fed to legacy)
    cached input rate: 0.30 / 1M (10x discount, new view only)

    legacy = (100_000 / 1000) * 0.003                 = $0.300000
    new    = 20_000 * 3.0/1M + 80_000 * 0.30/1M        = $0.084000
                 (uncached)        (cache_read, discounted)

    legacy / new ≈ 3.57x  → legacy overbills by $0.216 on this single response.
"""

import json
import os

import pytest

import agent_os.agent.pricing as pricing_mod
from agent_os.agent.loop import _estimate_cost_usd
from agent_os.agent.providers.types import TokenUsage
from agent_os.budget.normalize import normalize_usage, ProviderSemantics


# Exact pinned costs. If either side's math changes, one of these assertions
# breaks — which is the whole point of this regression test.
_EXPECTED_LEGACY_USD = 0.300000
_EXPECTED_NEW_USD = 0.084000

# Rates, in $/1M. The input rate is shared by both sides; only the cached rate
# differs, so the entire legacy-vs-new gap is attributable to the cache
# discount the new view applies and the legacy path does not.
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


def test_legacy_overbills_cached_tokens_vs_new_view(override_file):
    """Given identical usage with a large cache_read, the legacy path costs
    strictly MORE than the new derived view, and both numbers are pinned."""
    usage = TokenUsage(
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=80_000,
    )

    # (1) Legacy cost — the REAL loop function. Same input rate as the new view
    # (3.0/1M == 0.003/1k); legacy ignores the cache_read field entirely and
    # bills all 100_000 input tokens at the full input rate.
    legacy_cost = _estimate_cost_usd(
        usage,
        cost_per_1k_input=_INPUT_PER_1M / 1000,   # 0.003
        cost_per_1k_output=_OUTPUT_PER_1M / 1000,  # 0.015
    )

    # (2) New derived-view cost — REAL normalize_usage + resolve_rates.
    new_cost = _new_view_cost_usd(usage, override_file)

    # Pin BOTH numbers. These assertions document the overbilling magnitude and
    # break if either side's math is "fixed" to converge.
    assert legacy_cost == pytest.approx(_EXPECTED_LEGACY_USD, abs=1e-9)
    assert new_cost == pytest.approx(_EXPECTED_NEW_USD, abs=1e-9)

    # The core regression invariant: legacy strictly overbills.
    assert legacy_cost > new_cost

    # Document the exact overbilling delta and ratio so a future reader sees the
    # stakes, and so a silent convergence (both → same value) fails loudly here.
    assert (legacy_cost - new_cost) == pytest.approx(0.216000, abs=1e-9)
    assert (legacy_cost / new_cost) == pytest.approx(3.571428, abs=1e-5)


def test_legacy_and_new_agree_when_no_cache(override_file):
    """Control: with NO cached tokens, legacy and new view agree exactly.

    This proves the gap in the main test is caused SOLELY by cached-token
    discounting — not by a rate mismatch or a normalization artifact. Here
    uncached_input == input_tokens, cache_read == 0, so both sides bill the
    full prompt at the same input rate.
    """
    usage = TokenUsage(
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=0,
    )

    legacy_cost = _estimate_cost_usd(
        usage,
        cost_per_1k_input=_INPUT_PER_1M / 1000,
        cost_per_1k_output=_OUTPUT_PER_1M / 1000,
    )
    new_cost = _new_view_cost_usd(usage, override_file)

    assert legacy_cost == pytest.approx(0.300000, abs=1e-9)
    assert new_cost == pytest.approx(0.300000, abs=1e-9)
    assert legacy_cost == pytest.approx(new_cost, abs=1e-9)
