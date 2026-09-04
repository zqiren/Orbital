# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Route tests for GET /api/v2/projects/{pid}/cost (Budget Piece 1, Task 4).

Also covers the PUT /projects/{pid} budget_currency set-once semantics and the
anchor-reset flag, since those live in the same router and are part of this
task's contract.

Uses FastAPI's TestClient against a real router + a real ProjectStore on a
temp dir (so the set-once/anchor logic exercises actual persistence). The
ledger is written directly to the project workspace so spend() reads real
events. Rates are injected via a temp pricing-override file.
"""

import json
import os

import pytest
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient

import agent_os.agent.pricing as pricing_mod
from agent_os.api.routes.agents_v2 import router, configure
from agent_os.budget.ledger import ledger_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    path = str(tmp_path / "ov.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "anthropic": {
                "_currency": "USD",
                "claude-x": {"input_per_1m": 3.0, "output_per_1m": 15.0},
            },
            "moonshot": {
                "_currency": "CNY",
                "kimi-x": {"input_per_1m": 12.0, "output_per_1m": 12.0},
            },
        }, f)
    # Point the module default at this temp file so the route's spend() call
    # (which does not pass override_path) resolves these rates.
    pricing_mod._OVERRIDE_FILE = path
    yield path
    pricing_mod._OVERRIDE_FILE = pricing_mod._default_override_path()


@pytest.fixture
def store(tmp_path):
    from agent_os.daemon_v2.project_store import ProjectStore
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    return ProjectStore(data_dir)


class _FakeSettings:
    def __init__(self, fx_rates):
        self.fx_rates = fx_rates


class _FakeSettingsStore:
    """Settings-store double with the card surface the cost routes now call.

    Spec 082: a project no longer stores its own provider/model — it stores a
    ``card_id``, and the budget's set-once currency defaults from the provider
    of the card the project would actually run on. Two cards are enough to
    express the cases: one CNY provider, one USD.
    """

    CARDS = {
        "card_moonshot": ("moonshot", "kimi-x"),
        "card_anthropic": ("anthropic", "claude-x"),
    }

    def __init__(self, fx_rates):
        self._s = _FakeSettings(fx_rates)

    def get(self):
        return self._s

    def resolve_card(self, card_id):
        pair = self.CARDS.get(card_id)
        if pair is None:
            return None
        return SimpleNamespace(id=card_id, provider=pair[0], model=pair[1])

    get_card = resolve_card

    def default_card(self):
        return self.resolve_card("card_anthropic")

    stored_default_card = default_card


@pytest.fixture
def make_client():
    """Factory: build a TestClient wired to a given store + settings store."""
    def _make(store, fx_rates=None):
        from unittest.mock import MagicMock
        app = FastAPI()
        configure(
            project_store=store,
            agent_manager=MagicMock(),
            ws_manager=MagicMock(),
            settings_store=_FakeSettingsStore(fx_rates or {"CNY_per_USD": 7.2}),
            provider_registry=None,
        )
        app.include_router(router)
        return TestClient(app)
    return _make


def _write_event(workspace: str, record: dict) -> None:
    path = ledger_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _evt(ts="2026-06-10T12:00:00+00:00", provider="anthropic", model="claude-x",
         source="management", uncached=0, output=0, reported_cost=None,
         reported_cost_currency=None):
    rec = {
        "ts": ts, "session_id": "s", "source": source,
        "provider": provider, "model": model,
        "uncached_input": uncached, "cache_read": 0,
        "cache_write": 0, "output": output,
    }
    if reported_cost is not None:
        rec["reported_cost"] = reported_cost
        rec["reported_cost_currency"] = reported_cost_currency
    return rec


def _new_project(store, tmp_path, **extra):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)
    # Spec 082: a project references a card; provider/model/api_key are gone
    # from the project row. card_anthropic is the fake store's default (USD).
    config = {
        "name": "proj", "workspace": workspace, "card_id": "card_anthropic",
    }
    config.update(extra)
    return store.create_project(config), workspace


# ---------------------------------------------------------------------------
# GET /cost
# ---------------------------------------------------------------------------

class TestCostRoute:
    def test_returns_breakdown_and_totals(self, store, tmp_path, make_client,
                                          override_file):
        pid, ws = _new_project(store, tmp_path, budget_period="total",
                               budget_currency="USD")
        _write_event(ws, _evt(output=1_000_000))           # $15 USD
        _write_event(ws, _evt(provider="moonshot", model="kimi-x",
                              output=1_000_000))            # 12 CNY
        client = make_client(store, fx_rates={"CNY_per_USD": 7.2})
        resp = client.get(f"/api/v2/projects/{pid}/cost?window=total")
        assert resp.status_code == 200
        body = resp.json()
        assert body["window"] == "total"
        assert body["by_currency"]["USD"] == pytest.approx(15.0)
        assert body["by_currency"]["CNY"] == pytest.approx(12.0)
        # 12 CNY / 7.2 = 1.6667 USD + 15 = 16.6667
        assert body["converted_total"]["currency"] == "USD"
        assert body["converted_total"]["estimated"] is True
        assert body["converted_total"]["amount"] == pytest.approx(16.666667, abs=1e-5)
        assert len(body["breakdown"]) == 2
        # P3-B reshape: with no sub-agent rows planted, the new top-level
        # subagents block is present and empty — management numbers unchanged.
        assert body["subagents"] == []

    def test_window_defaults_to_project_budget_period(self, store, tmp_path,
                                                      make_client, override_file):
        pid, ws = _new_project(store, tmp_path, budget_period="total",
                               budget_currency="USD")
        _write_event(ws, _evt(ts="2020-01-01T00:00:00+00:00", output=1_000_000))
        client = make_client(store)
        # No ?window → uses budget_period="total", which includes the old event.
        resp = client.get(f"/api/v2/projects/{pid}/cost")
        assert resp.status_code == 200
        assert resp.json()["window"] == "total"
        assert len(resp.json()["breakdown"]) == 1

    def test_bad_window_returns_400_with_code(self, store, tmp_path, make_client,
                                              override_file):
        pid, ws = _new_project(store, tmp_path)
        client = make_client(store)
        resp = client.get(f"/api/v2/projects/{pid}/cost?window=yearly")
        assert resp.status_code == 400
        # i18n rule: machine code, not a human sentence.
        detail = resp.json()["detail"]
        assert detail["code"] == "invalid_window"
        assert set(detail["allowed"]) == {"daily", "weekly", "monthly", "total"}

    def test_unknown_project_returns_404(self, store, make_client, override_file):
        client = make_client(store)
        resp = client.get("/api/v2/projects/nope/cost?window=daily")
        assert resp.status_code == 404

    def test_target_currency_defaults_to_usd_when_unset(self, store, tmp_path,
                                                        make_client, override_file):
        # No budget_currency set on the project → converted_total defaults USD.
        pid, ws = _new_project(store, tmp_path, budget_period="total")
        _write_event(ws, _evt(output=1_000_000))
        client = make_client(store)
        body = client.get(f"/api/v2/projects/{pid}/cost").json()
        assert body["converted_total"]["currency"] == "USD"

    def test_read_only_no_side_effects(self, store, tmp_path, make_client,
                                       override_file):
        pid, ws = _new_project(store, tmp_path, budget_period="daily")
        before = json.dumps(store.get_project(pid), sort_keys=True, default=str)
        client = make_client(store)
        client.get(f"/api/v2/projects/{pid}/cost")
        after = json.dumps(store.get_project(pid), sort_keys=True, default=str)
        assert before == after

    def test_fx_rates_surfaced_codes_only(self, store, tmp_path, make_client,
                                          override_file):
        """P3-F footnote: the response carries the daemon's static FX table so
        the client can render the rate the total was converted at. Pair-code
        keys with numeric values ONLY — non-numeric junk in settings is
        filtered, and no display strings appear (binding i18n rule)."""
        pid, ws = _new_project(store, tmp_path, budget_period="total",
                               budget_currency="USD")
        client = make_client(store, fx_rates={
            "CNY_per_USD": 7.2,
            "junk": "not-a-number",   # filtered
            "flag": True,             # bools are not rates — filtered
        })
        body = client.get(f"/api/v2/projects/{pid}/cost?window=total").json()
        assert body["fx_rates"] == {"CNY_per_USD": pytest.approx(7.2)}
        assert all(isinstance(v, float) for v in body["fx_rates"].values())

    def test_fx_rates_empty_when_no_numeric_rates(self, store, tmp_path,
                                                  make_client, override_file):
        """A table with no usable numeric rates surfaces as an empty dict (the
        client then never shows a rate — it is never invented)."""
        pid, ws = _new_project(store, tmp_path, budget_period="total",
                               budget_currency="USD")
        # NOTE: make_client substitutes a default table for a falsy {} — use a
        # non-empty all-junk table to exercise the filter down to {}.
        client = make_client(store, fx_rates={"junk": "not-a-number"})
        body = client.get(f"/api/v2/projects/{pid}/cost?window=total").json()
        assert body["fx_rates"] == {}


# ---------------------------------------------------------------------------
# GET /cost — the P3-B subagents block reshape
# ---------------------------------------------------------------------------

class TestCostRouteSubagents:
    """Sub-agent rows move OUT of the management breakdown/by_currency/
    converted_total and INTO a separate top-level ``subagents`` list, carrying
    the provider-reported cost verbatim (or null)."""

    def test_subagent_rows_excluded_from_management_block(self, store, tmp_path,
                                                          make_client, override_file):
        pid, ws = _new_project(store, tmp_path, budget_period="total",
                               budget_currency="USD")
        # One management row ($15) + two sub-agent rows that must NOT pollute
        # the management block.
        _write_event(ws, _evt(output=1_000_000))  # management $15
        _write_event(ws, _evt(source="subagent:claude-code", output=2_000_000,
                              reported_cost=0.42, reported_cost_currency="USD"))
        _write_event(ws, _evt(source="subagent:codex", output=3_000_000))
        client = make_client(store)
        body = client.get(f"/api/v2/projects/{pid}/cost?window=total").json()

        # Management block is management-only: $15, one breakdown row.
        assert body["by_currency"] == {"USD": pytest.approx(15.0)}
        assert body["converted_total"]["amount"] == pytest.approx(15.0)
        assert len(body["breakdown"]) == 1
        assert body["breakdown"][0]["source"] == "management"

        # Sub-agents live in their own block, one entry per (provider,model,source).
        subs = {s["source"]: s for s in body["subagents"]}
        assert set(subs) == {"subagent:claude-code", "subagent:codex"}
        assert subs["subagent:claude-code"]["tokens"]["output"] == 2_000_000
        assert subs["subagent:codex"]["tokens"]["output"] == 3_000_000

    def test_reported_cost_surfaced_verbatim_with_currency(self, store, tmp_path,
                                                           make_client, override_file):
        pid, ws = _new_project(store, tmp_path, budget_period="total",
                               budget_currency="USD")
        _write_event(ws, _evt(source="subagent:claude-code", output=2_000_000,
                              reported_cost=0.42, reported_cost_currency="USD"))
        client = make_client(store)
        body = client.get(f"/api/v2/projects/{pid}/cost?window=total").json()
        entry = body["subagents"][0]
        # Verbatim: 0.42 USD as reported, NOT the rate-derived $30 (2M × $15).
        assert entry["reported_cost"] == {"amount": pytest.approx(0.42),
                                          "currency": "USD"}
        # The rate-derived cost block is deliberately absent on subagent entries.
        assert "cost" not in entry

    def test_absent_reported_cost_is_null(self, store, tmp_path, make_client,
                                          override_file):
        """A subscription-auth sub-agent (no reported_cost) → tokens with a
        null cost, never a fabricated dollar figure."""
        pid, ws = _new_project(store, tmp_path, budget_period="total",
                               budget_currency="USD")
        _write_event(ws, _evt(source="subagent:codex", output=3_000_000))
        client = make_client(store)
        body = client.get(f"/api/v2/projects/{pid}/cost?window=total").json()
        entry = body["subagents"][0]
        assert entry["reported_cost"] is None
        assert entry["tokens"]["output"] == 3_000_000

    def test_subagents_empty_when_only_management(self, store, tmp_path,
                                                  make_client, override_file):
        pid, ws = _new_project(store, tmp_path, budget_period="total",
                               budget_currency="USD")
        _write_event(ws, _evt(output=1_000_000))
        client = make_client(store)
        body = client.get(f"/api/v2/projects/{pid}/cost?window=total").json()
        assert body["subagents"] == []
        assert body["by_currency"]["USD"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# PUT budget_currency set-once + anchor reset
# ---------------------------------------------------------------------------

class TestBudgetCurrencySetOnce:
    def test_setting_limit_defaults_currency_from_provider(self, store, tmp_path,
                                                           make_client, override_file):
        # Project on a moonshot card (CNY). Setting a limit with no explicit
        # currency defaults budget_currency from that CARD's provider → CNY.
        pid, ws = _new_project(store, tmp_path, card_id="card_moonshot")
        client = make_client(store)
        resp = client.put(f"/api/v2/projects/{pid}", json={"budget_limit_usd": 10.0})
        assert resp.status_code == 200
        assert store.get_project(pid)["budget_currency"] == "CNY"

    def test_changing_provider_does_not_redenominate(self, store, tmp_path,
                                                     make_client, override_file):
        # Limit set on a moonshot card → CNY. Then switch the project to an
        # anthropic card (USD) WITHOUT touching the limit → must stay CNY.
        pid, ws = _new_project(store, tmp_path, card_id="card_moonshot")
        client = make_client(store)
        client.put(f"/api/v2/projects/{pid}", json={"budget_limit_usd": 10.0})
        assert store.get_project(pid)["budget_currency"] == "CNY"
        resp = client.put(f"/api/v2/projects/{pid}", json={"card_id": "card_anthropic"})
        assert resp.status_code == 200
        assert store.get_project(pid)["budget_currency"] == "CNY"  # unchanged

    def test_explicit_currency_in_put_is_honored(self, store, tmp_path,
                                                make_client, override_file):
        pid, ws = _new_project(store, tmp_path, card_id="card_moonshot")
        client = make_client(store)
        resp = client.put(f"/api/v2/projects/{pid}",
                          json={"budget_limit_usd": 10.0, "budget_currency": "USD"})
        assert resp.status_code == 200
        assert store.get_project(pid)["budget_currency"] == "USD"

    def test_explicit_currency_can_change_existing(self, store, tmp_path,
                                                  make_client, override_file):
        pid, ws = _new_project(store, tmp_path, provider="moonshot",
                               model="kimi-x", budget_currency="CNY")
        client = make_client(store)
        resp = client.put(f"/api/v2/projects/{pid}", json={"budget_currency": "USD"})
        assert resp.status_code == 200
        assert store.get_project(pid)["budget_currency"] == "USD"

    def test_setting_limit_when_currency_exists_keeps_it(self, store, tmp_path,
                                                        make_client, override_file):
        pid, ws = _new_project(store, tmp_path, provider="anthropic",
                               model="claude-x", budget_currency="CNY")
        client = make_client(store)
        # Re-setting the limit must NOT re-derive currency (already set → CNY).
        resp = client.put(f"/api/v2/projects/{pid}", json={"budget_limit_usd": 99.0})
        assert resp.status_code == 200
        assert store.get_project(pid)["budget_currency"] == "CNY"


class TestAnchorReset:
    def test_reset_flag_sets_anchor_to_now_total_mode(self, store, tmp_path,
                                                      make_client, override_file):
        """P2-D: the reset is total-mode-only. In total mode the flag sets the
        anchor to now and the response carries ``budget_reset.applied=True``."""
        pid, ws = _new_project(store, tmp_path, budget_period="total")
        client = make_client(store)
        resp = client.put(f"/api/v2/projects/{pid}",
                          json={"reset_budget_anchor": True})
        assert resp.status_code == 200
        assert resp.json()["budget_reset"]["applied"] is True
        assert resp.json()["budget_reset"]["code"] is None
        anchor = store.get_project(pid).get("budget_anchor_ts")
        assert anchor is not None
        # Parseable ISO8601.
        from datetime import datetime
        datetime.fromisoformat(anchor)

    def test_reset_flag_noop_non_total_mode(self, store, tmp_path, make_client,
                                            override_file):
        """P2-D: in a non-total window the reset is a no-op that returns the
        machine code ``not_total_mode`` and leaves the anchor untouched."""
        pid, ws = _new_project(store, tmp_path, budget_period="daily")
        client = make_client(store)
        resp = client.put(f"/api/v2/projects/{pid}",
                          json={"reset_budget_anchor": True})
        assert resp.status_code == 200
        assert resp.json()["budget_reset"]["applied"] is False
        assert resp.json()["budget_reset"]["code"] == "not_total_mode"
        assert store.get_project(pid).get("budget_anchor_ts") is None

    def test_direct_anchor_value_honored(self, store, tmp_path, make_client,
                                         override_file):
        pid, ws = _new_project(store, tmp_path)
        client = make_client(store)
        resp = client.put(f"/api/v2/projects/{pid}",
                          json={"budget_anchor_ts": "2026-03-01T00:00:00+00:00"})
        assert resp.status_code == 200
        assert store.get_project(pid)["budget_anchor_ts"] == "2026-03-01T00:00:00+00:00"

    def test_budget_period_persists(self, store, tmp_path, make_client, override_file):
        pid, ws = _new_project(store, tmp_path)
        client = make_client(store)
        resp = client.put(f"/api/v2/projects/{pid}", json={"budget_period": "weekly"})
        assert resp.status_code == 200
        assert store.get_project(pid)["budget_period"] == "weekly"
