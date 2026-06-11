# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the single budget enforcement point (Budget Piece 2).

``agent_os/budget/guard.py`` is THE one place that computes "is this project
over budget". These pin: the trip threshold (``spend >= limit``), the no-limit
no-op, live config consumption, the convertibility choice, and the
``budget_action`` migration table.
"""

import pytest

from agent_os.budget.guard import (
    ACTION_PAUSE,
    ACTION_STOP,
    BudgetDecision,
    evaluate_budget,
    normalize_budget_action,
)


def _spend_returning(amount, *, currency="USD", unconverted=None):
    """A stub spend() that returns a fixed converted total."""
    def _spend(project_dir, window, *, target_currency=None, fx_rates=None,
               now=None, anchor_ts=None, sources=None):
        ct = {"currency": target_currency or currency, "amount": amount,
              "estimated": True}
        if unconverted:
            ct["unconverted"] = unconverted
        return {"window": window, "by_currency": {}, "converted_total": ct,
                "breakdown": []}
    return _spend


# ---------------------------------------------------------------------------
# Migration table
# ---------------------------------------------------------------------------

class TestNormalizeBudgetAction:
    def test_ask_maps_to_pause(self):
        assert normalize_budget_action("ask") == ACTION_PAUSE

    def test_stop_stays_stop(self):
        assert normalize_budget_action("stop") == ACTION_STOP

    def test_unknown_maps_to_pause(self):
        assert normalize_budget_action("frobnicate") == ACTION_PAUSE

    def test_none_maps_to_pause(self):
        assert normalize_budget_action(None) == ACTION_PAUSE

    def test_pause_is_idempotent(self):
        assert normalize_budget_action("pause") == ACTION_PAUSE

    def test_idempotent(self):
        for raw in ("ask", "stop", "pause", None, ""):
            once = normalize_budget_action(raw)
            assert normalize_budget_action(once) == once


# ---------------------------------------------------------------------------
# Trip threshold
# ---------------------------------------------------------------------------

class TestEvaluateBudget:
    def test_spend_over_limit_trips(self):
        d = evaluate_budget(
            "/ws",
            {"budget_limit_usd": 5.0, "budget_action": "ask"},
            spend_fn=_spend_returning(6.0),
        )
        assert d.tripped is True
        assert d.action == ACTION_PAUSE
        assert d.spend == 6.0
        assert d.limit == 5.0

    def test_spend_equal_to_limit_trips(self):
        # The contract is spend >= limit (inclusive).
        d = evaluate_budget(
            "/ws",
            {"budget_limit_usd": 5.0},
            spend_fn=_spend_returning(5.0),
        )
        assert d.tripped is True

    def test_spend_under_limit_does_not_trip(self):
        d = evaluate_budget(
            "/ws",
            {"budget_limit_usd": 5.0},
            spend_fn=_spend_returning(4.99),
        )
        assert d.tripped is False

    def test_no_limit_is_noop(self):
        d = evaluate_budget("/ws", {"budget_action": "stop"},
                            spend_fn=_spend_returning(1000.0))
        assert d.tripped is False
        # action still normalizes even on the no-op path.
        assert d.action == ACTION_STOP

    def test_zero_limit_is_noop(self):
        d = evaluate_budget("/ws", {"budget_limit_usd": 0},
                            spend_fn=_spend_returning(1000.0))
        assert d.tripped is False

    def test_none_config_is_noop(self):
        d = evaluate_budget("/ws", None, spend_fn=_spend_returning(1000.0))
        assert d.tripped is False

    def test_stop_action_carried_through(self):
        d = evaluate_budget(
            "/ws",
            {"budget_limit_usd": 1.0, "budget_action": "stop"},
            spend_fn=_spend_returning(2.0),
        )
        assert d.tripped is True
        assert d.action == ACTION_STOP

    def test_window_and_currency_passed_through(self):
        captured = {}

        def _spend(project_dir, window, *, target_currency=None, fx_rates=None,
                   now=None, anchor_ts=None, sources=None):
            captured["window"] = window
            captured["currency"] = target_currency
            captured["anchor"] = anchor_ts
            captured["sources"] = sources
            return {"converted_total": {"amount": 0.0, "currency": target_currency}}

        evaluate_budget(
            "/ws",
            {
                "budget_limit_usd": 5.0,
                "budget_period": "monthly",
                "budget_currency": "CNY",
                "budget_anchor_ts": "2026-01-01T00:00:00+00:00",
            },
            spend_fn=_spend,
        )
        assert captured["window"] == "monthly"
        # P3-A: the guard isolates enforcement to management spend.
        assert captured["sources"] == ["management"]
        assert captured["currency"] == "CNY"
        assert captured["anchor"] == "2026-01-01T00:00:00+00:00"

    def test_unconvertible_slice_uses_convertible_total(self):
        # converted_total.amount omits the unconvertible slice; the guard
        # compares that convertible total to the limit (documented choice).
        d = evaluate_budget(
            "/ws",
            {"budget_limit_usd": 5.0},
            # convertible total 6.0 (>= 5.0) even though a CNY slice was dropped
            spend_fn=_spend_returning(6.0, unconverted=["CNY"]),
        )
        assert d.tripped is True
        assert d.spend == 6.0

    def test_spend_read_failure_fails_open(self):
        def _boom(*a, **k):
            raise RuntimeError("ledger read exploded")

        d = evaluate_budget(
            "/ws",
            {"budget_limit_usd": 5.0},
            spend_fn=_boom,
        )
        # Fail OPEN: a transient read error must not spuriously block the user.
        assert d.tripped is False

    def test_spend_read_failure_logs_warning(self, caplog):
        """The fail-open path is NOT silent — it logs a warning so a recurring
        ledger failure is visible in daemon.log."""
        import logging

        def _boom(*a, **k):
            raise RuntimeError("ledger read exploded")

        with caplog.at_level(logging.WARNING, logger="agent_os.budget.guard"):
            d = evaluate_budget(
                "/ws",
                {"budget_limit_usd": 5.0},
                spend_fn=_boom,
            )
        assert d.tripped is False
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "failing open" in r.getMessage()]
        assert warnings, (
            f"fail-open must log a warning; got records: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        # The traceback is attached (exc_info=True) for diagnosis.
        assert warnings[0].exc_info is not None

    def test_malformed_spend_return_raises_not_fail_open(self):
        """A malformed spend() RETURN (contract bug) must RAISE loudly — never
        be silently masked as $0/no-spend. The fail-open guard covers only the
        spend QUERY itself."""
        # Missing the converted_total key entirely.
        def _missing_key(project_dir, window, *, target_currency=None,
                         fx_rates=None, now=None, anchor_ts=None, sources=None):
            return {"window": window, "by_currency": {}, "breakdown": []}

        with pytest.raises(KeyError):
            evaluate_budget(
                "/ws",
                {"budget_limit_usd": 5.0},
                spend_fn=_missing_key,
            )

        # Wrong type: converted_total is not a mapping.
        def _wrong_type(project_dir, window, *, target_currency=None,
                        fx_rates=None, now=None, anchor_ts=None, sources=None):
            return {"converted_total": None}

        with pytest.raises(TypeError):
            evaluate_budget(
                "/ws",
                {"budget_limit_usd": 5.0},
                spend_fn=_wrong_type,
            )

        # Non-numeric amount.
        def _bad_amount(project_dir, window, *, target_currency=None,
                        fx_rates=None, now=None, anchor_ts=None, sources=None):
            return {"converted_total": {"amount": "not-a-number"}}

        with pytest.raises(ValueError):
            evaluate_budget(
                "/ws",
                {"budget_limit_usd": 5.0},
                spend_fn=_bad_amount,
            )
