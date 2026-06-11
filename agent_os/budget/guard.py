# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pre-flight budget guard (Budget Piece 2 — enforcement).

THE single enforcement point. Exactly one place in the codebase computes
"is this project over budget" — this function — and both the agent loop
(pre-LLM-call chokepoint) and the queue dispatcher (Piece 2-D) consult it.

Contract:

  - **Live resolution.** The caller passes the CURRENT project budget config
    (resolved fresh on every evaluation — never a session-start snapshot, the
    B5 fix). The loop wires a ``get_budget_config`` callable that reads the
    ProjectStore on each call.
  - **Spend side.** Spend is the ledger's converted total over the project's
    window (``budget/ledger.py::spend``), compared ``>= limit``. The window is
    the project's ``budget_period`` (default ``"daily"``); the ``total`` window
    uses ``budget_anchor_ts``.
  - **Convertibility choice.** If some spend slice cannot be converted to the
    target currency (unknown FX pair), the convertible total is treated as the
    spend. This is deliberately permissive on the *spend* side: we never invent
    a rate, and an unconvertible slice simply doesn't count toward the trip. The
    GET /cost view already surfaces the ``unconverted`` currencies for the user.
  - **No limit → no-op.** A limit of None / 0 / absent disables the guard.
  - **Between LLM calls only.** The guard is evaluated by the loop BETWEEN
    management-LLM calls; it never interrupts an in-flight tool execution.

This module computes the decision; it takes NO trip action (no session writes,
no queue mutation). Codes/numbers only — no display strings (i18n rule).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# budget_action vocabulary after the one-time silent migration. The persisted
# legacy values ("ask", "stop", and any stray value) normalize into exactly
# these two enforcement modes. Default for new projects: "pause".
ACTION_PAUSE = "pause"
ACTION_STOP = "stop"


def normalize_budget_action(raw: object) -> str:
    """Map a persisted ``budget_action`` to the Piece-2 enforcement vocabulary.

    Migration table (one-time, silent — applied at read AND save so persisted
    configs converge):

        "ask"  → "pause"   (the legacy approval-pause action)
        "stop" → "stop"
        anything else (None, "", unknown) → "pause"   (the safe default)

    Idempotent: ``normalize_budget_action(normalize_budget_action(x)) == \
normalize_budget_action(x)``.
    """
    if raw == ACTION_STOP:
        return ACTION_STOP
    # "ask", "pause", None, "", and any unknown value all converge to pause.
    return ACTION_PAUSE


@dataclass(frozen=True)
class BudgetDecision:
    """Outcome of one guard evaluation.

    ``tripped`` is the only load-bearing field for control flow. The rest are
    structured detail for the timeline event the caller appends on a trip
    (codes/numbers only — ``window``/``currency`` are enums/ISO codes, never
    display strings).
    """

    tripped: bool
    action: str  # normalized: "pause" | "stop" (meaningless when not tripped)
    window: str | None = None
    spend: float | None = None
    limit: float | None = None
    currency: str | None = None


def evaluate_budget(
    project_dir: str,
    budget_config: dict | None,
    *,
    fx_rates: dict | None = None,
    now=None,
    spend_fn=None,
) -> BudgetDecision:
    """Evaluate whether ``project_dir`` is over budget RIGHT NOW.

    Args:
        project_dir: Project workspace root (the ledger lives under it).
        budget_config: LIVE project budget config. Recognized keys:
            ``budget_limit_usd`` (the limit VALUE — the field keeps its legacy
            name though the currency need not be USD), ``budget_currency``
            (ISO 4217; the limit owns its currency, default USD),
            ``budget_period`` (window, default ``"daily"``),
            ``budget_anchor_ts`` (ISO8601 lower bound for the ``total`` window),
            ``budget_action`` (normalized here via ``normalize_budget_action``).
        fx_rates: Static FX pair rates from settings (``{"CNY_per_USD": 7.2}``).
        now: Override "current time" for window math (tests).
        spend_fn: Injectable spend() for tests. Defaults to the real
            ``budget.ledger.spend``.

    Returns:
        A ``BudgetDecision``. ``tripped`` is True iff a limit is configured
        (>0) AND the ledger's converted spend over the window is ``>= limit``.
        A missing/None/zero limit yields ``tripped=False`` (guard no-op).
    """
    cfg = budget_config or {}
    limit = cfg.get("budget_limit_usd")
    action = normalize_budget_action(cfg.get("budget_action"))

    # No limit set (None / 0 / absent) → guard is a no-op. Treat any
    # non-positive limit as "unset" so a 0 limit never traps every call.
    if limit is None:
        return BudgetDecision(tripped=False, action=action)
    try:
        limit_val = float(limit)
    except (TypeError, ValueError):
        return BudgetDecision(tripped=False, action=action)
    if limit_val <= 0:
        return BudgetDecision(tripped=False, action=action)

    window = cfg.get("budget_period") or "daily"
    currency = cfg.get("budget_currency") or "USD"
    anchor_ts = cfg.get("budget_anchor_ts")

    if spend_fn is None:
        from agent_os.budget.ledger import spend as spend_fn  # lazy: import cycle

    try:
        result = spend_fn(
            project_dir,
            window,
            target_currency=currency,
            fx_rates=fx_rates or {},
            now=now,
            anchor_ts=anchor_ts,
            # Enforcement counts MANAGEMENT spend only. Sub-agent usage is
            # captured to the ledger (display) but must never trip the guard
            # nor block the user (P3-A guard isolation).
            sources=["management"],
        )
    except Exception as e:  # noqa: BLE001 — fail-open scope: the spend QUERY only
        # A spend-query failure (ledger I/O, rate resolution) must never crash
        # the loop NOR spuriously trip the guard (that would block the user on a
        # transient disk error). Fail OPEN: treat as no spend recorded → no
        # trip. Scope is deliberately NARROW — only the spend_fn call itself.
        # A malformed RETURN from spend() is a contract bug and is handled
        # below, OUTSIDE this guard, so it surfaces loudly instead of silently
        # masking as $0.
        logger.warning(
            "budget guard: spend query failed; failing open (no block): %s",
            e, exc_info=True,
        )
        return BudgetDecision(tripped=False, action=action)

    # Result-shape access OUTSIDE the fail-open guard: a KeyError/TypeError/
    # ValueError here means spend() violated its return contract — a logic bug
    # that must RAISE (loudly), never be silently treated as "no spend".
    # The convertible total is the spend (per the convertibility choice
    # documented in the module docstring): unconvertible slices are omitted
    # from converted_total.amount and simply don't count toward the trip.
    spend_amount = float(result["converted_total"]["amount"])

    tripped = spend_amount >= limit_val
    return BudgetDecision(
        tripped=tripped,
        action=action,
        window=window,
        spend=spend_amount,
        limit=limit_val,
        currency=currency,
    )
