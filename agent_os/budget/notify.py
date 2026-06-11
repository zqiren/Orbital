# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Budget threshold/trip emitter — the producer side of the relay's Tier-1
``budget.exhausted`` push (Budget Piece 2 — Task E, the E17 fix).

The relay client (``agent_os/relay/client.py::_should_push``) already DEFINES a
Tier-1 push for an event of type ``budget.exhausted`` but nothing ever emitted
one. This module is the missing producer. It detects two crossings within the
project's CURRENT budget window and emits AT MOST ONCE each per
(project, window-identity):

  * **threshold** — converted spend crosses ``THRESHOLD_PCT`` (80%) of the limit
    → event type ``budget.threshold`` (the Tier-1 sibling).
  * **trip** — converted spend reaches the limit (>= 100%, the same condition
    the guard trips on) → event type ``budget.exhausted`` (the EXISTING
    relay-registered type, so it routes to the Tier-1 push unchanged).

Placement (coordinator-ruled: dual-site for the TRIP, append-only for the
threshold):

  * **Threshold (80%)** — detected at ledger-append ONLY
    (``maybe_emit_budget_event``). A crossing is *caused by* spend, and spend is
    exactly one ledger append, so the append catches every crossing at the
    causal instant — including a project that crosses 80% then sits IDLE, where
    the loop's pre-call guard and the dispatcher's paused 5s cadence would
    never re-evaluate.
  * **Trip (100%)** — emitted at whichever happens FIRST: the crossing append
    (``maybe_emit_budget_event``) or the loop's guard trip (``maybe_emit_trip``,
    called from ``AgentLoop._trip_budget``). The guard-trip site covers
    no-new-append trips: a limit lowered below current spend, or a fresh run on
    an already-over project whose crossing append predates the marker (e.g.
    absent/corrupted marker). Both sites share the SAME marker file and key, so
    the trip still fires exactly once per (project, window-identity).

The pure ``evaluate_budget`` stays pure: it never emits. The dispatcher's
paused-cadence guard usage (``_budget_auto_resume_ready``) is deliberately NOT
wired to this module — it stays a cheap read-only query — and the once-marker
holds under repeated evaluation regardless of cadence.

Marker: a tiny JSON file at ``orbital/budget/notify_state.json`` (owned by
``ProjectPaths.budget_notify_state``), keyed by window identity. A daemon
restart re-reads it (no re-fire); a new window identity (period roll-over /
anchor move) re-arms it. Missing/corrupt marker → re-fire is acceptable (spam is
not — the marker is rewritten valid immediately, so steady state is silent).

Payloads carry CODES/NUMBERS ONLY — no display strings, no English sentences
(the binding i18n rule). The client renders the message from the codes.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# Threshold crossing, fixed at 80% for v1 — no config knob (per spec).
THRESHOLD_PCT = 80

# Event types. The TRIP type is the EXISTING relay-registered Tier-1 type so it
# routes through ``_should_push`` unchanged; the THRESHOLD type is its sibling
# (registered alongside it in relay/client.py). These are the ``type`` field on
# the emitted payload — the relay receiver keys its push tier on it.
EVENT_TRIP = "budget.exhausted"
EVENT_THRESHOLD = "budget.threshold"

# Internal stable codes carried on the payload's ``code`` field (the i18n render
# key). Kept distinct from the relay event ``type`` so the timeline/client has a
# stable code independent of the push-routing type.
CODE_TRIP = "budget_blocked"
CODE_THRESHOLD = "budget_threshold"

# Marker keys: the per-threshold last-fired window identity. A threshold re-fires
# exactly when the stored identity for its key differs from the current one.
_MK_THRESHOLD = "threshold_window"
_MK_TRIP = "trip_window"


def window_identity(window: str, anchor_ts: str | None, now) -> str:
    """A stable identity string for the CURRENT instance of a budget window.

    Two evaluations in the SAME window instance share an identity; a period
    roll-over (daily/weekly/monthly) or a ``total``-window anchor move yields a
    different identity, which re-arms the once-marker.

    Built from the window code + the window's start instant (via the SAME
    ``window_start`` the spend query uses, so identity and spend agree on the
    boundary). For ``total`` the start IS the anchor, so an anchor move changes
    the identity directly.
    """
    from agent_os.budget.ledger import window_start

    start = window_start(window, now=now, anchor_ts=anchor_ts)
    return f"{window}:{start.isoformat()}"


def _read_marker(path: str) -> dict:
    """Read the marker file. Missing/corrupt → empty dict (re-fire acceptable)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        # Corrupt or unreadable: treat as un-fired (re-fire is acceptable; the
        # write below rewrites it valid, so steady state is not spammy).
        logger.warning("budget notify marker unreadable at %s; treating as empty",
                       path, exc_info=True)
    return {}


def _write_marker(path: str, marker: dict) -> None:
    """Persist the marker. ONE write location. Never raises (best-effort)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False)
            f.flush()
    except OSError:
        logger.warning("budget notify marker write failed at %s; "
                       "may re-fire next eval", path, exc_info=True)


def maybe_emit_budget_event(
    project_dir: str,
    budget_config: dict | None,
    *,
    emit,
    spend_fn=None,
    fx_rates: dict | None = None,
    now=None,
    marker_path: str | None = None,
) -> dict | None:
    """Detect threshold/trip crossings and emit AT MOST ONCE each per window.

    Args:
        project_dir: Project workspace root (ledger + marker live under it).
        budget_config: LIVE budget config (same shape ``evaluate_budget`` reads:
            ``budget_limit_usd``, ``budget_currency``, ``budget_period``,
            ``budget_anchor_ts``). No limit (None/0/<=0) → no detection, no-op.
        emit: A callable ``emit(payload: dict)`` that routes the event through
            the existing relay/WS broadcast path. ``None`` → silent no-op (relay
            absent / no wiring): nothing fires, no marker is touched.
        spend_fn: Injectable spend() (tests). Defaults to ``budget.ledger.spend``.
        fx_rates: Static FX pair rates (forwarded to spend, as the guard does).
        now: Override "current time" for window math (tests).
        marker_path: Override marker file path (tests). Defaults to
            ``ProjectPaths(project_dir).budget_notify_state``.

    Returns:
        The post-append management-spend SNAPSHOT this eval computed, as
        ``{"window", "spend", "limit", "currency"}`` — so the caller can feed the
        debounced ``budget.spend_updated`` broadcast WITHOUT a second spend query
        (Budget Piece 3 — Task P3-C, the zero-additional-query goal). Returns
        ``None`` when no spend query ran (``emit=None``, no limit configured, or a
        spend/window-math failure) — in which case the caller runs its own single
        query for the broadcast (the no-limit corner still shows spend). The
        emit/marker behaviour is unchanged regardless of the return value.

    Never raises: a spend-query failure or marker I/O error degrades to "emit
    nothing this eval" (logged), never crashing the caller (the ledger-append
    site). Pure ``evaluate_budget`` is untouched — this re-runs the spend query
    itself, so the dispatcher's cheap paused cadence path is unaffected.
    """
    # Relay absent / no wiring → silent no-op. Do not even read the ledger.
    if emit is None:
        return None

    cfg = budget_config or {}
    limit = cfg.get("budget_limit_usd")
    if limit is None:
        return None
    try:
        limit_val = float(limit)
    except (TypeError, ValueError):
        return None
    if limit_val <= 0:
        return None

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
            # Threshold/trip detection mirrors the guard: MANAGEMENT spend only.
            # Sub-agent ledger lines never cross 80%/100% nor fire a push
            # (P3-A guard isolation).
            sources=["management"],
        )
        spend_amount = float(result["converted_total"]["amount"])
    except Exception:  # noqa: BLE001 — never crash the ledger-append site
        logger.warning("budget notify: spend query failed; emitting nothing",
                       exc_info=True)
        return None

    # The post-append spend snapshot for the debounced spend_updated broadcast.
    # Returned to the caller so it reuses THIS query (no second spend()).
    snapshot = {
        "window": window,
        "spend": spend_amount,
        "limit": limit_val,
        "currency": currency,
    }

    # Compute the current window identity (re-arm key) and load the marker.
    try:
        identity = window_identity(window, anchor_ts, now)
    except Exception:  # noqa: BLE001 — a window-math failure must not crash
        logger.warning("budget notify: window identity failed; emitting nothing",
                       exc_info=True)
        # The spend was computed fine; the broadcast can still use it even
        # though threshold/trip detection is skipped this eval.
        return snapshot

    if marker_path is None:
        from agent_os.agent.project_paths import ProjectPaths
        marker_path = ProjectPaths(project_dir).budget_notify_state

    marker = _read_marker(marker_path)

    pct = (spend_amount / limit_val) * 100.0
    crossed_threshold = pct >= THRESHOLD_PCT
    tripped = spend_amount >= limit_val

    dirty = False

    # Threshold: fire once per window identity. (A jump straight past 80% to
    # 100%+ still fires the threshold once — both events are independent.)
    if crossed_threshold and marker.get(_MK_THRESHOLD) != identity:
        emit({
            "type": EVENT_THRESHOLD,
            "code": CODE_THRESHOLD,
            "pct": THRESHOLD_PCT,
            "spend": spend_amount,
            "limit": limit_val,
            "currency": currency,
            "window": window,
        })
        marker[_MK_THRESHOLD] = identity
        dirty = True

    # Trip: fire once per window identity, via the EXISTING relay-registered type.
    if tripped and marker.get(_MK_TRIP) != identity:
        emit({
            "type": EVENT_TRIP,
            "code": CODE_TRIP,
            "pct": 100,
            "spend": spend_amount,
            "limit": limit_val,
            "currency": currency,
            "window": window,
        })
        marker[_MK_TRIP] = identity
        dirty = True

    if dirty:
        _write_marker(marker_path, marker)

    return snapshot


def maybe_emit_trip(
    project_dir: str,
    *,
    window: str | None,
    spend: float | None,
    limit: float | None,
    currency: str | None,
    anchor_ts: str | None = None,
    emit,
    now=None,
    marker_path: str | None = None,
) -> None:
    """Emit the trip event from the GUARD-TRIP site (``AgentLoop._trip_budget``).

    The second trip-emission site (coordinator ruling): the crossing-append path
    misses trips that happen WITHOUT a new append — a limit lowered below the
    current spend, or a fresh run on an already-over project whose crossing
    append predates the marker. The loop's ``_trip_budget`` already holds the
    tripped ``BudgetDecision`` (window/spend/limit/currency), so no spend query
    is re-run here.

    Dedup: uses the SAME marker file and the SAME ``trip_window`` key as
    ``maybe_emit_budget_event``, so the trip fires exactly once per
    (project, window-identity) regardless of which site observes it first.

    Never raises. ``emit=None`` (relay absent / no wiring) → silent no-op.
    Missing decision fields (window/spend/limit None — shouldn't happen on a
    real trip, but be defensive) → no-op.
    """
    if emit is None:
        return
    if window is None or spend is None or limit is None:
        return

    try:
        identity = window_identity(window, anchor_ts, now)
    except Exception:  # noqa: BLE001 — window-math failure must not crash the loop
        logger.warning("budget notify: trip window identity failed; not emitting",
                       exc_info=True)
        return

    if marker_path is None:
        from agent_os.agent.project_paths import ProjectPaths
        marker_path = ProjectPaths(project_dir).budget_notify_state

    marker = _read_marker(marker_path)
    if marker.get(_MK_TRIP) == identity:
        return  # already fired this window (by either site)

    emit({
        "type": EVENT_TRIP,
        "code": CODE_TRIP,
        "pct": 100,
        "spend": float(spend),
        "limit": float(limit),
        "currency": currency or "USD",
        "window": window,
    })
    marker[_MK_TRIP] = identity
    _write_marker(marker_path, marker)
