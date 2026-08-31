# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Append-only token ledger (Budget Piece 1).

One JSONL line per LLM response, stored per-project at::

    {workspace}/orbital/ledger/usage.jsonl

The path is owned by ``ProjectPaths.ledger_file`` (``agent_os/agent/
project_paths.py``) — the single owner of all Orbital on-disk layout;
``ledger_path()`` here delegates to it.

Design rules in force:

  - **Tokens are the stored fact.** No dollars in the ledger — derived cost
    views live in a later piece and read these token fields against a rate
    table. The four token fields are DISJOINT (see ``budget/normalize.py``):
    ``uncached_input``, ``cache_read``, ``cache_write``, ``output``.
  - **Append-only.** Events are only ever appended + flushed, never rewritten.
    Low frequency (one per LLM response), so no batching.
  - **Resilient.** A write failure (unwritable dir, disk error) must never
    crash or alter the agent loop — ``append_event`` logs a warning and
    returns. The caller continues regardless.

A later task adds a ``spend()`` query over the same file; the flat per-line
schema here is chosen so reading events back is a plain ``json.loads`` per
line. This module does NOT implement querying.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover — type-only; avoids an import cycle.
    from agent_os.budget.normalize import NormalizedUsage

logger = logging.getLogger(__name__)

# Event source enum. "management" is the management agent's own LLM responses;
# the "subagent:*" values are emitted by the sub-agent transports (P3-B) and
# captured for the DISPLAY view only — never counted toward enforcement (the
# guard/notify pass sources=["management"]; see the P3-A pinned regression).
SOURCE_MANAGEMENT = "management"
SOURCE_SUBAGENT_CLAUDE_CODE = "subagent:claude-code"
SOURCE_SUBAGENT_CODEX = "subagent:codex"

# Valid query windows (codes only — no display strings; per the binding i18n
# rule, all human-readable labels render client-side).
WINDOWS = ("daily", "weekly", "monthly", "total")

# Rounding precision applied at the response EDGE only (never during
# accumulation), matching the legacy budget path's 6dp.
_COST_ROUND_DP = 6


def ledger_path(project_dir: str) -> str:
    """Return the absolute ledger path for a project workspace.

    ``{project_dir}/orbital/ledger/usage.jsonl``. Pure path calculation — no
    I/O. Delegates to ``ProjectPaths.ledger_file`` so the layout has exactly
    one owner; a later ``spend()`` query reads the file back through this
    same function.
    """
    # Imported lazily: project_paths lives under the agent package, whose
    # __init__ eagerly imports loop -> this module. A module-level import
    # here would be a circular import (same cycle as budget.normalize).
    from agent_os.agent.project_paths import ProjectPaths

    return ProjectPaths(project_dir).ledger_file


@dataclass(frozen=True)
class LedgerEvent:
    """One LLM-response usage event, prior to serialization.

    ``ts`` is filled in at append time (ISO8601 UTC) so callers don't have to.
    ``usage`` carries the four disjoint token counts from ``NormalizedUsage``.

    ``reported_cost`` / ``reported_cost_currency`` are an OPTIONAL provider-
    reported cost (P3-B): the SDK transport records claude-code's
    ``total_cost_usd`` verbatim here when it is present AND > 0 (subscription
    auth reports 0/absent → omitted). It is displayed VERBATIM by the cost
    endpoint and NEVER recomputed from our rate table. Both fields are
    serialized only when ``reported_cost`` is not None.
    """

    session_id: str
    source: str
    provider: str
    model: str
    usage: NormalizedUsage
    reported_cost: float | None = None
    reported_cost_currency: str | None = None

    def to_record(self, ts: str) -> dict:
        """Flatten to the on-disk JSON record (disjoint token fields inline)."""
        record = {
            "ts": ts,
            "session_id": self.session_id,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "uncached_input": self.usage.uncached_input,
            "cache_read": self.usage.cache_read,
            "cache_write": self.usage.cache_write,
            "output": self.usage.output,
        }
        # Only serialize the provider-reported cost when present — absent for
        # every management row and for subscription-auth subagent rows.
        if self.reported_cost is not None:
            record["reported_cost"] = self.reported_cost
            record["reported_cost_currency"] = self.reported_cost_currency
        return record


# ---------------------------------------------------------------------------
# Post-append hook — the single spend-notification chokepoint.
#
# THREE call sites append to the ledger: the management loop
# (``AgentLoop._emit_ledger_event``) and BOTH sub-agent transports
# (``SdkTransport._capture_usage``, ``CodexTransport._capture_usage``). Only the
# management one used to notify the UI, so sub-agent spend changed the ledger
# without ever refreshing the settings breakdown. Registering the notification
# HERE — the one function every appender crosses — fixes all three at once and
# gives the next transport the behaviour for free, mirroring the telemetry
# counter already emitted below.
#
# Keyed by project workspace so a multi-project daemon routes each append to the
# right loop; single-slot per project (the newest registration wins), and
# ``unregister_append_hook`` is identity-checked so a torn-down loop cannot
# unregister its successor.
# ---------------------------------------------------------------------------

_append_hooks: dict[str, Callable[[str, "LedgerEvent"], None]] = {}


def register_append_hook(
    project_dir: str, hook: Callable[[str, "LedgerEvent"], None],
) -> None:
    """Install THE post-append hook for one project workspace.

    Replaces any previously registered hook for the same ``project_dir``.
    """
    _append_hooks[project_dir] = hook


def unregister_append_hook(
    project_dir: str, hook: Callable[[str, "LedgerEvent"], None],
) -> None:
    """Remove ``hook`` iff it is STILL the one registered for ``project_dir``.

    The identity check matters on the hand-off between two loops for the same
    project: the outgoing loop's ``terminate()`` must not silence the incoming
    loop that already replaced it.
    """
    if _append_hooks.get(project_dir) is hook:
        _append_hooks.pop(project_dir, None)


def _fire_append_hook(project_dir: str, event: LedgerEvent) -> None:
    """Run the registered hook. NEVER raises, never blocks an append."""
    hook = _append_hooks.get(project_dir)
    if hook is None:
        return
    try:
        hook(project_dir, event)
    except Exception:  # noqa: BLE001 — a notification failure must never
        # affect the append that triggered it, nor the caller's turn.
        logger.warning(
            "Token-ledger append hook failed for project_dir=%s; continuing.",
            project_dir, exc_info=True,
        )


def append_event(project_dir: str, event: LedgerEvent) -> None:
    """Append one event line to the project's usage ledger.

    Creates ``{project_dir}/ledger/`` on demand, then appends a single JSON
    line and flushes. The timestamp is stamped here as ISO8601 UTC.

    Resilience contract: this function NEVER raises. Any failure (unwritable
    directory, disk error, serialization issue) is logged at WARNING and
    swallowed so the agent loop's behavior is unaffected.
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        record = event.to_record(ts)
        path = ledger_path(project_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:  # noqa: BLE001 — resilience: never break the loop
        logger.warning(
            "Token-ledger append failed for project_dir=%s (session=%s); "
            "continuing without it.",
            project_dir, getattr(event, "session_id", "?"),
            exc_info=True,
        )
        return
    # Telemetry turn counter (spec 046 §4): this is the single chokepoint every
    # LLM response crosses (management loop + both sub-agent transports), so
    # one hook here covers all sources. Counters only — no session/project ids.
    try:
        from agent_os import telemetry

        telemetry.emit(
            "turn_completed",
            {
                "provider": event.provider,
                "source": event.source,
                "uncached_input": event.usage.uncached_input,
                "cache_read": event.usage.cache_read,
                "cache_write": event.usage.cache_write,
                "output": event.usage.output,
            },
        )
        telemetry.latch("first_turn")
    except Exception:  # noqa: BLE001
        pass
    # Spend-notification chokepoint (see the hook block above): the UI refresh
    # for EVERY appender — management loop and both sub-agent transports. Runs
    # only after a SUCCESSFUL write (the early return above skips it), so the
    # broadcast never claims spend that was not recorded. Self-guarded.
    _fire_append_hook(project_dir, event)


# ===========================================================================
# Derived cost views (Budget Piece 1, Task 4)
#
# Cost is computed at QUERY time: tokens × the CURRENTLY-resolved rate. The
# rates table is never snapshotted into stored state — editing it changes
# historical cost, which is intended. ``spend()`` resolves rates per (provider,
# model) on every call via ``resolve_rates`` (which is itself mtime-invalidated)
# so the daemon need not restart for a rate edit to take effect.
# ===========================================================================

from datetime import timedelta  # noqa: E402 — grouped with the cost-view block


def window_start(
    window: str,
    *,
    now: datetime | None = None,
    anchor_ts: str | None = None,
) -> datetime:
    """Return the inclusive lower bound of a spend window as an aware UTC datetime.

    Windows are computed in the daemon's LOCAL timezone (calendar boundaries
    are local per spec), then converted to UTC for comparison against event
    timestamps (which are stored as ISO8601 UTC):

      - ``daily``   → today 00:00 local
      - ``weekly``  → most recent Monday 00:00 local
      - ``monthly`` → 1st of the current month 00:00 local
      - ``total``   → ``anchor_ts`` (parsed); if unset → epoch (everything)

    Args:
        window: One of ``WINDOWS``.
        now: Override for "current time" (aware or naive-local). Tests pass a
            timezone-controlled value to exercise the local-midnight boundary
            without sleeping. Defaults to ``datetime.now().astimezone()``.
        anchor_ts: ISO8601 timestamp for the ``total`` window's lower bound.

    Returns:
        An aware UTC datetime. For ``total`` with no/invalid anchor, returns
        the Unix epoch (1970-01-01 UTC) so every event is included.
    """
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; expected one of {WINDOWS}")

    if window == "total":
        if not anchor_ts:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        try:
            parsed = _parse_ts(anchor_ts)
        except (ValueError, TypeError):
            # Defensive: a malformed anchor must not crash the query. Treat as
            # "no anchor" → everything.
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    # daily / weekly / monthly are calendar boundaries in LOCAL time.
    #
    # When the caller omits ``now`` we use the daemon's machine-local zone
    # (``datetime.now().astimezone()`` attaches the local offset). When the
    # caller PASSES an aware ``now``, its own tzinfo defines "local" — we
    # compute the calendar boundary in that zone rather than re-projecting to
    # the machine zone, so tests can pin a timezone deterministically. A naive
    # ``now`` is interpreted as machine-local wall-clock.
    if now is None:
        local_now = datetime.now().astimezone()
    elif now.tzinfo is None:
        local_now = now.astimezone()  # attach machine-local offset
    else:
        local_now = now  # already aware → its tzinfo IS the local zone

    if window == "daily":
        start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif window == "weekly":
        midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_local = midnight - timedelta(days=midnight.weekday())  # Monday=0
    else:  # monthly
        start_local = local_now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

    return start_local.astimezone(timezone.utc)


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO8601 timestamp into an aware datetime.

    Events are written with an explicit UTC offset (``datetime.now(timezone.utc)
    .isoformat()`` → ``...+00:00``). A legacy/naive timestamp without an offset
    is assumed to be UTC (the writer has always stamped UTC).
    """
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _iter_events(path: str):
    """Yield parsed event dicts from a ledger file, skipping bad lines.

    Malformed / legacy lines (unparseable JSON, missing required keys, bad
    timestamps) are skipped — a corrupt line must never crash a read-only
    query. Skips are reported as ONE summary WARNING per read (not per line),
    so a heavily corrupted file polled by GET /cost cannot flood the log.
    Returns nothing if the file does not exist.
    """
    if not os.path.exists(path):
        return
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        logger.warning("ledger read failed to open %s; treating as empty", path,
                       exc_info=True)
        return
    skipped = 0
    first_bad_lineno: int | None = None
    with f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError("line is not a JSON object")
                # Required fields for cost attribution. A line missing any of
                # these is unusable; skip it.
                _ = (
                    rec["ts"],
                    rec["provider"],
                    rec["model"],
                    rec["source"],
                )
                # Validate the timestamp eagerly so the per-line skip happens
                # here (not mid-aggregation).
                _parse_ts(rec["ts"])
            except (ValueError, KeyError, TypeError):
                skipped += 1
                if first_bad_lineno is None:
                    first_bad_lineno = lineno
                continue
            yield rec
    if skipped:
        logger.warning(
            "skipped %d malformed ledger line(s) in %s (first at line %d)",
            skipped, path, first_bad_lineno,
        )


_TOKEN_FIELDS = ("uncached_input", "cache_read", "cache_write", "output")


def _convert(amount: float, from_ccy: str, to_ccy: str, fx_rates: dict) -> float | None:
    """Convert ``amount`` from one ISO-4217 currency to another.

    ``fx_rates`` carries static, user-editable pair rates keyed ``"{A}_per_{B}"``
    (today only ``"CNY_per_USD"`` is meaningful). Same-currency conversion is
    identity. An unknown pair returns ``None`` (caller marks the slice and omits
    it from the converted total) — we NEVER invent a rate.
    """
    if from_ccy == to_ccy:
        return amount
    cny_per_usd = None
    if isinstance(fx_rates, dict):
        raw = fx_rates.get("CNY_per_USD")
        if isinstance(raw, (int, float)) and raw > 0:
            cny_per_usd = float(raw)
    if cny_per_usd is None:
        return None
    if from_ccy == "USD" and to_ccy == "CNY":
        return amount * cny_per_usd
    if from_ccy == "CNY" and to_ccy == "USD":
        return amount / cny_per_usd
    # Any other pair (e.g. EUR↔CNY) is unsupported today.
    return None


def spend(
    project_dir: str,
    window: str,
    *,
    target_currency: str | None = None,
    fx_rates: dict | None = None,
    now: datetime | None = None,
    anchor_ts: str | None = None,
    override_path: str | None = None,
    sources: list[str] | None = None,
) -> dict:
    """Aggregate token usage + derived cost for a project over a window.

    Cost is computed at QUERY time (``tokens × current resolved rate``) — the
    rates table is NEVER snapshotted into stored state. Editing the rates (or
    the pricing-overrides file) changes the reported cost of historical events;
    this is intended.

    Args:
        project_dir: Project workspace root (ledger lives under it).
        window: One of ``WINDOWS`` (``daily``/``weekly``/``monthly``/``total``).
        target_currency: ISO-4217 code for ``converted_total``. Defaults to the
            project's ``budget_currency`` (caller passes it); falls back to USD
            when neither is set.
        fx_rates: Static FX pair rates from daemon config
            (``{"CNY_per_USD": 7.2}``). Used for ``converted_total`` and for
            converting per-currency subtotals into the target.
        now: Override "current time" for window math (tests).
        anchor_ts: ISO8601 anchor for the ``total`` window.
        override_path: Pricing-override file path forwarded to ``resolve_rates``
            (tests inject a temp file here).
        sources: Event-source filter. ``None`` (default) means ALL sources —
            the GET /cost display view aggregates everything. A non-None list
            keeps ONLY events whose ``source`` exactly matches a list member
            (e.g. ``["management"]`` so the pre-flight guard and threshold
            detector never count sub-agent usage toward enforcement). Empty
            list → matches nothing (everything excluded).

    Returns:
        A dict with ``window``, ``by_currency``, ``converted_total`` (carrying
        ``estimated: true`` and an ``unconverted`` list of currencies that
        could not be converted), and ``breakdown`` (per-model × per-tier ×
        per-source). All amounts rounded to 6dp at the edge; all token counts
        are exact integers. Contains codes/enums/ISO currency codes only — no
        display strings.
    """
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; expected one of {WINDOWS}")

    # Resolve lazily to avoid importing pricing at module import time (keeps the
    # writer path dependency-light) and to honor the query-time-rates rule.
    from agent_os.agent.pricing import resolve_rates

    fx = fx_rates or {}
    start_utc = window_start(window, now=now, anchor_ts=anchor_ts)
    path = ledger_path(project_dir)

    # Optional exact-match source filter. ``None`` → keep everything (display
    # view); a list → keep only the listed sources (the guard/notify pass
    # ``["management"]`` so sub-agent usage never counts toward enforcement).
    # A set membership test keeps the per-line check O(1).
    source_filter = None if sources is None else set(sources)

    # Aggregate token counts by (provider, model, source). Cost is derived AFTER
    # accumulation so we never round mid-sum.
    groups: dict[tuple[str, str, str], dict[str, int]] = {}
    # Provider-reported cost (P3-B) accumulated per group, keyed by currency so
    # a group never mixes currencies. Only sub-agent rows carry it (claude-code
    # total_cost_usd); displayed verbatim, never recomputed from our rates.
    reported: dict[tuple[str, str, str], dict[str, float]] = {}
    for rec in _iter_events(path):
        try:
            ts = _parse_ts(rec["ts"])
        except (ValueError, TypeError):
            continue  # already filtered in _iter_events, but be defensive
        if ts < start_utc:
            continue
        if source_filter is not None and rec["source"] not in source_filter:
            continue
        key = (rec["provider"], rec["model"], rec["source"])
        bucket = groups.setdefault(key, {f: 0 for f in _TOKEN_FIELDS})
        for fld in _TOKEN_FIELDS:
            val = rec.get(fld, 0)
            if isinstance(val, int):
                bucket[fld] += val
        # Provider-reported cost is optional and additive across a group's rows.
        rc = rec.get("reported_cost")
        if isinstance(rc, (int, float)):
            ccy = rec.get("reported_cost_currency")
            if isinstance(ccy, str) and ccy:
                rbucket = reported.setdefault(key, {})
                rbucket[ccy] = rbucket.get(ccy, 0.0) + float(rc)

    breakdown: list[dict] = []
    by_currency: dict[str, float] = {}
    for (provider, model, source), tokens in sorted(groups.items()):
        rates = resolve_rates(provider, model, override_path=override_path)
        cost_uncached = tokens["uncached_input"] * rates.input_per_1m / 1_000_000
        cost_cache_read = tokens["cache_read"] * rates.cached_input_per_1m / 1_000_000
        cost_cache_write = tokens["cache_write"] * rates.cache_write_per_1m / 1_000_000
        cost_output = tokens["output"] * rates.output_per_1m / 1_000_000
        cost_total = cost_uncached + cost_cache_read + cost_cache_write + cost_output

        by_currency[rates.currency] = by_currency.get(rates.currency, 0.0) + cost_total

        row = {
            "provider": provider,
            "model": model,
            "source": source,
            "currency": rates.currency,
            "tokens": dict(tokens),
            "cost": {
                "uncached_input": round(cost_uncached, _COST_ROUND_DP),
                "cache_read": round(cost_cache_read, _COST_ROUND_DP),
                "cache_write": round(cost_cache_write, _COST_ROUND_DP),
                "output": round(cost_output, _COST_ROUND_DP),
                "total": round(cost_total, _COST_ROUND_DP),
            },
        }
        # Surface the provider-reported cost verbatim (per-currency) so the
        # cost route can move sub-agent rows into a separate block carrying it.
        # Single-currency in practice (claude-code is USD); a dict keeps the
        # contract honest if a row ever spans currencies.
        rbucket = reported.get((provider, model, source))
        if rbucket:
            row["reported_cost"] = {
                ccy: round(amount, _COST_ROUND_DP)
                for ccy, amount in sorted(rbucket.items())
            }
        breakdown.append(row)

    # Target currency for the converted total.
    target = target_currency or "USD"

    converted_amount = 0.0
    unconverted: list[str] = []
    for ccy, amount in by_currency.items():
        conv = _convert(amount, ccy, target, fx)
        if conv is None:
            unconverted.append(ccy)
        else:
            converted_amount += conv

    converted_total = {
        "currency": target,
        "amount": round(converted_amount, _COST_ROUND_DP),
        "estimated": True,  # always FX-derived / approximate; never a live rate
    }
    if unconverted:
        # Slices we could not convert (unknown FX pair). The amount above omits
        # them; surface the offending currency codes so the client can flag it.
        converted_total["unconverted"] = sorted(unconverted)

    return {
        "window": window,
        "by_currency": {
            ccy: round(amount, _COST_ROUND_DP) for ccy, amount in by_currency.items()
        },
        "converted_total": converted_total,
        "breakdown": breakdown,
    }


def last_context_usage(project_dir: str, session_id: str) -> dict | None:
    """Prompt size of the most recent MANAGEMENT call for ``session_id``.

    This is the whole read path behind the composer's context meter. No new
    bookkeeping was needed for it: every management response already appends a
    row here, and because ``NormalizedUsage``'s four fields are disjoint,
    ``uncached_input + cache_read + cache_write`` reconstructs the exact prompt
    the provider billed for — provider-reported, not the ``len/4`` estimate
    ``ContextManager`` uses internally.

    Sub-agent rows land in the same project ledger but carry their own
    contexts, so they are filtered out; counting one would make the meter jump
    on every dispatch. ``provider``/``model`` come from the row rather than the
    project config so a mid-session fallback rotation moves the window with the
    model that actually served.

    Returns ``None`` when the session has never made a management call — the
    caller renders nothing, which is different from rendering zero.
    """
    latest: dict | None = None
    for rec in _iter_events(ledger_path(project_dir)):
        if rec.get("session_id") != session_id:
            continue
        if rec.get("source") != SOURCE_MANAGEMENT:
            continue
        latest = rec
    if latest is None:
        return None
    used = sum(
        int(latest.get(f) or 0)
        for f in ("uncached_input", "cache_read", "cache_write")
    )
    return {
        "used": used,
        "provider": latest.get("provider", ""),
        "model": latest.get("model", ""),
        "ts": latest.get("ts", ""),
    }
