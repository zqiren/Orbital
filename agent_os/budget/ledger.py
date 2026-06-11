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
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-only; avoids an import cycle.
    from agent_os.budget.normalize import NormalizedUsage

logger = logging.getLogger(__name__)

# Event source enum. Only "management" is captured today; the subagent values
# are RESERVED for a later piece and intentionally NOT emitted here.
SOURCE_MANAGEMENT = "management"

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
    """

    session_id: str
    source: str
    provider: str
    model: str
    usage: NormalizedUsage

    def to_record(self, ts: str) -> dict:
        """Flatten to the on-disk JSON record (disjoint token fields inline)."""
        return {
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

        breakdown.append({
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
        })

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
