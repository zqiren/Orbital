# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Workbench timezone + overdue/age math (spec §7.3).

Pure, dependency-light helpers used by the Workbench read path
(``api/routes/workbench.py``, entry-row age/overdue/days_late) and — for
``project_timezone`` only — the calendar native sources
(``calendar_hub/sources/automation_source.py``, ``memory_source.py``).

- ``project_timezone`` — the single effective-tz rule (explicit setting →
  first schedule trigger's tz → daemon local). The calendar sources import
  this so there is one implementation, not two.
- ``is_overdue`` / ``age_days`` / ``days_late`` — display math computed in the
  project tz, never UTC (a date-only ``due:`` is all-day; overdue rolls at
  local midnight).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone


# ---------------------------------------------------------------------------
# Timezone resolution
# ---------------------------------------------------------------------------

def _resolve_tz(name: str):
    """Resolve an IANA tz name to a tzinfo; UTC on anything unresolvable."""
    if not name:
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        pass
    try:  # pytz is already a dependency (trigger_manager)
        import pytz
        return pytz.timezone(name)
    except Exception:
        return timezone.utc


def _local_tz_name() -> str:
    """Best-effort IANA name for the daemon's local zone.

    ``TZ`` env wins; else the ``/etc/localtime`` symlink target (macOS/Linux);
    else UTC. Returned as a name (not a tzinfo) so callers — including Task 6 —
    can pass it straight back through ``_resolve_tz``.
    """
    tz_env = os.environ.get("TZ")
    if tz_env:
        return tz_env
    try:
        target = os.path.realpath("/etc/localtime")
        marker = "/zoneinfo/"
        i = target.find(marker)
        if i != -1:
            return target[i + len(marker):]
    except Exception:
        pass
    return "UTC"


def project_timezone(project_config: dict, triggers: list[dict]) -> str:
    """The project's effective tz name (spec §7.3).

    Precedence: explicit ``timezone`` on the project config → the first
    schedule trigger's ``schedule.timezone`` → the daemon's local zone name.
    Dependency-light by design: Task 6's calendar sources import this so the
    single-tz rule lives in exactly one place.
    """
    explicit = (project_config or {}).get("timezone")
    if explicit:
        return explicit
    for t in triggers or []:
        if t.get("type", "schedule") != "schedule":
            continue
        tz = (t.get("schedule") or {}).get("timezone")
        if tz:
            return tz
    return _local_tz_name()


def _now(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now


def today_in_tz(tz_name: str, now: datetime | None = None) -> date:
    """Today's calendar date in ``tz_name`` (project tz), not UTC."""
    return _now(now).astimezone(_resolve_tz(tz_name)).date()


def _due_date(due: str | None) -> date | None:
    """The calendar-date component of a raw ``due:`` value (date or datetime)."""
    if not due:
        return None
    try:
        return date.fromisoformat(due[:10])
    except ValueError:
        return None


def is_overdue(due: str | None, tz_name: str, now: datetime | None = None) -> bool:
    """True when ``due``'s date is strictly before today in the project tz.

    Date-only semantics on purpose: an item due today is not overdue until
    local midnight rolls, matching the all-day projection (spec §7.3). The
    boundary is computed in ``tz_name`` — a UTC-vs-project-tz mismatch near
    midnight is exactly what this guards.
    """
    d = _due_date(due)
    if d is None:
        return False
    return d < today_in_tz(tz_name, now)


def age_days(created: str | None, tz_name: str, now: datetime | None = None) -> int:
    """Whole days from ``created`` to today in the project tz (0 if unknown)."""
    c = _due_date(created)
    if c is None:
        return 0
    return max(0, (today_in_tz(tz_name, now) - c).days)


def days_late(due: str | None, tz_name: str, now: datetime | None = None) -> int | None:
    """Whole days ``due`` is past, computed in the project tz; None if not late.

    Non-None exactly when ``is_overdue`` is True (same date-only, project-tz
    boundary) — so the frontend renders "N days late" server-authoritatively
    instead of recomputing in browser tz (spec §7.3).
    """
    d = _due_date(due)
    if d is None:
        return None
    delta = (today_in_tz(tz_name, now) - d).days
    return delta if delta > 0 else None
