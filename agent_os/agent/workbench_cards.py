# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Workbench computed cards + timezone helper (spec §5.6, §6, §7.3; Task 5).

Pure, dependency-light detectors used by the Workbench read path
(``api/routes/workbench.py``) and — for ``project_timezone`` only — the
calendar native sources (Task 6). Everything here is mechanical (zero LLM):

- ``project_timezone`` — the single effective-tz rule (explicit setting →
  first schedule trigger's tz → daemon local). Task 6 imports this so there
  is one implementation, not two.
- ``is_overdue`` / ``age_days`` — display math computed in the project tz,
  never UTC (a date-only ``due:`` is all-day; overdue rolls at local midnight).
- ``is_broken_automation`` — an enabled schedule trigger more than one full
  cron period behind its last successful fire (croniter, mirroring
  ``trigger_manager``'s baseline math).
- ``paused_thread_cards`` — a non-running session whose last persisted message
  is an unanswered assistant question.
- ``suppress_entries`` — spec §5.6: a computed card beats a memory entry that
  asserts the same system-state fact.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone

from croniter import croniter

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Broken-automation detector
# ---------------------------------------------------------------------------

def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _trigger_baseline(trigger: dict) -> datetime:
    """max(last_triggered, created_at) — mirrors ``trigger_manager``.

    A missing/unparsable ``last_triggered`` falls back to ``created_at`` (so a
    freshly-created, never-fired trigger is not spuriously "broken"); both
    missing → epoch.
    """
    cands = [
        p for p in (_parse_iso(trigger.get("last_triggered")),
                    _parse_iso(trigger.get("created_at")))
        if p is not None
    ]
    return max(cands) if cands else datetime.min.replace(tzinfo=timezone.utc)


def is_broken_automation(trigger: dict, now: datetime | None = None) -> bool:
    """True when an enabled schedule trigger is > 1 full cron period behind.

    "Broken" = its last successful fire (baseline) predates the occurrence one
    full period before the most recent expected fire — i.e. it has missed the
    latest fire AND is already behind the one before it. That one-period grace
    keeps a just-fired trigger, or a single missed tick, from being flagged.
    Uses croniter in the trigger's own tz, the same math ``trigger_manager``
    uses for its due rule.
    """
    if not trigger.get("enabled", True):
        return False
    if trigger.get("type", "schedule") != "schedule":
        return False
    schedule = trigger.get("schedule") or {}
    cron = schedule.get("cron")
    if not cron or not croniter.is_valid(cron):
        return False
    tz = _resolve_tz(schedule.get("timezone", "UTC"))
    now_local = _now(now).astimezone(tz)
    try:
        itr = croniter(cron, now_local)
        itr.get_prev(datetime)               # most recent expected fire
        prev_prev = itr.get_prev(datetime)   # one full period earlier
    except Exception:
        logger.debug("broken_automation: croniter failed for %r", cron, exc_info=True)
        return False
    if prev_prev.tzinfo is None:
        prev_prev = prev_prev.replace(tzinfo=tz)
    return _trigger_baseline(trigger) < prev_prev


def broken_automation_cards(project_id: str, triggers: list[dict],
                            now: datetime | None = None) -> list[dict]:
    """Computed cards for every broken enabled schedule trigger in a project."""
    cards: list[dict] = []
    for t in triggers or []:
        if not is_broken_automation(t, now):
            continue
        name = t.get("name") or t.get("id") or "automation"
        baseline = _trigger_baseline(t)
        since = (baseline.date().isoformat()
                 if baseline.year > 1 else None)
        cards.append({
            "type": "broken_automation",
            "project_id": project_id,
            "key": t.get("id") or name,
            "text": f'Automation "{name}" has not run on schedule.',
            "since": since,
        })
    return cards


def broken_trigger_names(triggers: list[dict],
                         now: datetime | None = None) -> list[str]:
    """Names of broken enabled schedule triggers (for §5.6 suppression)."""
    names: list[str] = []
    for t in triggers or []:
        if is_broken_automation(t, now):
            name = t.get("name")
            if name:
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# Suppression (spec §5.6): computed truth beats remembered truth
# ---------------------------------------------------------------------------

def suppress_entries(entries: list, broken_names: list[str]) -> tuple[list, set]:
    """Split ``entries`` into (kept, suppressed_line_starts).

    A flagged entry whose text contains a broken trigger's name
    (case-insensitive) is suppressed — the ``broken_automation`` computed card
    asserts the same fact and wins. ``entries`` are ``user_flags.Entry``
    objects; the returned suppressed set holds their ``line_start`` values so
    callers can key by identity without importing the dataclass.
    """
    if not broken_names:
        return list(entries), set()
    lowered = [n.lower() for n in broken_names if n]
    kept: list = []
    suppressed: set = set()
    for e in entries:
        text = (e.text or "").lower()
        if any(name in text for name in lowered):
            suppressed.add(e.line_start)
        else:
            kept.append(e)
    return kept, suppressed


# ---------------------------------------------------------------------------
# Paused-thread detector
# ---------------------------------------------------------------------------

def _message_text(msg: dict) -> str:
    """Extract plain text from a persisted message (string or block list)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "".join(parts)
    return ""


def paused_thread_cards(project_id: str, sessions: list[dict],
                        tail_reader) -> list[dict]:
    """Computed cards for sessions paused on an unanswered agent question.

    Heuristic (documented, cheapest reliable signal): for each session that is
    NOT actively running (status not ``running``/``pending_approval``), read
    its last persisted message via ``tail_reader(session_uuid)``; if that
    message is an assistant turn whose text ends in ``?`` it is a paused
    thread. Newest (by ``last_activity_at``) first. ``tail_reader`` returns the
    last message dict or ``None``; any read failure yields no card (best
    effort — a missing signal must never 500 the Workbench).
    """
    cards: list[dict] = []
    for s in sessions or []:
        if s.get("status") in ("running", "pending_approval"):
            continue
        uuid = s.get("session_uuid")
        if not uuid:
            continue
        try:
            tail = tail_reader(uuid)
        except Exception:
            logger.debug("paused_thread: tail read failed for %s", uuid, exc_info=True)
            continue
        if not tail or tail.get("role") != "assistant":
            continue
        text = _message_text(tail).strip()
        if not text.endswith("?"):
            continue
        cards.append({
            "type": "paused_thread",
            "project_id": project_id,
            "key": uuid,
            "text": text,
            "since": tail.get("timestamp") or s.get("last_activity_at"),
        })
    cards.sort(key=lambda c: c.get("since") or "", reverse=True)
    return cards
