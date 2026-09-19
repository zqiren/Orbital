# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native ``memory`` calendar source (spec §7.2, §7.3; spec 089 §3.6).

Two kinds of dated project memory project onto the calendar, re-read at
request time (no derived file, no cache of its own — the hub's TTL cache
covers repeated views):

- **open asks with a ``due:``** from ``orbital/ASKS.md`` (``agent/asks.py``).
  Event identity is the ask id — ``memory/{project_id}/{ask_id}`` — which for
  a migrated ``[user due:]`` entry is the id it already had, so the event
  keeps its id across the upgrade. An ask's event disappears the moment the
  ask is closed and comes back if it is reopened.
- **dated facts** — PROJECT_STATE bullets carrying a ``[due:…]`` tag, via the
  shared ``user_flags`` grammar. Identity is the entry's stamped id when it
  has one, else a stable hash of project id + text (content-dependent, not
  random, so re-parsing an unchanged line always yields the same event id).
  There is no "resolved" state any more: a dated line's event lasts exactly
  as long as the line does. A leftover ``[user due:]`` line (written outside
  the tools, not yet moved into ASKS.md) still shows, so nothing dated
  disappears in the window before it moves.

Every event is pre-linked to its project (``NormalizedEvent.project_id`` set
directly here) — these events exist BECAUSE they belong to a project, unlike
an externally-sourced calendar item a user manually links, so they must not
depend on the ``Linkage`` store. This class declares ``linked_by_hub = False``
(see ``CalendarHub.list_events``) so the hub leaves its stamped
``project_id`` alone instead of overwriting it from the — necessarily empty —
linkage map; without that, the read-only agent tool's project lens would
never see these events at all.

This source never raises: a project with an unreadable/missing file, a decode
error, or a parse failure contributes zero events for that part and is
skipped — the same degrade-gracefully rule every source in this package
follows.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from agent_os.agent import asks, user_flags
from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.workbench_cards import project_timezone

from ..models import NormalizedEvent

logger = logging.getLogger(__name__)


def _resolve_tz(name: str):
    """Resolve an IANA tz name to a tzinfo; UTC on anything unresolvable.

    Mirrors ``workbench_cards._resolve_tz`` — kept local rather than imported
    so this source stays self-contained, matching its sibling source files
    (``eventkit.py``, ``mcp_calendar.py``), each of which owns its own small
    helpers rather than reaching into an unrelated module's private names.
    """
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


def _parse_range_iso(value: str) -> datetime | None:
    """Parse a range boundary (``start``/``end``) into a tz-aware instant."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_due(due: str, tz) -> tuple[datetime, bool]:
    """Return ``(tz-aware instant, all_day)`` for a raw ``due:`` value.

    Grammar: ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM`` — both parsers
    (``user_flags`` and ``asks``) only ever return one of those two shapes.
    Parsed by hand instead of ``datetime.fromisoformat`` so behavior does not
    depend on the running Python version's leniency toward a missing ``:SS``.
    Raises ``ValueError`` on anything else (e.g. an impossible date).
    """
    date_part, _, time_part = due.partition("T")
    year, month, day = (int(x) for x in date_part.split("-"))
    if not time_part:
        return datetime(year, month, day, tzinfo=tz), True
    hour, minute = (int(x) for x in time_part.split(":"))
    return datetime(year, month, day, hour, minute, tzinfo=tz), False


def _mem_id(project_id: str, entry) -> str:
    """The entry's stamped id, or a stable fallback for an un-id'd dated fact."""
    if entry.id:
        return entry.id
    digest = hashlib.sha256(f"{project_id}:{entry.text}".encode("utf-8")).hexdigest()[:12]
    return f"nofrag-{digest}"


class MemorySource:
    """Projects every project's open dated asks and dated facts onto the calendar."""

    id = "memory"
    kind = "memory"
    # Opts out of CalendarHub's linkage-store re-stamping (see hub.py) — this
    # source sets `project_id` itself and is never manually linked/unlinked.
    linked_by_hub = False

    def __init__(self, project_store):
        self._project_store = project_store

    @property
    def available(self) -> bool:
        return True  # computed from local files — no external dependency to fail

    async def list_events(self, start: str, end: str) -> list[NormalizedEvent]:
        start_dt = _parse_range_iso(start)
        end_dt = _parse_range_iso(end)
        if start_dt is None or end_dt is None:
            return []
        events: list[NormalizedEvent] = []
        for project in self._project_store.list_projects():
            events.extend(self._project_events(project, start_dt, end_dt))
        return events

    def _project_events(
        self, project: dict, start_dt: datetime, end_dt: datetime
    ) -> list[NormalizedEvent]:
        workspace = project.get("workspace", "")
        project_id = project.get("project_id", "")
        if not workspace or not project_id:
            return []
        tz_name = project_timezone(project, project.get("triggers", []) or [])
        tz = _resolve_tz(tz_name)
        paths = ProjectPaths(workspace)

        dated: list[tuple[str, str, str]] = []   # (event key, title, due)
        try:
            for a in asks.read_asks(paths.orbital_dir):
                if a.is_open and a.due:
                    dated.append((a.id, a.text, a.due))
        except Exception:
            logger.warning(
                "memory source: reading asks failed for %s", project_id, exc_info=True
            )
        try:
            with open(paths.project_state, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            content = ""
        try:
            for entry in user_flags.parse_entries(content):
                if entry.due:
                    dated.append((_mem_id(project_id, entry), entry.text, entry.due))
        except Exception:
            logger.warning(
                "memory source: parse_entries failed for %s", project_id, exc_info=True
            )

        out: list[NormalizedEvent] = []
        seen: set[str] = set()
        for key, title, due in dated:
            if key in seen:
                continue
            seen.add(key)
            try:
                instant, all_day = _parse_due(due, tz)
            except ValueError:
                continue
            if not (start_dt <= instant < end_dt):
                continue
            iso = instant.date().isoformat() if all_day else instant.isoformat()
            out.append(NormalizedEvent(
                source=self.id,
                source_id=f"{project_id}/{key}",
                title=title,
                start=iso,
                end=iso,
                all_day=all_day,
                timezone=tz_name,
                project_id=project_id,
            ))
        return out
