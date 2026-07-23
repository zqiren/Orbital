# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native ``memory`` calendar source (spec §7.2, §7.3; Task 6).

Every project's ``orbital/PROJECT_STATE.md`` already carries dated
commitments — flagged ``[user due:...]`` entries and unflagged dated facts
(``[due:...]``) — via the shared ``user_flags`` grammar (Task 1). This source
re-parses those files at request time (no derived file, no cache of its own —
the hub's own TTL cache covers repeated views) and projects every entry that
still has an open ``due:`` into the calendar.

A ``resolved:`` stamp closes an entry (spec §5.3 — the fulfilled exit sets it,
and the Workbench read path already treats a resolved dated fact as closed);
such an entry never emits an event even when its ``due:`` date falls inside
the requested range.

Event identity mirrors ``memory/{project_id}/{mem_id}`` (``source="memory"``,
``source_id="{project_id}/{mem_id}"``). ``mem_id`` is the entry's stamped id
when present; a dated fact that predates id-stamping (the write chokepoint
only stamps NEW *flagged* bullets — spec §5.2) has no id, so a stable
fallback key is derived from a hash of the project id + entry text —
content-dependent, not random, so re-parsing the same unchanged entry always
yields the same event id across requests (it only changes if the entry text
itself is edited, which is acceptable: nothing external references this
fallback id, unlike a real stamped one).

Every entry is pre-linked to its project (``NormalizedEvent.project_id`` set
directly here) — these events exist BECAUSE they belong to a project, unlike
an externally-sourced calendar item a user manually links, so they must not
depend on the ``Linkage`` store. This class declares ``linked_by_hub = False``
(see ``CalendarHub.list_events``, Task 6) so the hub leaves its stamped
``project_id`` alone instead of overwriting it from the — necessarily empty —
linkage map; without that, the read-only agent tool's project lens would
never see these events at all.

This source never raises: a project with an unreadable/missing state file, a
decode error, or a parse failure contributes zero events for that project and
is skipped — the same degrade-gracefully rule every source in this package
follows.

Note (scope decision — see Task 6 report): the spec's "kind = user-flagged /
dated-fact" per-event distinction has no home on the currently shipped
``NormalizedEvent`` — its ``to_dict()`` field set is locked by an existing
test (``test_calendar_hub.py::test_normalized_event_id_and_dict``) outside
this task's file scope, so no field was added for it here. Flagged vs
dated-fact is still recoverable downstream by re-parsing the same
PROJECT_STATE file if a future surface needs to render the distinction.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from agent_os.agent import user_flags
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

    Grammar (``user_flags._DUE_RE``): ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM`` —
    always exactly one of those two shapes when not ``None``, and always
    valid (``parse_entries`` already discards malformed ``due:`` values, so a
    non-None ``entry.due`` here is guaranteed well-formed). Parsed by hand
    instead of ``datetime.fromisoformat`` so behavior does not depend on the
    running Python version's leniency toward a missing ``:SS``.
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
    """Projects every project's open ``due:`` entries onto the calendar."""

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
        try:
            with open(
                ProjectPaths(workspace).project_state, "r",
                encoding="utf-8", errors="replace",
            ) as f:
                content = f.read()
        except OSError:
            return []
        try:
            entries = user_flags.parse_entries(content)
        except Exception:
            logger.warning(
                "memory source: parse_entries failed for %s", project_id, exc_info=True
            )
            return []

        tz_name = project_timezone(project, project.get("triggers", []) or [])
        tz = _resolve_tz(tz_name)

        out: list[NormalizedEvent] = []
        for entry in entries:
            if not entry.due or entry.resolved:
                continue
            instant, all_day = _parse_due(entry.due, tz)
            if not (start_dt <= instant < end_dt):
                continue
            iso = instant.date().isoformat() if all_day else instant.isoformat()
            mem_id = _mem_id(project_id, entry)
            out.append(NormalizedEvent(
                source=self.id,
                source_id=f"{project_id}/{mem_id}",
                title=entry.text,
                start=iso,
                end=iso,
                all_day=all_day,
                timezone=tz_name,
                project_id=project_id,
            ))
        return out
