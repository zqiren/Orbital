# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native ``automation`` calendar source (spec §7.2, §7.3; Task 6).

Projects FUTURE occurrences of every project's enabled schedule triggers into
the calendar — a recurring job already carries all its scheduling
information (cron + timezone) on the trigger itself, so this source needs no
separate store, just ``croniter`` expansion in the trigger's own timezone
(mirrors the cron math in ``trigger_manager``). Occurrences before today are
never emitted (no synthesized history — that belongs to ``TriggerManager``),
but today's already-fired slots ARE, so a same-day overview shows the whole
day (rev 6 Workbench "Today" strip).

Event identity: ``automation/{project_id}/{trigger_id}/{occurrence-iso}``
(``source="automation"``, ``source_id="{project_id}/{trigger_id}/{occurrence
-iso}"``) — every occurrence gets its own event id since each is a distinct
future instant, not one standing event. Pre-linked to its project for the
same reason ``memory_source`` is — see that module's docstring and the
``CalendarHub`` fix it documents (source-set ``project_id`` is now preserved
instead of always being overwritten from the linkage store).

This source never raises: a malformed cron expression, an unknown timezone,
a disabled/non-schedule trigger, or a trigger missing its id all just
contribute zero events for that one trigger — never a 500 for the merged
feed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from croniter import croniter

from agent_os.agent.workbench_cards import project_timezone

from ..models import NormalizedEvent
from .memory_source import _parse_range_iso, _resolve_tz

logger = logging.getLogger(__name__)

# Circuit breaker: every caller (the REST route and the read-only agent tool)
# validates the query range to at most 90 days (spec §7.5), so even a
# once-a-minute cron produces well under this many occurrences in-range.
# Guards against an unbounded loop if a source is ever driven directly with a
# wider range.
_MAX_OCCURRENCES_PER_TRIGGER = 5000


class AutomationSource:
    """Future cron occurrences of every project's enabled schedule triggers."""

    id = "automation"
    kind = "automation"
    # Opts out of CalendarHub's linkage-store re-stamping (see hub.py) — this
    # source sets `project_id` itself and is never manually linked/unlinked.
    linked_by_hub = False

    def __init__(self, project_store, *, now_fn=None):
        self._project_store = project_store
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    @property
    def available(self) -> bool:
        return True  # computed from project config — no external dependency to fail

    async def list_events(self, start: str, end: str) -> list[NormalizedEvent]:
        start_dt = _parse_range_iso(start)
        end_dt = _parse_range_iso(end)
        if start_dt is None or end_dt is None:
            return []
        now = self._now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        events: list[NormalizedEvent] = []
        for project in self._project_store.list_projects():
            project_id = project.get("project_id", "")
            if not project_id:
                continue
            triggers = project.get("triggers", []) or []
            for trigger in triggers:
                events.extend(self._trigger_events(
                    project, project_id, trigger, triggers, start_dt, end_dt, now,
                ))
        return events

    def _trigger_events(
        self, project: dict, project_id: str, trigger: dict, triggers: list[dict],
        start_dt: datetime, end_dt: datetime, now: datetime,
    ) -> list[NormalizedEvent]:
        if trigger.get("type", "schedule") != "schedule":
            return []
        if not trigger.get("enabled", True):
            return []
        trigger_id = trigger.get("id")
        if not trigger_id:
            return []
        schedule = trigger.get("schedule") or {}
        cron = schedule.get("cron")
        if not cron or not croniter.is_valid(cron):
            return []
        tz_name = schedule.get("timezone") or project_timezone(project, triggers)
        tz = _resolve_tz(tz_name)

        # Occurrences strictly after max(range start, start of TODAY in the
        # trigger's tz). Earlier days stay out (no synthesized history), but
        # today's already-fired slots ARE emitted — the Workbench "Today"
        # overview needs the whole day, not just the remainder (rev 6).
        today_floor = now.astimezone(tz).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        lower = max(start_dt.astimezone(tz), today_floor)
        try:
            itr = croniter(cron, lower)
        except Exception:
            logger.debug(
                "automation source: croniter init failed for %r", cron, exc_info=True
            )
            return []

        title = trigger.get("name") or trigger_id
        out: list[NormalizedEvent] = []
        for _ in range(_MAX_OCCURRENCES_PER_TRIGGER):
            try:
                occ = itr.get_next(datetime)
            except Exception:
                logger.debug(
                    "automation source: croniter.get_next failed for %r", cron, exc_info=True
                )
                break
            if occ.tzinfo is None:
                occ = occ.replace(tzinfo=tz)
            if occ >= end_dt:
                break
            occ_iso = occ.isoformat()
            # Nominal 30-minute block: zero-duration events collapse to
            # sliver-height rows on the week grid and overprint each other
            # (observed live, 2026-07-24).
            end_iso = (occ + timedelta(minutes=30)).isoformat()
            out.append(NormalizedEvent(
                source=self.id,
                source_id=f"{project_id}/{trigger_id}/{occ_iso}",
                title=title,
                start=occ_iso,
                end=end_iso,
                all_day=False,
                timezone=tz_name,
                project_id=project_id,
            ))
        return out
