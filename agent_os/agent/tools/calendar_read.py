# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read-only calendar tool (spec §7.5; Task 6).

No calendar CRUD tools exist by design (spec §7.5): a dated commitment is a
PROJECT_STATE entry with ``due:``, a recurring job is a trigger, and both
project onto the calendar automatically via the ``memory``/``automation``
native sources (Task 6). The only agent-facing calendar surface is this READ
tool — the merged feed for a range, lensed to the calling agent's own project
(an agent never sees another project's calendar through this tool), for
planning/prep.

Range validation is reused, not reimplemented, from the REST route
(``agent_os.api.routes.calendar._validate_range``) — spec §7.5 "reuses REST
range validation": missing/unparseable ISO datetimes or a span over 90 days
are rejected exactly like the REST ``/calendar/events`` endpoint rejects
them.
"""

from __future__ import annotations

import json

from fastapi import HTTPException

from agent_os.api.routes.calendar import _validate_range

from .base import Tool, ToolResult


def _compact_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class CalendarReadTool(Tool):
    """Read-only merged calendar feed for the calling agent's own project."""

    is_async = True  # backed by CalendarHub.list_events, which is async

    def __init__(self, *, calendar_hub, project_id: str):
        self._hub = calendar_hub
        self._project_id = project_id
        self.name = "calendar_read"
        self.description = (
            "Read the merged calendar feed (native due-date/automation "
            "events, plus any linked external calendar events) for THIS "
            "project only, within an ISO 8601 date range (max 90 days). "
            "Read-only — there is no calendar write tool: a dated "
            "commitment is a PROJECT_STATE entry with due:, a recurring job "
            "is a trigger; both appear here automatically."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "start": {
                    "type": "string",
                    "description": "ISO 8601 range start.",
                },
                "end": {
                    "type": "string",
                    "description": "ISO 8601 range end (at most 90 days after start).",
                },
            },
            "required": ["start", "end"],
            "additionalProperties": False,
        }

    async def execute(self, **arguments) -> ToolResult:
        try:
            start = arguments.get("start")
            end = arguments.get("end")
            if not isinstance(start, str) or not isinstance(end, str):
                return ToolResult(content="Error: start and end must be ISO 8601 strings")
            if self._hub is None:
                return ToolResult(content="Error: calendar is not available")
            try:
                _validate_range(start, end)
            except HTTPException as exc:
                return ToolResult(content=f"Error: {exc.detail}")
            events = await self._hub.list_events(start, end, project_id=self._project_id)
            return ToolResult(content=_compact_json({
                "events": [ev.to_dict() for ev in events],
            }))
        except Exception as exc:
            return ToolResult(content=f"Error: unable to read calendar: {exc}")
