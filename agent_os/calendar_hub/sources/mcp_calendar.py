# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""MCP-connector calendar source (Spec 011 §0.4/§2; Task G1).

Wraps a ``ConnectorManager`` + connector id (e.g. ``google-calendar``) as a
calendar source: available iff that connector is currently connected;
``list_events`` calls the connector's ``list_events`` MCP tool and maps the
result JSON onto ``NormalizedEvent`` conservatively.

Every MCP/parse failure degrades to available-but-empty with a logged warning —
this source never raises. Mapping is defensive: it handles both nested
Google-Calendar-v3 shapes (``summary``, ``start.dateTime``/``start.date``,
``attendees[].email``) and flat shapes, and drops any item without a usable id
(an unlinkable event is worse than a missing one).
"""

import json
import logging

from ..models import NormalizedEvent

logger = logging.getLogger(__name__)

# Google Calendar MCP tool that lists events in a range (per the catalog
# manifest's tool_overrides).
_LIST_TOOL = "list_events"


class McpCalendarSource:
    kind = "mcp"

    def __init__(self, connector_manager, connector_id: str, *, list_tool: str = _LIST_TOOL):
        self._manager = connector_manager
        self._connector_id = connector_id
        self._list_tool = list_tool

    @property
    def id(self) -> str:
        return self._connector_id

    @property
    def available(self) -> bool:
        try:
            for entry in self._manager.list_connectors():
                if entry["manifest"].id == self._connector_id:
                    return bool(entry["connected"])
        except Exception:
            logger.warning("connector availability check failed for %s",
                           self._connector_id, exc_info=True)
        return False

    async def list_events(self, start: str, end: str) -> list[NormalizedEvent]:
        if not self.available:
            return []
        try:
            # Param names per Google's Calendar MCP list_events inputSchema
            # (calendarId defaults to the primary calendar when omitted).
            result = await self._manager.call_tool(
                self._connector_id, self._list_tool,
                {"startTime": start, "endTime": end},
            )
        except Exception:
            logger.warning("MCP list_events call failed for %s", self._connector_id, exc_info=True)
            return []
        try:
            items = _extract_items(result)
        except Exception:
            logger.warning("failed to parse MCP calendar result for %s",
                           self._connector_id, exc_info=True)
            return []

        events: list[NormalizedEvent] = []
        for item in items:
            ev = self._map_item(item)
            if ev is not None:
                events.append(ev)
        return events

    def _map_item(self, item) -> NormalizedEvent | None:
        if not isinstance(item, dict):
            return None
        source_id = item.get("id") or item.get("event_id") or item.get("iCalUID")
        if not source_id:
            return None
        start, start_tz, start_all_day = _extract_when(item.get("start"))
        end, end_tz, _ = _extract_when(item.get("end"))
        return NormalizedEvent(
            source=self._connector_id,
            source_id=str(source_id),
            title=str(item.get("summary") or item.get("title") or "(no title)"),
            start=start,
            end=end,
            all_day=bool(start_all_day),
            timezone=start_tz or end_tz,
            attendees=_extract_attendees(item.get("attendees")),
            location=(str(item["location"]) if item.get("location") else None),
            url=(str(item.get("htmlLink") or item.get("url")) if (item.get("htmlLink") or item.get("url")) else None),
        )


def _extract_items(result) -> list:
    """Pull the list of raw event dicts out of an MCP CallToolResult.

    Prefers ``structuredContent`` (structured tool output); falls back to
    parsing concatenated text blocks as JSON. Accepts a top-level list, or a
    dict wrapping the list under ``events`` / ``items``."""
    structured = getattr(result, "structuredContent", None)
    payload = structured if structured is not None else _text_json(result)
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("events", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _text_json(result):
    parts = []
    for block in getattr(result, "content", None) or []:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            parts.append(block.text)
    if not parts:
        return None
    try:
        return json.loads("\n".join(parts))
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_when(value) -> tuple[str, str | None, bool]:
    """Return ``(iso, timezone, all_day)`` from a Google-style time object or a
    plain ISO string. A date-only value ⇒ all-day."""
    if isinstance(value, dict):
        if value.get("dateTime"):
            return str(value["dateTime"]), (value.get("timeZone") or None), False
        if value.get("date"):
            return str(value["date"]), (value.get("timeZone") or None), True
        return "", None, False
    if isinstance(value, str):
        return value, None, False
    return "", None, False


def _extract_attendees(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for a in value:
        if isinstance(a, dict):
            name = a.get("displayName") or a.get("email") or a.get("name")
            if name:
                out.append(str(name))
        elif isinstance(a, str):
            out.append(a)
    return out
