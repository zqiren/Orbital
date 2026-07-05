# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for the calendar surface (Spec 011 §0.4; Task G1).

Drives the ``CalendarHub`` singleton (injected via ``configure``): availability
for the sidebar/tab gate, an events feed by ISO range (optionally lensed to a
project), a manual refresh, and event↔project linkage.

Date validation is strict: missing or unparseable ``start``/``end`` and ranges
wider than 90 days are ``422`` — the surface fetches by visible window, never an
unbounded scan.
"""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/calendar")

_MAX_RANGE = timedelta(days=90)

_hub = None


def configure(calendar_hub):
    """Called by the app factory to inject the CalendarHub singleton."""
    global _hub
    _hub = calendar_hub


def _require_hub():
    if _hub is None:
        raise HTTPException(status_code=503, detail="Calendar hub not available")
    return _hub


class LinkRequest(BaseModel):
    project_id: str | None = None


def _validate_range(start: str, end: str) -> None:
    try:
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail="start and end must be ISO 8601 datetimes")
    try:
        span = e - s
    except TypeError:
        raise HTTPException(
            status_code=422,
            detail="start and end must both be timezone-aware or both naive",
        )
    if span.total_seconds() < 0:
        raise HTTPException(status_code=422, detail="end must not precede start")
    if span > _MAX_RANGE:
        raise HTTPException(status_code=422, detail="range must not exceed 90 days")


@router.get("/availability")
async def availability():
    """Whether any calendar source is live, and per-source status (gates the
    sidebar Calendar item and the per-project calendar tab)."""
    return _require_hub().availability()


@router.get("/events")
async def list_events(
    start: str = Query(..., description="ISO 8601 range start"),
    end: str = Query(..., description="ISO 8601 range end"),
    project_id: str | None = Query(None, description="lens to one project's linked events"),
):
    hub = _require_hub()
    _validate_range(start, end)
    events = await hub.list_events(start, end, project_id=project_id)
    return {"events": [ev.to_dict() for ev in events]}


@router.post("/refresh")
async def refresh():
    """Clear the per-source cache so the next fetch is fresh (manual refresh)."""
    _require_hub().refresh()
    return {"status": "ok"}


@router.post("/events/{source}/{event_id:path}/link")
async def link_event(source: str, event_id: str, req: LinkRequest):
    """Link an event to a project, or unlink it when ``project_id`` is null."""
    hub = _require_hub()
    full_id = f"{source}:{event_id}"
    if req.project_id is None:
        hub.unlink(full_id)
    else:
        hub.link(full_id, req.project_id)
    return {"status": "ok"}
