# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The calendar range check (Spec 011 §0.4, §7.5).

The Calendar page and the REST routes only it called (availability, events,
refresh, event↔project link) were removed on 2026-11-04. What is left is the
range check the agent's ``calendar_read`` tool reuses (spec §7.5 "reuses REST
range validation"): missing or unparseable ``start``/``end`` and ranges wider
than 90 days are ``422`` — a calendar read is by window, never an unbounded
scan. The CalendarHub and the Google Calendar connector are untouched.
"""

from datetime import datetime, timedelta

from fastapi import HTTPException

_MAX_RANGE = timedelta(days=90)


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
