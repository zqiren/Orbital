# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Calendar page is gone (2026-11-04), and so are the REST routes only it
called (/api/v2/calendar/availability, /events, /refresh, /events/…/link).

What stays: the CalendarHub (the agent's ``calendar_read`` tool and the
Workbench refresh use it), the Google Calendar connector, and the range
check the tool reuses from ``agent_os.api.routes.calendar`` — so the old
endpoint's 422 range rules are kept here as tests of that check.

Rewritten from the Task G1 route tests: the range cases moved from HTTP 422s
to the check itself; the availability / events / refresh / link cases became
"the daemon no longer serves it"."""

import pytest
from fastapi import HTTPException

from agent_os.api.routes import calendar as calendar_routes
from agent_os.api.routes.calendar import _validate_range

START = "2026-07-01T00:00:00+00:00"
END = "2026-07-08T00:00:00+00:00"


# ---- the surface is gone ---------------------------------------------------

def test_the_module_no_longer_carries_a_router():
    assert not hasattr(calendar_routes, "router")
    assert not hasattr(calendar_routes, "configure")


def test_the_daemon_serves_no_calendar_route(tmp_path, monkeypatch):
    # Headless keychain hang hazard (CLAUDE.md): create_app() touches
    # credential storage during construction.
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "in-memory")
    monkeypatch.setenv("AGENT_OS_API_KEY", "test")
    from agent_os.api.app import create_app

    app = create_app(data_dir=str(tmp_path))
    # The OpenAPI paths, not ``app.routes``: FastAPI 0.162 keeps an included
    # router as one nested entry with no path of its own.
    paths = list(app.openapi()["paths"])
    assert not [p for p in paths if p.startswith("/api/v2/calendar")]
    # The connectors (Google Calendar among them) are a different feature.
    assert any(p.startswith("/api/v2/connectors") for p in paths)


def test_calendar_read_still_imports_the_range_check():
    from agent_os.agent.tools import calendar_read
    assert calendar_read._validate_range is _validate_range


# ---- the range check the calendar_read tool reuses -------------------------

def _rejects(start, end) -> int:
    with pytest.raises(HTTPException) as exc:
        _validate_range(start, end)
    return exc.value.status_code


def test_range_happy_path():
    assert _validate_range(START, END) is None


def test_range_bad_iso_is_422():
    assert _rejects("not-a-date", END) == 422


def test_range_over_90_days_is_422():
    assert _rejects(START, "2026-11-01T00:00:00+00:00") == 422


def test_range_end_before_start_is_422():
    assert _rejects(END, START) == 422


def test_range_exactly_90_days_ok():
    assert _validate_range(START, "2026-09-29T00:00:00+00:00") is None


def test_range_mixed_naive_and_aware_is_422():
    assert _rejects("2026-07-01T00:00:00", END) == 422
