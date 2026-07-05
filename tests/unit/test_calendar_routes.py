# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the /api/v2/calendar routes (Task G1).

Wires a real CalendarHub (fake source + on-disk linkage) into the router and
drives it over an in-process ASGI transport — no EventKit, no network. The
conftest autouse fixture patches the daemon PID guard, but these tests mount
only the calendar router (not create_app), so that is incidental."""

import httpx
import pytest
from fastapi import FastAPI

from agent_os.api.routes import calendar as calendar_routes
from agent_os.calendar_hub.hub import CalendarHub
from agent_os.calendar_hub.linkage import Linkage
from agent_os.calendar_hub.models import NormalizedEvent

START = "2026-07-01T00:00:00+00:00"
END = "2026-07-08T00:00:00+00:00"


class FakeSource:
    def __init__(self, source_id, events, *, available=True, kind="fake"):
        self.id = source_id
        self.kind = kind
        self.available = available
        self._events = events
        self.calls = 0

    async def list_events(self, start, end):
        self.calls += 1
        return [
            NormalizedEvent(source=self.id, source_id=e.source_id, title=e.title,
                            start=e.start, end=e.end, all_day=e.all_day)
            for e in self._events
        ]


def _event(sid, title="Ev", all_day=False):
    return NormalizedEvent(source="s1", source_id=sid, title=title,
                           start=START, end=END, all_day=all_day)


@pytest.fixture
def source():
    return FakeSource("s1", [_event("a", "Standup"), _event("b", "Review")])


@pytest.fixture
def client(tmp_path, source):
    hub = CalendarHub(sources=[source], linkage=Linkage(str(tmp_path)))
    app = FastAPI()
    calendar_routes.configure(hub)
    app.include_router(calendar_routes.router)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---- availability ----------------------------------------------------------

async def test_availability(client):
    async with client:
        r = await client.get("/api/v2/calendar/availability")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["sources"] == [{"id": "s1", "kind": "fake", "available": True}]


# ---- events ----------------------------------------------------------------

async def test_events_happy_path(client):
    async with client:
        r = await client.get("/api/v2/calendar/events", params={"start": START, "end": END})
        assert r.status_code == 200, r.text
        events = r.json()["events"]
        assert {e["id"] for e in events} == {"s1:a", "s1:b"}
        ev = next(e for e in events if e["id"] == "s1:a")
        assert set(ev) == {
            "id", "source", "source_id", "title", "start", "end", "all_day",
            "timezone", "attendees", "location", "url", "project_id",
        }
        assert ev["source"] == "s1"
        assert ev["project_id"] is None


async def test_events_missing_params_is_422(client):
    async with client:
        assert (await client.get("/api/v2/calendar/events")).status_code == 422
        assert (await client.get("/api/v2/calendar/events", params={"start": START})).status_code == 422


async def test_events_bad_iso_is_422(client):
    async with client:
        r = await client.get("/api/v2/calendar/events", params={"start": "not-a-date", "end": END})
        assert r.status_code == 422


async def test_events_range_over_90_days_is_422(client):
    async with client:
        r = await client.get("/api/v2/calendar/events",
                             params={"start": START, "end": "2026-11-01T00:00:00+00:00"})
        assert r.status_code == 422


async def test_events_end_before_start_is_422(client):
    async with client:
        r = await client.get("/api/v2/calendar/events", params={"start": END, "end": START})
        assert r.status_code == 422


async def test_events_exactly_90_days_ok(client):
    async with client:
        r = await client.get("/api/v2/calendar/events",
                             params={"start": START, "end": "2026-09-29T00:00:00+00:00"})
        assert r.status_code == 200


# ---- refresh ---------------------------------------------------------------

async def test_refresh_clears_cache(client, source):
    async with client:
        await client.get("/api/v2/calendar/events", params={"start": START, "end": END})
        await client.get("/api/v2/calendar/events", params={"start": START, "end": END})
        assert source.calls == 1  # second served from cache

        r = await client.post("/api/v2/calendar/refresh")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

        await client.get("/api/v2/calendar/events", params={"start": START, "end": END})
        assert source.calls == 2  # refetched after refresh


# ---- link / unlink ---------------------------------------------------------

async def test_link_and_unlink_round_trip(client, tmp_path):
    async with client:
        # link s1:a → proj-a
        r = await client.post("/api/v2/calendar/events/s1/a/link", json={"project_id": "proj-a"})
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

        lensed = await client.get("/api/v2/calendar/events",
                                  params={"start": START, "end": END, "project_id": "proj-a"})
        assert {e["id"] for e in lensed.json()["events"]} == {"s1:a"}

        # unlink (null project_id)
        await client.post("/api/v2/calendar/events/s1/a/link", json={"project_id": None})
        lensed2 = await client.get("/api/v2/calendar/events",
                                   params={"start": START, "end": END, "project_id": "proj-a"})
        assert lensed2.json()["events"] == []


async def test_link_persists_to_disk(client, tmp_path):
    async with client:
        await client.post("/api/v2/calendar/events/s1/a/link", json={"project_id": "proj-a"})
    # A fresh linkage reading the same dir sees the link.
    assert Linkage(str(tmp_path)).get("s1:a") == "proj-a"
