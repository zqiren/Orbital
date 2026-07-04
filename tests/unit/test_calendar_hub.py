# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the calendar hub core (Task G1): models, linkage store, hub
merge/stamp/filter/cache, and the two sources' graceful behavior. All logic is
exercised with a fake source — EventKit itself is only import-guard-tested."""

import json
from types import SimpleNamespace

import pytest

from agent_os.calendar_hub.hub import CalendarHub
from agent_os.calendar_hub.linkage import Linkage
from agent_os.calendar_hub.models import NormalizedEvent
from agent_os.calendar_hub.sources.mcp_calendar import McpCalendarSource

START = "2026-07-01T00:00:00+00:00"
END = "2026-07-08T00:00:00+00:00"


# ---- Fakes -----------------------------------------------------------------

class FakeSource:
    def __init__(self, source_id, events, *, available=True, kind="fake"):
        self.id = source_id
        self.kind = kind
        self.available = available
        self._events = events
        self.calls = 0

    async def list_events(self, start, end):
        self.calls += 1
        # Fresh copies each fetch, mirroring a real source.
        return [
            NormalizedEvent(source=self.id, source_id=e.source_id, title=e.title,
                            start=e.start, end=e.end)
            for e in self._events
        ]


class MutableClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _event(source, sid, title="Ev"):
    return NormalizedEvent(source=source, source_id=sid, title=title, start=START, end=END)


# ---- Models ----------------------------------------------------------------

def test_normalized_event_id_and_dict():
    ev = _event("eventkit", "ABC-123", title="Standup")
    assert ev.id == "eventkit:ABC-123"
    d = ev.to_dict()
    assert d["id"] == "eventkit:ABC-123"
    assert d["source"] == "eventkit"
    assert d["source_id"] == "ABC-123"
    assert d["title"] == "Standup"
    assert d["attendees"] == []
    assert d["project_id"] is None
    # exact shipped shape (locked)
    assert set(d) == {
        "id", "source", "source_id", "title", "start", "end", "all_day",
        "timezone", "attendees", "location", "url", "project_id",
    }


# ---- Linkage ---------------------------------------------------------------

def test_linkage_link_get_unlink(tmp_path):
    link = Linkage(str(tmp_path))
    assert link.get("eventkit:1") is None
    link.link("eventkit:1", "proj-a")
    assert link.get("eventkit:1") == "proj-a"
    link.unlink("eventkit:1")
    assert link.get("eventkit:1") is None


def test_linkage_persists_atomically_across_instances(tmp_path):
    link = Linkage(str(tmp_path))
    link.link("eventkit:1", "proj-a")
    link.link("google-calendar:xyz", "proj-b")

    # Reload from disk in a fresh instance.
    reloaded = Linkage(str(tmp_path))
    assert reloaded.get("eventkit:1") == "proj-a"
    assert reloaded.get("google-calendar:xyz") == "proj-b"

    # File is valid JSON and no temp file is left behind.
    path = tmp_path / "calendar_links.json"
    assert path.is_file()
    assert not (tmp_path / "calendar_links.json.tmp").exists()
    assert json.loads(path.read_text()) == {
        "eventkit:1": "proj-a", "google-calendar:xyz": "proj-b",
    }


def test_linkage_tolerates_corrupt_file(tmp_path):
    (tmp_path / "calendar_links.json").write_text("{not json")
    link = Linkage(str(tmp_path))
    assert link.all() == {}  # degrades to empty rather than raising


# ---- Hub: merge + linkage stamping -----------------------------------------

async def test_hub_merges_sources_and_stamps_linkage(tmp_path):
    link = Linkage(str(tmp_path))
    link.link("s1:a", "proj-a")
    hub = CalendarHub(
        sources=[FakeSource("s1", [_event("s1", "a")]),
                 FakeSource("s2", [_event("s2", "b")])],
        linkage=link,
    )
    events = await hub.list_events(START, END)
    by_id = {e.id: e for e in events}
    assert set(by_id) == {"s1:a", "s2:b"}
    assert by_id["s1:a"].project_id == "proj-a"   # stamped from linkage
    assert by_id["s2:b"].project_id is None        # unlinked


async def test_hub_filters_by_project(tmp_path):
    link = Linkage(str(tmp_path))
    link.link("s1:a", "proj-a")
    hub = CalendarHub(
        sources=[FakeSource("s1", [_event("s1", "a"), _event("s1", "b")])],
        linkage=link,
    )
    events = await hub.list_events(START, END, project_id="proj-a")
    assert [e.id for e in events] == ["s1:a"]


async def test_hub_skips_unavailable_source(tmp_path):
    src = FakeSource("s1", [_event("s1", "a")], available=False)
    hub = CalendarHub(sources=[src], linkage=Linkage(str(tmp_path)))
    assert await hub.list_events(START, END) == []
    assert src.calls == 0  # never fetched


async def test_hub_source_that_raises_contributes_nothing(tmp_path):
    class Boom:
        id, kind, available = "boom", "fake", True

        async def list_events(self, start, end):
            raise RuntimeError("network down")

    hub = CalendarHub(
        sources=[Boom(), FakeSource("s1", [_event("s1", "a")])],
        linkage=Linkage(str(tmp_path)),
    )
    events = await hub.list_events(START, END)
    assert [e.id for e in events] == ["s1:a"]


# ---- Hub: cache TTL + refresh ----------------------------------------------

async def test_hub_caches_within_ttl_and_refresh_clears(tmp_path):
    clock = MutableClock()
    src = FakeSource("s1", [_event("s1", "a")])
    hub = CalendarHub(sources=[src], linkage=Linkage(str(tmp_path)),
                      cache_ttl=60.0, clock=clock)

    await hub.list_events(START, END)
    assert src.calls == 1
    clock.t = 30.0
    await hub.list_events(START, END)
    assert src.calls == 1  # served from cache

    clock.t = 61.0
    await hub.list_events(START, END)
    assert src.calls == 2  # TTL expired → refetched

    hub.refresh()
    await hub.list_events(START, END)
    assert src.calls == 3  # cache cleared → refetched


async def test_hub_cache_keyed_on_range(tmp_path):
    src = FakeSource("s1", [_event("s1", "a")])
    hub = CalendarHub(sources=[src], linkage=Linkage(str(tmp_path)))
    await hub.list_events(START, END)
    await hub.list_events("2026-08-01T00:00:00+00:00", "2026-08-08T00:00:00+00:00")
    assert src.calls == 2  # different range → separate cache entry


# ---- Hub: availability aggregation -----------------------------------------

def test_hub_availability_aggregates(tmp_path):
    hub = CalendarHub(
        sources=[FakeSource("s1", [], available=True, kind="eventkit"),
                 FakeSource("s2", [], available=False, kind="mcp")],
        linkage=Linkage(str(tmp_path)),
    )
    a = hub.availability()
    assert a["available"] is True
    assert a["sources"] == [
        {"id": "s1", "kind": "eventkit", "available": True},
        {"id": "s2", "kind": "mcp", "available": False},
    ]


def test_hub_availability_all_unavailable(tmp_path):
    hub = CalendarHub(
        sources=[FakeSource("s1", [], available=False)],
        linkage=Linkage(str(tmp_path)),
    )
    assert hub.availability()["available"] is False


# ---- MCP calendar source ---------------------------------------------------

class FakeConnectorManager:
    def __init__(self, *, connected=True, result=None, raises=False):
        self.connected = connected
        self.result = result
        self.raises = raises
        self.calls = []

    def list_connectors(self):
        return [{
            "manifest": SimpleNamespace(id="google-calendar"),
            "connected": self.connected,
            "account": "u@example.com",
            "enabled_error": None,
        }]

    async def call_tool(self, connector_id, tool, args):
        self.calls.append((connector_id, tool, args))
        if self.raises:
            raise RuntimeError("mcp down")
        return self.result


class _StructuredResult:
    def __init__(self, structured):
        self.structuredContent = structured
        self.content = []
        self.isError = False


class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _TextResult:
    def __init__(self, text):
        self.structuredContent = None
        self.content = [_TextBlock(text)]
        self.isError = False


def test_mcp_source_available_reflects_connection():
    src = McpCalendarSource(FakeConnectorManager(connected=True), "google-calendar")
    assert src.available is True
    src2 = McpCalendarSource(FakeConnectorManager(connected=False), "google-calendar")
    assert src2.available is False


async def test_mcp_source_maps_google_event_structured():
    result = _StructuredResult({"events": [{
        "id": "abc123",
        "summary": "Design review",
        "start": {"dateTime": "2026-07-04T10:00:00-07:00", "timeZone": "America/Los_Angeles"},
        "end": {"dateTime": "2026-07-04T11:00:00-07:00", "timeZone": "America/Los_Angeles"},
        "location": "Room 2",
        "htmlLink": "https://calendar.google.com/e/abc123",
        "attendees": [{"email": "a@x.com", "displayName": "Alice"}, {"email": "b@x.com"}],
    }]})
    src = McpCalendarSource(FakeConnectorManager(result=result), "google-calendar")
    events = await src.list_events(START, END)
    assert len(events) == 1
    ev = events[0]
    assert ev.source == "google-calendar"
    assert ev.id == "google-calendar:abc123"
    assert ev.title == "Design review"
    assert ev.start == "2026-07-04T10:00:00-07:00"
    assert ev.all_day is False
    assert ev.timezone == "America/Los_Angeles"
    assert ev.location == "Room 2"
    assert ev.url == "https://calendar.google.com/e/abc123"
    assert ev.attendees == ["Alice", "b@x.com"]


async def test_mcp_source_all_day_from_date_only():
    result = _StructuredResult([{
        "id": "holiday", "summary": "Holiday",
        "start": {"date": "2026-07-04"}, "end": {"date": "2026-07-05"},
    }])
    src = McpCalendarSource(FakeConnectorManager(result=result), "google-calendar")
    ev = (await src.list_events(START, END))[0]
    assert ev.all_day is True
    assert ev.start == "2026-07-04"


async def test_mcp_source_parses_text_json():
    payload = json.dumps({"items": [{"id": "t1", "summary": "From text", "start": "2026-07-04T09:00:00Z", "end": "2026-07-04T09:30:00Z"}]})
    src = McpCalendarSource(FakeConnectorManager(result=_TextResult(payload)), "google-calendar")
    events = await src.list_events(START, END)
    assert [e.id for e in events] == ["google-calendar:t1"]


async def test_mcp_source_drops_item_without_id():
    result = _StructuredResult([{"summary": "no id here"}, {"id": "ok", "summary": "keep"}])
    src = McpCalendarSource(FakeConnectorManager(result=result), "google-calendar")
    events = await src.list_events(START, END)
    assert [e.id for e in events] == ["google-calendar:ok"]


async def test_mcp_source_not_connected_returns_empty_without_call():
    mgr = FakeConnectorManager(connected=False)
    src = McpCalendarSource(mgr, "google-calendar")
    assert await src.list_events(START, END) == []
    assert mgr.calls == []  # never called the tool


async def test_mcp_source_call_failure_returns_empty():
    mgr = FakeConnectorManager(raises=True)
    src = McpCalendarSource(mgr, "google-calendar")
    assert await src.list_events(START, END) == []  # never raises out


# ---- EventKit source: import guard only (no TCC from tests) -----------------

try:
    import EventKit  # noqa: F401
    _HAS_EVENTKIT = True
except Exception:
    _HAS_EVENTKIT = False


@pytest.mark.skipif(
    _HAS_EVENTKIT,
    reason="EventKit present — constructing EKEventStore triggers the macOS TCC "
           "prompt; unit tests must not do that (see module docstring)",
)
async def test_eventkit_source_unavailable_without_framework():
    from agent_os.calendar_hub.sources.eventkit import EventKitSource
    src = EventKitSource()
    assert src.available is False
    assert await src.list_events(START, END) == []
