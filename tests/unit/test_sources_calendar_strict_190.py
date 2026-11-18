# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 190: a calendar poll reads the authorisation once, never asks macOS
for access, and a read that fails is a failed check (never a successful empty
one). Non-full statuses are logged raw, tagged with the path that read them.

The real EventKit framework is never touched here: every test that goes
through the real reader installs a fake ``EventKit`` / ``Foundation`` module
in ``sys.modules``, so no test run can show a macOS Calendar prompt."""

from __future__ import annotations

import logging
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

from agent_os.calendar_hub.sources.eventkit import EventKitSource, EventKitUnavailable
from agent_os.sources.adapters import calendar as cal_mod
from agent_os.sources.adapters.calendar import (
    DENIED, FULL_ACCESS, NOT_DETERMINED, RESTRICTED, WRITE_ONLY, CalendarAdapter, CalEvent,
    EventKitReader,
)
from agent_os.sources.context import SourceContext
from agent_os.sources.core import SourcesCore
from agent_os.sources.model import SourceFailure, VerifyError
from agent_os.sources.registry import AdapterRegistry
from agent_os.sources.secrets import MemorySecretStore

T0 = datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)
LOGGER = "agent_os.sources.adapters.calendar"


class Clock:
    def __init__(self, t):
        self.t = t


def _ctx(kinds, state, clock, config=None):
    ctx = SourceContext(source_id="src_c", type_="calendar", config=config or {}, kinds=kinds,
                        account=None, secrets_fn=lambda k: None, state=state)
    ctx.now = lambda: clock.t
    ctx.lang = "en"
    return ctx


class CountingReader:
    """A fake reader that counts every call; ``statuses`` is consumed one per
    status read (the last value repeats)."""

    def __init__(self, statuses, events=None, fail=None):
        self.statuses = list(statuses)
        self.status_reads = 0
        self.requests = 0
        self.event_reads = 0
        self.events_list = list(events or [])
        self.fail = fail

    def status(self):
        self.status_reads += 1
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def request_access(self):
        self.requests += 1
        return False

    def calendars(self):
        return [{"id": "home", "title": "Home", "account": "iCloud"}]

    def events(self, start, end, calendars=None):
        self.event_reads += 1
        if self.fail:
            raise self.fail
        return [e for e in self.events_list if start <= e.start <= end]


def _soon(minutes=10):
    s = T0 + timedelta(minutes=minutes)
    return CalEvent(id="ev1", title="Standup", start=s, end=s + timedelta(minutes=30),
                    created=T0 - timedelta(days=1))


# ── the poll gate, with a fake reader ──────────────────────────────────────

async def test_full_access_reads_events_and_produces_arrivals():
    r = CountingReader([FULL_ACCESS], events=[_soon()])
    items = await CalendarAdapter(r).poll(_ctx({"before": {"minutes": 30}}, {}, Clock(T0)))
    assert [i.kind for i in items] == ["before"]
    assert (r.status_reads, r.event_reads, r.requests) == (1, 1, 0)


@pytest.mark.parametrize("status", [NOT_DETERMINED, RESTRICTED, DENIED, WRITE_ONLY, None])
async def test_without_full_access_a_poll_never_reads_events_nor_asks(status):
    r = CountingReader([status], events=[_soon()])
    with pytest.raises(SourceFailure) as f:
        await CalendarAdapter(r).poll(_ctx({"before": {"minutes": 30}, "new_event": {}}, {}, Clock(T0)))
    assert f.value.code == "calendar_permission"
    assert (r.event_reads, r.requests) == (0, 0)


async def test_a_hundred_polls_with_a_changing_status_never_ask():
    cycle = [FULL_ACCESS, DENIED, FULL_ACCESS, WRITE_ONLY, NOT_DETERMINED, FULL_ACCESS, RESTRICTED, None]
    seq = [cycle[i % len(cycle)] for i in range(100)]
    r = CountingReader(seq + [FULL_ACCESS])
    a = CalendarAdapter(r)
    for _ in range(100):
        try:
            await a.poll(_ctx({"before": {"minutes": 30}}, {}, Clock(T0)))
        except SourceFailure as f:
            assert f.code == "calendar_permission"
    assert r.requests == 0
    assert r.status_reads == 100  # one status read per poll
    assert r.event_reads == seq.count(FULL_ACCESS)  # events only behind a 3


@pytest.mark.skipif(sys.platform != "darwin", reason="calendar sources exist only on macOS")
async def test_a_read_that_fails_is_a_failed_check_through_the_core(tmp_path, monkeypatch):
    clock = Clock(T0)
    monkeypatch.setattr(SourceContext, "now", staticmethod(lambda: clock.t))
    r = CountingReader([FULL_ACCESS])
    core = SourcesCore(str(tmp_path), registry=AdapterRegistry({"calendar": CalendarAdapter(r)}),
                       secret_store=MemorySecretStore(), now_fn=lambda: clock.t)
    core.load()
    sid = (await core.create({"type": "calendar", "kinds": {"before": {"minutes": 30}}}))["id"]
    await core._poll_one(core._cfg(sid))
    ok_at = core.store.rt(sid)["last_check_at"]
    assert ok_at is not None and core.store.rt(sid)["failure"] is None
    r.fail = RuntimeError("EKEventStore fetch failed")
    clock.t = T0 + timedelta(minutes=1)
    await core._poll_one(core._cfg(sid))
    rt = core.store.rt(sid)
    assert rt["failure"]["code"] == "error"
    assert rt["last_check_at"] == ok_at  # never stamped as a successful check


# ── the real reader, with a fake EventKit framework ────────────────────────

class _FakeStore:
    """Duck-types the EKEventStore selectors the reader calls. Class-level
    counters, since the reader creates a store per read."""
    status = FULL_ACCESS
    requests = 0
    reads = 0
    fetch_error: Exception | None = None
    events: list = []

    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    @classmethod
    def authorizationStatusForEntityType_(cls, _entity):
        return cls.status

    def requestFullAccessToEventsWithCompletion_(self, completion):
        type(self).requests += 1
        completion(False, None)

    def calendarWithIdentifier_(self, cid):
        return None

    def predicateForEventsWithStartDate_endDate_calendars_(self, s, e, cals):
        return ("pred", s, e, cals)

    def eventsMatchingPredicate_(self, pred):
        type(self).reads += 1
        if type(self).fetch_error is not None:
            raise type(self).fetch_error
        return list(type(self).events)


class _NSDateValue:
    def __init__(self, ts):
        self.ts = ts

    def timeIntervalSince1970(self):
        return self.ts


class _FakeEKEvent:
    def __init__(self, start: datetime):
        self.start = start

    def eventIdentifier(self):
        return "EV-1"

    def title(self):
        return "Standup"

    def startDate(self):
        return _NSDateValue(self.start.timestamp())

    def endDate(self):
        return _NSDateValue((self.start + timedelta(minutes=30)).timestamp())

    def isAllDay(self):
        return False

    def timeZone(self):
        return None

    def attendees(self):
        return []

    def location(self):
        return None

    def URL(self):
        return None

    def creationDate(self):
        return _NSDateValue((self.start - timedelta(days=1)).timestamp())

    def occurrenceDate(self):
        return _NSDateValue(self.start.timestamp())

    def hasRecurrenceRules(self):
        return False


@pytest.fixture
def fake_eventkit(monkeypatch):
    class Store(_FakeStore):
        status = FULL_ACCESS
        requests = 0
        reads = 0
        fetch_error = None
        events = []
    ek = types.ModuleType("EventKit")
    ek.EKEventStore = Store
    found = types.ModuleType("Foundation")

    class NSDate:
        @staticmethod
        def dateWithTimeIntervalSince1970_(ts):
            return _NSDateValue(ts)
    found.NSDate = NSDate
    monkeypatch.setitem(sys.modules, "EventKit", ek)
    monkeypatch.setitem(sys.modules, "Foundation", found)
    monkeypatch.setattr(sys, "platform", "darwin")
    return Store


async def test_strict_source_never_requests_access(fake_eventkit):
    fake_eventkit.status = DENIED  # the store's own read disagrees with the gate
    src = EventKitSource(strict=True)
    assert await src.list_events("2026-11-18T10:00:00Z", "2026-11-18T11:00:00Z") == []
    assert fake_eventkit.requests == 0
    assert fake_eventkit.reads == 1  # it read; macOS decides what a denied read returns


async def test_strict_source_raises_a_fetch_error(fake_eventkit):
    fake_eventkit.fetch_error = RuntimeError("fetch failed")
    with pytest.raises(RuntimeError):
        await EventKitSource(strict=True).list_events("2026-11-18T10:00:00Z", "2026-11-18T11:00:00Z")


async def test_strict_source_raises_when_the_store_cannot_be_made(fake_eventkit):
    def boom(cls):
        raise RuntimeError("no store")
    fake_eventkit.alloc = classmethod(boom)
    with pytest.raises(EventKitUnavailable):
        await EventKitSource(strict=True).list_events("2026-11-18T10:00:00Z", "2026-11-18T11:00:00Z")


async def test_default_source_keeps_asking_and_swallowing(fake_eventkit):
    """Risk 2: the workbench calendar hub's reader is unchanged."""
    fake_eventkit.status = NOT_DETERMINED
    src = EventKitSource()
    assert await src.list_events("2026-11-18T10:00:00Z", "2026-11-18T11:00:00Z") == []
    assert fake_eventkit.requests == 1 and fake_eventkit.reads == 0
    fake_eventkit.status = FULL_ACCESS
    fake_eventkit.fetch_error = RuntimeError("fetch failed")
    assert await EventKitSource().list_events("2026-11-18T10:00:00Z", "2026-11-18T11:00:00Z") == []
    assert fake_eventkit.requests == 1


async def test_the_real_reader_poll_never_requests_even_when_the_store_disagrees(fake_eventkit, monkeypatch):
    """The owner's case: the poll's gate reads 3, the store's own read then
    says something else. Before spec 190 that sent a request from the timer,
    waited on it and reported a successful empty check."""
    monkeypatch.setattr(EventKitReader, "status", staticmethod(lambda: FULL_ACCESS))
    fake_eventkit.status = DENIED
    fake_eventkit.events = [_FakeEKEvent(T0 + timedelta(minutes=10))]
    items = await CalendarAdapter(EventKitReader()).poll(
        _ctx({"before": {"minutes": 30}}, {}, Clock(T0)))
    assert fake_eventkit.requests == 0
    assert [i.kind for i in items] == ["before"]


async def test_the_real_reader_turns_a_failed_fetch_into_a_failed_check(fake_eventkit, monkeypatch):
    monkeypatch.setattr(EventKitReader, "status", staticmethod(lambda: FULL_ACCESS))
    fake_eventkit.fetch_error = RuntimeError("fetch failed")
    with pytest.raises(SourceFailure) as f:
        await CalendarAdapter(EventKitReader()).poll(_ctx({"before": {"minutes": 30}}, {}, Clock(T0)))
    assert f.value.code == "error"
    assert fake_eventkit.requests == 0


async def test_a_hundred_real_reader_polls_with_a_changing_status_never_request(fake_eventkit, monkeypatch):
    cycle = [FULL_ACCESS, DENIED, WRITE_ONLY, FULL_ACCESS, NOT_DETERMINED, RESTRICTED]
    n = {"i": 0}

    def flapping():
        v = cycle[n["i"] % len(cycle)]
        n["i"] += 1
        return v
    monkeypatch.setattr(EventKitReader, "status", staticmethod(flapping))
    fake_eventkit.status = NOT_DETERMINED  # the store's own read never says 3
    a = CalendarAdapter(EventKitReader())
    for _ in range(100):
        try:
            await a.poll(_ctx({"before": {"minutes": 30}}, {}, Clock(T0)))
        except SourceFailure as f:
            assert f.code == "calendar_permission"
    assert fake_eventkit.requests == 0


# ── the raw-status log line ────────────────────────────────────────────────

def _status_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == LOGGER and "calendar access status" in r.getMessage()]


async def test_raw_status_is_logged_once_per_path_and_value_transition(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    seq = [FULL_ACCESS, DENIED, DENIED, DENIED, FULL_ACCESS, DENIED, WRITE_ONLY, WRITE_ONLY, None]
    r = CountingReader(seq)
    a = CalendarAdapter(r)
    for _ in seq:
        try:
            await a.poll(_ctx({"before": {"minutes": 30}}, {}, Clock(T0)))
        except SourceFailure:
            pass
    lines = _status_lines(caplog)
    assert len(lines) == 4, lines  # 2, 2 again after a 3, then 4, then None
    assert all("poll" in ln for ln in lines)
    assert "2 (denied)" in lines[0] and "2 (denied)" in lines[1]
    assert "4 (write_only)" in lines[2] and "None" in lines[3]
    # a 3 is never logged as a status line
    assert not any("3 (full_access)" in ln for ln in lines)


async def test_raw_status_lines_carry_their_path(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    r = CountingReader([DENIED])
    a = CalendarAdapter(r)
    with pytest.raises(SourceFailure):
        await a.poll(_ctx({"before": {"minutes": 30}}, {}, Clock(T0)))
    with pytest.raises(VerifyError):
        await a.verify(_ctx({}, {}, Clock(T0)))
    with pytest.raises(VerifyError):
        await a.choices(_ctx({}, {}, Clock(T0)), "calendars", {})
    # the same value seen by three paths: one line each, never repeated
    with pytest.raises(SourceFailure):
        await a.poll(_ctx({"before": {"minutes": 30}}, {}, Clock(T0)))
    lines = _status_lines(caplog)
    assert [ln.rsplit(" ", 1)[-1] for ln in lines] == ["poll", "verify", "choices"], lines
    assert r.requests == 0  # denied: nothing to ask for


def test_status_names_cover_every_known_value():
    for v in (NOT_DETERMINED, RESTRICTED, DENIED, WRITE_ONLY):
        assert cal_mod._status_name(v) != "unknown"
