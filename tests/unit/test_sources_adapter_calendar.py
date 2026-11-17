# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Calendar source adapter with a fake reader (never the real EventKit, so no
macOS permission prompt from a test run). The fake's statuses are
EKAuthorizationStatus values (0 notDetermined … 3 fullAccess, 4 writeOnly,
read from pyobjc's EventKit constants on this Mac); its events carry what
``EKEvent`` exposes (eventIdentifier, creationDate, occurrenceDate,
hasRecurrenceRules — Apple's EventKit reference). The real permission flow
and real events are the owner's manual check.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pytest

from agent_os.sources.adapters.calendar import (
    DENIED, FULL_ACCESS, NOT_DETERMINED, WRITE_ONLY, CalendarAdapter, CalEvent, to_cal_event,
)
from agent_os.calendar_hub.models import NormalizedEvent
from agent_os.calendar_hub.sources.eventkit import EventKitSource
from agent_os.sources.context import SourceContext
from agent_os.sources.core import SourcesCore
from agent_os.sources.model import SourceFailure, VerifyError
from agent_os.sources.registry import AdapterRegistry
from agent_os.sources.secrets import MemorySecretStore

T0 = datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)


class FakeReader:
    def __init__(self, status=FULL_ACCESS, grant=True):
        self._status = status
        self.grant = grant
        self.requests = 0
        self.events_list: list[CalEvent] = []
        self.windows = []
        self.asked: list = []          # the calendars each read was given
        self.cal_of: dict[str, str] = {}  # event id -> calendarIdentifier
        self.calendar_rows = [{"id": "home", "title": "Home", "account": "iCloud"}]

    def status(self):
        return self._status

    def request_access(self):
        self.requests += 1
        if self.grant:
            self._status = FULL_ACCESS
        else:
            self._status = DENIED
        return self.grant

    def calendars(self):
        return list(self.calendar_rows)

    def events(self, start, end, calendars=None):
        self.windows.append((start, end))
        self.asked.append(calendars)
        # like EventKit: an id that no longer exists reads nothing
        return [e for e in self.events_list if start <= e.start <= end
                and (calendars is None or self.cal_of.get(e.id, "home") in calendars)]


class Clock:
    def __init__(self, t):
        self.t = t


def _ctx(kinds, state, clock, lang="en", config=None):
    ctx = SourceContext(source_id="src_c", type_="calendar", config=config or {}, kinds=kinds, account=None,
                        secrets_fn=lambda k: None, state=state)
    ctx.now = lambda: clock.t
    ctx.lang = lang
    return ctx


def _ev(eid, start, *, created, title="Design review", recurring=False, occurrence=None, all_day=False):
    return CalEvent(id=eid, title=title, start=start, end=start + timedelta(hours=1), all_day=all_day,
                    location="Room 3", attendees=["Li Na", "Kai Wu"], created=created,
                    occurrence=occurrence, recurring=recurring)


async def test_verify_asks_only_when_undetermined_or_write_only():
    r = FakeReader(status=FULL_ACCESS)
    assert (await CalendarAdapter(r).verify(_ctx({}, {}, Clock(T0)))).label is None
    assert r.requests == 0
    r = FakeReader(status=NOT_DETERMINED, grant=True)
    await CalendarAdapter(r).verify(_ctx({}, {}, Clock(T0)))
    assert r.requests == 1
    r = FakeReader(status=NOT_DETERMINED, grant=False)
    with pytest.raises(VerifyError) as e:
        await CalendarAdapter(r).verify(_ctx({}, {}, Clock(T0)))
    assert e.value.reason == "permission"
    r = FakeReader(status=DENIED)
    with pytest.raises(VerifyError) as e:
        await CalendarAdapter(r).verify(_ctx({}, {}, Clock(T0)))
    assert e.value.reason == "permission" and r.requests == 0
    with pytest.raises(VerifyError):
        await CalendarAdapter(FakeReader(status=None)).verify(_ctx({}, {}, Clock(T0)))


class StaysWriteOnly(FakeReader):
    """macOS shows no upgrade prompt: the ask returns False at once and the
    status stays "Add Events Only"."""

    def request_access(self):
        self.requests += 1
        return False


async def test_verify_with_add_events_only_asks_once_for_full_access():
    # macOS upgrades it: the user taps Allow, connect succeeds
    r = FakeReader(status=WRITE_ONLY, grant=True)
    assert (await CalendarAdapter(r).verify(_ctx({}, {}, Clock(T0)))).label is None
    assert r.requests == 1
    # macOS stays silent: one ask, then the permission line
    r = StaysWriteOnly(status=WRITE_ONLY)
    with pytest.raises(VerifyError) as e:
        await CalendarAdapter(r).verify(_ctx({}, {}, Clock(T0)))
    assert e.value.reason == "permission" and r.requests == 1


async def test_poll_never_prompts_and_fails_with_the_contract_code():
    for st in (NOT_DETERMINED, DENIED, WRITE_ONLY, None):
        r = FakeReader(status=st)
        with pytest.raises(SourceFailure) as f:
            await CalendarAdapter(r).poll(_ctx({"before": {"minutes": 30}}, {}, Clock(T0)))
        assert f.value.code == "calendar_permission" and f.value.retryable
        assert r.requests == 0
    # "Add Events Only", polled repeatedly: never an ask from the timer
    r = FakeReader(status=WRITE_ONLY, grant=True)
    for _ in range(3):
        with pytest.raises(SourceFailure) as f:
            await CalendarAdapter(r).poll(_ctx({"new_event": {}}, {}, Clock(T0)))
        assert f.value.code == "calendar_permission"
    assert r.requests == 0


async def test_new_event_means_created_after_watching_not_drifted_into_view():
    r = FakeReader()
    clock = Clock(T0)
    state = {}
    a = CalendarAdapter(r)
    r.events_list = [_ev("old", T0 + timedelta(days=3), created=T0 - timedelta(days=30))]
    assert await a.poll(_ctx({"new_event": {}}, state, clock)) == []
    assert state["since"] == "2026-11-16T09:00:00Z"
    clock.t = T0 + timedelta(minutes=5)
    r.events_list += [
        _ev("fresh", T0 + timedelta(days=2), created=T0 + timedelta(minutes=3), title="Launch sync"),
        _ev("weekly", T0 + timedelta(days=1), created=T0 + timedelta(minutes=4), recurring=True,
            occurrence=T0 + timedelta(days=1), title="1:1"),
        _ev("weekly", T0 + timedelta(days=8), created=T0 + timedelta(minutes=4), recurring=True,
            occurrence=T0 + timedelta(days=8), title="1:1"),
        _ev("nodate", T0 + timedelta(days=2), created=None),
        # created months ago, far out: drifts into the 60-day view later — never "new"
        _ev("drift", T0 + timedelta(days=61), created=T0 - timedelta(days=90)),
    ]
    items = await a.poll(_ctx({"new_event": {}}, state, clock))
    assert [(i.id, i.kind, i.title) for i in items] == [
        ("new:fresh", "new_event", "Launch sync"), ("new:weekly", "new_event", "1:1")]
    assert "Where: Room 3" in items[0].body and "With: Li Na, Kai Wu" in items[0].body
    clock.t = T0 + timedelta(days=2)
    items = await a.poll(_ctx({"new_event": {}}, state, clock))
    assert "new:drift" not in [i.id for i in items]


async def test_before_fires_once_per_occurrence_and_follows_a_move():
    r = FakeReader()
    clock = Clock(T0)
    a = CalendarAdapter(r)
    kinds = {"before": {"minutes": 15}}
    standup = _ev("standup", T0 + timedelta(minutes=40), created=T0 - timedelta(days=9))
    r.events_list = [standup, _ev("allday", T0 + timedelta(minutes=10), created=T0, all_day=True)]
    assert await a.poll(_ctx(kinds, {}, clock)) == []  # 40 min away; all-day skipped
    clock.t = T0 + timedelta(minutes=26)
    items = await a.poll(_ctx(kinds, {}, clock))
    assert [(i.id, i.kind, i.title) for i in items] == [
        ("before:standup:2026-11-16T09:40:00Z", "before", "Design review (starts in 14 min)")]
    assert items[0].subject == "event:standup"
    first_id = items[0].id
    # the next minute returns the same id: the core records it once
    clock.t = T0 + timedelta(minutes=27)
    assert [i.id for i in await a.poll(_ctx(kinds, {}, clock))] == [first_id]
    # moved half an hour later: a reminder for the new time
    standup.start = T0 + timedelta(minutes=70)
    clock.t = T0 + timedelta(minutes=56)
    items = await a.poll(_ctx(kinds, {}, clock, lang="zh"))
    assert [i.id for i in items] == ["before:standup:2026-11-16T10:10:00Z"]
    assert items[0].title == "Design review（14 分钟后开始）"
    # already started: nothing (no stale reminder after sleep)
    clock.t = T0 + timedelta(minutes=75)
    assert await a.poll(_ctx(kinds, {}, clock)) == []


async def test_recurring_occurrences_are_separate_reminders():
    r = FakeReader()
    a = CalendarAdapter(r)
    clock = Clock(T0)
    day1 = _ev("daily", T0 + timedelta(minutes=10), created=T0 - timedelta(days=50), recurring=True,
               occurrence=T0 + timedelta(minutes=10))
    r.events_list = [day1]
    i1 = (await a.poll(_ctx({"before": {"minutes": 30}}, {}, clock)))[0]
    day1.start = day1.occurrence = T0 + timedelta(days=1, minutes=10)
    clock.t = T0 + timedelta(days=1)
    i2 = (await a.poll(_ctx({"before": {"minutes": 30}}, {}, clock)))[0]
    assert i1.id != i2.id and i1.subject != i2.subject


@pytest.mark.skipif(sys.platform != "darwin", reason="calendar sources exist only on macOS")
async def test_through_the_core_connect_sends_nothing_already_there(tmp_path, monkeypatch):
    """Whole source through SourcesCore: verify on create, baseline on the first
    poll, one arrival for an event created afterwards."""
    r = FakeReader()
    clock = Clock(T0)
    monkeypatch.setattr(SourceContext, "now", staticmethod(lambda: clock.t))
    core = SourcesCore(str(tmp_path), registry=AdapterRegistry({"calendar": CalendarAdapter(r)}),
                       secret_store=MemorySecretStore(), now_fn=lambda: clock.t)
    core.load()
    r.events_list = [_ev("soon", T0 + timedelta(minutes=10), created=T0 - timedelta(days=1))]
    detail = await core.create({"type": "calendar", "kinds": {"new_event": {}, "before": {"minutes": 30}}})
    sid = detail["id"]
    assert detail["label"] is None
    await core._poll_one(core._cfg(sid))
    assert (await core.detail(sid))["history"] == []  # baseline: "soon" is not sent
    r.events_list.append(_ev("added", T0 + timedelta(days=1), created=T0 + timedelta(seconds=30),
                             title="Board prep"))
    clock.t = T0 + timedelta(minutes=1)
    await core._poll_one(core._cfg(sid))
    assert [h["title"] for h in (await core.detail(sid))["history"]] == ["Board prep"]
    # permission revoked later: the source fails with the code the panel maps to System Settings
    r._status = DENIED
    clock.t = T0 + timedelta(minutes=2)
    await core._poll_one(core._cfg(sid))
    d = await core.detail(sid)
    assert d["failure"]["code"] == "calendar_permission"
    assert "open_settings" in d["failure"]["actions"] and r.requests == 0


class _NSDate:
    def __init__(self, dt):
        self.dt = dt

    def timeIntervalSince1970(self):
        return self.dt.timestamp()


class _FakeEKEvent:
    """Duck-types the EKEvent selectors the reader and the adapter call."""

    def __init__(self, recurring):
        self._r = recurring

    def eventIdentifier(self):
        return "ABC-123:XYZ"

    def title(self):
        return "季度复盘"

    def startDate(self):
        return _NSDate(T0)

    def endDate(self):
        return _NSDate(T0 + timedelta(hours=1))

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
        return _NSDate(T0 - timedelta(days=2))

    def occurrenceDate(self):
        return _NSDate(T0)

    def hasRecurrenceRules(self):
        return self._r


def test_reader_mapping_keeps_creation_and_occurrence():
    ev = _FakeEKEvent(recurring=True)
    base = EventKitSource()._map(ev)
    assert isinstance(base, NormalizedEvent)
    c = to_cal_event(base, ev)
    assert (c.id, c.title, c.start, c.recurring) == ("ABC-123:XYZ", "季度复盘", T0, True)
    assert c.created == T0 - timedelta(days=2) and c.occurrence == T0


# ---------------------------------------------------------------- calendars


async def test_no_key_reads_every_calendar_and_a_list_is_passed_through():
    r = FakeReader()
    clock = Clock(T0)
    await CalendarAdapter(r).poll(_ctx({"before": {"minutes": 30}}, {}, clock))
    assert r.asked == [None]  # no key: every calendar (EventKit's nil)
    await CalendarAdapter(r).poll(_ctx({"before": {"minutes": 30}}, {}, clock,
                                       config={"calendars": ["work-1", "home"]}))
    assert r.asked[-1] == ["work-1", "home"]


async def test_a_narrowed_list_reads_only_its_calendars():
    r = FakeReader()
    clock = Clock(T0)
    r.events_list = [_ev("w", T0 + timedelta(minutes=10), created=T0, title="Work sync"),
                     _ev("h", T0 + timedelta(minutes=12), created=T0, title="Dentist")]
    r.cal_of = {"w": "work-1", "h": "home"}
    items = await CalendarAdapter(r).poll(_ctx({"before": {"minutes": 30}}, {}, clock,
                                               config={"calendars": ["home"]}))
    assert [i.title for i in items] == ["Dentist (starts in 12 min)"]


async def test_nothing_picked_never_asks_eventkit():
    r = FakeReader()
    items = await CalendarAdapter(r).poll(_ctx({"new_event": {}, "before": {"minutes": 30}}, {},
                                               Clock(T0), config={"calendars": []}))
    assert items == [] and r.windows == [] and r.requests == 0


async def test_the_list_is_grouped_by_account_and_asks_like_connect():
    r = FakeReader(status=NOT_DETERMINED, grant=True)
    r.calendar_rows = [{"id": "A1", "title": "Home", "account": "iCloud"},
                       {"id": "B2", "title": "Team", "account": "Exchange"},
                       {"id": "C3", "title": "", "account": ""}]
    got = await CalendarAdapter(r).choices(_ctx({}, {}, Clock(T0)), "calendars", {})
    assert [c.to_json() for c in got] == [
        {"value": "A1", "label": "Home", "group": "iCloud"},
        {"value": "B2", "label": "Team", "group": "Exchange"},
        {"value": "C3", "label": "C3", "group": ""},
    ]
    assert r.requests == 1  # opening the list is a user's gesture: one ask
    # "Add Events Only": one ask, then the permission line
    r = StaysWriteOnly(status=WRITE_ONLY)
    with pytest.raises(VerifyError) as e:
        await CalendarAdapter(r).choices(_ctx({}, {}, Clock(T0)), "calendars", {})
    assert e.value.reason == "permission" and r.requests == 1
    assert await CalendarAdapter(FakeReader()).choices(_ctx({}, {}, Clock(T0)), "folders", {}) == []


async def test_a_poll_with_a_selection_never_prompts():
    for st in (NOT_DETERMINED, WRITE_ONLY):
        r = FakeReader(status=st, grant=True)
        with pytest.raises(SourceFailure):
            await CalendarAdapter(r).poll(_ctx({"new_event": {}}, {}, Clock(T0),
                                               config={"calendars": ["home"]}))
        assert r.requests == 0 and r.windows == []


def _core(tmp_path, monkeypatch, r, clock):
    monkeypatch.setattr(SourceContext, "now", staticmethod(lambda: clock.t))
    core = SourcesCore(str(tmp_path), registry=AdapterRegistry({"calendar": CalendarAdapter(r)}),
                       secret_store=MemorySecretStore(), now_fn=lambda: clock.t)
    core.load()
    return core


@pytest.mark.skipif(sys.platform != "darwin", reason="calendar sources exist only on macOS")
async def test_ids_are_kept_byte_exact_and_null_deletes_the_key(tmp_path, monkeypatch):
    from agent_os.sources.core import ApiError
    r = FakeReader()
    clock = Clock(T0)
    core = _core(tmp_path, monkeypatch, r, clock)
    odd = ["  0F3A:Ünïcode/with spaces\t ", "x" * 600, "  0F3A:Ünïcode/with spaces\t "]
    d = await core.create({"type": "calendar", "kinds": {"new_event": {}}, "config": {"calendars": odd}})
    sid = d["id"]
    assert d["config"]["calendars"] == odd[:2]  # exact, no trim, no cap (dupes dropped)
    await core._poll_one(core._cfg(sid))
    assert r.asked[-1] == odd[:2]
    # a PATCH without the key keeps the list (config merges)
    d = await core.patch(sid, {"config": {}})
    assert d["config"]["calendars"] == odd[:2]
    # null: back to every calendar, the key is gone
    d = await core.patch(sid, {"config": {"calendars": None}})
    assert "calendars" not in d["config"]
    # connect with everything ticked sends null: no key stored
    d2 = await core.create({"type": "calendar", "kinds": {"new_event": {}}, "config": {"calendars": None}})
    assert "calendars" not in d2["config"]
    for bad in ([], "home", [1], [""], {"a": 1}):
        with pytest.raises(ApiError) as e:
            await core.patch(sid, {"config": {"calendars": bad}})
        assert e.value.status == 422 and e.value.detail["field"] == "calendars"


@pytest.mark.skipif(sys.platform != "darwin", reason="calendar sources exist only on macOS")
async def test_ticking_another_calendar_does_not_flood_old_events(tmp_path, monkeypatch):
    r = FakeReader()
    clock = Clock(T0)
    core = _core(tmp_path, monkeypatch, r, clock)
    d = await core.create({"type": "calendar", "kinds": {"new_event": {}}, "config": {"calendars": ["home"]}})
    sid = d["id"]
    await core._poll_one(core._cfg(sid))  # baseline at T0
    # two weeks on, the work calendar holds 5 events created last week
    clock.t = T0 + timedelta(days=14)
    for i in range(5):
        r.events_list.append(_ev(f"w{i}", clock.t + timedelta(days=2 + i),
                                 created=T0 + timedelta(days=7, minutes=i), title=f"Work {i}"))
        r.cal_of[f"w{i}"] = "work-1"
    await core._poll_one(core._cfg(sid))
    assert (await core.detail(sid))["history"] == []  # not picked yet
    await core.patch(sid, {"config": {"calendars": ["home", "work-1"]}})
    clock.t += timedelta(minutes=1)
    await core._poll_one(core._cfg(sid))
    assert r.asked[-1] == ["home", "work-1"]
    assert (await core.detail(sid))["history"] == []  # 0 arrivals: the selection starts from now
    # something created in it afterwards does arrive
    r.events_list.append(_ev("w-new", clock.t + timedelta(days=1), created=clock.t + timedelta(seconds=5),
                             title="Fresh"))
    r.cal_of["w-new"] = "work-1"
    clock.t += timedelta(minutes=1)
    await core._poll_one(core._cfg(sid))
    assert [h["title"] for h in (await core.detail(sid))["history"]] == ["Fresh"]


@pytest.mark.skipif(sys.platform != "darwin", reason="calendar sources exist only on macOS")
async def test_narrowed_then_all_again_reads_a_calendar_added_later(tmp_path, monkeypatch):
    r = FakeReader()
    clock = Clock(T0)
    core = _core(tmp_path, monkeypatch, r, clock)
    sid = (await core.create({"type": "calendar", "kinds": {"new_event": {}},
                              "config": {"calendars": ["home"]}}))["id"]
    await core._poll_one(core._cfg(sid))
    await core.patch(sid, {"config": {"calendars": None}})  # all re-ticked
    clock.t += timedelta(minutes=1)
    await core._poll_one(core._cfg(sid))
    assert r.asked[-1] is None
    # a calendar added to Calendar.app afterwards is read
    r.calendar_rows.append({"id": "new-cal", "title": "Kids", "account": "iCloud"})
    r.events_list.append(_ev("k", clock.t + timedelta(days=1), created=clock.t + timedelta(seconds=5),
                             title="School play"))
    r.cal_of["k"] = "new-cal"
    clock.t += timedelta(minutes=1)
    await core._poll_one(core._cfg(sid))
    assert [h["title"] for h in (await core.detail(sid))["history"]] == ["School play"]


@pytest.mark.skipif(sys.platform != "darwin", reason="calendar sources exist only on macOS")
async def test_an_existing_source_without_the_key_keeps_reading_every_calendar(tmp_path, monkeypatch):
    """v0.23.0 sources carry no key: upgrades keep the all-calendar behaviour,
    and a PATCH of something else (the kinds) does not restart the cursor."""
    r = FakeReader()
    clock = Clock(T0)
    core = _core(tmp_path, monkeypatch, r, clock)
    sid = (await core.create({"type": "calendar", "kinds": {"new_event": {}}}))["id"]
    await core._poll_one(core._cfg(sid))
    since = core.store.rt(sid)["state"]["since"]
    clock.t += timedelta(minutes=5)
    await core.patch(sid, {"kinds": {"new_event": {}, "before": {"minutes": 30}}})
    assert core.store.rt(sid)["state"]["since"] == since
    await core._poll_one(core._cfg(sid))
    assert r.asked[-1] is None


@pytest.mark.skipif(sys.platform != "darwin", reason="calendar sources exist only on macOS")
async def test_the_choices_answer_carries_the_group(tmp_path, monkeypatch):
    r = FakeReader()
    core = _core(tmp_path, monkeypatch, r, Clock(T0))
    got = await core.choices({"type": "calendar", "list": "calendars"})
    assert got == {"items": [{"value": "home", "label": "Home", "group": "iCloud"}]}


class _FakeCal:
    def __init__(self, cid):
        self.cid = cid


class _FakeStore:
    def __init__(self, known):
        self.known = known
        self.predicates = []
        self.reads = 0

    def calendarWithIdentifier_(self, cid):
        return _FakeCal(cid) if cid in self.known else None

    def predicateForEventsWithStartDate_endDate_calendars_(self, s, e, cals):
        self.predicates.append(cals)
        return ("pred", cals)

    def eventsMatchingPredicate_(self, pred):
        self.reads += 1
        return []


async def test_eventkit_skips_missing_calendars_and_never_passes_an_empty_list(monkeypatch):
    import types
    fake_found = types.ModuleType("Foundation")

    class NSDate:
        @staticmethod
        def dateWithTimeIntervalSince1970_(ts):
            return ts
    fake_found.NSDate = NSDate
    monkeypatch.setitem(sys.modules, "Foundation", fake_found)
    monkeypatch.setitem(sys.modules, "EventKit", types.ModuleType("EventKit"))
    src = EventKitSource()
    store = _FakeStore(known={"a"})
    src._store, src._checked = store, True
    s, e = "2026-11-16T09:00:00Z", "2026-11-17T09:00:00Z"
    # the hub's call (no calendars) is unchanged: EventKit's nil = every calendar
    assert await src.list_events(s, e) == []
    assert store.predicates == [None] and store.reads == 1
    # a deleted calendar is skipped; the rest are passed as objects
    await src.list_events(s, e, calendars=["a", "gone"])
    assert [c.cid for c in store.predicates[-1]] == ["a"] and store.reads == 2
    # none left: nothing is asked of EventKit
    assert await src.list_events(s, e, calendars=["gone"]) == []
    assert await src.list_events(s, e, calendars=[]) == []
    assert len(store.predicates) == 2 and store.reads == 2
