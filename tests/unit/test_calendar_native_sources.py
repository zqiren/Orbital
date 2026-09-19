# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the standalone-calendar native sources (Task 6): the
``memory`` source (open asks with ``due:`` + PROJECT_STATE ``[due:]`` facts —
spec 089 §3.6), the ``automation`` source
(future cron occurrences of enabled schedule triggers), and the read-only
``calendar_read`` agent tool. All logic is exercised against tmp workspaces —
no real daemon, no real cron clock.
"""

import os
from datetime import datetime, timezone

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.tools.calendar_read import CalendarReadTool
from agent_os.calendar_hub.hub import CalendarHub
from agent_os.calendar_hub.linkage import Linkage
from agent_os.calendar_hub.models import NormalizedEvent
from agent_os.calendar_hub.sources.automation_source import AutomationSource
from agent_os.calendar_hub.sources.memory_source import MemorySource


# ---- Fakes / helpers --------------------------------------------------------

class FakeProjectStore:
    def __init__(self, projects):
        self._projects = projects

    def list_projects(self):
        return self._projects


def _write_state(workspace: str, content: str) -> None:
    path = ProjectPaths(workspace).project_state
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _project(project_id, workspace, *, timezone_name=None, triggers=None):
    p = {"project_id": project_id, "workspace": str(workspace), "triggers": triggers or []}
    if timezone_name:
        p["timezone"] = timezone_name
    return p


# ---- MemorySource: timezone-sensitive due parsing ---------------------------

def _write_asks(workspace: str, lines: list[str]) -> None:
    from agent_os.agent import asks
    path = os.path.join(ProjectPaths(workspace).orbital_dir, "ASKS.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(asks.FORMAT_HEADER + "\n" + "\n".join(lines) + "\n")


async def test_memory_source_date_only_due_is_all_day_in_project_tz(tmp_path, monkeypatch):
    """A date-only ``due:`` must be resolved as project-tz midnight, not UTC
    midnight — even though the daemon process itself runs UTC. The query
    window below only contains Shanghai-local midnight (16:00 UTC the day
    before); it does NOT contain UTC midnight of the due date itself, so this
    fails loudly if the source ever falls back to UTC math.
    """
    monkeypatch.setenv("TZ", "UTC")
    ws = tmp_path / "proj-a"
    _write_asks(str(ws), ["- open a7f3a2 2026-07-19 due:2026-07-28 Send the draft."])
    src = MemorySource(FakeProjectStore([
        _project("proj-a", ws, timezone_name="Asia/Shanghai"),
    ]))

    events = await src.list_events("2026-07-27T15:00:00+00:00", "2026-07-27T17:00:00+00:00")

    assert len(events) == 1
    ev = events[0]
    assert ev.all_day is True
    assert ev.start == "2026-07-28"
    assert ev.end == "2026-07-28"
    assert ev.timezone == "Asia/Shanghai"
    assert ev.title == "Send the draft."
    assert ev.id == "memory:proj-a/a7f3a2"      # event id = the ask id


async def test_memory_source_timed_due_is_tz_aware_instant(tmp_path):
    ws = tmp_path / "proj-a"
    _write_asks(str(ws), ["- open 71e3aa 2026-07-20 due:2026-07-28T20:00 Call the vendor."])
    src = MemorySource(FakeProjectStore([
        _project("proj-a", ws, timezone_name="Asia/Shanghai"),
    ]))

    events = await src.list_events("2026-07-28T00:00:00+00:00", "2026-07-29T00:00:00+00:00")

    assert len(events) == 1
    ev = events[0]
    assert ev.all_day is False
    assert ev.start == "2026-07-28T20:00:00+08:00"
    assert ev.timezone == "Asia/Shanghai"


async def test_memory_source_closed_asks_emit_nothing(tmp_path):
    ws = tmp_path / "proj-a"
    _write_asks(str(ws), [
        "- open 00000a 2026-07-01 due:2026-07-20 Done already",
        "- done 00000a 2026-07-02 by:user",
        "- open 00000b 2026-07-01 due:2026-07-21 Dropped",
        "- dropped 00000b 2026-07-02 by:user",
        "- open 00000c 2026-07-01 Open but undated",
    ])
    src = MemorySource(FakeProjectStore([_project("proj-a", ws, timezone_name="UTC")]))

    events = await src.list_events("2026-07-01T00:00:00+00:00", "2026-07-31T00:00:00+00:00")

    assert events == []


async def test_memory_source_reopened_ask_emits_again(tmp_path):
    ws = tmp_path / "proj-a"
    _write_asks(str(ws), [
        "- open 00000a 2026-07-01 due:2026-07-20 Back on the list",
        "- done 00000a 2026-07-02 by:agent \"done\"",
        "- reopen 00000a 2026-07-03 by:user",
    ])
    src = MemorySource(FakeProjectStore([_project("proj-a", ws, timezone_name="UTC")]))
    events = await src.list_events("2026-07-01T00:00:00+00:00", "2026-07-31T00:00:00+00:00")
    assert [e.title for e in events] == ["Back on the list"]


async def test_memory_source_dated_fact_ignores_legacy_resolved_stamp(tmp_path):
    """Spec 089 removed the ``resolved:`` logic: a dated line's event lasts
    exactly as long as the line does."""
    ws = tmp_path / "proj-a"
    _write_state(str(ws), (
        "- [due:2026-07-15] Renew the old domain.\n"
        "<!--mem resolved:2026-07-16-->\n"
    ))
    src = MemorySource(FakeProjectStore([
        _project("proj-a", ws, timezone_name="UTC"),
    ]))

    events = await src.list_events("2026-07-01T00:00:00+00:00", "2026-07-31T00:00:00+00:00")

    assert [e.title for e in events] == ["Renew the old domain."]


async def test_memory_source_merges_asks_and_dated_facts(tmp_path):
    ws = tmp_path / "proj-a"
    _write_state(str(ws), "- [due:2026-07-30] Renew the domain.\n")
    _write_asks(str(ws), ["- open 00000a 2026-07-01 due:2026-07-29 Approve the budget"])
    src = MemorySource(FakeProjectStore([_project("proj-a", ws, timezone_name="UTC")]))
    events = await src.list_events("2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00")
    assert sorted(e.title for e in events) == ["Approve the budget", "Renew the domain."]


async def test_memory_source_unflagged_dated_fact_still_emits(tmp_path):
    """A dated fact (no ``user`` token) in PROJECT_STATE projects onto the
    calendar."""
    ws = tmp_path / "proj-a"
    _write_state(str(ws), "- [due:2026-07-30] Renew the domain.\n")
    src = MemorySource(FakeProjectStore([
        _project("proj-a", ws, timezone_name="UTC"),
    ]))

    events = await src.list_events("2026-07-01T00:00:00+00:00", "2026-07-31T00:00:00+00:00")

    assert len(events) == 1
    assert events[0].title == "Renew the domain."


async def test_memory_source_missing_id_uses_stable_fallback_key(tmp_path):
    """An un-id'd dated fact (predates id-stamping) still needs a stable
    per-entry event id — re-listing the same unchanged content must produce
    the SAME id both times (so the hub's cache / a future linkage entry keys
    consistently), while a second, differently-worded entry gets a different
    id."""
    ws = tmp_path / "proj-a"
    _write_state(str(ws), (
        "- [due:2026-07-30] Renew the domain.\n"
        "- [due:2026-07-31] Pay the invoice.\n"
    ))
    src = MemorySource(FakeProjectStore([
        _project("proj-a", ws, timezone_name="UTC"),
    ]))

    first = await src.list_events("2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00")
    second = await src.list_events("2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00")

    ids_first = sorted(ev.id for ev in first)
    ids_second = sorted(ev.id for ev in second)
    assert ids_first == ids_second  # stable across re-parses
    assert len(set(ids_first)) == 2  # distinct per entry
    for ev in first:
        assert ev.id.startswith("memory:proj-a/")


async def test_memory_source_events_carry_project_link(tmp_path):
    ws = tmp_path / "proj-a"
    _write_state(str(ws), "- [due:2026-07-30] Renew the domain.\n")
    src = MemorySource(FakeProjectStore([
        _project("proj-a", ws, timezone_name="UTC"),
    ]))

    events = await src.list_events("2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00")

    assert len(events) == 1
    assert events[0].project_id == "proj-a"


async def test_memory_source_out_of_range_due_excluded(tmp_path):
    ws = tmp_path / "proj-a"
    _write_state(str(ws), "- [due:2026-09-01] Way in the future.\n")
    src = MemorySource(FakeProjectStore([
        _project("proj-a", ws, timezone_name="UTC"),
    ]))

    events = await src.list_events("2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00")

    assert events == []


def test_memory_source_always_available():
    src = MemorySource(FakeProjectStore([]))
    assert src.available is True
    assert src.available is True  # no I/O side-effect from checking twice


# ---- AutomationSource: future-only cron expansion ---------------------------

def _schedule_trigger(trigger_id, cron, *, tz="UTC", enabled=True, name=None):
    return {
        "id": trigger_id, "type": "schedule", "enabled": enabled,
        "name": name or trigger_id,
        "schedule": {"cron": cron, "timezone": tz},
    }


async def test_automation_source_emits_from_start_of_today_within_range(tmp_path):
    """Today's already-fired slots ARE emitted (the Workbench Today overview
    needs the full day); occurrences on EARLIER days stay out (no synthesized
    history)."""
    frozen_now = datetime(2026, 7, 24, 10, 0, 0, tzinfo=timezone.utc)  # after today's 09:00 fire
    ws = tmp_path / "proj-a"
    src = AutomationSource(
        FakeProjectStore([
            _project("proj-a", ws, triggers=[_schedule_trigger("trig-1", "0 9 * * *")]),
        ]),
        now_fn=lambda: frozen_now,
    )

    events = await src.list_events("2026-07-23T00:00:00+00:00", "2026-07-26T00:00:00+00:00")

    starts = sorted(ev.start for ev in events)
    # 07-24 09:00 is today-past → included; 07-23 09:00 is yesterday → excluded.
    assert starts == ["2026-07-24T09:00:00+00:00", "2026-07-25T09:00:00+00:00"]


async def test_automation_source_disabled_trigger_emits_nothing(tmp_path):
    frozen_now = datetime(2026, 7, 24, 0, 0, 0, tzinfo=timezone.utc)
    ws = tmp_path / "proj-a"
    src = AutomationSource(
        FakeProjectStore([
            _project("proj-a", ws, triggers=[
                _schedule_trigger("trig-1", "0 9 * * *", enabled=False),
            ]),
        ]),
        now_fn=lambda: frozen_now,
    )

    events = await src.list_events("2026-07-23T00:00:00+00:00", "2026-07-30T00:00:00+00:00")

    assert events == []


async def test_automation_source_events_carry_project_link(tmp_path):
    frozen_now = datetime(2026, 7, 24, 0, 0, 0, tzinfo=timezone.utc)
    ws = tmp_path / "proj-a"
    src = AutomationSource(
        FakeProjectStore([
            _project("proj-a", ws, triggers=[_schedule_trigger("trig-1", "0 9 * * *")]),
        ]),
        now_fn=lambda: frozen_now,
    )

    events = await src.list_events("2026-07-23T00:00:00+00:00", "2026-07-26T00:00:00+00:00")

    assert len(events) >= 1
    for ev in events:
        assert ev.project_id == "proj-a"
        assert ev.id.startswith("automation:proj-a/trig-1/")


def test_automation_source_always_available():
    src = AutomationSource(FakeProjectStore([]))
    assert src.available is True
    assert src.available is True


# ---- CalendarHub: source-set project_id is preserved (Task 6 hub fix) ------

class _StaticSource:
    id = "memory"
    kind = "memory"
    available = True
    linked_by_hub = False  # mirrors MemorySource/AutomationSource's opt-out

    def __init__(self, events):
        self._events = events

    async def list_events(self, start, end):
        return list(self._events)


async def test_hub_preserves_source_set_project_id(tmp_path):
    """A native source's pre-set project_id must survive CalendarHub's merge
    — without this, the calendar_read tool's project lens would silently
    drop every memory/automation event (they are never in the Linkage
    store), which is exactly why this fix was required for Task 6."""
    ev = NormalizedEvent(
        source="memory", source_id="proj-a/x1", title="t",
        start="2026-07-28", end="2026-07-28", project_id="proj-a",
    )
    hub = CalendarHub(sources=[_StaticSource([ev])], linkage=Linkage(str(tmp_path)))

    events = await hub.list_events("2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00")
    assert events[0].project_id == "proj-a"

    lensed = await hub.list_events(
        "2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00", project_id="proj-a"
    )
    assert [e.id for e in lensed] == ["memory:proj-a/x1"]


# ---- calendar_read tool ------------------------------------------------------

class FakeCalendarHub:
    def __init__(self, events):
        self._events = events
        self.calls = []

    async def list_events(self, start, end, project_id=None):
        self.calls.append((start, end, project_id))
        return [e for e in self._events if project_id is None or e.project_id == project_id]


def _event(source_id, project_id):
    return NormalizedEvent(
        source="memory", source_id=source_id, title="t",
        start="2026-07-28", end="2026-07-28", project_id=project_id,
    )


async def test_calendar_read_tool_returns_merged_feed_lensed_to_project():
    hub = FakeCalendarHub([_event("proj-a/x1", "proj-a"), _event("proj-b/x2", "proj-b")])
    tool = CalendarReadTool(calendar_hub=hub, project_id="proj-a")

    result = await tool.execute(start="2026-07-01T00:00:00+00:00", end="2026-07-08T00:00:00+00:00")

    assert "proj-a/x1" in result.content
    assert "proj-b/x2" not in result.content
    assert hub.calls == [("2026-07-01T00:00:00+00:00", "2026-07-08T00:00:00+00:00", "proj-a")]


async def test_calendar_read_tool_rejects_over_90_day_range():
    hub = FakeCalendarHub([])
    tool = CalendarReadTool(calendar_hub=hub, project_id="proj-a")

    result = await tool.execute(start="2026-01-01T00:00:00+00:00", end="2026-06-01T00:00:00+00:00")

    assert result.content.startswith("Error:")
    assert "90" in result.content
    assert hub.calls == []  # rejected before ever touching the hub


async def test_calendar_read_tool_accepts_exactly_90_days():
    hub = FakeCalendarHub([])
    tool = CalendarReadTool(calendar_hub=hub, project_id="proj-a")

    result = await tool.execute(start="2026-01-01T00:00:00+00:00", end="2026-04-01T00:00:00+00:00")

    assert not result.content.startswith("Error:")


async def test_calendar_read_tool_rejects_malformed_dates():
    hub = FakeCalendarHub([])
    tool = CalendarReadTool(calendar_hub=hub, project_id="proj-a")

    result = await tool.execute(start="not-a-date", end="2026-07-08T00:00:00+00:00")

    assert result.content.startswith("Error:")
    assert hub.calls == []


@pytest.mark.parametrize("kwargs", [
    {"start": 123, "end": "2026-07-08T00:00:00+00:00"},
    {"start": "2026-07-01T00:00:00+00:00", "end": None},
])
async def test_calendar_read_tool_rejects_non_string_args(kwargs):
    hub = FakeCalendarHub([])
    tool = CalendarReadTool(calendar_hub=hub, project_id="proj-a")

    result = await tool.execute(**kwargs)

    assert result.content.startswith("Error:")
