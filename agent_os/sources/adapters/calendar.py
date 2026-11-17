# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Mac's own calendar (every account Calendar.app holds), read through
EventKit with the existing reader (``agent_os.calendar_hub.sources.eventkit``).
macOS only.

Kinds:
- ``new_event``: an event that was *created* after the source started
  watching appears in the next ``NEW_WINDOW_DAYS`` days. Created, not merely
  seen: an old event that drifts into the window as days pass is not new.
  One item per event (a repeating series is one item). Events whose
  calendar gives no creation date are not counted.
- ``before``: a timed event starts within N minutes (N from the kind's
  fixed choices). One item per occurrence and start time: an event moved
  before its reminder fires is reminded at its new time; one moved after
  its reminder fired is reminded again for the new time (same subject, so
  an open task merges it). All-day events are skipped, and nothing fires
  for an event that already started (a Mac waking from sleep sends no
  stale reminders).

Calendars: ``config.calendars`` absent means every calendar, including
ones added to Calendar.app later. A list of ``calendarIdentifier`` values
reads only those; ids that no longer exist are skipped, and when none is
left the poll reads nothing (EventKit is never asked with an empty list).
The form lists them through ``choices(…, "calendars")``, grouped by account.

Permission: the macOS Calendar prompt is only ever asked for from a user's
own gesture — ``verify`` (pressing connect) or ``choices`` (opening the
calendar list) — when the access is undecided or "Add Events Only". A poll
only reads the current authorisation and, without full access, fails with
``calendar_permission`` (the panel offers opening System Settings) — it
never prompts.
"""

from __future__ import annotations

import asyncio
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from agent_os.sources.adapters import Adapter
from agent_os.sources.model import Choice, Item, SourceFailure, Verified, VerifyError

NEW_WINDOW_DAYS = 60
DEFAULT_MINUTES = 30

# EKAuthorizationStatus
NOT_DETERMINED, RESTRICTED, DENIED, FULL_ACCESS, WRITE_ONLY = 0, 1, 2, 3, 4


@dataclass
class CalEvent:
    id: str
    title: str
    start: datetime
    end: datetime | None = None
    all_day: bool = False
    location: str | None = None
    url: str | None = None
    attendees: list[str] = field(default_factory=list)
    created: datetime | None = None
    occurrence: datetime | None = None
    recurring: bool = False


def _parse(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_cal_event(base, ev) -> CalEvent:
    """The reader's NormalizedEvent plus what it leaves out (creation date,
    original occurrence date, whether the event repeats), from the EKEvent."""
    from agent_os.calendar_hub.sources.eventkit import _nsdate_to_iso

    def _d(name):
        try:
            v = getattr(ev, name)()
            return _parse(_nsdate_to_iso(v)) if v is not None else None
        except Exception:
            return None
    try:
        recurring = bool(ev.hasRecurrenceRules())
    except Exception:
        recurring = False
    return CalEvent(id=base.source_id, title=base.title, start=_parse(base.start),
                    end=_parse(base.end), all_day=base.all_day, location=base.location,
                    url=base.url, attendees=list(base.attendees), created=_d("creationDate"),
                    occurrence=_d("occurrenceDate"), recurring=recurring)


class EventKitReader:
    """The real reader. Every EventKit call runs in a worker thread with its
    own store (EKEventStore is created per read, never shared across
    threads)."""

    @staticmethod
    def status() -> int | None:
        """The current authorisation, read without prompting; None when
        EventKit cannot be used here."""
        if sys.platform != "darwin":
            return None
        try:
            import EventKit  # noqa: PLC0415 (lazy: macOS-only framework)
            return int(EventKit.EKEventStore.authorizationStatusForEntityType_(0))
        except Exception:
            return None

    @staticmethod
    def request_access() -> bool:
        """Asks macOS for Calendar access (shows the prompt the first time).
        Blocking, bounded by the reader's own timeout. Only from verify."""
        from agent_os.calendar_hub.sources.eventkit import EventKitSource
        return EventKitSource().available

    @staticmethod
    def calendars() -> list[dict]:
        """Blocking. ``[{"id", "title", "account"}]``; only called with full
        access."""
        from agent_os.calendar_hub.sources.eventkit import EventKitSource
        return EventKitSource().list_calendars()

    def events(self, start: datetime, end: datetime,
               calendars: list[str] | None = None) -> list[CalEvent]:
        """Blocking. Only called with full access (so the store's own access
        check passes without a prompt). ``calendars``: None = every one."""
        from agent_os.calendar_hub.sources.eventkit import EventKitSource

        class _Src(EventKitSource):
            def _map(self, ev):
                return to_cal_event(super()._map(ev), ev)
        src = _Src()
        rows = asyncio.run(src.list_events(_iso(start), _iso(end), calendars=calendars))
        return [r for r in rows if isinstance(r, CalEvent) and r.start is not None]


def selected_calendars(config: dict | None) -> list[str] | None:
    """The source's calendars: None (no key) = every calendar; otherwise the
    stored ``calendarIdentifier`` strings, exactly as stored."""
    sel = (config or {}).get("calendars")
    if not isinstance(sel, list):
        return None
    return [s for s in sel if isinstance(s, str) and s]


class CalendarAdapter(Adapter):
    mode = "poll"
    interval_s = 60
    platforms = ["darwin"]

    def __init__(self, reader=None):
        self.reader = reader or EventKitReader()

    # -- verify -------------------------------------------------------------

    async def _ask_once(self, ctx) -> None:
        """A user's gesture (connect, opening the calendar list): the one
        place macOS may ask. "Add Events Only" (write-only) gets the same
        single ask for full access; if macOS shows nothing, the panel's line
        says where to switch Orbital to Full Access."""
        status = await ctx.to_thread(self.reader.status)
        if status is None:
            raise VerifyError("error", None, "the calendar cannot be read on this computer")
        if status in (NOT_DETERMINED, WRITE_ONLY):
            granted = await ctx.to_thread(self.reader.request_access)
            status = FULL_ACCESS if granted else await ctx.to_thread(self.reader.status)
        if status != FULL_ACCESS:
            raise VerifyError("permission", None, "Orbital has no access to Calendar")

    async def verify(self, ctx) -> Verified:
        await self._ask_once(ctx)
        return Verified(label=None)

    async def choices(self, ctx, list_name: str, params: dict) -> list:
        if list_name != "calendars":
            return []
        await self._ask_once(ctx)
        rows = await ctx.to_thread(self.reader.calendars)
        return [Choice(value=r["id"], label=r.get("title") or r["id"], group=r.get("account") or "")
                for r in rows or [] if isinstance(r, dict) and r.get("id")]

    # -- poll ---------------------------------------------------------------

    async def poll(self, ctx) -> list:
        status = await ctx.to_thread(self.reader.status)
        if status != FULL_ACCESS:
            # never prompt from a timer
            raise SourceFailure("calendar_permission", {})
        now = ctx.now()
        st = ctx.state
        since = _parse(st.get("since"))
        if since is None:
            since = now
            st["since"] = _iso(now)
        want_new = "new_event" in ctx.kinds
        before = ctx.kinds.get("before")
        minutes = DEFAULT_MINUTES
        if before is not None:
            try:
                minutes = int(before.get("minutes") or DEFAULT_MINUTES)
            except (TypeError, ValueError):
                minutes = DEFAULT_MINUTES
        if not want_new and before is None:
            return []
        calendars = selected_calendars(ctx.config)
        if calendars == []:
            return []  # nothing picked: never ask EventKit with an empty list
        end = now + (timedelta(days=NEW_WINDOW_DAYS) if want_new else timedelta(minutes=minutes + 1))
        try:
            events = await ctx.to_thread(self.reader.events, now - timedelta(minutes=1), end, calendars)
        except Exception:
            raise SourceFailure("error")
        out: list = []
        for ev in events:
            if want_new and ev.created is not None and ev.created > since:
                out.append(self._new_item(ev, ctx))
            if before is not None and not ev.all_day and now < ev.start <= now + timedelta(minutes=minutes):
                out.append(self._before_item(ev, now, ctx))
        # a repeating series is one "new event"
        seen, unique = set(), []
        for it in out:
            if it.id not in seen:
                seen.add(it.id)
                unique.append(it)
        return unique

    @staticmethod
    def _when(ev: CalEvent) -> str:
        s = ev.start.astimezone()
        if ev.all_day:
            return s.strftime("%Y-%m-%d") + " (all day)"
        text = s.strftime("%Y-%m-%d %H:%M")
        if ev.end is not None:
            e = ev.end.astimezone()
            text += "–" + (e.strftime("%H:%M") if e.date() == s.date() else e.strftime("%Y-%m-%d %H:%M"))
        return text + s.strftime(" (UTC%z)")

    def _body(self, ev: CalEvent) -> str:
        lines = [f"When: {self._when(ev)}"]
        if ev.location:
            lines.append(f"Where: {ev.location}")
        if ev.attendees:
            lines.append("With: " + ", ".join(ev.attendees[:20]))
        if ev.url:
            lines.append(f"Link: {ev.url}")
        return "\n".join(lines)

    def _new_item(self, ev: CalEvent, ctx) -> Item:
        return Item(id=f"new:{ev.id}", kind="new_event", subject=f"event:{ev.id}",
                    title=ev.title, body=self._body(ev), url=ev.url,
                    at=_iso(ev.created) if ev.created else None)

    def _before_item(self, ev: CalEvent, now: datetime, ctx) -> Item:
        n = max(1, math.ceil((ev.start - now).total_seconds() / 60))
        title = (f"{ev.title}（{n} 分钟后开始）" if getattr(ctx, "lang", "en") == "zh"
                 else f"{ev.title} (starts in {n} min)")
        occ = ev.occurrence if ev.recurring and ev.occurrence else None
        subject = f"event:{ev.id}@{_iso(occ)}" if occ else f"event:{ev.id}"
        return Item(id=f"before:{ev.id}:{_iso(ev.start)}", kind="before", subject=subject,
                    title=title, body=self._body(ev), url=ev.url, at=_iso(now))


ADAPTER = CalendarAdapter
