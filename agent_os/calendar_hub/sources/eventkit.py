# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""macOS EventKit calendar source (Spec 011 §0.5; Task G1).

Reads the Mac's own calendar store via ``pyobjc-framework-EventKit`` — no OAuth,
and it sees *every* account the user already has in Calendar.app
(iCloud/Google/Exchange). ``EventKit`` is imported lazily and the ``EKEventStore``
is created on first use.

**TCC consent:** the first time this source touches the event store it triggers
the macOS privacy prompt for the daemon-hosting process (Orbital.app in the
packaged build, the Python/uvicorn process in dev). That is expected and only
happens on real access — never at import or construction, and never from the
unit tests (which run on machines without the framework and only exercise the
graceful-unavailable path).

Every failure mode — non-darwin, framework missing, access denied, fetch error
— degrades to ``available=False`` / ``list_events == []``. This source never
raises out.
"""

import logging
import sys
import threading
from datetime import datetime, timezone

from ..models import NormalizedEvent

logger = logging.getLogger(__name__)

# EKEntityTypeEvent
_ENTITY_EVENT = 0
# EKAuthorizationStatus.authorized / .fullAccess both == 3
_AUTH_GRANTED = 3
_ACCESS_TIMEOUT = 30.0


class EventKitSource:
    id = "eventkit"
    kind = "eventkit"

    def __init__(self):
        self._store = None
        self._checked = False  # have we attempted store creation + access yet

    @property
    def available(self) -> bool:
        return self._ensure_store() is not None

    def _ensure_store(self):
        """Lazily create the EKEventStore and request access. Idempotent;
        returns the store or ``None`` if EventKit is unusable here."""
        if self._checked:
            return self._store
        self._checked = True
        if sys.platform != "darwin":
            return None
        try:
            import EventKit  # noqa: PLC0415 (lazy on purpose — see docstring)
        except Exception:
            logger.info("EventKit framework unavailable; calendar source disabled")
            return None
        try:
            store = EventKit.EKEventStore.alloc().init()
            if not self._request_access(EventKit, store):
                logger.info("EventKit access not granted; calendar source disabled")
                return None
            self._store = store
        except Exception:
            logger.warning("EventKit store init failed; calendar source disabled", exc_info=True)
            self._store = None
        return self._store

    def _request_access(self, EventKit, store) -> bool:
        """Block (bounded) on the async access request. Triggers the TCC prompt
        on first ever call for this process."""
        status = EventKit.EKEventStore.authorizationStatusForEntityType_(_ENTITY_EVENT)
        if status == _AUTH_GRANTED:
            return True
        done = threading.Event()
        outcome = {"granted": False}

        def _completion(granted, error):
            outcome["granted"] = bool(granted)
            done.set()

        # macOS 14+ splits full/write access; fall back to the pre-14 API.
        if hasattr(store, "requestFullAccessToEventsWithCompletion_"):
            store.requestFullAccessToEventsWithCompletion_(_completion)
        else:
            store.requestAccessToEntityType_completion_(_ENTITY_EVENT, _completion)
        done.wait(timeout=_ACCESS_TIMEOUT)
        return outcome["granted"]

    async def list_events(self, start: str, end: str) -> list[NormalizedEvent]:
        store = self._ensure_store()
        if store is None:
            return []
        try:
            import EventKit  # noqa: PLC0415
            from Foundation import NSDate  # noqa: PLC0415

            start_dt = _parse_iso(start)
            end_dt = _parse_iso(end)
            if start_dt is None or end_dt is None:
                return []
            ns_start = NSDate.dateWithTimeIntervalSince1970_(start_dt.timestamp())
            ns_end = NSDate.dateWithTimeIntervalSince1970_(end_dt.timestamp())
            predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
                ns_start, ns_end, None
            )
            ek_events = store.eventsMatchingPredicate_(predicate) or []
            return [self._map(ev) for ev in ek_events]
        except Exception:
            logger.warning("EventKit list_events failed", exc_info=True)
            return []

    def _map(self, ev) -> NormalizedEvent:
        return NormalizedEvent(
            source=self.id,
            source_id=str(ev.eventIdentifier()),
            title=str(ev.title() or "(no title)"),
            start=_nsdate_to_iso(ev.startDate()),
            end=_nsdate_to_iso(ev.endDate()),
            all_day=bool(ev.isAllDay()),
            timezone=(str(ev.timeZone().name()) if ev.timeZone() is not None else None),
            attendees=_attendee_names(ev),
            location=(str(ev.location()) if ev.location() else None),
            url=(str(ev.URL().absoluteString()) if ev.URL() is not None else None),
        )


def _attendee_names(ev) -> list[str]:
    out = []
    for p in ev.attendees() or []:
        name = p.name() if hasattr(p, "name") else None
        out.append(str(name) if name else str(p.URL().absoluteString()))
    return out


def _parse_iso(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _nsdate_to_iso(ns_date) -> str:
    if ns_date is None:
        return ""
    ts = ns_date.timeIntervalSince1970()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
