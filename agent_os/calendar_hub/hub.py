# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""CalendarHub — merges calendar sources into one normalized feed (Task G1).

Holds an ordered list of sources (each exposing ``id``, ``kind``,
``available``, and ``async list_events(start, end)``) plus the project linkage
store. ``list_events`` fans out across available sources, stamps each event's
``project_id`` from the linkage map, and optionally filters to one project. A
per-source in-memory cache keyed on the ``(start, end)`` range (short TTL)
absorbs repeated views; ``refresh`` clears it for the manual refresh button.

A source that is unavailable, or that raises while listing, contributes zero
events — one bad source never breaks the merged feed.
"""

import logging
import time
from typing import Callable

from .linkage import Linkage
from .models import NormalizedEvent

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL = 60.0  # seconds


class CalendarHub:
    def __init__(
        self,
        sources: list,
        linkage: Linkage,
        *,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._sources = list(sources)
        self._linkage = linkage
        self._cache_ttl = cache_ttl
        self._clock = clock
        # source id -> {(start, end): (fetched_at, events)}
        self._cache: dict[str, dict[tuple[str, str], tuple[float, list[NormalizedEvent]]]] = {}

    async def list_events(
        self, start: str, end: str, project_id: str | None = None
    ) -> list[NormalizedEvent]:
        events: list[NormalizedEvent] = []
        for source in self._sources:
            src_events = await self._source_events(source, start, end)
            # Stamp project linkage fresh every call (idempotent) so a
            # link/unlink takes effect without touching the event cache.
            # Skipped for a source that opts out via `linked_by_hub = False`
            # (the native `memory`/`automation` sources — Task 6 — are BORN
            # linked to a project and set `project_id` themselves; they are
            # never manually linked, and re-stamping from the — necessarily
            # empty — linkage store would null them back out). Every other
            # source is re-stamped unconditionally on each call, same as
            # before: events are cached (see `_source_events`), so a cached
            # object's `project_id` must be overwritten every time rather
            # than only when still `None`, or an unlink would never be
            # observed on an already-linked cached event.
            if getattr(source, "linked_by_hub", True):
                for ev in src_events:
                    ev.project_id = self._linkage.get(ev.id)
            events.extend(src_events)
        if project_id is not None:
            events = [ev for ev in events if ev.project_id == project_id]
        return events

    async def _source_events(
        self, source, start: str, end: str
    ) -> list[NormalizedEvent]:
        try:
            available = bool(source.available)
        except Exception:
            logger.warning("calendar source availability check failed", exc_info=True)
            available = False
        if not available:
            return []

        cache = self._cache.setdefault(source.id, {})
        key = (start, end)
        now = self._clock()
        cached = cache.get(key)
        if cached is not None and (now - cached[0]) < self._cache_ttl:
            return cached[1]

        try:
            fetched = await source.list_events(start, end)
        except Exception:
            logger.warning(
                "calendar source %s failed to list events", getattr(source, "id", "?"),
                exc_info=True,
            )
            return []
        events = list(fetched)
        cache[key] = (now, events)
        return events

    def refresh(self) -> None:
        """Clear every per-source cache (the manual refresh button)."""
        self._cache.clear()

    def availability(self) -> dict:
        sources = []
        for s in self._sources:
            try:
                available = bool(s.available)
            except Exception:
                available = False
            sources.append({"id": s.id, "kind": s.kind, "available": available})
        return {
            "available": any(s["available"] for s in sources),
            "sources": sources,
        }

    # ---- Linkage passthrough (routes hold only the hub) ----

    def link(self, event_id: str, project_id: str) -> None:
        self._linkage.link(event_id, project_id)

    def unlink(self, event_id: str) -> None:
        self._linkage.unlink(event_id)
