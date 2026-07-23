# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The normalized calendar event model (Spec 011 §0.4/§2; Task G1).

Every calendar source (EventKit, a Google-Calendar MCP connector, future
CalDAV/Outlook) maps its native events onto one ``NormalizedEvent`` so the
calendar surface and the project-linkage layer never see source-specific
shapes. The stable, cross-source identifier is ``id = "<source>:<source_id>"``
— used as the linkage key and in the link route path.
"""

from dataclasses import dataclass, field


@dataclass
class NormalizedEvent:
    """One calendar event, source-agnostic.

    ``project_id`` is Orbital metadata. For an ordinary externally-sourced
    event (never comes from the external calendar itself) it is left ``None``
    by the source and stamped by ``CalendarHub`` from the linkage store on
    every read (see ``CalendarHub.list_events``). A native,
    computed-from-Orbital-data source (e.g. the ``memory`` / ``automation``
    calendar sources, Task 6) sets it directly and declares
    ``linked_by_hub = False`` on the source class so the hub leaves it alone
    — those events are BORN linked to a project rather than manually linked.
    """

    source: str                       # source registry id, e.g. "eventkit"
    source_id: str                    # native id within that source
    title: str
    start: str                        # ISO 8601
    end: str                          # ISO 8601
    all_day: bool = False
    timezone: str | None = None
    attendees: list[str] = field(default_factory=list)
    location: str | None = None
    url: str | None = None
    project_id: str | None = None

    @property
    def id(self) -> str:
        """Stable cross-source identifier and linkage key."""
        return f"{self.source}:{self.source_id}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "all_day": self.all_day,
            "timezone": self.timezone,
            "attendees": list(self.attendees),
            "location": self.location,
            "url": self.url,
            "project_id": self.project_id,
        }
