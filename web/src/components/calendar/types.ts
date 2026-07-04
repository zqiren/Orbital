// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Frontend mirror of the calendar hub's wire shapes (Spec 011 §0.4, Task G1).
 * These match `GET /api/v2/calendar/events` and `/availability` byte-for-byte
 * so the surface never guesses at field names. Times are ISO 8601 strings
 * rendered in the browser's local zone (Date + Intl, no tz library).
 */

/** One normalized calendar event. `id` is the composite `<source>:<source_id>`. */
export interface CalendarEvent {
  id: string;
  source: string;
  source_id: string;
  title: string;
  start: string;
  end: string;
  all_day: boolean;
  timezone: string | null;
  attendees: string[];
  location: string | null;
  url: string | null;
  /** Orbital linkage metadata: the project this event is linked to, or null. */
  project_id: string | null;
}

/** Per-source availability entry from `GET /api/v2/calendar/availability`. */
export interface AvailabilitySource {
  id: string;
  kind: string;
  available: boolean;
}

/** Response of `GET /api/v2/calendar/availability`. */
export interface Availability {
  available: boolean;
  sources: AvailabilitySource[];
}

/** Response of `GET /api/v2/calendar/events`. */
export interface EventsResponse {
  events: CalendarEvent[];
}
