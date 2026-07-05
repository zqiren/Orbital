// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Pure week-range math + display formatting for the calendar surface.
 *
 * Week starts MONDAY (Spec 011 §0.4). Everything works in the browser's LOCAL
 * zone: a `Date` constructed from an ISO string carries an absolute instant,
 * and the getters below (`getDay`, `getHours`, …) read it back in local time,
 * which is exactly the rendering contract. No timezone library — `Date` +
 * `Intl.DateTimeFormat` only.
 *
 * These functions are exported standalone (not bound to React) so they unit-
 * test as pure input→output, per the plan's Vitest list.
 */

import type { CalendarEvent } from './types';

/** Midnight (local) of the Monday that starts the week containing `d`. */
export function startOfWeek(d: Date): Date {
  const r = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  // getDay(): 0=Sun … 6=Sat. Days since Monday = (dow + 6) % 7.
  const sinceMonday = (r.getDay() + 6) % 7;
  r.setDate(r.getDate() - sinceMonday);
  r.setHours(0, 0, 0, 0);
  return r;
}

/** A new Date `n` days after `d` (n may be negative). */
export function addDays(d: Date, n: number): Date {
  const r = new Date(d);
  r.setDate(r.getDate() + n);
  return r;
}

/** A new Date `n` weeks after `d` (n may be negative). Drives prev/next nav. */
export function addWeeks(d: Date, n: number): Date {
  return addDays(d, n * 7);
}

/** The seven local day-start Dates (Mon…Sun) of the week containing `anchor`. */
export function weekDays(anchor: Date): Date[] {
  const start = startOfWeek(anchor);
  return Array.from({ length: 7 }, (_, i) => addDays(start, i));
}

/**
 * The half-open [start, end) instant range of the whole week as ISO strings,
 * for the `?start=&end=` query. `end` is the Monday of the FOLLOWING week, so
 * the range covers all seven days inclusively.
 */
export function weekRangeISO(anchor: Date): { start: string; end: string } {
  const start = startOfWeek(anchor);
  const end = addDays(start, 7);
  return { start: start.toISOString(), end: end.toISOString() };
}

/** True when `a` and `b` fall on the same local calendar day. */
export function isSameDay(a: Date, b: Date): boolean {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  );
}

/** Minutes from local midnight for an ISO instant (0…1440). */
export function minutesFromMidnight(iso: string): number {
  const d = new Date(iso);
  return d.getHours() * 60 + d.getMinutes();
}

/**
 * A human range label for the visible week, e.g. "Jun 30 – Jul 6, 2026" or
 * "Dec 29, 2025 – Jan 4, 2026" across a year boundary. `locale` defaults to
 * the browser default; tests pass a fixed locale for determinism.
 */
export function weekRangeLabel(anchor: Date, locale?: string): string {
  const days = weekDays(anchor);
  const first = days[0];
  const last = days[6];
  const sameYear = first.getFullYear() === last.getFullYear();
  const monthDay = new Intl.DateTimeFormat(locale, { month: 'short', day: 'numeric' });
  const monthDayYear = new Intl.DateTimeFormat(locale, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
  const left = sameYear ? monthDay.format(first) : monthDayYear.format(first);
  const right = monthDayYear.format(last);
  return `${left} – ${right}`;
}

/** Short weekday + day-of-month header for one agenda/grid day column. */
export function dayHeaderLabel(d: Date, locale?: string): string {
  return new Intl.DateTimeFormat(locale, { weekday: 'short', day: 'numeric' }).format(d);
}

/** Longer day header for the mobile agenda, e.g. "Monday, Jun 30". */
export function agendaDayLabel(d: Date, locale?: string): string {
  return new Intl.DateTimeFormat(locale, {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
  }).format(d);
}

/** Local wall-clock time of an ISO instant, e.g. "9:30 AM". */
export function formatTime(iso: string, locale?: string): string {
  return new Intl.DateTimeFormat(locale, { hour: 'numeric', minute: '2-digit' }).format(
    new Date(iso),
  );
}

/** "9:30 – 10:30 AM"-style span for a timed event's detail/agenda line. */
export function formatTimeRange(startISO: string, endISO: string, locale?: string): string {
  return `${formatTime(startISO, locale)} – ${formatTime(endISO, locale)}`;
}

/** One agenda day group: the day and its events (sorted, all-day first). */
export interface DayGroup {
  day: Date;
  events: CalendarEvent[];
}

/**
 * Group events into the seven visible days of `anchor`'s week. Every day gets
 * an entry (empty days included, so the agenda can show them or skip them);
 * within a day, all-day events sort first, then timed events by start.
 * Events whose start falls outside the week are dropped.
 */
export function groupEventsByDay(events: CalendarEvent[], anchor: Date): DayGroup[] {
  const days = weekDays(anchor);
  return days.map((day) => {
    const dayEvents = events
      .filter((ev) => isSameDay(new Date(ev.start), day))
      .sort((a, b) => {
        if (a.all_day !== b.all_day) return a.all_day ? -1 : 1;
        return new Date(a.start).getTime() - new Date(b.start).getTime();
      });
    return { day, events: dayEvents };
  });
}
