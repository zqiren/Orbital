// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import {
  startOfWeek,
  addDays,
  addWeeks,
  weekDays,
  weekRangeISO,
  isSameDay,
  groupEventsByDay,
  weekRangeLabel,
} from './range';
import type { CalendarEvent } from './types';

function ev(overrides: Partial<CalendarEvent>): CalendarEvent {
  return {
    id: 'id',
    source: 's',
    source_id: 'x',
    title: '',
    start: '',
    end: '',
    all_day: false,
    timezone: null,
    attendees: [],
    location: null,
    url: null,
    project_id: null,
    ...overrides,
  };
}

const DAY_MS = 24 * 3600 * 1000;
/** Local ISO for a specific wall-clock time on a given local day. */
function localISO(d: Date, h: number, m = 0): string {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate(), h, m).toISOString();
}

describe('startOfWeek (Monday week start)', () => {
  it('returns the Monday at local midnight for a mid-week date', () => {
    const wed = new Date(2026, 6, 1, 15, 30); // Wed Jul 1 2026, 15:30 local
    const mon = startOfWeek(wed);
    expect(mon.getDay()).toBe(1); // Monday
    expect(mon.getHours()).toBe(0);
    expect(mon.getMinutes()).toBe(0);
    expect(mon.getTime()).toBeLessThanOrEqual(wed.getTime());
    expect(wed.getTime() - mon.getTime()).toBeLessThan(7 * DAY_MS);
  });

  it('is idempotent when the input is already a Monday', () => {
    const mon = startOfWeek(new Date(2026, 6, 1));
    expect(startOfWeek(mon).getTime()).toBe(mon.getTime());
  });

  it('treats Sunday as the LAST day of the week, not the start', () => {
    const mon = startOfWeek(new Date(2026, 6, 1));
    const sunday = addDays(mon, 6);
    expect(sunday.getDay()).toBe(0); // Sunday
    expect(startOfWeek(sunday).getTime()).toBe(mon.getTime());
  });
});

describe('addWeeks (prev/next range nav)', () => {
  it('shifts the week by exactly seven calendar days, staying on Monday', () => {
    const mon = startOfWeek(new Date(2026, 6, 1));
    const next = startOfWeek(addWeeks(mon, 1));
    const prev = startOfWeek(addWeeks(mon, -1));
    expect(next.getDay()).toBe(1);
    expect(prev.getDay()).toBe(1);
    expect(Math.round((next.getTime() - mon.getTime()) / DAY_MS)).toBe(7);
    expect(Math.round((mon.getTime() - prev.getTime()) / DAY_MS)).toBe(7);
  });
});

describe('weekDays / weekRangeISO', () => {
  it('weekDays yields seven days Monday..Sunday', () => {
    const days = weekDays(new Date(2026, 6, 1));
    expect(days).toHaveLength(7);
    expect(days[0].getDay()).toBe(1);
    expect(days[6].getDay()).toBe(0);
  });

  it('weekRangeISO spans [Monday, following Monday)', () => {
    const { start, end } = weekRangeISO(new Date(2026, 6, 1));
    const s = new Date(start);
    const e = new Date(end);
    expect(s.getDay()).toBe(1);
    expect(e.getDay()).toBe(1);
    expect(Math.round((e.getTime() - s.getTime()) / DAY_MS)).toBe(7);
  });
});

describe('weekRangeLabel', () => {
  it('renders a dashed range that includes the year', () => {
    const label = weekRangeLabel(new Date(2026, 6, 1), 'en-US');
    expect(label).toContain('–');
    expect(label).toContain('2026');
  });
});

describe('groupEventsByDay (agenda grouping)', () => {
  it('buckets events into the seven visible days, all-day first then by start', () => {
    const mon = startOfWeek(new Date(2026, 6, 1));
    const tue = addDays(mon, 1);
    const timed10 = ev({ id: 't10', start: localISO(tue, 10), end: localISO(tue, 11) });
    const timed9 = ev({ id: 't9', start: localISO(tue, 9), end: localISO(tue, 9, 30) });
    const allDay = ev({ id: 'ad', all_day: true, start: localISO(tue, 0), end: localISO(tue, 0) });

    const groups = groupEventsByDay([timed10, allDay, timed9], mon);

    expect(groups).toHaveLength(7);
    const tueGroup = groups[1]; // index 1 = Tuesday
    expect(isSameDay(tueGroup.day, tue)).toBe(true);
    expect(tueGroup.events.map((e) => e.id)).toEqual(['ad', 't9', 't10']);
    // Monday and the rest are empty.
    expect(groups[0].events).toHaveLength(0);
  });

  it('drops events whose start falls outside the visible week', () => {
    const mon = startOfWeek(new Date(2026, 6, 1));
    const nextWeek = addDays(mon, 9);
    const outside = ev({ id: 'out', start: localISO(nextWeek, 10), end: localISO(nextWeek, 11) });
    const groups = groupEventsByDay([outside], mon);
    expect(groups.every((g) => g.events.length === 0)).toBe(true);
  });
});
