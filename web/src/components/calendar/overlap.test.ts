// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { layoutOverlaps } from './overlap';
import type { CalendarEvent } from './types';

/** A timed (or all-day) event on Jul 1 2026 at the given local hours. */
function ev(id: string, startH: number, endH: number, all_day = false): CalendarEvent {
  const at = (h: number) => new Date(2026, 6, 1, Math.floor(h), Math.round((h % 1) * 60)).toISOString();
  return {
    id,
    source: 's',
    source_id: id,
    title: id,
    start: at(startH),
    end: at(endH),
    all_day,
    timezone: null,
    attendees: [],
    location: null,
    url: null,
    project_id: null,
  };
}

describe('layoutOverlaps', () => {
  it('gives non-overlapping events a single full-width column each', () => {
    const out = layoutOverlaps([ev('a', 9, 10), ev('b', 11, 12)]);
    expect(out).toHaveLength(2);
    expect(out.every((p) => p.cols === 1 && p.col === 0)).toBe(true);
  });

  it('splits two overlapping events into two side-by-side columns', () => {
    const out = layoutOverlaps([ev('a', 9, 10.5), ev('b', 10, 11)]);
    expect(out.map((p) => p.cols)).toEqual([2, 2]);
    expect(new Set(out.map((p) => p.col))).toEqual(new Set([0, 1]));
  });

  it('uses three columns for three mutually-overlapping events', () => {
    const out = layoutOverlaps([ev('a', 9, 12), ev('b', 9.5, 11), ev('c', 10, 10.75)]);
    expect(Math.max(...out.map((p) => p.cols))).toBe(3);
  });

  it('reuses a freed column when an earlier event has ended', () => {
    // a 9–10, b 9.5–11 (overlaps a), c 10–10.5 (overlaps b only). One cluster,
    // c can reuse a's column 0. Total width = 2 columns.
    const out = layoutOverlaps([ev('a', 9, 10), ev('b', 9.5, 11), ev('c', 10, 10.5)]);
    const byId = Object.fromEntries(out.map((p) => [p.event.id, p]));
    expect(byId.a.cols).toBe(2);
    expect(byId.c.col).toBe(0); // reused column 0
  });

  it('excludes all-day events from the timed layout', () => {
    const out = layoutOverlaps([ev('ad', 9, 10, true), ev('b', 9.5, 11)]);
    expect(out).toHaveLength(1);
    expect(out[0].event.id).toBe('b');
  });
});
