// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Side-by-side overlap layout for a single day's TIMED events (Spec 011 §0.4:
 * "simple side-by-side overlap handling — equal-width split is fine, no fancy
 * packing").
 *
 * Events that overlap in time are grouped into a cluster; within a cluster each
 * event is greedily assigned the first column whose previous event has already
 * ended. The whole cluster then reports the same column COUNT, so the renderer
 * gives every event in it an equal `1/cols` width. Non-overlapping events stand
 * alone at full width (`cols === 1`).
 *
 * Pure function — all-day events must be filtered out by the caller.
 */

import type { CalendarEvent } from './types';

export interface PositionedEvent {
  event: CalendarEvent;
  /** 0-based column this event occupies within its overlap cluster. */
  col: number;
  /** Total columns in this event's cluster (equal-width divisor). */
  cols: number;
}

function ms(iso: string): number {
  return new Date(iso).getTime();
}

export function layoutOverlaps(events: CalendarEvent[]): PositionedEvent[] {
  const timed = events.filter((e) => !e.all_day);
  const sorted = [...timed].sort(
    (a, b) => ms(a.start) - ms(b.start) || ms(a.end) - ms(b.end),
  );

  const result: PositionedEvent[] = [];
  let cluster: CalendarEvent[] = [];
  let clusterEnd = -Infinity;

  const flush = () => {
    if (cluster.length === 0) return;
    // colEnd[i] = end instant of the last event placed in column i.
    const colEnd: number[] = [];
    const placed: Array<{ event: CalendarEvent; col: number }> = [];
    for (const e of cluster) {
      let col = 0;
      for (; col < colEnd.length; col++) {
        if (ms(e.start) >= colEnd[col]) break; // column has freed up
      }
      colEnd[col] = ms(e.end);
      placed.push({ event: e, col });
    }
    const cols = colEnd.length;
    for (const p of placed) result.push({ event: p.event, col: p.col, cols });
    cluster = [];
    clusterEnd = -Infinity;
  };

  for (const e of sorted) {
    // A gap (this event starts at/after everything so far ended) closes the
    // current cluster before this event opens a new one.
    if (cluster.length > 0 && ms(e.start) >= clusterEnd) flush();
    cluster.push(e);
    clusterEnd = Math.max(clusterEnd, ms(e.end));
  }
  flush();

  return result;
}
