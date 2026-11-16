// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { buildHistoryRows, dateLabel, gapLabel } from './historyModel';
import { item } from './testFixtures';
import type { HistoryEntry } from './types';

// Local wall-clock times so the labels do not depend on the machine's zone.
const at = (d: number, h: number, m: number) => new Date(2026, 10, d, h, m).toISOString();
const NOW = new Date(2026, 10, 16, 14, 20); // Monday 16 Nov 2026
const tr = (k: string, v?: Record<string, string | number>) =>
  k + (v ? `(${Object.entries(v).map(([a, b]) => `${a}=${b}`).join(',')})` : '');

describe('dateLabel', () => {
  it('today, yesterday, this week, older', () => {
    expect(dateLabel(at(16, 9, 5), NOW, 'zh', tr)).toBe('sources.time.today(t=09:05)');
    expect(dateLabel(at(15, 22, 20), NOW, 'zh', tr)).toBe('sources.time.yesterday(t=22:20)');
    expect(dateLabel(at(12, 19, 0), NOW, 'zh', tr)).toBe('周四 19:00');
    expect(dateLabel(at(12, 19, 0), NOW, 'en', tr)).toBe('Thu 19:00');
    expect(dateLabel(at(2, 8, 0), NOW, 'zh', tr)).toBe('sources.time.date(date=11月2日,t=08:00)');
  });
});

describe('gapLabel', () => {
  it('a gap still open runs to now; one inside a day shows the end time only', () => {
    expect(gapLabel({ id: 'g', kind: 'gap', start: at(16, 13, 35), end: null }, NOW, 'zh', tr))
      .toBe('sources.time.today(t=13:35) – sources.time.now');
    expect(gapLabel({ id: 'g', kind: 'gap', start: at(16, 13, 16), end: at(16, 13, 28) }, NOW, 'zh', tr))
      .toBe('sources.time.today(t=13:16) – 13:28');
    expect(gapLabel({ id: 'g', kind: 'gap', start: at(15, 23, 15), end: at(16, 8, 50) }, NOW, 'zh', tr))
      .toBe('sources.time.yesterday(t=23:15) – sources.time.today(t=08:50)');
  });
});

describe('buildHistoryRows', () => {
  it('folds the waiting items into one row right under the one being routed', () => {
    const h: HistoryEntry[] = [
      item({ id: 'w2', state: 'waiting', title: 'newer', arrived_at: at(16, 14, 18) }),
      item({ id: 'r', state: 'routing', title: 'routing', arrived_at: at(16, 14, 17) }),
      item({ id: 'w1', state: 'waiting', title: 'oldest waiting', arrived_at: at(16, 14, 16) }),
      item({ id: 'd', state: 'routed', arrived_at: at(16, 13, 55) }),
    ];
    const rows = buildHistoryRows(h);
    expect(rows.map((r) => r.kind === 'wait' ? `wait:${r.count}:${r.title}` : r.id)).toEqual(['r', 'wait:2:oldest waiting', 'd']);
  });
  it('with nothing being routed the waiting row is first', () => {
    const rows = buildHistoryRows([
      item({ id: 'd', state: 'routed', arrived_at: at(16, 13, 55) }),
      item({ id: 'w', state: 'waiting', arrived_at: at(16, 14, 0) }),
    ]);
    expect(rows[0].kind).toBe('wait');
  });
  it('held and dismissed items are quiet; gaps pass through', () => {
    const rows = buildHistoryRows([
      item({ id: 'h', state: 'held' }), item({ id: 'x', state: 'dismissed' }),
      { id: 'g', kind: 'gap', start: at(16, 1, 0), end: null }, item({ id: 'r', state: 'routed' }),
    ]);
    expect(rows.map((r) => (r.kind === 'item' ? `${r.id}:${r.quiet}` : r.kind))).toEqual(['h:true', 'x:true', 'gap', 'r:false']);
  });
});
