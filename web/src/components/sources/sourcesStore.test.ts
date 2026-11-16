// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { anyFailed, applySourcesEvent, applyDetail, EMPTY_SOURCES, totalWaiting, type SourcesState } from './sourcesStore';
import { detail, item, summary } from './testFixtures';

function withList(...s: ReturnType<typeof summary>[]): SourcesState {
  return { ...EMPTY_SOURCES, loaded: true, list: s, waiting: totalWaiting(s) };
}

describe('sources store reducer', () => {
  it('sources.changed upserts a known source in place and recounts waiting', () => {
    const a = summary({ id: 'a', waiting: 1 });
    const b = summary({ id: 'b', waiting: 2 });
    const s = applySourcesEvent(withList(a, b), { type: 'sources.changed', source: { ...b, waiting: 5 } });
    expect(s.list.map((x) => x.id)).toEqual(['a', 'b']);
    expect(s.list[1].waiting).toBe(5);
    expect(s.waiting).toBe(6);
  });

  it('sources.changed puts a new source first (newest first)', () => {
    const s = applySourcesEvent(withList(summary({ id: 'a' })), { type: 'sources.changed', source: summary({ id: 'n' }) });
    expect(s.list.map((x) => x.id)).toEqual(['n', 'a']);
  });

  it('the entry button is marked while any source is failed, paused ones too', () => {
    expect(anyFailed([summary({ id: 'a' }), summary({ id: 'b', status: 'unavailable' })])).toBe(false);
    expect(anyFailed([summary({ id: 'a' }), summary({ id: 'b', on: false, status: 'failed' })])).toBe(true);
  });

  it('a paused source does not count toward the entry button', () => {
    const s = withList(summary({ id: 'a', waiting: 3, on: false }), summary({ id: 'b', waiting: 2 }));
    expect(s.waiting).toBe(2);
  });

  it('sources.changed refreshes the summary fields of a loaded detail, keeping its history', () => {
    const d = detail({ id: 'a', history: [item({ id: 'x' })] });
    let s = applyDetail(withList(summary({ id: 'a' })), d);
    s = applySourcesEvent(s, { type: 'sources.changed', source: summary({ id: 'a', on: false, held: 2 }) });
    expect(s.details.a.on).toBe(false);
    expect(s.details.a.held).toBe(2);
    expect(s.details.a.history.map((h) => h.id)).toEqual(['x']);
  });

  it('sources.removed drops the source and its detail', () => {
    let s = applyDetail(withList(summary({ id: 'a' }), summary({ id: 'b' })), detail({ id: 'a' }));
    s = applySourcesEvent(s, { type: 'sources.removed', source_id: 'a' });
    expect(s.list.map((x) => x.id)).toEqual(['b']);
    expect(s.details.a).toBeUndefined();
  });

  it('sources.entry replaces an entry by id and adds a new one newest first', () => {
    const old = item({ id: 'x', state: 'waiting', arrived_at: '2026-11-16T07:00:00Z' });
    let s = applyDetail(withList(summary({ id: 'a' })), detail({ id: 'a', history: [old] }));
    s = applySourcesEvent(s, { type: 'sources.entry', source_id: 'a', entry: { ...old, state: 'routed' } });
    expect(s.details.a.history).toHaveLength(1);
    expect((s.details.a.history[0] as { state: string }).state).toBe('routed');
    const nu = item({ id: 'y', arrived_at: '2026-11-16T08:00:00Z' });
    s = applySourcesEvent(s, { type: 'sources.entry', source_id: 'a', entry: nu });
    expect(s.details.a.history.map((h) => h.id)).toEqual(['y', 'x']);
  });

  it('a gap sorts by its start', () => {
    const it1 = item({ id: 'x', arrived_at: '2026-11-16T07:00:00Z' });
    let s = applyDetail(withList(summary({ id: 'a' })), detail({ id: 'a', history: [it1] }));
    s = applySourcesEvent(s, {
      type: 'sources.entry', source_id: 'a',
      entry: { id: 'g', kind: 'gap', start: '2026-11-16T07:30:00Z', end: null },
    });
    expect(s.details.a.history.map((h) => h.id)).toEqual(['g', 'x']);
  });

  it('sources.entry for a detail not loaded changes nothing', () => {
    const s0 = withList(summary({ id: 'a' }));
    const s = applySourcesEvent(s0, { type: 'sources.entry', source_id: 'a', entry: item({ id: 'z' }) });
    expect(s).toBe(s0);
  });

  it('applyDetail also refreshes the row in the list', () => {
    const s = applyDetail(withList(summary({ id: 'a', on: true })), detail({ id: 'a', on: false }));
    expect(s.list[0].on).toBe(false);
  });
});
