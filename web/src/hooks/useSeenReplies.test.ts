// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 108 — the per-device "which reply have I shown" map:
 * localStorage['orbital:seenReplies:<pid>'] → { sessionKey: ISO }.
 *   - First visit seeds, not floods: the first non-empty list for a project
 *     writes every row's current reply as seen.
 *   - markSeen persists and survives a remount.
 *   - Every write prunes to the keys in the current list.
 *   - Storage failure is swallowed.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useSeenReplies, seenRepliesStorageKey } from './useSeenReplies';
import type { SessionListEntry } from '../types';

function entry(uuid: string, replyAt: string | null): SessionListEntry {
  return {
    session_id: uuid,
    session_uuid: uuid,
    status: 'idle',
    last_terminal_event: null,
    last_activity_at: replyAt,
    last_user_at: '2026-10-01T00:00:00Z',
    last_reply_at: replyAt,
  };
}

const KEY = seenRepliesStorageKey('proj-1');

function stored(): Record<string, string> | null {
  const raw = localStorage.getItem(KEY);
  return raw === null ? null : (JSON.parse(raw) as Record<string, string>);
}

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe('useSeenReplies — seeding', () => {
  it('writes nothing while the list is empty (not loaded yet)', () => {
    renderHook(() => useSeenReplies('proj-1', []));
    expect(stored()).toBeNull();
  });

  it('writes nothing without a project', () => {
    renderHook(() => useSeenReplies(null, [entry('a', '2026-10-01T00:00:10Z')]));
    expect(localStorage.length).toBe(0);
  });

  it('the first non-empty list seeds every current reply as seen (no badge storm)', () => {
    const list = [
      entry('a', '2026-10-01T00:00:10Z'),
      entry('b', '2026-10-01T00:00:20Z'),
      entry('c', null),
    ];
    const { result } = renderHook(() => useSeenReplies('proj-1', list));
    expect(stored()).toEqual({ a: '2026-10-01T00:00:10Z', b: '2026-10-01T00:00:20Z' });
    expect(result.current.seenAt('a')).toBe('2026-10-01T00:00:10Z');
    expect(result.current.seenAt('c')).toBeUndefined();
  });

  it('a project whose key exists (even empty) is NOT re-seeded: later replies count', () => {
    localStorage.setItem(KEY, '{}');
    const { result } = renderHook(() =>
      useSeenReplies('proj-1', [entry('a', '2026-10-01T00:00:10Z')]),
    );
    expect(stored()).toEqual({});
    expect(result.current.seenAt('a')).toBeUndefined();
  });

  it('a list that becomes non-empty later seeds then, once', () => {
    const { result, rerender } = renderHook(
      ({ list }: { list: SessionListEntry[] }) => useSeenReplies('proj-1', list),
      { initialProps: { list: [] as SessionListEntry[] } },
    );
    expect(stored()).toBeNull();
    rerender({ list: [entry('a', '2026-10-01T00:00:10Z')] });
    expect(stored()).toEqual({ a: '2026-10-01T00:00:10Z' });
    // A newer reply on the next refetch is NOT swallowed by a second seed.
    rerender({ list: [entry('a', '2026-10-01T00:00:30Z')] });
    expect(result.current.seenAt('a')).toBe('2026-10-01T00:00:10Z');
  });

  it('switching projects reads that project\'s map', () => {
    localStorage.setItem(seenRepliesStorageKey('proj-2'), JSON.stringify({ z: '2026-10-01T00:00:01Z' }));
    const { result, rerender } = renderHook(
      ({ pid }: { pid: string }) => useSeenReplies(pid, [entry('z', '2026-10-01T00:00:01Z')]),
      { initialProps: { pid: 'proj-1' } },
    );
    expect(result.current.seenAt('z')).toBe('2026-10-01T00:00:01Z'); // seeded for proj-1
    rerender({ pid: 'proj-2' });
    expect(result.current.seenAt('z')).toBe('2026-10-01T00:00:01Z');
    expect(stored()).toEqual({ z: '2026-10-01T00:00:01Z' });
  });
});

describe('useSeenReplies — markSeen and prune', () => {
  it('markSeen persists and survives a remount', () => {
    localStorage.setItem(KEY, '{}');
    const list = [entry('a', '2026-10-01T00:00:10Z')];
    const first = renderHook(() => useSeenReplies('proj-1', list));
    act(() => first.result.current.markSeen('a', '2026-10-01T00:00:10Z'));
    expect(first.result.current.seenAt('a')).toBe('2026-10-01T00:00:10Z');
    first.unmount();

    const second = renderHook(() => useSeenReplies('proj-1', list));
    expect(second.result.current.seenAt('a')).toBe('2026-10-01T00:00:10Z');
  });

  it('markSeen with the already-stored value is a no-op (no write, same identity)', () => {
    localStorage.setItem(KEY, JSON.stringify({ a: '2026-10-01T00:00:10Z' }));
    const list = [entry('a', '2026-10-01T00:00:10Z')];
    const { result } = renderHook(() => useSeenReplies('proj-1', list));
    const before = result.current.seenAt;
    const spy = vi.spyOn(Storage.prototype, 'setItem');
    act(() => result.current.markSeen('a', '2026-10-01T00:00:10Z'));
    expect(spy).not.toHaveBeenCalled();
    expect(result.current.seenAt).toBe(before);
  });

  it('every write prunes keys that are no longer in the list', () => {
    localStorage.setItem(
      KEY,
      JSON.stringify({ a: '2026-10-01T00:00:10Z', gone: '2026-09-01T00:00:00Z' }),
    );
    const list = [entry('a', '2026-10-01T00:00:10Z'), entry('b', '2026-10-01T00:00:20Z')];
    const { result } = renderHook(() => useSeenReplies('proj-1', list));
    act(() => result.current.markSeen('b', '2026-10-01T00:00:20Z'));
    expect(stored()).toEqual({ a: '2026-10-01T00:00:10Z', b: '2026-10-01T00:00:20Z' });
    expect(result.current.seenAt('gone')).toBeUndefined();
  });

  it('the seed itself only records rows in the list (nothing to prune, but never extra)', () => {
    const { result } = renderHook(() =>
      useSeenReplies('proj-1', [entry('a', '2026-10-01T00:00:10Z')]),
    );
    expect(Object.keys(stored() ?? {})).toEqual(['a']);
    expect(result.current.seenAt('a')).toBe('2026-10-01T00:00:10Z');
  });

  it('swallows a storage that throws on read and on write', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('locked');
    });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('locked');
    });
    const list = [entry('a', '2026-10-01T00:00:10Z')];
    const { result } = renderHook(() => useSeenReplies('proj-1', list));
    expect(() =>
      act(() => result.current.markSeen('a', '2026-10-01T00:00:10Z')),
    ).not.toThrow();
    // The in-memory map still works for this visit.
    expect(result.current.seenAt('a')).toBe('2026-10-01T00:00:10Z');
  });

  it('ignores a corrupt stored value and starts fresh', () => {
    localStorage.setItem(KEY, '[not json');
    const { result } = renderHook(() =>
      useSeenReplies('proj-1', [entry('a', '2026-10-01T00:00:10Z')]),
    );
    // Corrupt key counts as present: no seed. Marking works and overwrites it.
    act(() => result.current.markSeen('a', '2026-10-01T00:00:10Z'));
    expect(stored()).toEqual({ a: '2026-10-01T00:00:10Z' });
  });
});
