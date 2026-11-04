// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 154 §5.1 — the checked state lives in the daemon. `useSeenReplies`
 * keeps spec 108's signature but reads each row's `checked_reply_at` and
 * writes through `POST /api/v2/workbench/checked`, with an optimistic mark in
 * between. The retired per-device localStorage map is neither read nor
 * written (left in place for a downgrade).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';

let apiFn = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiFn(...args),
  isRelayMode: false,
}));

import {
  __resetCheckedMarks, effectiveChecked, markChecked, seenRepliesStorageKey, useSeenReplies,
} from './useSeenReplies';
import type { SessionListEntry } from '../types';

function entry(uuid: string, replyAt: string | null, checked: string | null = null): SessionListEntry {
  return {
    session_id: uuid,
    session_uuid: uuid,
    status: 'idle',
    last_terminal_event: null,
    last_activity_at: replyAt,
    last_user_at: '2026-10-01T00:00:00Z',
    last_reply_at: replyAt,
    checked_reply_at: checked,
  };
}

beforeEach(() => {
  localStorage.clear();
  __resetCheckedMarks();
  apiFn = vi.fn().mockResolvedValue({ status: 'ok' });
});

describe('useSeenReplies — daemon-backed', () => {
  it('seenAt reads the row checked_reply_at', () => {
    const list = [entry('a', '2026-10-01T00:00:10Z', '2026-10-01T00:00:10Z'), entry('b', '2026-10-01T00:00:20Z')];
    const { result } = renderHook(() => useSeenReplies('proj-1', list));
    expect(result.current.seenAt('a')).toBe('2026-10-01T00:00:10Z');
    expect(result.current.seenAt('b')).toBeUndefined();
  });

  it('markSeen POSTs to the daemon and marks at once', async () => {
    const list = [entry('a', '2026-10-01T00:00:10Z')];
    const { result } = renderHook(() => useSeenReplies('proj-1', list));
    act(() => result.current.markSeen('a', '2026-10-01T00:00:10Z'));
    expect(result.current.seenAt('a')).toBe('2026-10-01T00:00:10Z');
    expect(apiFn).toHaveBeenCalledWith('/api/v2/workbench/checked', {
      method: 'POST',
      body: JSON.stringify({ project_id: 'proj-1', session_id: 'a', reply_at: '2026-10-01T00:00:10Z' }),
    });
  });

  it('never reads or writes the retired localStorage map', () => {
    localStorage.setItem(seenRepliesStorageKey('proj-1'), JSON.stringify({ a: '2026-10-01T00:00:10Z' }));
    const list = [entry('a', '2026-10-01T00:00:10Z')];
    const { result } = renderHook(() => useSeenReplies('proj-1', list));
    expect(result.current.seenAt('a')).toBeUndefined();
    act(() => result.current.markSeen('a', '2026-10-01T00:00:10Z'));
    // Left exactly as it was, for a downgrade.
    expect(JSON.parse(localStorage.getItem(seenRepliesStorageKey('proj-1'))!)).toEqual({ a: '2026-10-01T00:00:10Z' });
    expect(localStorage.length).toBe(1);
  });

  it('a failed POST drops the optimistic mark', async () => {
    apiFn = vi.fn().mockRejectedValue(new Error('offline'));
    const ok = await markChecked('proj-1', 'a', '2026-10-01T00:00:10Z');
    expect(ok).toBe(false);
    expect(effectiveChecked('proj-1', 'a', null)).toBeNull();
  });

  it('the effective time is the later of the daemon and the mark', async () => {
    await markChecked('proj-1', 'a', '2026-10-01T00:00:20Z');
    expect(effectiveChecked('proj-1', 'a', '2026-10-01T00:00:10Z')).toBe('2026-10-01T00:00:20Z');
    expect(effectiveChecked('proj-1', 'a', '2026-10-01T00:00:30Z')).toBe('2026-10-01T00:00:30Z');
    await waitFor(() => expect(apiFn).toHaveBeenCalledTimes(1));
  });
});
