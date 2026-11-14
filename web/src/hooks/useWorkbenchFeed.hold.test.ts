// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 169 B: a card held in 进行中 for a turn that never starts lets go on
 * its own after HOLD_SAFETY_MS — locally, with no feed fetch, so it works
 * even when the fetch fails. The timer dies with its hold (R3).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock, isRelayMode: false, BASE_URL: '' }));

import {
  HOLD_SAFETY_MS, __resetWorkbenchFeed, getFeedState, holdCard, refreshFeed, releaseCard,
  sessionKeyOf, settleReply,
} from './useWorkbenchFeed';
import type { WorkbenchFeedRow } from '../types';

const ROW: WorkbenchFeedRow = {
  project_id: 'p', session_id: 's', session_uuid: 's', name: 'Goals', agent: null,
  status: 'idle', worker_running: false, last_user_at: null, last_reply_at: null,
  checked_reply_at: null, ask: null, result_line: null, stop: null,
};
const KEY = sessionKeyOf('p', 's');
const isHeld = () => getFeedState().held.has(KEY);

beforeEach(() => {
  vi.useFakeTimers();
  apiMock.mockReset();
  __resetWorkbenchFeed();
});
afterEach(() => {
  __resetWorkbenchFeed();
  vi.useRealTimers();
});

describe('held card: local expiry (spec 169 B)', () => {
  it('a hold with no run seen is dropped at exactly HOLD_SAFETY_MS, with no feed fetch', () => {
    holdCard(ROW);
    expect(isHeld()).toBe(true);
    vi.advanceTimersByTime(HOLD_SAFETY_MS - 1);
    expect(isHeld()).toBe(true);
    // Listeners hear the drop (useSyncExternalStore re-renders on a new state).
    const before = getFeedState();
    vi.advanceTimersByTime(1);
    expect(isHeld()).toBe(false);
    expect(getFeedState()).not.toBe(before);
    expect(apiMock).not.toHaveBeenCalled();
  });

  it('is dropped on time even when the feed fetch fails', async () => {
    apiMock.mockRejectedValue(new Error('offline'));
    holdCard(ROW);
    await refreshFeed();
    expect(getFeedState().error).toBe(true);
    expect(isHeld()).toBe(true);
    vi.advanceTimersByTime(HOLD_SAFETY_MS);
    expect(isHeld()).toBe(false);
  });

  it('a hold whose run was seen is not dropped by the timer (the feed releases it when the run ends)', async () => {
    holdCard(ROW);
    apiMock.mockResolvedValue({ rows: [{ ...ROW, status: 'running' }] });
    await refreshFeed();
    expect(getFeedState().held.get(KEY)?.sawRunning).toBe(true);
    vi.advanceTimersByTime(HOLD_SAFETY_MS * 2);
    expect(isHeld()).toBe(true);
  });

  it('a released hold takes its timer with it: a later hold gets its own full window (R3)', () => {
    holdCard(ROW);
    vi.advanceTimersByTime(60_000);
    releaseCard('p', 's');
    expect(isHeld()).toBe(false);
    holdCard(ROW);
    // The first hold's deadline passes: the second hold stays.
    vi.advanceTimersByTime(HOLD_SAFETY_MS - 60_000);
    expect(isHeld()).toBe(true);
    vi.advanceTimersByTime(60_000);
    expect(isHeld()).toBe(false);
  });

  it('no timer is left behind once the feed releases the hold', async () => {
    holdCard(ROW);
    apiMock.mockResolvedValue({ rows: [{ ...ROW, status: 'running' }] });
    await refreshFeed();
    apiMock.mockResolvedValue({ rows: [ROW] });
    await refreshFeed();
    expect(isHeld()).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('settleReply: one step after a reply from a card (spec 169 B)', () => {
  it('a reply the daemon recorded (no turn) lets the hold go and refreshes', () => {
    apiMock.mockResolvedValue({ rows: [] });
    holdCard(ROW);
    settleReply('p', 's', { status: 'recorded' });
    expect(isHeld()).toBe(false);
    expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench/feed');
    expect(vi.getTimerCount()).toBe(0);
  });

  it('a failed send lets the hold go at once', () => {
    apiMock.mockResolvedValue({ rows: [] });
    holdCard(ROW);
    settleReply('p', 's', null);
    expect(isHeld()).toBe(false);
  });

  it('a reply that started a turn keeps the hold', () => {
    apiMock.mockResolvedValue({ rows: [] });
    holdCard(ROW);
    settleReply('p', 's', { status: 'started' });
    settleReply('p', 's', undefined);
    expect(isHeld()).toBe(true);
  });
});

describe('settleReply: a reply delivered into the project the session moved to (spec 180)', () => {
  const MOVED = sessionKeyOf('q', 's');

  it('the hold follows the session to the project the response names, under the normal release rule', async () => {
    apiMock.mockResolvedValue({ rows: [] });
    holdCard(ROW);
    settleReply('p', 's', { status: 'started', project_id: 'q' });
    expect(isHeld()).toBe(false);
    expect(getFeedState().held.get(MOVED)?.row.project_id).toBe('q');
    // settleReply's own refetch lands first.
    await vi.advanceTimersByTimeAsync(0);
    // The run in the new project is seen, then ends: the hold goes.
    apiMock.mockResolvedValue({ rows: [{ ...ROW, project_id: 'q', status: 'running' }] });
    await refreshFeed();
    expect(getFeedState().held.get(MOVED)?.sawRunning).toBe(true);
    apiMock.mockResolvedValue({ rows: [{ ...ROW, project_id: 'q' }] });
    await refreshFeed();
    expect(getFeedState().held.size).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('the moved hold keeps its original safety deadline', () => {
    apiMock.mockResolvedValue({ rows: [] });
    holdCard(ROW);
    vi.advanceTimersByTime(60_000);
    settleReply('p', 's', { status: 'started', project_id: 'q' });
    vi.advanceTimersByTime(HOLD_SAFETY_MS - 60_001);
    expect(getFeedState().held.has(MOVED)).toBe(true);
    vi.advanceTimersByTime(1);
    expect(getFeedState().held.has(MOVED)).toBe(false);
  });

  it('a response naming the same project changes nothing', () => {
    apiMock.mockResolvedValue({ rows: [] });
    holdCard(ROW);
    settleReply('p', 's', { status: 'started', project_id: 'p' });
    expect(isHeld()).toBe(true);
  });

  it('a recorded reply delivered elsewhere lets the hold go', () => {
    apiMock.mockResolvedValue({ rows: [] });
    holdCard(ROW);
    settleReply('p', 's', { status: 'recorded', project_id: 'q' });
    expect(getFeedState().held.size).toBe(0);
  });
});
