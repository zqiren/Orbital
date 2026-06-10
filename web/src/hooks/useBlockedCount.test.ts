// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const onMock = vi.fn();
const offMock = vi.fn();

vi.mock('./useWebSocket', () => ({
  useWebSocket: () => ({
    on: onMock,
    off: offMock,
    connectionState: 'connected',
    subscribe: vi.fn(),
  }),
}));

let apiFn = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiFn(...args),
  ApiError: class ApiError extends Error {},
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000/ws',
}));

import { useBlockedCount } from './useBlockedCount';
import type { BlockedSessionEntry } from '../types';

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useBlockedCount', () => {
  beforeEach(() => {
    onMock.mockClear();
    offMock.mockClear();
    apiFn = vi.fn();
  });

  it('fetches GET /api/v2/blocked on mount and returns initial count', async () => {
    const blocked: BlockedSessionEntry[] = [
      { project_id: 'p1', session_id: 's1' },
    ];
    apiFn.mockResolvedValueOnce({ blocked_count: 1, blocked_sessions: blocked });

    const { result } = renderHook(() => useBlockedCount());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.blockedCount).toBe(1);
    expect(result.current.blockedSessions).toEqual(blocked);
    expect(apiFn).toHaveBeenCalledWith('/api/v2/blocked');
  });

  it('starts with loading=true and transitions to false after fetch', async () => {
    let resolve!: (v: unknown) => void;
    apiFn.mockReturnValueOnce(new Promise((res) => { resolve = res; }));

    const { result } = renderHook(() => useBlockedCount());
    expect(result.current.loading).toBe(true);

    await act(async () => {
      resolve({ blocked_count: 0, blocked_sessions: [] });
    });

    await waitFor(() => expect(result.current.loading).toBe(false));
  });

  it('defaults to 0 / empty array when the fetch fails', async () => {
    apiFn.mockRejectedValueOnce(new Error('503'));

    const { result } = renderHook(() => useBlockedCount());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.blockedCount).toBe(0);
    expect(result.current.blockedSessions).toEqual([]);
  });

  it('subscribes to blocked-count-changed WS event on mount', async () => {
    apiFn.mockResolvedValueOnce({ blocked_count: 0, blocked_sessions: [] });

    renderHook(() => useBlockedCount());

    await waitFor(() => {
      const calls = onMock.mock.calls.map((c) => c[0]);
      expect(calls).toContain('blocked-count-changed');
    });
  });

  it('calls off() for blocked-count-changed on unmount', async () => {
    apiFn.mockResolvedValueOnce({ blocked_count: 0, blocked_sessions: [] });

    const { unmount } = renderHook(() => useBlockedCount());

    await waitFor(() => {
      const calls = onMock.mock.calls.map((c) => c[0]);
      expect(calls).toContain('blocked-count-changed');
    });

    unmount();

    const offCalls = offMock.mock.calls.map((c) => c[0]);
    expect(offCalls).toContain('blocked-count-changed');
  });

  it('updates blockedCount and blockedSessions when the WS event fires', async () => {
    apiFn.mockResolvedValueOnce({ blocked_count: 0, blocked_sessions: [] });

    const { result } = renderHook(() => useBlockedCount());
    await waitFor(() => expect(result.current.loading).toBe(false));

    // Grab the registered handler
    const handlerCall = onMock.mock.calls.find(
      (c) => c[0] === 'blocked-count-changed',
    );
    expect(handlerCall).toBeDefined();
    const handler = handlerCall![1] as (e: unknown) => void;

    const newBlocked: BlockedSessionEntry[] = [
      { project_id: 'p2', session_id: 's2' },
      { project_id: 'p3', session_id: 's3' },
    ];

    await act(async () => {
      handler({
        type: 'blocked-count-changed',
        blocked_count: 2,
        blocked_sessions: newBlocked,
      });
    });

    expect(result.current.blockedCount).toBe(2);
    expect(result.current.blockedSessions).toEqual(newBlocked);
  });

  it('ignores other WS event types (handler is a no-op for non-matching types)', async () => {
    apiFn.mockResolvedValueOnce({ blocked_count: 3, blocked_sessions: [] });

    const { result } = renderHook(() => useBlockedCount());
    await waitFor(() => expect(result.current.loading).toBe(false));

    const handlerCall = onMock.mock.calls.find(
      (c) => c[0] === 'blocked-count-changed',
    );
    const handler = handlerCall![1] as (e: unknown) => void;

    await act(async () => {
      // Fire an unrelated event type
      handler({ type: 'agent.status', project_id: 'p1', status: 'idle' });
    });

    // Count should remain unchanged
    expect(result.current.blockedCount).toBe(3);
  });
});
