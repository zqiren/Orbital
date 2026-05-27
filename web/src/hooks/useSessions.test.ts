// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

// ---------------------------------------------------------------------------
// Mocks — must be declared before the module under test is imported
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
  // Proxy through to the current apiFn so tests can swap the impl.
  api: (...args: unknown[]) => apiFn(...args),
  ApiError: class ApiError extends Error {},
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000/ws',
}));

import { useSessions } from './useSessions';
import type { SessionListEntry } from '../types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSession(overrides: Partial<SessionListEntry> = {}): SessionListEntry {
  return {
    session_id: 'sess-1',
    status: 'idle',
    session_uuid: null,
    last_terminal_event: null,
    last_activity_at: null,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useSessions', () => {
  beforeEach(() => {
    onMock.mockClear();
    offMock.mockClear();
    apiFn = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('fetches sessions on mount and returns them', async () => {
    const sessions: SessionListEntry[] = [
      makeSession({ session_id: 'sess-a', last_activity_at: '2026-05-24T10:00:00Z' }),
      makeSession({ session_id: 'sess-b', status: 'running' }),
    ];
    apiFn.mockResolvedValueOnce(sessions);

    const { result } = renderHook(() => useSessions('proj-1'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.sessions).toEqual(sessions);
    expect(result.current.error).toBeNull();
    expect(apiFn).toHaveBeenCalledWith('/api/v2/projects/proj-1/sessions');
  });

  it('unwraps the REAL wrapped response shape { project_id, sessions: [...] } (regression for c89a6bc)', async () => {
    // The actual endpoint returns the array WRAPPED in an object, not a bare
    // array. If the hook stops unwrapping resp.sessions, `sessions` becomes a
    // plain object and SessionSidebar's `sessions.filter()` throws — the
    // Phase-1B blocker that blanked the whole app. Every other test here mocks
    // a bare array (which the defensive guard also accepts), so THIS is the
    // test that actually pins the real contract.
    const sessions: SessionListEntry[] = [
      makeSession({ session_id: 's1', status: 'running' }),
      makeSession({ session_id: 's2' }),
    ];
    apiFn.mockResolvedValueOnce({ project_id: 'proj-wrapped', sessions });

    const { result } = renderHook(() => useSessions('proj-wrapped'));

    await waitFor(() => expect(result.current.loading).toBe(false));

    // Must be the INNER array, not the wrapper object.
    expect(Array.isArray(result.current.sessions)).toBe(true);
    expect(result.current.sessions).toEqual(sessions);
    expect(result.current.sessions[0].session_id).toBe('s1');
    expect(result.current.error).toBeNull();
  });

  it('surfaces last_activity_at on returned session entries', async () => {
    const sessions: SessionListEntry[] = [
      makeSession({ session_id: 's1', last_activity_at: '2026-01-01T00:00:00Z' }),
    ];
    apiFn.mockResolvedValueOnce(sessions);

    const { result } = renderHook(() => useSessions('proj-2'));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.sessions[0].last_activity_at).toBe('2026-01-01T00:00:00Z');
  });

  it('sets error and returns empty array on fetch failure', async () => {
    apiFn.mockRejectedValueOnce(new Error('network error'));

    const { result } = renderHook(() => useSessions('proj-fail'));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.sessions).toEqual([]);
    expect(result.current.error).toBe('network error');
  });

  it('returns empty sessions and no error when projectId is null', async () => {
    const { result } = renderHook(() => useSessions(null));

    // Should not fetch
    expect(apiFn).not.toHaveBeenCalled();
    expect(result.current.sessions).toEqual([]);
    expect(result.current.error).toBeNull();
  });

  it('subscribes to agent.status WS event on mount', async () => {
    apiFn.mockResolvedValue([]);

    renderHook(() => useSessions('proj-ws'));

    await waitFor(() => {
      // on() should have been called with 'agent.status'
      const calls = onMock.mock.calls.map((c) => c[0]);
      expect(calls).toContain('agent.status');
    });
  });

  it('calls off() for agent.status on unmount (cleanup)', async () => {
    apiFn.mockResolvedValue([]);

    const { unmount } = renderHook(() => useSessions('proj-ws-cleanup'));

    await waitFor(() => {
      const calls = onMock.mock.calls.map((c) => c[0]);
      expect(calls).toContain('agent.status');
    });

    unmount();

    const offCalls = offMock.mock.calls.map((c) => c[0]);
    expect(offCalls).toContain('agent.status');
  });

  it('refreshes when agent.status event fires for the same project', async () => {
    const initial: SessionListEntry[] = [makeSession({ session_id: 's1' })];
    const updated: SessionListEntry[] = [
      makeSession({ session_id: 's1' }),
      makeSession({ session_id: 's2', status: 'idle' }),
    ];
    apiFn.mockResolvedValueOnce(initial).mockResolvedValueOnce(updated);

    const { result } = renderHook(() => useSessions('proj-refresh'));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.sessions).toEqual(initial);

    // Simulate the WS handler firing: grab the registered handler and invoke it.
    const handlerCall = onMock.mock.calls.find((c) => c[0] === 'agent.status');
    expect(handlerCall).toBeDefined();
    const handler = handlerCall![1] as (e: unknown) => void;

    await act(async () => {
      handler({ type: 'agent.status', project_id: 'proj-refresh', status: 'idle' });
    });

    await waitFor(() => {
      expect(result.current.sessions).toEqual(updated);
    });
  });

  it('does NOT refresh when agent.status event is for a different project', async () => {
    const initial: SessionListEntry[] = [makeSession({ session_id: 's1' })];
    apiFn.mockResolvedValueOnce(initial);

    const { result } = renderHook(() => useSessions('proj-mine'));

    await waitFor(() => expect(result.current.loading).toBe(false));

    const callCountBeforeEvent = apiFn.mock.calls.length;

    const handlerCall = onMock.mock.calls.find((c) => c[0] === 'agent.status');
    const handler = handlerCall![1] as (e: unknown) => void;

    await act(async () => {
      handler({ type: 'agent.status', project_id: 'proj-OTHER', status: 'idle' });
    });

    // No additional fetch
    expect(apiFn.mock.calls.length).toBe(callCountBeforeEvent);
  });

  it('surfaces the name field on returned session entries', async () => {
    const sessions: SessionListEntry[] = [
      makeSession({ session_id: 's1', name: 'My Login Flow' }),
    ];
    apiFn.mockResolvedValueOnce(sessions);

    const { result } = renderHook(() => useSessions('proj-name'));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.sessions[0].name).toBe('My Login Flow');
  });
});

describe('useSessions — rename', () => {
  beforeEach(() => {
    onMock.mockClear();
    offMock.mockClear();
    apiFn = vi.fn();
  });
  afterEach(() => vi.restoreAllMocks());

  it('PATCHes the rename endpoint and optimistically updates the list', async () => {
    const sessions: SessionListEntry[] = [makeSession({ session_id: 's1', name: 'old' })];
    apiFn.mockResolvedValueOnce(sessions); // initial fetch
    apiFn.mockResolvedValueOnce(undefined); // PATCH

    const { result } = renderHook(() => useSessions('proj-rn'));
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await result.current.renameSession('s1', 'new name');
    });

    // PATCH was issued with the right path + body.
    const patchCall = apiFn.mock.calls.find(
      (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH',
    );
    expect(patchCall).toBeDefined();
    expect(patchCall![0]).toBe('/api/v2/agents/proj-rn/sessions/s1');
    expect(JSON.parse((patchCall![1] as RequestInit).body as string)).toEqual({ name: 'new name' });

    // Optimistic local update.
    expect(result.current.sessions[0].name).toBe('new name');
  });

  it('rejects an empty rename without calling the API', async () => {
    apiFn.mockResolvedValueOnce([makeSession({ session_id: 's1' })]);
    const { result } = renderHook(() => useSessions('proj-rn2'));
    await waitFor(() => expect(result.current.loading).toBe(false));

    const callsBefore = apiFn.mock.calls.length;
    await expect(
      act(async () => {
        await result.current.renameSession('s1', '   ');
      }),
    ).rejects.toThrow();
    // No PATCH issued.
    expect(apiFn.mock.calls.length).toBe(callsBefore);
  });
});

describe('useSessions — delete', () => {
  beforeEach(() => {
    onMock.mockClear();
    offMock.mockClear();
    apiFn = vi.fn();
  });
  afterEach(() => vi.restoreAllMocks());

  it('DELETEs the endpoint and removes the row from the list', async () => {
    const sessions: SessionListEntry[] = [
      makeSession({ session_id: 's1' }),
      makeSession({ session_id: 's2' }),
    ];
    apiFn.mockResolvedValueOnce(sessions); // initial fetch
    apiFn.mockResolvedValueOnce(undefined); // DELETE

    const { result } = renderHook(() => useSessions('proj-del'));
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await result.current.deleteSession('s1');
    });

    const delCall = apiFn.mock.calls.find(
      (c) => (c[1] as RequestInit | undefined)?.method === 'DELETE',
    );
    expect(delCall).toBeDefined();
    expect(delCall![0]).toBe('/api/v2/agents/proj-del/sessions/s1');

    // Row removed locally.
    expect(result.current.sessions.map((s) => s.session_id)).toEqual(['s2']);
  });

  it('does NOT remove the row when the DELETE fails (e.g. 409)', async () => {
    const sessions: SessionListEntry[] = [makeSession({ session_id: 's1' })];
    apiFn.mockResolvedValueOnce(sessions); // initial fetch
    apiFn.mockRejectedValueOnce(new Error('409 running')); // DELETE fails

    const { result } = renderHook(() => useSessions('proj-del-fail'));
    await waitFor(() => expect(result.current.loading).toBe(false));

    await expect(
      act(async () => {
        await result.current.deleteSession('s1');
      }),
    ).rejects.toThrow();

    // Row still present.
    expect(result.current.sessions.map((s) => s.session_id)).toEqual(['s1']);
  });
});
