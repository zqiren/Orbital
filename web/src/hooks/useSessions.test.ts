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

  // Spec 081 — the three pending events are the only signal that a queued
  // session's row appeared, flipped or vanished; none of them emits an
  // agent.status.
  const PENDING_EVENTS = [
    'chat.pending_enqueued',
    'chat.pending_dispatched',
    'chat.pending_cancelled',
  ] as const;

  for (const type of PENDING_EVENTS) {
    it(`subscribes to ${type} and refreshes on it for the same project`, async () => {
      const initial: SessionListEntry[] = [makeSession({ session_id: 's1' })];
      const updated: SessionListEntry[] = [
        makeSession({ session_id: 's1' }),
        makeSession({ session_id: 's2', status: 'queued' }),
      ];
      apiFn.mockResolvedValueOnce(initial).mockResolvedValueOnce(updated);

      const { result } = renderHook(() => useSessions('proj-pending'));
      await waitFor(() => expect(result.current.loading).toBe(false));
      expect(result.current.sessions).toEqual(initial);

      const handlerCall = onMock.mock.calls.find((c) => c[0] === type);
      expect(handlerCall).toBeDefined();
      const handler = handlerCall![1] as (e: unknown) => void;

      await act(async () => {
        handler({ type, project_id: 'proj-pending', session_id: 's2', nonce: 'n1' });
      });

      await waitFor(() => {
        expect(result.current.sessions).toEqual(updated);
      });
      expect(result.current.sessions[1].status).toBe('queued');
    });

    it(`ignores ${type} for a different project`, async () => {
      apiFn.mockResolvedValueOnce([makeSession({ session_id: 's1' })]);

      const { result } = renderHook(() => useSessions('proj-mine'));
      await waitFor(() => expect(result.current.loading).toBe(false));
      const before = apiFn.mock.calls.length;

      const handler = onMock.mock.calls.find((c) => c[0] === type)![1] as (e: unknown) => void;
      await act(async () => {
        handler({ type, project_id: 'proj-OTHER', session_id: 's2', nonce: 'n1' });
      });

      expect(apiFn.mock.calls.length).toBe(before);
    });

    it(`calls off() for ${type} on unmount`, async () => {
      apiFn.mockResolvedValue([]);

      const { unmount } = renderHook(() => useSessions('proj-pending-cleanup'));
      await waitFor(() => {
        expect(onMock.mock.calls.map((c) => c[0])).toContain(type);
      });

      unmount();

      expect(offMock.mock.calls.map((c) => c[0])).toContain(type);
    });
  }

  // Spec 106 / 102 §4.2 step 8 — a pinned dispatch runs zero management
  // turns, so no agent.status ever fires for it; the worker lifecycle events
  // are the only signal that its row appeared or its worker_running flipped.
  const WORKER_EVENTS = [
    'sub_agent.dispatched',
    'sub_agent.started',
    'sub_agent.completed',
    'sub_agent.error',
    'sub_agent.failed',
    'sub_agent.stopped',
    'sub_agent.turn_interrupted',
  ] as const;

  for (const type of WORKER_EVENTS) {
    it(`subscribes to ${type} and refreshes on it for the same project`, async () => {
      const initial: SessionListEntry[] = [makeSession({ session_id: 's1' })];
      const updated: SessionListEntry[] = [
        makeSession({ session_id: 's1' }),
        makeSession({ session_id: 's2', status: 'idle', worker_running: true }),
      ];
      apiFn.mockResolvedValueOnce(initial).mockResolvedValueOnce(updated);

      const { result } = renderHook(() => useSessions('proj-worker'));
      await waitFor(() => expect(result.current.loading).toBe(false));
      expect(result.current.sessions).toEqual(initial);

      const handlerCall = onMock.mock.calls.find((c) => c[0] === type);
      expect(handlerCall).toBeDefined();
      const handler = handlerCall![1] as (e: unknown) => void;

      await act(async () => {
        handler({ type, project_id: 'proj-worker', session_id: 's2', handle: 'claude-code' });
      });

      await waitFor(() => {
        expect(result.current.sessions).toEqual(updated);
      });
      expect(result.current.sessions[1].worker_running).toBe(true);
    });

    it(`ignores ${type} for a different project`, async () => {
      apiFn.mockResolvedValueOnce([makeSession({ session_id: 's1' })]);

      const { result } = renderHook(() => useSessions('proj-mine'));
      await waitFor(() => expect(result.current.loading).toBe(false));
      const before = apiFn.mock.calls.length;

      const handler = onMock.mock.calls.find((c) => c[0] === type)![1] as (e: unknown) => void;
      await act(async () => {
        handler({ type, project_id: 'proj-OTHER', session_id: 's2', handle: 'claude-code' });
      });

      expect(apiFn.mock.calls.length).toBe(before);
    });

    it(`calls off() for ${type} on unmount`, async () => {
      apiFn.mockResolvedValue([]);

      const { unmount } = renderHook(() => useSessions('proj-worker-cleanup'));
      await waitFor(() => {
        expect(onMock.mock.calls.map((c) => c[0])).toContain(type);
      });

      unmount();

      expect(offMock.mock.calls.map((c) => c[0])).toContain(type);
    });
  }

  it('does not refetch on chat.sub_agent_message (one per worker text chunk)', async () => {
    apiFn.mockResolvedValue([]);
    renderHook(() => useSessions('proj-chunks'));
    await waitFor(() => expect(onMock.mock.calls.length).toBeGreaterThan(0));
    expect(onMock.mock.calls.map((c) => c[0])).not.toContain('chat.sub_agent_message');
  });

  it('coalesces a burst of WS-triggered refreshes into one in flight + one trailing', async () => {
    const initial: SessionListEntry[] = [makeSession({ session_id: 's1' })];
    const final: SessionListEntry[] = [
      makeSession({ session_id: 's1' }),
      makeSession({ session_id: 's2', name: 'write the essay' }),
    ];
    const pending: Array<(v: SessionListEntry[]) => void> = [];
    apiFn = vi.fn(
      () =>
        new Promise<SessionListEntry[]>((resolve) => {
          pending.push(resolve);
        }),
    );

    const { result } = renderHook(() => useSessions('proj-burst'));
    await waitFor(() => expect(pending.length).toBe(1));
    await act(async () => {
      pending[0](initial);
    });
    await waitFor(() => expect(result.current.sessions).toEqual(initial));
    const baseline = apiFn.mock.calls.length; // the initial load

    const handlerFor = (type: string) =>
      onMock.mock.calls.find((c) => c[0] === type)![1] as (e: unknown) => void;

    // dispatched → one request goes out and stays pending.
    await act(async () => {
      handlerFor('sub_agent.dispatched')({ type: 'sub_agent.dispatched', project_id: 'proj-burst', session_id: 's2' });
    });
    expect(apiFn.mock.calls.length - baseline).toBe(1);

    // started + completed land while it is still in flight → no new request.
    await act(async () => {
      handlerFor('sub_agent.started')({ type: 'sub_agent.started', project_id: 'proj-burst', session_id: 's2' });
      handlerFor('sub_agent.completed')({ type: 'sub_agent.completed', project_id: 'proj-burst', session_id: 's2' });
    });
    expect(apiFn.mock.calls.length - baseline).toBe(1);

    // The first settles with a stale snapshot; exactly one trailing refetch follows.
    await act(async () => {
      pending[1](initial);
    });
    await waitFor(() => expect(apiFn.mock.calls.length - baseline).toBe(2));
    await act(async () => {
      pending[2](final);
    });
    await waitFor(() => expect(result.current.sessions).toEqual(final));

    // Settled: no further requests.
    await act(async () => {
      await Promise.resolve();
    });
    expect(apiFn.mock.calls.length - baseline).toBe(2);
  });

  it('the explicit refresh() is not coalesced (rename/pin revert paths need an immediate fetch)', async () => {
    const pending: Array<(v: SessionListEntry[]) => void> = [];
    apiFn = vi.fn(
      () =>
        new Promise<SessionListEntry[]>((resolve) => {
          pending.push(resolve);
        }),
    );
    const { result } = renderHook(() => useSessions('proj-explicit'));
    await waitFor(() => expect(pending.length).toBe(1));
    await act(async () => {
      pending[0]([]);
    });
    await waitFor(() => expect(result.current.loading).toBe(false));
    const baseline = apiFn.mock.calls.length;

    const handler = onMock.mock.calls.find((c) => c[0] === 'sub_agent.dispatched')![1] as (e: unknown) => void;
    await act(async () => {
      handler({ type: 'sub_agent.dispatched', project_id: 'proj-explicit', session_id: 's2' });
    });
    expect(apiFn.mock.calls.length - baseline).toBe(1);

    let explicit: Promise<SessionListEntry[]> | undefined;
    await act(async () => {
      explicit = result.current.refresh();
    });
    // The explicit call went out immediately despite the in-flight WS refetch.
    expect(apiFn.mock.calls.length - baseline).toBe(2);
    await act(async () => {
      pending[1]([]);
      pending[2]([makeSession({ session_id: 'fresh' })]);
    });
    await expect(explicit!).resolves.toEqual([makeSession({ session_id: 'fresh' })]);
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

describe('useSessions — pinAgent (spec 074)', () => {
  beforeEach(() => {
    onMock.mockClear();
    offMock.mockClear();
    apiFn = vi.fn();
  });
  afterEach(() => vi.restoreAllMocks());

  it('PATCHes pinned_target with the slug and optimistically updates', async () => {
    const sessions: SessionListEntry[] = [makeSession({ session_id: 's1' })];
    apiFn.mockResolvedValueOnce(sessions); // initial fetch
    apiFn.mockResolvedValueOnce(undefined); // PATCH

    const { result } = renderHook(() => useSessions('proj-pin'));
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await result.current.pinAgent('s1', 'codex');
    });

    const patchCall = apiFn.mock.calls.find(
      (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH',
    );
    expect(patchCall).toBeDefined();
    expect(patchCall![0]).toBe('/api/v2/agents/proj-pin/sessions/s1');
    expect(JSON.parse((patchCall![1] as RequestInit).body as string)).toEqual({
      pinned_target: 'codex',
    });
    expect(result.current.sessions[0].pinned_target).toBe('codex');
  });

  it('unpin sends an EXPLICIT null (absence means untouched on the backend)', async () => {
    const sessions: SessionListEntry[] = [
      makeSession({ session_id: 's1', pinned_target: 'codex' }),
    ];
    apiFn.mockResolvedValueOnce(sessions); // initial fetch
    apiFn.mockResolvedValueOnce(undefined); // PATCH

    const { result } = renderHook(() => useSessions('proj-unpin'));
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await result.current.pinAgent('s1', null);
    });

    const patchCall = apiFn.mock.calls.find(
      (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH',
    );
    expect(patchCall).toBeDefined();
    // The key must be PRESENT with a null value — JSON.stringify would drop
    // undefined, and the backend treats absence as "leave the pin alone".
    expect((patchCall![1] as RequestInit).body as string).toContain('"pinned_target":null');
    expect(result.current.sessions[0].pinned_target).toBeNull();
  });

  it('refetches authoritative state when the PATCH fails', async () => {
    const sessions: SessionListEntry[] = [makeSession({ session_id: 's1' })];
    apiFn.mockResolvedValueOnce(sessions); // initial fetch
    apiFn.mockRejectedValueOnce(new Error('422 unknown slug')); // PATCH fails
    apiFn.mockResolvedValueOnce(sessions); // revert refetch

    const { result } = renderHook(() => useSessions('proj-pin-fail'));
    await waitFor(() => expect(result.current.loading).toBe(false));

    await expect(
      act(async () => {
        await result.current.pinAgent('s1', 'not-real');
      }),
    ).rejects.toThrow();

    await waitFor(() => {
      expect(result.current.sessions[0].pinned_target).toBeUndefined();
    });
  });
});
