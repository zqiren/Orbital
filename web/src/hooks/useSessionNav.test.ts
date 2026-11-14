// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Spec 137 (the folder shell): session navigation moved out of ChatTab into a
// hook the project column and the chat share. The "+ new session" cases came
// with it from ChatTab.test.tsx (Task C, spec 107 FE-1).

import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Route } from '../route';
import type { SessionListEntry } from '../types';
import { consumeFreshSession } from '../utils/freshSession';

const mockSetActiveSessionId = vi.fn();
vi.mock('./useSession', () => ({
  useSession: () => ({ activeSessionId: null, setActiveSessionId: mockSetActiveSessionId }),
}));

const mockNewSession = vi.fn(async (_projectId: string) => ({ status: 'ok', session_id: 'sess_minted_new' }));
vi.mock('./useAgent', () => ({
  useAgent: () => ({ newSession: mockNewSession }),
}));

import { useSessionNav } from './useSessionNav';

type ProjectRoute = Extract<Route, { name: 'project' }>;

function route(over: Partial<ProjectRoute> = {}): ProjectRoute {
  return { name: 'project', projectId: 'proj-1', sessionId: 'sess-old', ...over };
}

function lastUpdate(setRoute: ReturnType<typeof vi.fn>, from: Route): Route {
  const calls = setRoute.mock.calls;
  const updater = calls[calls.length - 1][0] as (prev: Route) => Route;
  return updater(from);
}

function entry(session_id: string): SessionListEntry {
  return { session_id, status: 'idle', session_uuid: `uuid-${session_id}`, last_terminal_event: null, last_activity_at: null };
}

beforeEach(() => {
  mockSetActiveSessionId.mockClear();
  mockNewSession.mockClear();
  mockNewSession.mockResolvedValue({ status: 'ok', session_id: 'sess_minted_new' });
});

describe('useSessionNav', () => {
  it('selectSession opens the session in the chat: leaves any page, persists the choice', () => {
    const setRoute = vi.fn();
    const { result } = renderHook(() => useSessionNav('proj-1', setRoute));
    act(() => result.current.selectSession('sess-b'));
    const from = route({ page: 'tasks', queuePane: 'automations' });
    expect(lastUpdate(setRoute, from)).toEqual({ ...from, sessionId: 'sess-b', page: undefined });
    expect(mockSetActiveSessionId).toHaveBeenCalledWith('sess-b');
  });

  it('never touches another project’s route', () => {
    const setRoute = vi.fn();
    const { result } = renderHook(() => useSessionNav('proj-1', setRoute));
    act(() => result.current.selectSession('sess-b'));
    const other = route({ projectId: 'proj-2' });
    expect(lastUpdate(setRoute, other)).toBe(other);
  });

  it('newSession mints a fresh session id and navigates to it (not the most-recent existing)', async () => {
    const setRoute = vi.fn();
    const { result } = renderHook(() => useSessionNav('proj-1', setRoute));
    await act(() => result.current.newSession());
    // Fresh-create: no explicit session id.
    expect(mockNewSession).toHaveBeenCalledTimes(1);
    expect(mockNewSession).toHaveBeenCalledWith('proj-1');
    const next = lastUpdate(setRoute, route({ page: 'settings' }));
    expect(next).toMatchObject({ name: 'project', projectId: 'proj-1', sessionId: 'sess_minted_new', page: undefined });
    expect(mockSetActiveSessionId).toHaveBeenCalledWith('sess_minted_new');
  });

  it('marks the minted id as fresh BEFORE navigating to it (spec 107 FE-1)', async () => {
    let freshAtSetRoute: boolean | null = null;
    const setRoute = vi.fn(() => {
      // Consuming here proves the mark preceded the navigation.
      freshAtSetRoute = consumeFreshSession('proj-1', 'sess_minted_new');
    });
    const { result } = renderHook(() => useSessionNav('proj-1', setRoute));
    await act(() => result.current.newSession());
    expect(setRoute).toHaveBeenCalled();
    expect(freshAtSetRoute).toBe(true);
    // Consumed once: nothing lingers for a later visit to the same id.
    expect(consumeFreshSession('proj-1', 'sess_minted_new')).toBe(false);
  });

  it('a mint without an id leaves the route alone', async () => {
    mockNewSession.mockResolvedValue({ status: 'ok' } as never);
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    const setRoute = vi.fn();
    const { result } = renderHook(() => useSessionNav('proj-1', setRoute));
    await act(() => result.current.newSession());
    expect(setRoute).not.toHaveBeenCalled();
    errors.mockRestore();
  });

  it('sessionDeleted moves off the viewed session to the most recent remaining one, else does nothing', () => {
    const setRoute = vi.fn();
    const { result } = renderHook(() => useSessionNav('proj-1', setRoute));
    act(() => result.current.sessionDeleted('sess-other', [entry('sess-old')], 'sess-old'));
    expect(setRoute).not.toHaveBeenCalled();

    act(() => result.current.sessionDeleted('sess-old', [entry('sess-next'), entry('sess-older')], 'sess-old'));
    expect(lastUpdate(setRoute, route())).toMatchObject({ sessionId: 'sess-next' });
    expect(mockSetActiveSessionId).toHaveBeenCalledWith('sess-next');

    act(() => result.current.sessionDeleted('sess-next', [], 'sess-next'));
    expect(lastUpdate(setRoute, route({ sessionId: 'sess-next' }))).toMatchObject({ sessionId: undefined });
    expect(mockSetActiveSessionId).toHaveBeenLastCalledWith(null);
  });

  it('spec 180: sessionMoved follows the viewed session to its new project, else leaves the route alone', () => {
    const setRoute = vi.fn();
    const { result } = renderHook(() => useSessionNav('proj-1', setRoute));
    act(() => result.current.sessionMoved('sess-old', 'proj-2'));
    expect(lastUpdate(setRoute, route({ page: 'tasks' }))).toEqual({ name: 'project', projectId: 'proj-2', sessionId: 'sess-old' });
    // Another session is open: nothing to follow.
    const other = route({ sessionId: 'sess-b' });
    expect(lastUpdate(setRoute, other)).toBe(other);
    // Another project is open: nothing to follow either.
    const elsewhere = route({ projectId: 'proj-3' });
    expect(lastUpdate(setRoute, elsewhere)).toBe(elsewhere);
  });
});
