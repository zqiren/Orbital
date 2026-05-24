// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * ChatTab unit tests — spec §6.2.1 ("route reflects active session").
 *
 * ChatTab is the session-resolution hub: it owns the logic that picks a
 * default active session when route.sessionId is undefined, reflects it back
 * into the route via setRoute, and passes the resolved id down to
 * SessionSidebar (selectedSessionId) and ChatView (sessionId).
 *
 * Strategy: mock useSessions, useSession, SessionSidebar, and ChatView so we
 * can inspect the props ChatTab passes them and the setRoute calls it makes,
 * without any network / WebSocket / rendering work from those heavy children.
 */

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, act, cleanup } from '@testing-library/react';
import type { SessionListEntry, Project } from '../types';
import type { Route } from '../route';

afterEach(() => cleanup());

// ---------------------------------------------------------------------------
// Mutable state shared between test setup and mocks
// ---------------------------------------------------------------------------

let mockSessions: SessionListEntry[] = [];
let mockActiveSessionId: string | null = null;
const mockSetActiveSessionId = vi.fn((id: string | null) => {
  mockActiveSessionId = id;
});

// ---------------------------------------------------------------------------
// Mock hooks — must be declared before importing the component
// ---------------------------------------------------------------------------

vi.mock('../hooks/useSessions', () => ({
  useSessions: (_projectId: string | null) => ({
    sessions: mockSessions,
    loading: false,
    error: null,
    refresh: vi.fn(),
  }),
}));

vi.mock('../hooks/useSession', () => ({
  useSession: (_projectId: string | null) => ({
    activeSessionId: mockActiveSessionId,
    setActiveSessionId: mockSetActiveSessionId,
  }),
}));

// ---------------------------------------------------------------------------
// Stub children — capture props so we can assert on them without rendering
// the real heavy components (which need API / WS / etc.)
// ---------------------------------------------------------------------------

// Capture the last props passed to each stub.
let lastSessionSidebarProps: Record<string, unknown> = {};
let lastChatViewProps: Record<string, unknown> = {};

vi.mock('./SessionSidebar', () => ({
  SessionSidebar: (props: Record<string, unknown>) => {
    lastSessionSidebarProps = props;
    return null;
  },
}));

vi.mock('./ChatView', () => ({
  default: (props: Record<string, unknown>) => {
    lastChatViewProps = props;
    return null;
  },
}));

// ---------------------------------------------------------------------------
// Import under test (AFTER mocks are declared)
// ---------------------------------------------------------------------------

import ChatTab from './ChatTab';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PROJECT: Project = {
  project_id: 'proj-1',
  name: 'Test Project',
  workspace: '/tmp/test',
  model: 'claude-3-5-sonnet',
  api_key: '',
  base_url: null,
  autonomy: 'hands_off',
  instructions: '',
};

function makeSession(overrides: Partial<SessionListEntry> = {}): SessionListEntry {
  return {
    session_id: 'sess-default',
    status: 'idle',
    session_uuid: 'uuid-default',
    last_terminal_event: null,
    last_activity_at: null,
    ...overrides,
  };
}

function makeRoute(
  overrides: Partial<Extract<Route, { name: 'project' }>> = {},
): Extract<Route, { name: 'project' }> {
  return {
    name: 'project',
    projectId: 'proj-1',
    tab: 'chat',
    sessionId: undefined,
    ...overrides,
  };
}

function resetMocks() {
  mockSessions = [];
  mockActiveSessionId = null;
  mockSetActiveSessionId.mockClear();
  lastSessionSidebarProps = {};
  lastChatViewProps = {};
}

// ---------------------------------------------------------------------------
// §6.2.1 — Route reflects active session
// ---------------------------------------------------------------------------

describe('ChatTab — route.sessionId undefined: resolution + setRoute reflection', () => {
  beforeEach(() => resetMocks());

  it('calls setRoute with the persisted activeSessionId when it is present in the session list', async () => {
    mockSessions = [
      makeSession({ session_id: 'sess-persisted', last_activity_at: '2026-01-01T00:00:00Z' }),
      makeSession({ session_id: 'sess-recent', last_activity_at: '2026-06-01T00:00:00Z' }),
    ];
    mockActiveSessionId = 'sess-persisted';

    const setRoute = vi.fn();
    const route = makeRoute(); // sessionId: undefined

    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          mentionAgents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    // setRoute must have been called (reflection effect)
    expect(setRoute).toHaveBeenCalled();

    // The updater function passed to setRoute must resolve to a route with
    // sessionId === 'sess-persisted' (persisted wins over most-recently-active)
    const updater = setRoute.mock.calls[0][0] as (prev: Route) => Route;
    const updated = updater(route);
    expect(updated).toMatchObject({ name: 'project', projectId: 'proj-1', sessionId: 'sess-persisted' });
  });

  it('falls back to most-recently-active (max last_activity_at) when persisted id is null', async () => {
    mockSessions = [
      makeSession({ session_id: 'sess-old', last_activity_at: '2026-01-01T00:00:00Z' }),
      makeSession({ session_id: 'sess-newest', last_activity_at: '2026-06-01T12:00:00Z' }),
      makeSession({ session_id: 'sess-mid', last_activity_at: '2026-03-15T00:00:00Z' }),
    ];
    mockActiveSessionId = null;

    const setRoute = vi.fn();
    const route = makeRoute();

    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          mentionAgents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    expect(setRoute).toHaveBeenCalled();
    const updater = setRoute.mock.calls[0][0] as (prev: Route) => Route;
    const updated = updater(route);
    expect(updated).toMatchObject({ sessionId: 'sess-newest' });
  });
});

describe('ChatTab — route.sessionId already set: no override', () => {
  beforeEach(() => resetMocks());

  it('does NOT call setRoute when route.sessionId is already defined', async () => {
    mockSessions = [
      makeSession({ session_id: 'sess-a' }),
      makeSession({ session_id: 'sess-b' }),
    ];
    mockActiveSessionId = 'sess-b';

    const setRoute = vi.fn();
    const route = makeRoute({ sessionId: 'sess-a' }); // already set

    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          mentionAgents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    expect(setRoute).not.toHaveBeenCalled();
  });

  it('passes the already-set route.sessionId as selectedSessionId and sessionId to children', async () => {
    mockSessions = [makeSession({ session_id: 'sess-x' })];
    mockActiveSessionId = null;

    const route = makeRoute({ sessionId: 'sess-x' });

    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          mentionAgents={[]}
          route={route}
          setRoute={vi.fn()}
        />,
      );
    });

    expect(lastSessionSidebarProps.selectedSessionId).toBe('sess-x');
    expect(lastChatViewProps.sessionId).toBe('sess-x');
  });
});

describe('ChatTab — persisted session gone: fallback to most-recent existing', () => {
  beforeEach(() => resetMocks());

  it('ignores a persisted id that is NOT in the session list and picks most-recent instead', async () => {
    mockSessions = [
      makeSession({ session_id: 'sess-current', last_activity_at: '2026-05-01T00:00:00Z' }),
    ];
    // Persisted id names a session that no longer exists
    mockActiveSessionId = 'sess-deleted';

    const setRoute = vi.fn();
    const route = makeRoute();

    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          mentionAgents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    expect(setRoute).toHaveBeenCalled();
    const updater = setRoute.mock.calls[0][0] as (prev: Route) => Route;
    const updated = updater(route);
    // Must NOT use the stale persisted id
    expect(updated.name === 'project' && updated.sessionId).not.toBe('sess-deleted');
    // Must pick the only existing session
    expect(updated.name === 'project' && updated.sessionId).toBe('sess-current');
  });
});

describe('ChatTab — controlled sidebar handoff (onSessionSelect)', () => {
  beforeEach(() => resetMocks());

  it('passes selectedSessionId (= route.sessionId) to SessionSidebar', async () => {
    mockSessions = [makeSession({ session_id: 'sess-ctrl' })];
    const route = makeRoute({ sessionId: 'sess-ctrl' });

    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          mentionAgents={[]}
          route={route}
          setRoute={vi.fn()}
        />,
      );
    });

    expect(lastSessionSidebarProps.selectedSessionId).toBe('sess-ctrl');
  });

  it('invoking SessionSidebar onSessionSelect calls setRoute with the picked session and setActiveSessionId', async () => {
    mockSessions = [
      makeSession({ session_id: 'sess-a' }),
      makeSession({ session_id: 'sess-b' }),
    ];
    const route = makeRoute({ sessionId: 'sess-a' });
    const setRoute = vi.fn();

    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          mentionAgents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    // Simulate user picking 'sess-b' in the sidebar
    const onSessionSelect = lastSessionSidebarProps.onSessionSelect as (id: string) => void;
    act(() => {
      onSessionSelect('sess-b');
    });

    // setRoute must have been called with an updater that resolves to sessionId: 'sess-b'
    expect(setRoute).toHaveBeenCalled();
    const updater = setRoute.mock.calls[0][0] as (prev: Route) => Route;
    const updated = updater(route);
    expect(updated).toMatchObject({ name: 'project', projectId: 'proj-1', sessionId: 'sess-b' });

    // setActiveSessionId must also have been called with the picked id
    expect(mockSetActiveSessionId).toHaveBeenCalledWith('sess-b');
  });
});

describe('ChatTab — empty project (no sessions)', () => {
  beforeEach(() => resetMocks());

  it('resolves to undefined without crashing when there are no sessions', async () => {
    mockSessions = [];
    mockActiveSessionId = null;

    const setRoute = vi.fn();
    const route = makeRoute();

    // Must not throw
    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          mentionAgents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    // No route override should be called — nothing to resolve to
    expect(setRoute).not.toHaveBeenCalled();

    // Children receive undefined sessionId
    expect(lastSessionSidebarProps.selectedSessionId).toBeUndefined();
    expect(lastChatViewProps.sessionId).toBeUndefined();
  });
});
