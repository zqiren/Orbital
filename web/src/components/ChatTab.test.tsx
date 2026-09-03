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
import { render, act, cleanup, screen, fireEvent } from '@testing-library/react';
import type {
  ActivityEvent,
  AgentRunStatus,
  ChatMessage,
  Project,
  SessionListEntry,
} from '../types';
import type { Route } from '../route';
import { __resetPanelState } from '../hooks/usePanelState';
import { __resetAnnotationsStore } from '../hooks/useAnnotations';

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

// useAgent.newSession is called by "+ new session" to mint a fresh session id
// on the backend. Mock it to return a known minted id so we can assert that
// ChatTab navigates to that id rather than re-resolving to an old session.
const mockNewSession = vi.fn(async (_projectId: string, _sessionId?: string) => ({
  status: 'ok',
  session_id: 'sess_minted_new',
}));

vi.mock('../hooks/useAgent', () => ({
  useAgent: () => ({
    newSession: mockNewSession,
  }),
}));

// ---------------------------------------------------------------------------
// Spec 078 — the workspace panel's dependencies.
//
// ChatTab now subscribes to the activity stream (useWebSocket throws outside a
// provider) and owns the panel's copy of the session transcript. Both are
// mocked so the resolution tests above stay pure prop assertions, and so the
// panel tests below can drive events and messages directly.
// ---------------------------------------------------------------------------

const wsHandlers = new Map<string, (event: unknown) => void>();
const mockOn = vi.fn((type: string, fn: (event: unknown) => void) => {
  wsHandlers.set(type, fn);
});
const mockOff = vi.fn((type: string) => {
  wsHandlers.delete(type);
});

vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    connectionState: 'connected',
    subscribe: vi.fn(),
    on: mockOn,
    off: mockOff,
  }),
}));

let mockMessages: ChatMessage[] = [];
const mockLoadHistory = vi.fn(async () => mockMessages);

vi.mock('../hooks/useChatHistory', () => ({
  useChatHistory: () => ({
    messages: mockMessages,
    loading: false,
    lastEvent: null,
    loadHistory: mockLoadHistory,
    mergeRealtimeEvent: vi.fn(),
    clearMessages: vi.fn(),
  }),
}));

// The two views are other workstreams' surfaces; capture their props instead
// of rendering their network-backed bodies.
let lastFilesViewProps: Record<string, unknown> = {};
let lastBrowserViewProps: Record<string, unknown> = {};

vi.mock('./panel/FilesView', () => ({
  default: (props: Record<string, unknown>) => {
    lastFilesViewProps = props;
    return <div data-testid="files-view" />;
  },
}));

vi.mock('./panel/BrowserView', () => ({
  default: (props: Record<string, unknown>) => {
    lastBrowserViewProps = props;
    return <div data-testid="browser-view" />;
  },
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
  mockNewSession.mockClear();
  mockNewSession.mockResolvedValue({ status: 'ok', session_id: 'sess_minted_new' });
  lastSessionSidebarProps = {};
  lastChatViewProps = {};
  lastFilesViewProps = {};
  lastBrowserViewProps = {};
  mockMessages = [];
  mockLoadHistory.mockClear();
  mockOn.mockClear();
  mockOff.mockClear();
  wsHandlers.clear();
  __resetPanelState();
  __resetAnnotationsStore();
  localStorage.clear();
  setViewportWidth(1024); // below the push threshold: no panel by default
}

/** usePanelDockable reads window.innerWidth; 1440 is the spec's reference width. */
function setViewportWidth(width: number) {
  Object.defineProperty(window, 'innerWidth', {
    value: width,
    configurable: true,
    writable: true,
  });
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

describe('ChatTab — composer prefill (route.draft → ChatView, spec 2026-07-24)', () => {
  beforeEach(() => resetMocks());

  it('threads route.draft to ChatView as initialDraft', async () => {
    mockSessions = [makeSession({ session_id: 'sess-x' })];
    const route = makeRoute({ sessionId: 'sess-x', draft: 'Workbench · "do the thing"\n\n' });

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

    expect(lastChatViewProps.initialDraft).toBe('Workbench · "do the thing"\n\n');
  });

  it('passes undefined initialDraft when route.draft is unset', async () => {
    mockSessions = [makeSession({ session_id: 'sess-x' })];
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

    expect(lastChatViewProps.initialDraft).toBeUndefined();
  });

  it('onDraftConsumed clears route.draft back to undefined via setRoute, preserving the rest of the route', async () => {
    mockSessions = [makeSession({ session_id: 'sess-x' })];
    const setRoute = vi.fn();
    const route = makeRoute({ sessionId: 'sess-x', draft: 'Workbench · "do the thing"\n\n' });

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

    const onDraftConsumed = lastChatViewProps.onDraftConsumed as () => void;
    act(() => {
      onDraftConsumed();
    });

    expect(setRoute).toHaveBeenCalled();
    const updater = setRoute.mock.calls[setRoute.mock.calls.length - 1][0] as (prev: Route) => Route;
    const updated = updater(route);
    expect(updated).toMatchObject({
      name: 'project',
      projectId: 'proj-1',
      tab: 'chat',
      sessionId: 'sess-x',
    });
    expect((updated as Extract<Route, { name: 'project' }>).draft).toBeUndefined();
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

// ---------------------------------------------------------------------------
// Task C — "+ new session" mints a FRESH blank session (not re-resolve old)
// ---------------------------------------------------------------------------

describe('ChatTab — "+ new session" creates a genuinely blank session', () => {
  beforeEach(() => resetMocks());

  it('mints a fresh session id via newSession and navigates to it (not the most-recent existing)', async () => {
    // There is an existing, most-recently-active session. The OLD buggy
    // behaviour cleared route.sessionId to undefined, which re-resolved to
    // this existing session. The fix must navigate to the minted id instead.
    mockSessions = [
      makeSession({ session_id: 'sess-old', last_activity_at: '2026-06-01T00:00:00Z' }),
    ];
    mockActiveSessionId = 'sess-old';

    const setRoute = vi.fn();
    const route = makeRoute({ sessionId: 'sess-old' });

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

    // The resolution effect must NOT have fired (route.sessionId already set).
    setRoute.mockClear();

    // User clicks "+ new session".
    const onNewSession = lastSessionSidebarProps.onNewSession as () => void;
    await act(async () => {
      onNewSession();
      // allow the awaited newSession() promise + state update to flush
      await Promise.resolve();
      await Promise.resolve();
    });

    // newSession was called with NO explicit session id (fresh-create).
    expect(mockNewSession).toHaveBeenCalledTimes(1);
    expect(mockNewSession).toHaveBeenCalledWith('proj-1');

    // setRoute navigates to the MINTED id, not the old/most-recent one.
    expect(setRoute).toHaveBeenCalled();
    const updater = setRoute.mock.calls[setRoute.mock.calls.length - 1][0] as (prev: Route) => Route;
    const updated = updater(route);
    expect(updated).toMatchObject({
      name: 'project',
      projectId: 'proj-1',
      sessionId: 'sess_minted_new',
    });
    expect(updated.name === 'project' && updated.sessionId).not.toBe('sess-old');
  });

  it('tolerates a route.sessionId that is not in the sessions list (blank session, no re-resolve)', async () => {
    // After minting, the new id is DEFINED but not yet in `sessions` (it
    // materializes server-side only on first inject). The resolution effect
    // must early-return and leave the new id in place — children get it.
    mockSessions = [
      makeSession({ session_id: 'sess-old', last_activity_at: '2026-06-01T00:00:00Z' }),
    ];
    mockActiveSessionId = 'sess-old';

    const setRoute = vi.fn();
    const route = makeRoute({ sessionId: 'sess_minted_new' }); // not in list

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

    // No clobbering back to an existing session.
    expect(setRoute).not.toHaveBeenCalled();
    // Children receive the fresh id even though it isn't in `sessions`.
    expect(lastSessionSidebarProps.selectedSessionId).toBe('sess_minted_new');
    expect(lastChatViewProps.sessionId).toBe('sess_minted_new');
  });
});

// ---------------------------------------------------------------------------
// Spec 078 §5.1/§9.4 — the workspace panel column and the collapsed handle
// ---------------------------------------------------------------------------

function renderChatTab(
  routeOverride: Partial<Extract<Route, { name: 'project' }>> = {},
  props: { agentStatus?: AgentRunStatus } = {},
) {
  const route = makeRoute({ sessionId: 'sess-x', ...routeOverride });
  const setRoute = vi.fn();
  const view = render(
    <ChatTab
      project={PROJECT}
      agentStatus={props.agentStatus ?? 'idle'}
      mentionAgents={[]}
      route={route}
      setRoute={setRoute}
    />,
  );
  return { ...view, route, setRoute };
}

function emitActivity(overrides: Partial<ActivityEvent> = {}) {
  const handler = wsHandlers.get('agent.activity');
  if (!handler) throw new Error('ChatTab did not subscribe to agent.activity');
  act(() => {
    handler({
      type: 'agent.activity',
      project_id: 'proj-1',
      session_id: 'sess-x',
      id: `evt-${Math.random()}`,
      category: 'file_edit',
      description: 'Edited src/a.ts',
      tool_name: 'edit',
      source: 'management',
      timestamp: '2026-09-03T10:00:00Z',
      ...overrides,
    } satisfies ActivityEvent);
  });
}

function toolCallMessage(name: string, args: Record<string, unknown>): ChatMessage {
  return {
    role: 'assistant',
    content: null,
    source: 'management',
    timestamp: '2026-09-03T10:00:00Z',
    tool_calls: [
      { id: `call-${name}`, type: 'function', function: { name, arguments: JSON.stringify(args) } },
    ],
  };
}

describe('ChatTab — panel column vs. overlay (push threshold)', () => {
  beforeEach(() => resetMocks());

  it('renders neither the handle nor the panel below the 1200px threshold', async () => {
    setViewportWidth(1024);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
    await act(async () => {
      renderChatTab();
    });
    expect(screen.queryByTestId('panel-handle')).toBeNull();
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
  });

  it('renders the collapsed edge handle at rest on a wide window', async () => {
    setViewportWidth(1440);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
    await act(async () => {
      renderChatTab();
    });
    expect(screen.getByTestId('panel-handle')).toBeInTheDocument();
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
  });

  it('clicking the handle expands the panel as a third column beside the chat', async () => {
    setViewportWidth(1440);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
    await act(async () => {
      renderChatTab();
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-handle'));
    });

    expect(screen.getByTestId('workspace-panel-column')).toBeInTheDocument();
    expect(screen.queryByTestId('panel-handle')).toBeNull();
    // The session column and the chat column are still there — three columns.
    expect(lastSessionSidebarProps.selectedSessionId).toBe('sess-x');
    expect(lastChatViewProps.sessionId).toBe('sess-x');
    expect(screen.getByRole('tab', { name: 'Files' })).toBeInTheDocument();
  });

  it('the panel’s collapse button puts the handle back', async () => {
    setViewportWidth(1440);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
    await act(async () => {
      renderChatTab();
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-handle'));
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Hide workspace' }));
    });
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
    expect(screen.getByTestId('panel-handle')).toBeInTheDocument();
  });
});

describe('ChatTab — the panel follows the agent (D8)', () => {
  beforeEach(() => {
    resetMocks();
    setViewportWidth(1440);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
  });

  it('a browser action expands the panel on the Browser view', async () => {
    await act(async () => {
      renderChatTab();
    });
    emitActivity({ category: 'browser_automation', tool_name: 'browser', arguments: { action: 'click' } });

    expect(screen.getByTestId('workspace-panel-column')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Browser' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('browser-view')).toBeInTheDocument();
  });

  it('a file edit expands on Files and opens that file’s preview', async () => {
    await act(async () => {
      renderChatTab();
    });
    emitActivity({ category: 'file_edit', tool_name: 'edit', arguments: { path: 'src/a.ts' } });

    expect(screen.getByTestId('workspace-panel-column')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'true');
    expect(lastFilesViewProps.file).toBe('src/a.ts');
  });

  it('a command-only turn does not open the panel (§13.3)', async () => {
    await act(async () => {
      renderChatTab();
    });
    emitActivity({ category: 'command_exec', tool_name: 'shell', arguments: { command: 'ls' } });
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
    expect(screen.getByTestId('panel-handle')).toBeInTheDocument();
  });

  it('ignores activity belonging to another session', async () => {
    await act(async () => {
      renderChatTab();
    });
    emitActivity({ session_id: 'sess-other', category: 'browser_automation', tool_name: 'browser' });
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
  });

  it('stays expanded when the turn ends (no auto-collapse, D8 amended)', async () => {
    const view = await act(async () =>
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="running"
          mentionAgents={[]}
          route={makeRoute({ sessionId: 'sess-x' })}
          setRoute={vi.fn()}
        />,
      ),
    );
    emitActivity({ category: 'file_read', tool_name: 'read', arguments: { path: 'a.md' } });
    expect(screen.getByTestId('workspace-panel-column')).toBeInTheDocument();

    await act(async () => {
      view.rerender(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          mentionAgents={[]}
          route={makeRoute({ sessionId: 'sess-x' })}
          setRoute={vi.fn()}
        />,
      );
    });
    expect(screen.getByTestId('workspace-panel-column')).toBeInTheDocument();
    expect(screen.queryByTestId('panel-handle')).toBeNull();
  });

  it('a panel collapsed during a run stays collapsed for the rest of it', async () => {
    await act(async () =>
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="running"
          mentionAgents={[]}
          route={makeRoute({ sessionId: 'sess-x' })}
          setRoute={vi.fn()}
        />,
      ),
    );
    emitActivity({ category: 'file_read', tool_name: 'read', arguments: { path: 'a.md' } });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Hide workspace' }));
    });

    emitActivity({ category: 'browser_automation', tool_name: 'browser' });
    emitActivity({ category: 'file_edit', tool_name: 'edit', arguments: { path: 'b.md' } });
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
  });

  it('shows the working dot on the handle only while the agent is running', async () => {
    await act(async () =>
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="running"
          mentionAgents={[]}
          route={makeRoute({ sessionId: 'sess-x' })}
          setRoute={vi.fn()}
        />,
      ),
    );
    expect(screen.getByTestId('panel-handle-working')).toBeInTheDocument();
  });
});

describe('ChatTab — what the panel is given', () => {
  beforeEach(() => {
    resetMocks();
    setViewportWidth(1440);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
  });

  it('derives the touched list from the session transcript', async () => {
    mockMessages = [
      toolCallMessage('read', { path: 'a.md' }),
      toolCallMessage('write', { path: 'b.md' }),
    ];
    await act(async () => {
      renderChatTab();
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-handle'));
    });

    expect(mockLoadHistory).toHaveBeenCalledWith('proj-1');
    expect(lastFilesViewProps.touched).toEqual([
      { path: 'b.md', op: 'written', lastAt: '2026-09-03T10:00:00Z' },
      { path: 'a.md', op: 'read', lastAt: '2026-09-03T10:00:00Z' },
    ]);
  });

  it('adds files touched live, mid-run, without refetching the transcript', async () => {
    await act(async () => {
      renderChatTab();
    });
    emitActivity({ category: 'file_write', tool_name: 'write', arguments: { path: 'live.md' } });
    expect(lastFilesViewProps.touched).toEqual([
      { path: 'live.md', op: 'written', lastAt: '2026-09-03T10:00:00Z' },
    ]);
  });

  it('gives BrowserView the session’s last screenshot as its fallback', async () => {
    mockMessages = [
      {
        role: 'tool',
        content: 'ok',
        source: 'management',
        timestamp: '2026-09-03T10:00:01Z',
        tool_call_id: 'call-1',
        _meta: { url: 'https://x', title: 'Queue', screenshot_path: '/ws/shots/12.png' },
      },
    ];
    await act(async () => {
      renderChatTab();
    });
    emitActivity({ category: 'browser_automation', tool_name: 'browser' });

    expect(lastBrowserViewProps.fallback).toEqual({ path: '/ws/shots/12.png', title: 'Queue' });
    expect(lastBrowserViewProps.active).toBe(true);
  });

  it('Annotate lives on the Browser view only; Files quotes without a mode', async () => {
    await act(async () => {
      renderChatTab();
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-handle'));
    });
    // Files view: no Annotate button, and FilesView is never handed the mode.
    expect(screen.queryByTestId('panel-annotate')).toBeNull();
    expect((lastFilesViewProps as Record<string, unknown>).annotating).toBeUndefined();

    await act(async () => {
      fireEvent.click(screen.getByRole('tab', { name: 'Browser' }));
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-annotate'));
    });
    expect(screen.getByRole('button', { name: 'Done' })).toBeInTheDocument();
    expect(lastBrowserViewProps.annotating).toBe(true);
  });
});

describe('ChatTab — route.previewPath opens the panel, not the overlay (§9.10)', () => {
  beforeEach(() => {
    resetMocks();
    setViewportWidth(1440);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
  });

  it('a chat path click expands the panel on Files with that file', async () => {
    await act(async () => {
      renderChatTab({ previewPath: 'docs/plan.md' });
    });
    expect(screen.getByTestId('workspace-panel-column')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'true');
    expect(lastFilesViewProps.file).toBe('docs/plan.md');
  });

  it('going back to the tree clears route.previewPath so it cannot re-open', async () => {
    const { setRoute, route } = await act(async () =>
      renderChatTab({ previewPath: 'docs/plan.md' }),
    );
    setRoute.mockClear();

    act(() => {
      (lastFilesViewProps.onSelectFile as (path: string | null) => void)(null);
    });

    expect(setRoute).toHaveBeenCalled();
    const updater = setRoute.mock.calls[setRoute.mock.calls.length - 1][0] as (prev: Route) => Route;
    const updated = updater(route) as Extract<Route, { name: 'project' }>;
    expect(updated.previewPath).toBeUndefined();
    expect(lastFilesViewProps.file).toBeNull();
  });

  it('"Open in Files" navigates to the full-width Files tab', async () => {
    const { setRoute, route } = await act(async () =>
      renderChatTab({ previewPath: 'docs/plan.md' }),
    );
    setRoute.mockClear();

    act(() => {
      (lastFilesViewProps.onOpenInFiles as (path: string) => void)('docs/plan.md');
    });

    const updater = setRoute.mock.calls[setRoute.mock.calls.length - 1][0] as (prev: Route) => Route;
    const updated = updater(route) as Extract<Route, { name: 'project' }>;
    expect(updated.tab).toBe('files');
    expect(updated.previewPath).toBeUndefined();
  });
});
