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
import { render, act, cleanup, screen, fireEvent, within } from '@testing-library/react';
import type {
  ActivityEvent,
  AgentRunStatus,
  ChatMessage,
  Project,
  SessionListEntry,
} from '../types';
import type { Route } from '../route';
import { __resetPanelState } from '../hooks/usePanelState';
import { OpenPathContext } from './filePreviewContext';
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

// Maximized Files renders the old Files tab's explorer (spec 137 gap 1).
let lastFileExplorerProps: Record<string, unknown> | null = null;
vi.mock('./FileExplorer', () => ({
  default: (props: Record<string, unknown>) => {
    lastFileExplorerProps = props;
    return <div data-testid="file-explorer" />;
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

// Capture the last props passed to each stub. SessionSidebar is stubbed only
// to prove ChatTab no longer renders it (spec 137: it is the project column's).
let lastSessionSidebarProps: Record<string, unknown> = {};
let lastChatViewProps: Record<string, unknown> = {};

vi.mock('./SessionSidebar', () => ({
  SessionSidebar: (props: Record<string, unknown>) => {
    lastSessionSidebarProps = props;
    return null;
  },
}));

// The undo sheet's own fetching is TurnUndoSheet's test; here only its 查看
// hand-off matters, so the stub is one row with the eye.
vi.mock('./history/TurnUndoSheet', () => ({
  default: (props: { onView: (path: string, v: number) => void }) => (
    <button type="button" data-testid="stub-undo-view" onClick={() => props.onView('docs/plan.md', 3)}>
      view
    </button>
  ),
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
  lastFileExplorerProps = null;
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
          agents={[]}
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
          agents={[]}
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

  // Spec 081 §3.4 — a reload while the first message is queued used to land
  // on the running session, because the queued session was not listed and the
  // persisted id was therefore discarded. Now the queued row is a list entry
  // like any other, so the persisted selection survives the reload.
  it('honors a persisted queued session id and does not fall back to the holder', async () => {
    mockSessions = [
      makeSession({
        session_id: 'sess-holder', status: 'running',
        last_activity_at: '2026-06-01T00:00:00Z',
      }),
      makeSession({
        session_id: 'sess-queued', status: 'queued',
        last_activity_at: '2026-06-01T00:05:00Z',
      }),
    ];
    mockActiveSessionId = 'sess-queued';

    const setRoute = vi.fn();
    const route = makeRoute();

    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          agents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    expect(setRoute).toHaveBeenCalled();
    const updater = setRoute.mock.calls[0][0] as (prev: Route) => Route;
    expect(updater(route)).toMatchObject({ sessionId: 'sess-queued' });
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
          agents={[]}
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
          agents={[]}
          route={route}
          setRoute={vi.fn()}
        />,
      );
    });

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
          agents={[]}
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

// Spec 137 (the folder shell): the session list moved into the project
// column (ProjectDetail), which stays while a page fills the right area. The
// chat only opens other sessions through ChatView's links, with the same
// navigation the column uses (hooks/useSessionNav).
describe('ChatTab — the session list lives in the project column, not here', () => {
  beforeEach(() => resetMocks());

  it('renders no session list of its own', async () => {
    mockSessions = [makeSession({ session_id: 'sess-ctrl' })];
    await act(async () => {
      render(
        <ChatTab
          project={PROJECT}
          agentStatus="idle"
          agents={[]}
          route={makeRoute({ sessionId: 'sess-ctrl' })}
          setRoute={vi.fn()}
        />,
      );
    });
    expect(lastSessionSidebarProps).toEqual({});
    expect(screen.queryByTestId('chat-tab-sessions')).toBeNull();
  });

  it('ChatView opening another session sets the route to it (back on chat) and persists it', async () => {
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
          agents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    act(() => {
      (lastChatViewProps.onOpenSession as (id: string) => void)('sess-b');
    });

    expect(setRoute).toHaveBeenCalled();
    const updater = setRoute.mock.calls[0][0] as (prev: Route) => Route;
    expect(updater(route)).toMatchObject({ name: 'project', projectId: 'proj-1', sessionId: 'sess-b', page: undefined });
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
          agents={[]}
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
          agents={[]}
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
          agents={[]}
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
          agents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    // No route override should be called — nothing to resolve to
    expect(setRoute).not.toHaveBeenCalled();

    // ChatView receives undefined sessionId
    expect(lastChatViewProps.sessionId).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Task C — a freshly minted session is not in the list yet. ("+ new session"
// itself moved to the project column; its tests are in useSessionNav.test.ts.)
// ---------------------------------------------------------------------------

describe('ChatTab — a minted session not yet in the list', () => {
  beforeEach(() => resetMocks());

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
          agents={[]}
          route={route}
          setRoute={setRoute}
        />,
      );
    });

    // No clobbering back to an existing session.
    expect(setRoute).not.toHaveBeenCalled();
    // ChatView receives the fresh id even though it isn't in `sessions`.
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
      agents={[]}
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
    // The chat is still there beside it (the project column is
    // ProjectDetail's, to the left of both).
    expect(screen.getByTestId('chat-tab-conversation').className).not.toMatch(/(^|\s)hidden(\s|$)/);
    expect(lastChatViewProps.sessionId).toBe('sess-x');
    expect(screen.getByRole('tab', { name: 'Files' })).toBeInTheDocument();
  });

  it('the Files | Browser switch and collapse share the panel’s ONE top row (spec 088)', async () => {
    setViewportWidth(1440);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
    await act(async () => {
      renderChatTab();
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-handle'));
    });

    expect(screen.getAllByRole('tablist')).toHaveLength(1);
    const row = screen.getByTestId('workspace-panel-header');
    expect(within(row).getByRole('tablist')).toBeInTheDocument();
    expect(within(row).getByRole('button', { name: 'Hide workspace' })).toBeInTheDocument();
    expect(within(row).queryByTestId('files-view')).toBeNull();

    // Browser: Annotate joins the same row; the view body sits below it.
    await act(async () => {
      fireEvent.click(within(row).getByRole('tab', { name: 'Browser' }));
    });
    expect(within(screen.getByTestId('workspace-panel-header')).getByTestId('panel-annotate')).toBeInTheDocument();
    expect(screen.getAllByRole('tablist')).toHaveLength(1);
    expect(screen.getByTestId('browser-view')).toBeInTheDocument();
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
          agents={[]}
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
          agents={[]}
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
          agents={[]}
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
          agents={[]}
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

  it('"Open full width" maximizes the panel on that file — no route change, no Files tab', async () => {
    const { setRoute } = await act(async () =>
      renderChatTab({ previewPath: 'docs/plan.md' }),
    );
    setRoute.mockClear();

    act(() => {
      (lastFilesViewProps.onOpenInFiles as (path: string) => void)('docs/plan.md');
    });

    // Every setRoute is an updater; none of them reintroduces a tab.
    for (const call of setRoute.mock.calls) {
      const arg = call[0] as Route | ((prev: Route) => Route);
      const next = typeof arg === 'function' ? arg(makeRoute({ sessionId: 'sess-x' })) : arg;
      expect(next).not.toHaveProperty('tab');
    }
    expect(screen.getByTestId('workspace-panel-column')).toHaveAttribute('data-maximized', 'true');
    expect(screen.getByTestId('file-explorer')).toBeInTheDocument();
    expect(lastFileExplorerProps?.initialPath).toBe('docs/plan.md');
  });

  // Spec 110 — route.previewPath is a request, consumed once by the docked
  // panel. A repeat click on the same card (same string) must reopen the
  // file whatever the user did with the panel in between.

  /** Re-render the mounted ChatTab with a NEW route object (what a card click produces). */
  function rerenderRoute(
    view: ReturnType<typeof renderChatTab>,
    override: Partial<Extract<Route, { name: 'project' }>>,
  ) {
    view.rerender(
      <ChatTab
        project={PROJECT}
        agentStatus="idle"
        agents={[]}
        route={makeRoute({ sessionId: 'sess-x', ...override })}
        setRoute={view.setRoute}
      />,
    );
  }

  /** Apply every setRoute updater ChatTab issued, in order, to `route`. */
  function applyUpdaters(setRoute: ReturnType<typeof vi.fn>, route: Route): Route {
    return setRoute.mock.calls.reduce<Route>((acc, call) => {
      const arg = call[0] as Route | ((prev: Route) => Route);
      return typeof arg === 'function' ? arg(acc) : arg;
    }, route);
  }

  it('a consumed previewPath is cleared from the route (spec 110 §3.1)', async () => {
    const { setRoute, route } = await act(async () =>
      renderChatTab({ previewPath: 'docs/plan.md' }),
    );
    expect(screen.getByTestId('workspace-panel-column')).toBeInTheDocument();
    expect(lastFilesViewProps.file).toBe('docs/plan.md');

    expect(setRoute).toHaveBeenCalled();
    const updater = setRoute.mock.calls[setRoute.mock.calls.length - 1][0] as (prev: Route) => Route;
    const updated = updater(route) as Extract<Route, { name: 'project' }>;
    expect(updated.previewPath).toBeUndefined();
    // Consuming never clears a path that belongs to another project.
    const other = updater(makeRoute({ projectId: 'proj-other', previewPath: 'x.md' }));
    expect((other as Extract<Route, { name: 'project' }>).previewPath).toBe('x.md');
  });

  /**
   * Model a second card click on the SAME file. App first re-renders with the
   * route ChatTab's own setRoute calls produced (post-fix: previewPath gone;
   * pre-fix: unchanged), then ProjectDetail.handleOpenPath writes the path
   * again. Pre-fix both renders carry the same string, so the effect never
   * fires and the card is dead — that is the production sequence this guards.
   */
  async function clickSameCardAgain(view: ReturnType<typeof renderChatTab>, path: string) {
    const held = applyUpdaters(view.setRoute, view.route) as Extract<Route, { name: 'project' }>;
    await act(async () => {
      rerenderRoute(view, { previewPath: held.previewPath });
    });
    await act(async () => {
      rerenderRoute(view, { previewPath: path });
    });
  }

  it('re-clicking the same card after "Hide workspace" reopens the file (spec 110)', async () => {
    const view = await act(async () => renderChatTab({ previewPath: 'docs/plan.md' }));
    expect(lastFilesViewProps.file).toBe('docs/plan.md');

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Hide workspace' }));
    });
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
    expect(screen.getByTestId('panel-handle')).toBeInTheDocument();

    lastFilesViewProps = {};
    await clickSameCardAgain(view, 'docs/plan.md');
    expect(screen.getByTestId('workspace-panel-column')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'true');
    expect(lastFilesViewProps.file).toBe('docs/plan.md');
    // Consumed again: the route the app holds afterwards carries no path.
    const final = applyUpdaters(view.setRoute, view.route) as Extract<Route, { name: 'project' }>;
    expect(final.previewPath).toBeUndefined();
  });

  it('re-clicking while the panel is on Browser lands on Files with the file (spec 110)', async () => {
    const view = await act(async () => renderChatTab({ previewPath: 'docs/plan.md' }));
    await act(async () => {
      fireEvent.click(screen.getByRole('tab', { name: 'Browser' }));
    });
    expect(screen.getByRole('tab', { name: 'Browser' })).toHaveAttribute('aria-selected', 'true');

    lastFilesViewProps = {};
    await clickSameCardAgain(view, 'docs/plan.md');
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'true');
    expect(lastFilesViewProps.file).toBe('docs/plan.md');
  });

  it('re-clicking after picking another file in the tree shows the card file again (spec 110)', async () => {
    const view = await act(async () => renderChatTab({ previewPath: 'docs/plan.md' }));
    act(() => {
      (lastFilesViewProps.onSelectFile as (path: string | null) => void)('src/other.ts');
    });
    expect(lastFilesViewProps.file).toBe('src/other.ts');

    await clickSameCardAgain(view, 'docs/plan.md');
    expect(lastFilesViewProps.file).toBe('docs/plan.md');
  });

  it('overlay mode (narrow viewport) never consumes the route path (spec 110 §3.2)', async () => {
    setViewportWidth(1000);
    const { setRoute } = await act(async () => renderChatTab({ previewPath: 'docs/plan.md' }));
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
    expect(setRoute).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Session import: entering a project never auto-opens an import that has not
// been converted yet (opening one materializes it, and landing there would
// skip the cold-start card and the onboarding kickoff).
// ---------------------------------------------------------------------------
describe('ChatTab — default resolution skips unmaterialized imports', () => {
  beforeEach(() => resetMocks());

  const imported = (id: string, at: string, materialized: boolean) =>
    makeSession({
      session_id: id, session_uuid: id, origin: 'imported', last_activity_at: at,
      imported: { agent: 'codex', native_id: id, materialized, size_bytes: 10 },
    });

  async function resolvedFor(): Promise<string | undefined> {
    const setRoute = vi.fn();
    const route = makeRoute();
    await act(async () => {
      render(
        <ChatTab project={PROJECT} agentStatus="idle" agents={[]} route={route} setRoute={setRoute} />,
      );
    });
    if (setRoute.mock.calls.length === 0) return undefined;
    const updated = (setRoute.mock.calls[0][0] as (prev: Route) => Route)(route);
    return (updated as Extract<Route, { name: 'project' }>).sessionId;
  }

  it('a project with only unmaterialized imports resolves to no session', async () => {
    mockSessions = [
      imported('imp_codex_a', '2026-10-20T10:00:00Z', false),
      imported('imp_codex_b', '2026-10-19T10:00:00Z', false),
    ];
    expect(await resolvedFor()).toBeUndefined();
    expect(lastChatViewProps.sessionId).toBeUndefined();
  });

  it('the newest ordinary session wins over a newer unmaterialized import', async () => {
    mockSessions = [
      imported('imp_codex_new', '2026-10-21T10:00:00Z', false),
      makeSession({ session_id: 'sess-chat', last_activity_at: '2026-10-01T00:00:00Z' }),
    ];
    expect(await resolvedFor()).toBe('sess-chat');
  });

  it('a persisted id naming an unmaterialized import is not reopened', async () => {
    mockSessions = [
      imported('imp_codex_new', '2026-10-21T10:00:00Z', false),
      makeSession({ session_id: 'sess-chat', last_activity_at: '2026-10-01T00:00:00Z' }),
    ];
    mockActiveSessionId = 'imp_codex_new';
    expect(await resolvedFor()).toBe('sess-chat');
  });

  it('a materialized import is an ordinary session', async () => {
    mockSessions = [
      imported('imp_codex_done', '2026-10-21T10:00:00Z', true),
      makeSession({ session_id: 'sess-chat', last_activity_at: '2026-10-01T00:00:00Z' }),
    ];
    expect(await resolvedFor()).toBe('imp_codex_done');
  });
});

// ---------------------------------------------------------------------------
// Spec 137 — maximize replaces the Files tab; the panel reaches narrow
// windows (overlay) and phones (bottom sheet) through a header button.
// ---------------------------------------------------------------------------

describe('ChatTab — maximize (docked)', () => {
  beforeEach(() => {
    resetMocks();
    setViewportWidth(1440);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
  });

  async function openDocked() {
    const view = await act(async () => renderChatTab());
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-handle'));
    });
    return view;
  }

  it('docked → maximized hides the chat and fills the right area; Restore brings back the docked width', async () => {
    const view = await openDocked();
    const chat = view.container.querySelector('[data-testid="chat-tab-conversation"]') as HTMLElement;
    expect(chat.className).not.toMatch(/(^|\s)hidden(\s|$)/);

    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-maximize'));
    });
    expect(screen.getByTestId('workspace-panel-column')).toHaveAttribute('data-maximized', 'true');
    expect(chat.className).toMatch(/(^|\s)hidden(\s|$)/);
    // Maximized Files is the old Files tab: the tree + preview explorer.
    expect(screen.getByTestId('file-explorer')).toBeInTheDocument();
    expect(screen.queryByTestId('files-view')).toBeNull();

    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-restore'));
    });
    expect(screen.getByTestId('workspace-panel-column')).toHaveAttribute('data-maximized', 'false');
    expect(chat.className).not.toMatch(/(^|\s)hidden(\s|$)/);
    expect(screen.getByTestId('files-view')).toBeInTheDocument();
  });

  it('Esc restores a maximized panel', async () => {
    await openDocked();
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-maximize'));
    });
    await act(async () => {
      fireEvent.keyDown(window, { key: 'Escape' });
    });
    expect(screen.getByTestId('workspace-panel-column')).toHaveAttribute('data-maximized', 'false');
  });

  it('the chat stays mounted while maximized (the composer keeps its draft)', async () => {
    await openDocked();
    const before = lastChatViewProps;
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-maximize'));
    });
    // ChatView was re-rendered with the same session, not torn down.
    expect(lastChatViewProps.sessionId).toBe(before.sessionId);
    expect(lastChatViewProps.projectId).toBe('proj-1');
  });

  it('is not persisted: nothing about maximize is written to storage', async () => {
    await openDocked();
    const keysBefore = Object.keys(localStorage).sort();
    const valuesBefore = keysBefore.map((k) => localStorage.getItem(k));
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-maximize'));
    });
    const keysAfter = Object.keys(localStorage).sort();
    expect(keysAfter).toEqual(keysBefore);
    expect(keysAfter.map((k) => localStorage.getItem(k))).toEqual(valuesBefore);
    // A fresh load (the in-memory store reset) lands docked.
    __resetPanelState();
    cleanup();
    await act(async () => renderChatTab());
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
    expect(screen.getByTestId('panel-handle')).toBeInTheDocument();
  });

  it('maximized Files reports the explorer\'s selection back, so Restore shows the same file', async () => {
    await openDocked();
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-maximize'));
    });
    act(() => {
      (lastFileExplorerProps?.onSelectFile as (p: string) => void)('src/app.ts');
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-restore'));
    });
    expect(lastFilesViewProps.file).toBe('src/app.ts');
  });
});

describe('ChatTab — upload from the docked Files view (gap 2)', () => {
  beforeEach(() => {
    resetMocks();
    setViewportWidth(1440);
    mockSessions = [makeSession({ session_id: 'sess-x' })];
  });
  afterEach(() => vi.unstubAllGlobals());

  it('Upload picks a file, posts it, and refreshes the tree', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const view = await act(async () => renderChatTab());
    await act(async () => {
      fireEvent.click(screen.getByTestId('panel-handle'));
    });
    const input = view.container.querySelector('input[type="file"][data-testid="panel-upload-input"]') as HTMLInputElement;
    expect(input).not.toBeNull();
    const clickSpy = vi.spyOn(input, 'click');
    fireEvent.click(screen.getByTestId('panel-upload'));
    expect(clickSpy).toHaveBeenCalled();

    const before = (lastFilesViewProps.refreshKey as number | undefined) ?? 0;
    const file = new File(['a,b'], 'data.csv', { type: 'text/csv' });
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String((fetchMock.mock.calls[0] as unknown[])[0])).toContain('/files/upload');
    expect(lastFilesViewProps.refreshKey).toBe(before + 1);
  });
});

describe('ChatTab — narrow windows and phones (gap 3)', () => {
  beforeEach(() => {
    resetMocks();
    mockSessions = [makeSession({ session_id: 'sess-x' })];
  });

  async function openOverlay() {
    const { usePanelState } = await import('../hooks/usePanelState');
    function Opener() {
      const p = usePanelState('proj-1', 'sess-x');
      return <button data-testid="opener" onClick={p.openOverlay} />;
    }
    const view = await act(async () => renderChatTab());
    render(<Opener />);
    await act(async () => {
      fireEvent.click(screen.getByTestId('opener'));
    });
    return view;
  }

  it('below the dock threshold the header button opens the panel as an overlay, with maximize', async () => {
    setViewportWidth(1000);
    await openOverlay();
    expect(screen.queryByTestId('workspace-panel-column')).toBeNull();
    const dialog = screen.getByRole('dialog');
    expect(dialog).not.toHaveAttribute('inert');
    expect(within(dialog).getByRole('tab', { name: 'Files' })).toBeInTheDocument();
    expect(within(dialog).getByTestId('files-view')).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(within(dialog).getByTestId('panel-maximize'));
    });
    expect(dialog).toHaveAttribute('data-maximized', 'true');

    // Esc first restores the overlay's width, then closes it.
    await act(async () => {
      fireEvent.keyDown(window, { key: 'Escape' });
    });
    expect(dialog).toHaveAttribute('data-maximized', 'false');
    expect(dialog).not.toHaveAttribute('inert');
    await act(async () => {
      fireEvent.keyDown(window, { key: 'Escape' });
    });
    expect(dialog).toHaveAttribute('inert');
  });

  it('on a phone it is the bottom sheet, with no maximize and no "Open full width"', async () => {
    setViewportWidth(390);
    await openOverlay();
    const dialog = screen.getByRole('dialog');
    expect(dialog).not.toHaveAttribute('inert');
    expect(within(dialog).queryByTestId('panel-maximize')).toBeNull();
    expect(lastFilesViewProps.onOpenInFiles).toBeUndefined();
  });
});

describe('ChatTab — /automate hand-off from Tasks (decision 2)', () => {
  beforeEach(() => resetMocks());

  it('passes route.draftSkill to ChatView with the draft, and clears both once consumed', async () => {
    const { setRoute } = await act(async () =>
      renderChatTab({ draft: 'Every weekday at 9am, summarize new GitHub issues', draftSkill: 'automate' }),
    );
    expect(lastChatViewProps.initialDraft).toBe('Every weekday at 9am, summarize new GitHub issues');
    expect(lastChatViewProps.initialSkill).toBe('automate');
    setRoute.mockClear();
    act(() => {
      (lastChatViewProps.onDraftConsumed as () => void)();
    });
    const updater = setRoute.mock.calls[0][0] as (prev: Route) => Route;
    const next = updater(makeRoute({ sessionId: 'sess-x', draft: 'x', draftSkill: 'automate' })) as Extract<
      Route,
      { name: 'project' }
    >;
    expect(next.draft).toBeUndefined();
    expect(next.draftSkill).toBeUndefined();
  });
});

// Spec 065 §5.9 on a phone: 查看 in the undo sheet opens the file in the
// overlay drawer through the same opener a chat path click uses (it fetches,
// then routes), not a bare route write that leaves the drawer empty.
describe('ChatTab — the undo sheet hands 查看 to the opener', () => {
  beforeEach(() => resetMocks());

  function renderWithOpener(openPath: (path: string) => void) {
    const route = makeRoute({ sessionId: 'sess-x' });
    const setRoute = vi.fn();
    const view = render(
      <OpenPathContext.Provider value={openPath}>
        <ChatTab project={PROJECT} agentStatus="idle" agents={[]} route={route} setRoute={setRoute} />
      </OpenPathContext.Provider>,
    );
    return { ...view, setRoute };
  }

  it('on a phone, 查看 closes the sheet and opens the file through the opener', async () => {
    setViewportWidth(390);
    const openPath = vi.fn();
    await act(async () => {
      renderWithOpener(openPath);
    });
    await act(async () => {
      (lastChatViewProps.onUndoTurn as (t: unknown) => void)({ sessionId: 'sess-x', fromTs: '2026-11-04T10:00:00Z', toTs: null });
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('stub-undo-view'));
    });
    expect(openPath).toHaveBeenCalledWith('docs/plan.md');
    expect(screen.queryByTestId('stub-undo-view')).toBeNull();
  });

  it('docked, 查看 still routes to the panel (unchanged)', async () => {
    setViewportWidth(1440);
    const openPath = vi.fn();
    let setRoute: ReturnType<typeof vi.fn> = vi.fn();
    await act(async () => {
      ({ setRoute } = renderWithOpener(openPath));
    });
    await act(async () => {
      (lastChatViewProps.onUndoTurn as (t: unknown) => void)({ sessionId: 'sess-x', fromTs: '2026-11-04T10:00:00Z', toTs: null });
    });
    setRoute.mockClear();
    await act(async () => {
      fireEvent.click(screen.getByTestId('stub-undo-view'));
    });
    expect(openPath).not.toHaveBeenCalled();
    const updater = setRoute.mock.calls[0][0] as (prev: Route) => Route;
    expect((updater(makeRoute({ sessionId: 'sess-x' })) as { previewPath?: string }).previewPath).toBe('docs/plan.md');
  });
});

// The undo sheet's ✕ closes a sheet, not a preview: its own label.
describe('ChatTab — the phone undo sheet close label', () => {
  beforeEach(() => resetMocks());

  it('reads Close, not Close preview', async () => {
    setViewportWidth(390);
    await act(async () => {
      renderChatTab();
    });
    await act(async () => {
      (lastChatViewProps.onUndoTurn as (t: unknown) => void)({ sessionId: 'sess-x', fromTs: '2026-11-04T10:00:00Z', toTs: null });
    });
    const sheet = screen.getByTestId('stub-undo-view').closest('[role="dialog"]') as HTMLElement;
    expect(within(sheet).getByRole('button', { name: 'Close' })).toBeInTheDocument();
    expect(within(sheet).queryByRole('button', { name: 'Close preview' })).toBeNull();
  });
});

// Spec 156 §5.6: editing a sent message opens the same sheet, retitled.
describe('ChatTab — the sheet opened by an edit', () => {
  beforeEach(() => resetMocks());

  it('is titled 恢复之后改过的文件 / Restore files changed after this; the undo sheet keeps its own title', async () => {
    setViewportWidth(1440);
    await act(async () => {
      renderChatTab();
    });
    const open = lastChatViewProps.onUndoTurn as (t: unknown) => void;
    await act(async () => {
      open({ sessionId: 'sess-x', fromTs: '2026-11-04T10:00:00Z', toTs: '2026-11-04T11:00:00Z', edit: { onProceed: vi.fn(), onSendOnly: vi.fn() } });
    });
    expect(screen.getByText('Restore files changed after this')).toBeInTheDocument();
    await act(async () => {
      open({ sessionId: 'sess-x', fromTs: '2026-11-04T10:00:00Z', toTs: null });
    });
    expect(screen.queryByText('Restore files changed after this')).toBeNull();
    expect(screen.getByText('Undo this turn')).toBeInTheDocument();
  });
});
