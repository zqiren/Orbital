// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { SessionListEntry } from '../types';

afterEach(() => cleanup());

// ---------------------------------------------------------------------------
// Mock useSessions and useSession before importing the component
// ---------------------------------------------------------------------------

let mockSessions: SessionListEntry[] = [];
const renameSessionMock = vi.fn(async () => 'renamed');
const deleteSessionMock = vi.fn(async () => undefined);
const pinSessionMock = vi.fn(async () => true);

vi.mock('../hooks/useSessions', () => ({
  useSessions: (_projectId: string | null) => ({
    sessions: mockSessions,
    loading: false,
    error: null,
    refresh: vi.fn(),
    renameSession: renameSessionMock,
    pinSession: pinSessionMock,
    deleteSession: deleteSessionMock,
  }),
}));

// SessionSidebar is CONTROLLED — it no longer uses useSession internally
// (selection is driven by the selectedSessionId prop, owned by ChatTab). No
// useSession mock is needed.

// useSessions internally uses useWebSocket — mock to prevent import errors.
vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    on: vi.fn(),
    off: vi.fn(),
    connectionState: 'connected',
    subscribe: vi.fn(),
  }),
}));

// Spec 154: checking a reply POSTs to the daemon through api().
const apiMock = vi.fn(async (..._args: unknown[]) => ({ status: 'ok' }));
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiMock(...args),
  isRelayMode: false,
}));

import { SessionSidebar } from './SessionSidebar';
import { __resetCheckedMarks } from '../hooks/useSeenReplies';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function resetMocks() {
  mockSessions = [];
  apiMock.mockClear();
  __resetCheckedMarks();
  renameSessionMock.mockClear();
  pinSessionMock.mockClear();
  deleteSessionMock.mockClear();
}

// ---------------------------------------------------------------------------
// Unified session list (no archived bucket — every session in one list)
// ---------------------------------------------------------------------------

describe('SessionSidebar — unified session list', () => {
  it('all sessions (running/waiting/pending_approval/idle) appear in the single list', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-run', status: 'running', session_uuid: 'u1' }),
      makeSession({ session_id: 'sess-wait', status: 'waiting', session_uuid: 'u2' }),
      makeSession({ session_id: 'sess-blocked', status: 'pending_approval', session_uuid: 'u3' }),
      makeSession({ session_id: 'sess-idle', status: 'idle', session_uuid: 'u4' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);

    const list = screen.getByTestId('session-list');
    expect(list).toBeInTheDocument();

    expect(screen.getByTestId('session-list-item-sess-run')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-wait')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-blocked')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-idle')).toBeInTheDocument();
  });

  it('error and new_session statuses appear in the unified list (none filtered out)', () => {
    // DO-NOT #12: ALL sessions must appear. There is no archived bucket and no
    // active/archived split — every status renders in the one list.
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-error', status: 'error', session_uuid: 'ue' }),
      makeSession({ session_id: 'sess-new', status: 'new_session', session_uuid: 'un' }),
      makeSession({ session_id: 'sess-idle', status: 'idle', session_uuid: 'ui' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);

    expect(screen.getByTestId('session-list-item-sess-error')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-new')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-idle')).toBeInTheDocument();
    // Count reflects ALL sessions.
    expect(screen.getByTestId('session-active-count')).toHaveTextContent('3');
  });

  it('never renders an archived section or any "archived" affordance', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-a', status: 'idle', session_uuid: 'u1' }),
      makeSession({ session_id: 'sess-b', status: 'error', session_uuid: 'u2' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);

    expect(screen.queryByTestId('session-archived-section')).toBeNull();
    expect(screen.queryByTestId('session-archived-toggle')).toBeNull();
    expect(screen.queryByTestId('session-archived-list')).toBeNull();
    expect(screen.queryByText(/archived/i)).toBeNull();
  });

  it('header count reflects the total number of sessions', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 's1', status: 'running', session_uuid: 'u1' }),
      makeSession({ session_id: 's2', status: 'idle', session_uuid: 'u2' }),
      makeSession({ session_id: 's3', status: 'error', session_uuid: 'u3' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);
    expect(screen.getByTestId('session-active-count')).toHaveTextContent('3');
  });

  it('sorts sessions by last-activity descending (most recent first)', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-old', session_uuid: 'u1', last_activity_at: '2026-01-01T00:00:00Z' }),
      makeSession({ session_id: 'sess-newest', session_uuid: 'u2', last_activity_at: '2026-06-01T00:00:00Z' }),
      makeSession({ session_id: 'sess-mid', session_uuid: 'u3', last_activity_at: '2026-03-15T00:00:00Z' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);

    const items = screen.getAllByTestId(/^session-list-item-/);
    expect(items.map((el) => el.getAttribute('data-testid'))).toEqual([
      'session-list-item-sess-newest',
      'session-list-item-sess-mid',
      'session-list-item-sess-old',
    ]);
  });

  // Spec 081 — a queued session is listed like any other row: it sorts to the
  // top on its enqueue time and highlights when selected. No new row type.
  it('a queued session sorts to the top on its enqueue time', () => {
    resetMocks();
    mockSessions = [
      makeSession({
        session_id: 'sess-holder', session_uuid: 'u1', status: 'running',
        last_activity_at: '2026-06-01T00:00:00Z',
      }),
      makeSession({
        session_id: 'sess-queued', session_uuid: 'sess-queued', status: 'queued',
        name: 'Say hello', last_activity_at: '2026-06-01T00:05:00Z',
      }),
    ];

    render(<SessionSidebar projectId="proj-1" />);

    const items = screen.getAllByTestId(/^session-list-item-/);
    expect(items.map((el) => el.getAttribute('data-testid'))).toEqual([
      'session-list-item-sess-queued',
      'session-list-item-sess-holder',
    ]);
    expect(screen.getByText('Say hello')).toBeInTheDocument();
  });

  it('highlights the queued session when it is the selected one', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-holder', session_uuid: 'u1', status: 'running' }),
      makeSession({ session_id: 'sess-queued', session_uuid: 'sess-queued', status: 'queued' }),
    ];

    render(<SessionSidebar projectId="proj-1" selectedSessionId="sess-queued" />);

    expect(screen.getByTestId('session-list-item-sess-queued')).toHaveAttribute(
      'aria-selected', 'true',
    );
    expect(screen.getByTestId('session-list-item-sess-holder')).toHaveAttribute(
      'aria-selected', 'false',
    );
  });
});

// ---------------------------------------------------------------------------
// + new session button
// ---------------------------------------------------------------------------

describe('SessionSidebar — new session row (item 157)', () => {
  it('renders New session as the first row of the list, not at the bottom', () => {
    resetMocks();
    mockSessions = [makeSession({ session_id: 'a', session_uuid: 'ua', name: 'Chat' })];
    render(<SessionSidebar projectId="proj-1" />);
    const button = screen.getByTestId('session-new-button');
    expect(button).toHaveTextContent('New session');
    // The row carries a + glyph, so the copy has no leading "+".
    expect(button.textContent).not.toMatch(/\+/);
    const list = screen.getByTestId('session-list');
    expect(list.firstElementChild).toBe(screen.getByTestId('session-new-row'));
    expect(list.contains(button)).toBe(true);
  });

  it('sits on the white tint (--card / --elevated), sticky while the list scrolls', () => {
    resetMocks();
    render(<SessionSidebar projectId="proj-1" />);
    const button = screen.getByTestId('session-new-button');
    expect(button.className).toContain('bg-card');
    expect(button.className).toContain('dark:bg-elevated');
    expect(button.className).not.toMatch(/shadow/);
    expect(screen.getByTestId('session-new-row').className).toContain('sticky');
  });

  it('calls onNewSession when clicked', async () => {
    resetMocks();
    const onNewSession = vi.fn();
    render(<SessionSidebar projectId="proj-1" onNewSession={onNewSession} />);
    await userEvent.click(screen.getByTestId('session-new-button'));
    expect(onNewSession).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// Queue-origin session present in list
// ---------------------------------------------------------------------------

describe('SessionSidebar — ALL sessions included regardless of origin', () => {
  it('a queue-origin session appears in the active list', () => {
    resetMocks();
    mockSessions = [
      makeSession({
        session_id: 'sess-manual',
        status: 'idle',
        origin: 'chat',
        session_uuid: 'um',
      }),
      makeSession({
        session_id: 'sess-queue',
        status: 'running',
        origin: 'queue',
        session_uuid: 'uq',
      }),
    ];

    render(<SessionSidebar projectId="proj-1" />);

    expect(screen.getByTestId('session-list-item-sess-manual')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-queue')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Row selection
// ---------------------------------------------------------------------------

describe('SessionSidebar — session selection (controlled)', () => {
  it('onSessionSelect callback fires with the session_id when a row is clicked', async () => {
    resetMocks();
    const onSessionSelect = vi.fn();
    mockSessions = [
      makeSession({ session_id: 'sess-cb', status: 'idle', session_uuid: 'uc' }),
    ];

    render(<SessionSidebar projectId="proj-1" onSessionSelect={onSessionSelect} />);
    await userEvent.click(screen.getByTestId('session-list-item-sess-cb'));
    expect(onSessionSelect).toHaveBeenCalledWith('sess-cb');
  });

  it('highlights the row named by the selectedSessionId prop (controlled)', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-a', status: 'idle', session_uuid: 'ua' }),
      makeSession({ session_id: 'sess-b', status: 'running', session_uuid: 'ub' }),
    ];

    render(<SessionSidebar projectId="proj-1" selectedSessionId="sess-b" />);

    // SessionListItem sets aria-selected from its `selected` prop.
    expect(screen.getByTestId('session-list-item-sess-a')).toHaveAttribute(
      'aria-selected',
      'false',
    );
    expect(screen.getByTestId('session-list-item-sess-b')).toHaveAttribute(
      'aria-selected',
      'true',
    );
  });

  it('highlights no row when selectedSessionId is undefined', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-a', status: 'idle', session_uuid: 'ua' }),
      makeSession({ session_id: 'sess-b', status: 'running', session_uuid: 'ub' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);

    expect(screen.getByTestId('session-list-item-sess-a')).toHaveAttribute(
      'aria-selected',
      'false',
    );
    expect(screen.getByTestId('session-list-item-sess-b')).toHaveAttribute(
      'aria-selected',
      'false',
    );
  });

  it('updates the highlight when selectedSessionId changes (rerender)', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-a', status: 'idle', session_uuid: 'ua' }),
      makeSession({ session_id: 'sess-b', status: 'running', session_uuid: 'ub' }),
    ];

    const { rerender } = render(
      <SessionSidebar projectId="proj-1" selectedSessionId="sess-a" />,
    );
    expect(screen.getByTestId('session-list-item-sess-a')).toHaveAttribute(
      'aria-selected',
      'true',
    );

    rerender(<SessionSidebar projectId="proj-1" selectedSessionId="sess-b" />);
    expect(screen.getByTestId('session-list-item-sess-a')).toHaveAttribute(
      'aria-selected',
      'false',
    );
    expect(screen.getByTestId('session-list-item-sess-b')).toHaveAttribute(
      'aria-selected',
      'true',
    );
  });
});

// ---------------------------------------------------------------------------
// Display names (label = name → session_id fallback)
// ---------------------------------------------------------------------------

describe('SessionSidebar — display names', () => {
  it('shows the human-readable name instead of the session_id when present', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess_8badf00d', session_uuid: 'u1', name: 'Fix the deploy' }),
    ];
    render(<SessionSidebar projectId="proj-1" />);
    expect(screen.getByTestId('session-name')).toHaveTextContent('Fix the deploy');
    expect(screen.getByTestId('session-name')).not.toHaveTextContent('sess_8badf00d');
  });
});

// ---------------------------------------------------------------------------
// Delete → navigation callback
// ---------------------------------------------------------------------------

describe('SessionSidebar — delete + navigation', () => {
  it('calls deleteSession and onSessionDeleted with the remaining sessions', async () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-keep', session_uuid: 'uk', last_activity_at: '2026-06-01T00:00:00Z' }),
      makeSession({ session_id: 'sess-del', session_uuid: 'ud', last_activity_at: '2026-05-01T00:00:00Z' }),
    ];
    const onSessionDeleted = vi.fn();

    render(<SessionSidebar projectId="proj-1" onSessionDeleted={onSessionDeleted} />);

    // Open the menu for the row to delete, click Delete, confirm.
    const row = screen.getByTestId('session-list-item-sess-del');
    await userEvent.pointer({ keys: '[MouseRight]', target: row });
    await userEvent.click(screen.getByTestId('session-action-delete'));
    await userEvent.click(screen.getByTestId('session-delete-confirm-button'));

    expect(deleteSessionMock).toHaveBeenCalledWith('sess-del');
    expect(onSessionDeleted).toHaveBeenCalledTimes(1);
    const [deletedId, remaining] = onSessionDeleted.mock.calls[0];
    expect(deletedId).toBe('sess-del');
    expect((remaining as SessionListEntry[]).map((s) => s.session_id)).toEqual(['sess-keep']);
  });
});

// ---------------------------------------------------------------------------
// Pin to top (BACKLOG spec 067)
// ---------------------------------------------------------------------------

describe('SessionSidebar — pinned sessions', () => {
  function ids() {
    return screen
      .getAllByTestId(/^session-list-item-/)
      .map((el) => el.getAttribute('data-testid'));
  }

  it('pinned sessions lead the list even when far less recent', () => {
    resetMocks();
    mockSessions = [
      makeSession({
        session_id: 'fresh', session_uuid: 'u-fresh',
        last_activity_at: '2026-08-20T12:00:00Z',
      }),
      makeSession({
        session_id: 'stale-pinned', session_uuid: 'u-stale',
        last_activity_at: '2026-01-01T00:00:00Z', pinned: true,
      }),
    ];
    render(<SessionSidebar projectId="p1" />);
    expect(ids()).toEqual([
      'session-list-item-stale-pinned',
      'session-list-item-fresh',
    ]);
  });

  it('a pin outranks a RUNNING session (the decision recorded in the sort)', () => {
    resetMocks();
    mockSessions = [
      makeSession({
        session_id: 'running-now', session_uuid: 'u-run', status: 'running',
        last_activity_at: '2026-08-20T12:00:00Z',
      }),
      makeSession({
        session_id: 'pinned-idle', session_uuid: 'u-pin',
        last_activity_at: '2026-02-02T00:00:00Z', pinned: true,
      }),
    ];
    render(<SessionSidebar projectId="p1" />);
    expect(ids()[0]).toBe('session-list-item-pinned-idle');
  });

  it('pinned rows stay activity-sorted among themselves', () => {
    resetMocks();
    mockSessions = [
      makeSession({
        session_id: 'pin-old', session_uuid: 'u-po',
        last_activity_at: '2026-03-01T00:00:00Z', pinned: true,
      }),
      makeSession({
        session_id: 'pin-new', session_uuid: 'u-pn',
        last_activity_at: '2026-07-01T00:00:00Z', pinned: true,
      }),
      makeSession({
        session_id: 'plain', session_uuid: 'u-pl',
        last_activity_at: '2026-08-01T00:00:00Z',
      }),
    ];
    render(<SessionSidebar projectId="p1" />);
    expect(ids()).toEqual([
      'session-list-item-pin-new',
      'session-list-item-pin-old',
      'session-list-item-plain',
    ]);
  });

  it('renders a divider under the pinned block, and none when nothing is pinned', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'a', session_uuid: 'ua', pinned: true }),
      makeSession({ session_id: 'b', session_uuid: 'ub' }),
    ];
    const { unmount } = render(<SessionSidebar projectId="p1" />);
    expect(screen.getAllByTestId('session-pin-divider')).toHaveLength(1);
    unmount();

    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'a', session_uuid: 'ua' }),
      makeSession({ session_id: 'b', session_uuid: 'ub' }),
    ];
    render(<SessionSidebar projectId="p1" />);
    expect(screen.queryByTestId('session-pin-divider')).toBeNull();
  });

  it('draws no divider when EVERY session is pinned (it would separate nothing)', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'a', session_uuid: 'ua', pinned: true }),
      makeSession({ session_id: 'b', session_uuid: 'ub', pinned: true }),
    ];
    render(<SessionSidebar projectId="p1" />);
    expect(screen.queryByTestId('session-pin-divider')).toBeNull();
  });

  it('the row menu pins an unpinned session and unpins a pinned one', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = [makeSession({ session_id: 's1', session_uuid: 'u1' })];
    const { unmount } = render(<SessionSidebar projectId="p1" />);
    await user.click(screen.getByTestId('session-three-dot-trigger'));
    expect(screen.getByTestId('session-action-pin')).toHaveTextContent('Pin to top');
    await user.click(screen.getByTestId('session-action-pin'));
    expect(pinSessionMock).toHaveBeenCalledWith('s1', true);
    unmount();

    resetMocks();
    mockSessions = [makeSession({ session_id: 's1', session_uuid: 'u1', pinned: true })];
    render(<SessionSidebar projectId="p1" />);
    await user.click(screen.getByTestId('session-three-dot-trigger'));
    expect(screen.getByTestId('session-action-pin')).toHaveTextContent('Unpin');
    await user.click(screen.getByTestId('session-action-pin'));
    expect(pinSessionMock).toHaveBeenCalledWith('s1', false);
  });

  it('a pinned RESTING row shows the pin glyph; a running one keeps its status glyph', () => {
    resetMocks();
    mockSessions = [makeSession({ session_id: 'idle-pin', session_uuid: 'u1', pinned: true })];
    const { unmount } = render(<SessionSidebar projectId="p1" />);
    expect(screen.getByTestId('session-pin-glyph')).toBeInTheDocument();
    unmount();

    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'run-pin', session_uuid: 'u2', pinned: true, status: 'running' }),
    ];
    render(<SessionSidebar projectId="p1" />);
    expect(screen.queryByTestId('session-pin-glyph')).toBeNull();
    expect(screen.getByTestId('session-status-glyph').textContent).not.toBe('');
  });
});

// ---------------------------------------------------------------------------
// Spec 066 4b — automation sessions are grouped apart from chats
// ---------------------------------------------------------------------------

describe('SessionSidebar — filter and automation groups', () => {
  function ids() {
    return screen
      .getAllByTestId(/^session-list-item-/)
      .map((el) => el.getAttribute('data-testid'));
  }

  afterEach(() => {
    try {
      localStorage.clear();
    } catch {
      /* storage unavailable in this environment */
    }
  });

  const daily = (id: string, at: string, extra: Partial<SessionListEntry> = {}) =>
    makeSession({
      session_id: id, session_uuid: `u-${id}`, trigger_type: 'schedule',
      name: "[Triggered by schedule 'Daily' (Every day)]", last_activity_at: at, ...extra,
    });

  // Two chats around three runs of one automation, plus a lone queue item.
  const mixed = () => [
    daily('run-3', '2026-09-10T00:00:00Z'),
    makeSession({ session_id: 'chat-new', session_uuid: 'u-cn', name: 'Fix the bug', last_activity_at: '2026-09-09T00:00:00Z' }),
    daily('run-2', '2026-09-08T00:00:00Z'),
    makeSession({ session_id: 'queue-one', session_uuid: 'u-q', origin: 'queue', last_activity_at: '2026-09-05T00:00:00Z' }),
    makeSession({ session_id: 'chat-old', session_uuid: 'u-co', origin: 'chat', name: 'Plan the launch', last_activity_at: '2026-09-01T00:00:00Z' }),
    daily('run-1', '2026-08-01T00:00:00Z'),
  ];

  it("folds an automation's runs into ONE collapsed row, placed at its latest run — not parked below the chats", () => {
    resetMocks();
    mockSessions = mixed();
    render(<SessionSidebar projectId="p1" />);

    // Runs are folded away; the lone queue item is a plain row (a group of one
    // has nothing to collapse).
    expect(ids()).toEqual([
      'session-list-item-chat-new',
      'session-list-item-queue-one',
      'session-list-item-chat-old',
    ]);
    const group = screen.getByTestId('session-automation-group-toggle');
    expect(group).toHaveTextContent('Daily');
    expect(group).toHaveTextContent('3 runs');
    expect(group).toHaveAttribute('aria-expanded', 'false');
    // The group leads the list: its latest run is the most recent activity.
    // (After the New session row, which is always first — item 157.)
    const list = screen.getByTestId('session-list');
    expect(list.firstElementChild).toBe(screen.getByTestId('session-new-row'));
    expect(list.children[1]).toBe(screen.getByTestId('session-automation-group'));
    // Every session is still counted (no hidden bucket).
    expect(screen.getByTestId('session-active-count')).toHaveTextContent('6');
  });

  it('expanding a group lists its runs, most recent first', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = mixed();
    render(<SessionSidebar projectId="p1" />);

    await user.click(screen.getByTestId('session-automation-group-toggle'));
    expect(screen.getByTestId('session-automation-group-toggle')).toHaveAttribute('aria-expanded', 'true');
    const runs = within(screen.getByTestId('session-automation-group'))
      .getAllByTestId(/^session-list-item-/)
      .map((el) => el.getAttribute('data-testid'));
    expect(runs).toEqual(['session-list-item-run-3', 'session-list-item-run-2', 'session-list-item-run-1']);
  });

  it('a collapsed group still shows its active runs and the selected one', () => {
    resetMocks();
    mockSessions = [...mixed(), daily('run-live', '2026-07-01T00:00:00Z', { status: 'running' })];
    render(<SessionSidebar projectId="p1" selectedSessionId="run-2" />);

    expect(screen.getByTestId('session-automation-group-toggle')).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByTestId('session-list-item-run-2')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-run-live')).toBeInTheDocument();
    expect(screen.queryByTestId('session-list-item-run-3')).toBeNull();
  });

  it('a collapsed group still shows a run whose worker is running (spec 102)', () => {
    resetMocks();
    mockSessions = [
      ...mixed(),
      daily('run-worker', '2026-07-02T00:00:00Z', { status: 'idle', worker_running: true }),
    ];
    render(<SessionSidebar projectId="p1" selectedSessionId="chat-new" />);

    expect(screen.getByTestId('session-automation-group-toggle')).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByTestId('session-list-item-run-worker')).toBeInTheDocument();
    expect(screen.queryByTestId('session-list-item-run-3')).toBeNull();
  });

  it('Chats shows only conversations; Automations shows only automation sessions', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = mixed();
    render(<SessionSidebar projectId="p1" />);

    await user.click(screen.getByTestId('session-filter-toggle'));
    await user.click(screen.getByTestId('session-filter-chats'));
    expect(ids()).toEqual(['session-list-item-chat-new', 'session-list-item-chat-old']);
    expect(screen.queryByTestId('session-automation-group')).toBeNull();
    expect(screen.getByTestId('session-active-count')).toHaveTextContent('2');

    await user.click(screen.getByTestId('session-filter-automations'));
    expect(ids()).toEqual(['session-list-item-queue-one']);
    expect(screen.getByTestId('session-automation-group-toggle')).toHaveTextContent('3 runs');
    expect(screen.getByTestId('session-active-count')).toHaveTextContent('4');
  });

  it('recognises a renamed-away trigger session by its trigger_type', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'a', session_uuid: 'ua', trigger_type: 'file_watch', name: 'Renamed' }),
      makeSession({ session_id: 'b', session_uuid: 'ub', name: 'Chat' }),
    ];
    render(<SessionSidebar projectId="p1" />);
    await user.click(screen.getByTestId('session-filter-toggle'));
    await user.click(screen.getByTestId('session-filter-automations'));
    expect(ids()).toEqual(['session-list-item-a']);
  });

  it('falls back to the name prefix when the backend sends no trigger_type', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'a', session_uuid: 'ua', name: "[Triggered by file_watch 'Inbox']" }),
      makeSession({ session_id: 'b', session_uuid: 'ub', name: 'Chat' }),
    ];
    render(<SessionSidebar projectId="p1" />);
    await user.click(screen.getByTestId('session-filter-toggle'));
    await user.click(screen.getByTestId('session-filter-chats'));
    expect(ids()).toEqual(['session-list-item-b']);
  });

  it('an empty filter says so instead of showing a blank list', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = [makeSession({ session_id: 'a', session_uuid: 'ua', name: 'Chat' })];
    render(<SessionSidebar projectId="p1" />);
    await user.click(screen.getByTestId('session-filter-toggle'));
    await user.click(screen.getByTestId('session-filter-automations'));
    expect(screen.getByTestId('session-empty')).toHaveTextContent('No automation runs yet');
  });

  // Item 157 (2026-11-04, reverses the 2026-10-28 "always shown" call): the
  // filter hides behind a bare chevron on the "Sessions · N" header.
  it('the filter is hidden by default; the header chevron expands the boxed segmented control', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = [makeSession({ session_id: 'a', session_uuid: 'ua', name: 'Chat' })];
    render(<SessionSidebar projectId="p1" />);
    expect(screen.queryByRole('tablist', { name: 'Filter sessions' })).toBeNull();
    const toggle = screen.getByTestId('session-filter-toggle');
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(toggle).toHaveAttribute('title', 'Filter');

    await user.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    const tablist = screen.getByRole('tablist', { name: 'Filter sessions' });
    for (const c of ['rounded-lg', 'border', 'bg-nav', 'p-0.5']) expect(tablist.className).toContain(c);
    expect(tablist.parentElement?.className).toContain('px-4');
    expect(screen.getByTestId('session-sidebar-header').className).toContain('px-4');
    const all = screen.getByTestId('session-filter-all');
    expect(all).toHaveAttribute('aria-selected', 'true');
    expect(all.className).toContain('bg-segment-on');

    await user.click(toggle);
    expect(screen.queryByRole('tablist', { name: 'Filter sessions' })).toBeNull();
  });

  it('the header reads "Sessions · 6": sentence case, the count without parentheses', () => {
    resetMocks();
    mockSessions = mixed();
    render(<SessionSidebar projectId="p1" />);
    const header = screen.getByTestId('session-sidebar-header');
    expect(header.textContent).toBe('Sessions·6');
    expect(header.innerHTML).not.toMatch(/uppercase|font-mono/);
  });

  it('a filter left on while closed reads in the header, chevron in accent', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = mixed();
    render(<SessionSidebar projectId="p1" />);
    const toggle = screen.getByTestId('session-filter-toggle');
    expect(toggle.className).not.toContain('text-accent');
    await user.click(toggle);
    await user.click(screen.getByTestId('session-filter-automations'));
    // Open: the control shows the choice, the header keeps the plain count.
    expect(screen.queryByTestId('session-header-filter')).toBeNull();
    await user.click(toggle);
    expect(screen.getByTestId('session-sidebar-header').textContent).toBe('Sessions·Automations·4');
    expect(toggle.className).toContain('text-accent');
  });

  it('remembers the filter open/closed state across remounts', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = mixed();
    const { unmount } = render(<SessionSidebar projectId="p1" />);
    await user.click(screen.getByTestId('session-filter-toggle'));
    unmount();
    render(<SessionSidebar projectId="p1" />);
    expect(screen.getByTestId('session-filter-toggle')).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('tablist', { name: 'Filter sessions' })).toBeInTheDocument();
  });

  it('remembers the chosen filter across remounts', async () => {
    const user = userEvent.setup();
    resetMocks();
    mockSessions = mixed();
    const { unmount } = render(<SessionSidebar projectId="p1" />);
    await user.click(screen.getByTestId('session-filter-toggle'));
    await user.click(screen.getByTestId('session-filter-chats'));
    unmount();

    render(<SessionSidebar projectId="p1" />);
    expect(screen.getByTestId('session-filter-chats')).toHaveAttribute('aria-selected', 'true');
    expect(ids()).toEqual(['session-list-item-chat-new', 'session-list-item-chat-old']);
  });

  it('a pinned automation stays in the pinned block', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'chat', session_uuid: 'u1', last_activity_at: '2026-09-09T00:00:00Z' }),
      makeSession({
        session_id: 'cron-pinned', session_uuid: 'u2', trigger_type: 'schedule',
        pinned: true, last_activity_at: '2026-01-01T00:00:00Z',
      }),
    ];
    render(<SessionSidebar projectId="p1" />);
    expect(ids()).toEqual(['session-list-item-cron-pinned', 'session-list-item-chat']);
    expect(screen.queryByTestId('session-automation-group')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Spec 108 — unread badge, its checked state moved into the daemon by spec
// 154 §5.1: rows carry `checked_reply_at`; viewing a row POSTs
// /api/v2/workbench/checked; a reply older than 14 days never badges. The
// retired per-device localStorage map is neither read nor written.
// ---------------------------------------------------------------------------

describe('SessionSidebar — unread badge (spec 108 on the daemon checked state)', () => {
  const SEEN_KEY = 'orbital:seenReplies:proj-1';
  // Recent times (inside the 14-day window): an hour ago plus `secs`.
  const BASE = Date.now() - 60 * 60 * 1000;
  const at = (secs: number) => new Date(BASE + secs * 1000).toISOString();

  function replied(id: string, replyAt: string, userAt = at(0), checked: string | null = null) {
    return makeSession({
      session_id: id,
      session_uuid: id,
      name: id,
      last_activity_at: replyAt,
      last_user_at: userAt,
      last_reply_at: replyAt,
      checked_reply_at: checked,
    });
  }

  function dotIn(id: string) {
    return within(screen.getByTestId(`session-list-item-${id}`)).queryByTestId('session-unread-dot');
  }

  function checkedPosts() {
    return apiMock.mock.calls
      .filter((c) => c[0] === '/api/v2/workbench/checked')
      .map((c) => JSON.parse(((c[1] as { body: string }).body)));
  }

  it('rows the daemon already counts as checked (its upgrade seed) show no dot', () => {
    resetMocks();
    mockSessions = [replied('a', at(10), at(0), at(10)), replied('b', at(20), at(0), at(20))];
    render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('a')).toBeNull();
    expect(dotIn('b')).toBeNull();
    expect(checkedPosts()).toEqual([]);
  });

  it('the retired per-device seen map is neither read nor written', () => {
    resetMocks();
    // An old build's map says "b" was seen; the daemon says it was not.
    localStorage.setItem(SEEN_KEY, JSON.stringify({ b: at(20) }));
    mockSessions = [replied('a', at(10)), replied('b', at(20))];
    render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).not.toBeNull();
    expect(localStorage.getItem(SEEN_KEY)).toBe(JSON.stringify({ b: at(20) }));
  });

  it('a non-selected row with a new reply renders the dot; the selected one never does', () => {
    resetMocks();
    mockSessions = [replied('a', at(10)), replied('b', at(20))];
    render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('a')).toBeNull();
    expect(dotIn('b')).not.toBeNull();
    // Viewing "a" checked its reply in the daemon.
    expect(checkedPosts()).toEqual([{ project_id: 'proj-1', session_id: 'a', reply_at: at(10) }]);
  });

  it('selecting the badged row clears its dot at once and checks it in the daemon', () => {
    resetMocks();
    mockSessions = [replied('a', at(10)), replied('b', at(20))];
    const { rerender } = render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).not.toBeNull();
    rerender(<SessionSidebar projectId="proj-1" selectedSessionId="b" />);
    expect(dotIn('b')).toBeNull();
    // Going back to "a": "b" stays read (the optimistic mark holds until the
    // refetch carries the daemon's checked_reply_at).
    rerender(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).toBeNull();
    expect(checkedPosts()).toEqual([
      { project_id: 'proj-1', session_id: 'a', reply_at: at(10) },
      { project_id: 'proj-1', session_id: 'b', reply_at: at(20) },
    ]);
  });

  it('a reply checked on another device clears the dot here', () => {
    resetMocks();
    mockSessions = [replied('a', at(10), at(0), at(10)), replied('b', at(20))];
    const { rerender } = render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).not.toBeNull();
    // The workbench.changed refetch brings the other device's check.
    mockSessions = [replied('a', at(10), at(0), at(10)), replied('b', at(20), at(0), at(20))];
    rerender(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).toBeNull();
  });

  it("a refetch that advances the SELECTED row's reply never shows a dot, and checks it", () => {
    resetMocks();
    mockSessions = [replied('a', at(10), at(0), at(10))];
    const { rerender } = render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    mockSessions = [replied('a', at(40), at(0), at(10))];
    rerender(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('a')).toBeNull();
    expect(checkedPosts()).toEqual([{ project_id: 'proj-1', session_id: 'a', reply_at: at(40) }]);
  });

  it("a refetch that advances a NON-selected row's reply lights it", () => {
    resetMocks();
    mockSessions = [replied('a', at(10), at(0), at(10)), replied('b', at(20), at(0), at(20))];
    const { rerender } = render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).toBeNull();
    mockSessions = [replied('a', at(10), at(0), at(10)), replied('b', at(50), at(0), at(20))];
    rerender(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).not.toBeNull();
  });

  it("the user's own newer message clears the dot without a write", () => {
    resetMocks();
    mockSessions = [replied('a', at(10), at(0), at(10)), replied('b', at(20), at(30))];
    render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).toBeNull();
    expect(checkedPosts()).toEqual([]);
  });

  it('a reply older than 14 days never badges (spec 154 window)', () => {
    resetMocks();
    const old = new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString();
    const older = new Date(Date.now() - 16 * 24 * 60 * 60 * 1000).toISOString();
    mockSessions = [replied('a', at(10), at(0), at(10)), replied('b', old, older)];
    render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).toBeNull();
  });

  it('a running row is never badged; it lights once it rests with a reply', () => {
    resetMocks();
    mockSessions = [
      replied('a', at(10), at(0), at(10)),
      { ...replied('b', at(20)), status: 'running' },
    ];
    const { rerender } = render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).toBeNull();
    mockSessions = [replied('a', at(10), at(0), at(10)), replied('b', at(25))];
    rerender(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).not.toBeNull();
  });

  it('rows from an older daemon (no timestamps) never badge', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'a', session_uuid: 'a', last_activity_at: at(10) }),
      makeSession({ session_id: 'b', session_uuid: 'b', last_activity_at: at(20) }),
    ];
    render(<SessionSidebar projectId="proj-1" selectedSessionId="a" />);
    expect(dotIn('b')).toBeNull();
  });

  it('a collapsed automation group shows a dot while a hidden run waits (spec 150); expanding shows the run dots', async () => {
    // The badge does not change grouping or visibility: only active and
    // selected runs escape a collapsed group.
    const user = userEvent.setup();
    resetMocks();
    const daily = (id: string, replyAt: string) => ({
      ...replied(id, replyAt),
      name: "[Triggered by schedule 'Daily' (9am)]\n\nrun",
      trigger_type: 'schedule' as const,
    });
    mockSessions = [
      replied('chat', at(10), at(0), at(10)),
      daily('run1', at(20)),
      daily('run2', at(30)),
    ];
    render(<SessionSidebar projectId="proj-1" selectedSessionId="chat" />);
    expect(screen.queryByTestId('session-list-item-run1')).toBeNull();
    expect(screen.queryByTestId('session-list-item-run2')).toBeNull();
    expect(screen.getByTestId('session-group-unread-dot')).toBeInTheDocument();
    await user.click(screen.getByTestId('session-automation-group-toggle'));
    expect(screen.queryByTestId('session-group-unread-dot')).toBeNull();
    expect(dotIn('run1')).not.toBeNull();
    expect(dotIn('run2')).not.toBeNull();
    expect(dotIn('chat')).toBeNull();
  });

  it('a collapsed group with every run checked shows no group dot', () => {
    resetMocks();
    mockSessions = [
      replied('chat', at(10), at(0), at(10)),
      { ...replied('run1', at(20), at(0), at(20)), name: "[Triggered by schedule 'Daily' (9am)]", trigger_type: 'schedule' as const },
      { ...replied('run2', at(30), at(0), at(30)), name: "[Triggered by schedule 'Daily' (9am)]", trigger_type: 'schedule' as const },
    ];
    render(<SessionSidebar projectId="proj-1" selectedSessionId="chat" />);
    expect(screen.queryByTestId('session-group-unread-dot')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Session import (contract §7): imported rows sit inline by recency, and the
// whole list is capped at the 50 most recent rows with a "Show older (n)" row.
// ---------------------------------------------------------------------------
describe('SessionSidebar — imported rows and the 50-row cap', () => {
  const at = (minutesAgo: number) => new Date(Date.UTC(2026, 9, 21, 12, 0) - minutesAgo * 60_000).toISOString();
  const rowIds = () =>
    screen.getAllByTestId(/^session-list-item-/).map((el) => el.getAttribute('data-testid')!.replace('session-list-item-', ''));

  it('imported rows sort inline with everything else by last activity', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'chat-old', session_uuid: 'u1', last_activity_at: at(30) }),
      makeSession({
        session_id: 'imp_codex_a', session_uuid: 'imp_codex_a', origin: 'imported', name: 'Imported one',
        last_activity_at: at(10),
        imported: { agent: 'codex', native_id: 'a', materialized: false, size_bytes: 1 },
      }),
      makeSession({ session_id: 'chat-new', session_uuid: 'u2', last_activity_at: at(1) }),
    ];
    render(<SessionSidebar projectId="p1" />);
    expect(rowIds()).toEqual(['chat-new', 'imp_codex_a', 'chat-old']);
    expect(screen.getAllByTestId('session-agent-mark')).toHaveLength(1);
  });

  function manySessions(n: number) {
    return Array.from({ length: n }, (_, k) => makeSession({
      session_id: `s${k}`, session_uuid: `u${k}`, name: `Chat ${k}`, last_activity_at: at(k),
    }));
  }

  it('shows the 50 most recent rows and a Show older row counting the rest', () => {
    resetMocks();
    mockSessions = manySessions(53);
    render(<SessionSidebar projectId="p1" />);
    const ids = rowIds();
    expect(ids).toHaveLength(50);
    expect(ids[0]).toBe('s0');
    expect(ids[49]).toBe('s49');
    expect(screen.getByTestId('session-show-older').textContent).toBe('Show older (3)');
    // The header still counts every session.
    expect(screen.getByTestId('session-active-count').textContent).toBe('53');
  });

  it('Show older reveals the rest', async () => {
    resetMocks();
    mockSessions = manySessions(53);
    render(<SessionSidebar projectId="p1" />);
    await userEvent.click(screen.getByTestId('session-show-older'));
    expect(rowIds()).toHaveLength(53);
    expect(screen.queryByTestId('session-show-older')).toBeNull();
  });

  it('no Show older row at 50 or fewer', () => {
    resetMocks();
    mockSessions = manySessions(50);
    render(<SessionSidebar projectId="p1" />);
    expect(rowIds()).toHaveLength(50);
    expect(screen.queryByTestId('session-show-older')).toBeNull();
  });

  it('the reveal is per project: another project starts capped again', async () => {
    resetMocks();
    mockSessions = manySessions(53);
    const { rerender } = render(<SessionSidebar projectId="p1" />);
    await userEvent.click(screen.getByTestId('session-show-older'));
    rerender(<SessionSidebar projectId="p2" />);
    expect(rowIds()).toHaveLength(50);
    expect(screen.getByTestId('session-show-older')).toBeTruthy();
  });

  it('pinned rows are never capped away', () => {
    resetMocks();
    mockSessions = [
      ...manySessions(55),
      makeSession({ session_id: 'pinned-old', session_uuid: 'up', pinned: true, last_activity_at: at(500) }),
    ];
    render(<SessionSidebar projectId="p1" />);
    const ids = rowIds();
    expect(ids[0]).toBe('pinned-old');
    expect(ids).toHaveLength(51);
    expect(screen.getByTestId('session-show-older').textContent).toBe('Show older (5)');
  });

  it('a capped-away automation group counts its runs in n', () => {
    resetMocks();
    const runs = Array.from({ length: 4 }, (_, k) => makeSession({
      session_id: `run${k}`, session_uuid: `ur${k}`, name: "[Triggered by schedule 'Nightly' (Every day)]", trigger_type: 'schedule',
      last_activity_at: at(100 + k),
    }));
    mockSessions = [...manySessions(50), ...runs];
    render(<SessionSidebar projectId="p1" />);
    expect(screen.getByTestId('session-show-older').textContent).toBe('Show older (4)');
  });

  it('zh copy', async () => {
    resetMocks();
    mockSessions = manySessions(52);
    window.localStorage.setItem('orbital.locale', 'zh');
    try {
      const { LocaleProvider } = await import('../i18n/LocaleContext');
      render(<LocaleProvider><SessionSidebar projectId="p1" /></LocaleProvider>);
      expect(screen.getByTestId('session-show-older').textContent).toBe('显示更早的 2 条');
    } finally {
      window.localStorage.removeItem('orbital.locale');
    }
  });

  it('zh header, New session row and filter tooltip (item 157)', async () => {
    resetMocks();
    mockSessions = manySessions(3);
    window.localStorage.setItem('orbital.locale', 'zh');
    window.localStorage.setItem('orbital:sessionSidebar.filter', 'chats');
    try {
      const { LocaleProvider } = await import('../i18n/LocaleContext');
      render(<LocaleProvider><SessionSidebar projectId="p1" /></LocaleProvider>);
      expect(screen.getByTestId('session-sidebar-header').textContent).toBe('会话·对话·3');
      expect(screen.getByTestId('session-filter-toggle')).toHaveAttribute('title', '筛选');
      expect(screen.getByTestId('session-new-button').textContent).toBe('新建会话');
    } finally {
      window.localStorage.removeItem('orbital.locale');
    }
  });
});

// ---------------------------------------------------------------------------
// Spec 137 (the folder shell): the list lives in the project column, which
// stays while a page fills the right area.
// ---------------------------------------------------------------------------

describe('SessionSidebar — in the project column', () => {
  afterEach(() => cleanup());

  it('with a page open, the viewed session is only "was here": no selected fill, secondary text', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-a', session_uuid: 'ua' }),
      makeSession({ session_id: 'sess-b', session_uuid: 'ub' }),
    ];
    const { rerender } = render(<SessionSidebar projectId="proj-1" selectedSessionId="sess-a" />);
    const row = () => screen.getByTestId('session-list-item-sess-a');
    expect(row()).toHaveAttribute('aria-selected', 'true');
    expect(row()).not.toHaveAttribute('data-was-here');
    expect(within(row()).getByTestId('session-name').className).toContain('text-primary');

    rerender(<SessionSidebar projectId="proj-1" selectedSessionId="sess-a" pageOpen />);
    expect(row()).toHaveAttribute('aria-selected', 'false');
    expect(row()).toHaveAttribute('data-was-here', 'true');
    expect(row().getAttribute('style') ?? '').not.toContain('session-selected');
    const name = within(row()).getByTestId('session-name');
    expect(name.className).toContain('text-secondary');
    expect(name.className).toContain('font-normal');
    expect(screen.getByTestId('session-list-item-sess-b')).not.toHaveAttribute('data-was-here');
  });

  it('clicking the "was here" row still selects it (that is the way back to the chat)', async () => {
    resetMocks();
    mockSessions = [makeSession({ session_id: 'sess-a', session_uuid: 'ua' })];
    const onSelect = vi.fn();
    render(<SessionSidebar projectId="proj-1" selectedSessionId="sess-a" pageOpen onSessionSelect={onSelect} />);
    await userEvent.click(screen.getByTestId('session-list-item-sess-a'));
    expect(onSelect).toHaveBeenCalledWith('sess-a');
  });

  it('owns no width, border or inset shadow — the column does', () => {
    resetMocks();
    render(<SessionSidebar projectId="proj-1" />);
    const el = screen.getByTestId('session-sidebar');
    expect(el.getAttribute('style')).toBeNull();
    expect(el.className).not.toMatch(/\bborder|shadow-well|bg-sidebar/);
  });

  // Spec 137 round 4 ("status card only"): the status line is the column's
  // single white card; the session group sits on the column tint.
  it('sits on the column tint: no card, shadow, rounding or inset of its own', () => {
    resetMocks();
    render(<SessionSidebar projectId="proj-1" />);
    const el = screen.getByTestId('session-sidebar');
    expect(el.className).not.toMatch(/bg-card|shadow|rounded|\bmx-|\bmt-/);
  });

  it('the selected row is the raised card row it is on the tint', () => {
    resetMocks();
    mockSessions = [makeSession({ session_id: 'sess-a', session_uuid: 'ua' })];
    render(<SessionSidebar projectId="proj-1" selectedSessionId="sess-a" />);
    const row = screen.getByTestId('session-list-item-sess-a');
    expect(row.style.background).toBe('var(--color-session-selected)');
    expect(row.style.boxShadow).not.toBe('');
  });
});

