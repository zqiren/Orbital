// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
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

import { SessionSidebar } from './SessionSidebar';

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

describe('SessionSidebar — new session button', () => {
  it('renders the + new session button', () => {
    resetMocks();
    render(<SessionSidebar projectId="proj-1" />);
    expect(screen.getByTestId('session-new-button')).toBeInTheDocument();
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
