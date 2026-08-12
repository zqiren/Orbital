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

vi.mock('../hooks/useSessions', () => ({
  useSessions: (_projectId: string | null) => ({
    sessions: mockSessions,
    loading: false,
    error: null,
    refresh: vi.fn(),
    renameSession: renameSessionMock,
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
