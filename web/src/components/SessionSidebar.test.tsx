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

const mockSetActiveSessionId = vi.fn();
let mockSessions: SessionListEntry[] = [];
let mockActiveSessionId: string | null = null;

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
  mockActiveSessionId = null;
  mockSetActiveSessionId.mockReset();
}

// ---------------------------------------------------------------------------
// Grouping: active vs archived
// ---------------------------------------------------------------------------

describe('SessionSidebar — active vs archived grouping', () => {
  it('active sessions (running/waiting/pending_approval/idle) appear in the main list', () => {
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

    // All four active sessions are in the main list
    expect(screen.getByTestId('session-list-item-sess-run')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-wait')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-blocked')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-idle')).toBeInTheDocument();
  });

  it('stopped sessions appear in the archived section, not the main list', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-active', status: 'idle', session_uuid: 'ua' }),
      makeSession({ session_id: 'sess-stopped', status: 'stopped', session_uuid: 'us' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);

    // Active in main list
    expect(screen.getByTestId('session-list-item-sess-active')).toBeInTheDocument();
    // Stopped not in main list initially (archived section collapsed)
    expect(screen.queryByTestId('session-list-item-sess-stopped')).toBeNull();
    // Archived section toggle present
    expect(screen.getByTestId('session-archived-toggle')).toBeInTheDocument();
    expect(screen.getByText(/archived \(1\)/i)).toBeInTheDocument();
  });

  it('error and new_session statuses APPEAR in the active list (not filtered out)', () => {
    // Regression: error and new_session must not fall through both
    // isActiveStatus and isArchivedStatus and vanish from the sidebar.
    // DO-NOT #12: ALL sessions must appear.
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-error', status: 'error', session_uuid: 'ue' }),
      makeSession({ session_id: 'sess-new', status: 'new_session', session_uuid: 'un' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);

    // Both appear in the active list (not dropped)
    expect(screen.getByTestId('session-list-item-sess-error')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-new')).toBeInTheDocument();
    // No archived section since neither is `stopped`
    expect(screen.queryByTestId('session-archived-section')).toBeNull();
    // Both count as active
    expect(screen.getByTestId('session-active-count')).toHaveTextContent('2');
  });

  it('archived section is absent when there are no stopped sessions', () => {
    resetMocks();
    mockSessions = [makeSession({ session_id: 'sess-idle', status: 'idle' })];

    render(<SessionSidebar projectId="proj-1" />);

    expect(screen.queryByTestId('session-archived-section')).toBeNull();
  });

  it('active count in header reflects number of active sessions', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 's1', status: 'running', session_uuid: 'u1' }),
      makeSession({ session_id: 's2', status: 'idle', session_uuid: 'u2' }),
      makeSession({ session_id: 's3', status: 'stopped', session_uuid: 'u3' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);
    expect(screen.getByTestId('session-active-count')).toHaveTextContent('2');
  });
});

// ---------------------------------------------------------------------------
// Archived section collapse / expand
// ---------------------------------------------------------------------------

describe('SessionSidebar — archived section collapse/expand', () => {
  it('archived section is collapsed by default', () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-stopped', status: 'stopped', session_uuid: 'u1' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);
    expect(screen.queryByTestId('session-archived-list')).toBeNull();
    expect(screen.getByTestId('session-archived-toggle')).toHaveAttribute(
      'aria-expanded',
      'false',
    );
  });

  it('clicking the archived toggle expands the archived list', async () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-stopped', status: 'stopped', session_uuid: 'u1' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);
    await userEvent.click(screen.getByTestId('session-archived-toggle'));

    expect(screen.getByTestId('session-archived-list')).toBeInTheDocument();
    expect(screen.getByTestId('session-list-item-sess-stopped')).toBeInTheDocument();
    expect(screen.getByTestId('session-archived-toggle')).toHaveAttribute(
      'aria-expanded',
      'true',
    );
  });

  it('clicking the toggle again collapses the archived list', async () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-stopped', status: 'stopped', session_uuid: 'u1' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);
    await userEvent.click(screen.getByTestId('session-archived-toggle'));
    expect(screen.getByTestId('session-archived-list')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('session-archived-toggle'));
    expect(screen.queryByTestId('session-archived-list')).toBeNull();
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
        origin: 'manual',
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

describe('SessionSidebar — session selection', () => {
  it('clicking a row calls setActiveSessionId with the session_id', async () => {
    resetMocks();
    mockSessions = [
      makeSession({ session_id: 'sess-pick', status: 'idle', session_uuid: 'up' }),
    ];

    render(<SessionSidebar projectId="proj-1" />);
    await userEvent.click(screen.getByTestId('session-list-item-sess-pick'));
    expect(mockSetActiveSessionId).toHaveBeenCalledWith('sess-pick');
  });

  it('onSessionSelect callback fires when a row is clicked', async () => {
    resetMocks();
    const onSessionSelect = vi.fn();
    mockSessions = [
      makeSession({ session_id: 'sess-cb', status: 'idle', session_uuid: 'uc' }),
    ];

    render(<SessionSidebar projectId="proj-1" onSessionSelect={onSessionSelect} />);
    await userEvent.click(screen.getByTestId('session-list-item-sess-cb'));
    expect(onSessionSelect).toHaveBeenCalledWith('sess-cb');
  });
});
