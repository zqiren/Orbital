// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, waitFor, cleanup, fireEvent, within } from '@testing-library/react';
import type { Project } from '../types';

afterEach(() => cleanup());

// ---------------------------------------------------------------------------
// Mock useBlockedCount with a configurable fn so each test group can control
// the returned count (the pattern BlockedBadge.test.tsx uses).
// ---------------------------------------------------------------------------
const mockUseBlockedCount = vi.fn();

vi.mock('../hooks/useBlockedCount', () => ({
  useBlockedCount: () => mockUseBlockedCount(),
}));

// Sidebar's Workbench badge count fetches GET /api/v2/workbench directly
// (no dedicated hook module to mock) — stub the api client. Defaults to an
// empty response (count 0) so pre-existing tests that don't care about the
// Workbench badge are unaffected; individual tests override with mockResolvedValueOnce.
const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import Sidebar from './Sidebar';

const mockProject: Project = {
  project_id: 'proj-1',
  name: 'Test Project',
  workspace: '/tmp/test',
  model: 'claude-3-5-sonnet',
  api_key: '',
  base_url: null,
  autonomy: 'hands_off',
  instructions: '',
};

const defaultProps = {
  projects: [mockProject],
  agentStatuses: {},
  statusSummaries: {},
  pendingApprovals: {},
  route: { name: 'list' } as const,
  connectionState: 'connected' as const,
  onSelectProject: vi.fn(),
  onSelectCalendar: vi.fn(),
  onSelectWorkbench: vi.fn(),
  onNewProject: vi.fn(),
  onSettings: vi.fn(),
};

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockResolvedValue({ entries: [], computed: [] });
});

describe('Sidebar — BlockedBadge integration (count 0)', () => {
  beforeEach(() => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 0, blockedSessions: [], loading: false });
  });

  it('renders the Sidebar without crashing', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByText('Orbital')).toBeInTheDocument();
  });

  it('renders the "Blocked" label from BlockedBadge in the global area', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByText('Blocked')).toBeInTheDocument();
  });

  it('does NOT show the blocked count pill when blockedCount is 0', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.queryByTestId('blocked-badge-pill')).toBeNull();
  });

  it('still renders project list entries', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByText('Test Project')).toBeInTheDocument();
  });

  it('still renders New Project and Settings buttons', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByText('+ New Project')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('still shows connection indicator', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByText('Connected')).toBeInTheDocument();
  });
});

describe('Sidebar — BlockedBadge integration (count > 0)', () => {
  beforeEach(() => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 3, blockedSessions: [], loading: false });
  });

  it('renders the blocked count pill inside the Sidebar when blockedCount > 0', () => {
    render(<Sidebar {...defaultProps} />);
    const pill = screen.getByTestId('blocked-badge-pill');
    expect(pill).toBeInTheDocument();
    expect(pill).toHaveTextContent('3');
  });

  it('still renders the project list alongside the pill', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByTestId('blocked-badge-pill')).toBeInTheDocument();
    expect(screen.getByText('Test Project')).toBeInTheDocument();
  });
});

describe('Sidebar — Workspace zone (two-zone IA)', () => {
  beforeEach(() => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 0, blockedSessions: [], loading: false });
  });

  const scratchProject: Project = {
    ...mockProject,
    project_id: 'scratch-1',
    name: 'Quick Tasks',
    is_scratch: true,
  };

  it('renders the Workspace zone label and a Calendar item', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByText('Workspace')).toBeInTheDocument();
    expect(screen.getByText('Calendar')).toBeInTheDocument();
  });

  it('clicking Calendar calls onSelectCalendar', () => {
    const onSelectCalendar = vi.fn();
    render(<Sidebar {...defaultProps} onSelectCalendar={onSelectCalendar} />);
    fireEvent.click(screen.getByText('Calendar'));
    expect(onSelectCalendar).toHaveBeenCalledOnce();
  });

  it('marks the Calendar item active (aria-current) when route.name is calendar', () => {
    render(<Sidebar {...defaultProps} route={{ name: 'calendar' }} />);
    const calBtn = screen.getByText('Calendar').closest('button');
    expect(calBtn).toHaveAttribute('aria-current', 'page');
  });

  it('pins the is_scratch project in the Workspace zone and removes it from the Projects list', () => {
    const { container } = render(
      <Sidebar {...defaultProps} projects={[scratchProject, mockProject]} />,
    );
    // Quick Tasks renders exactly once (promoted, not duplicated).
    expect(screen.getAllByText('Quick Tasks')).toHaveLength(1);
    // The Projects zone (<nav>) contains the regular project but NOT Quick Tasks.
    const nav = container.querySelector('nav') as HTMLElement;
    expect(within(nav).getByText('Test Project')).toBeInTheDocument();
    expect(within(nav).queryByText('Quick Tasks')).toBeNull();
    // Quick Tasks keeps its clickable project behavior (routes via onSelectProject).
    fireEvent.click(screen.getByText('Quick Tasks'));
    expect(defaultProps.onSelectProject).toHaveBeenCalledWith('scratch-1');
  });

  it('the Projects count reflects non-scratch projects only', () => {
    render(<Sidebar {...defaultProps} projects={[scratchProject, mockProject]} />);
    // Two projects total, one scratch → Projects header count is 1.
    const header = screen.getByText('Projects').parentElement as HTMLElement;
    expect(within(header).getByText('1')).toBeInTheDocument();
  });
});

describe('Sidebar — Workbench nav item', () => {
  beforeEach(() => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 0, blockedSessions: [], loading: false });
  });

  it('renders a Workbench item in the Workspace zone', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByText('Workbench')).toBeInTheDocument();
  });

  it('clicking Workbench calls onSelectWorkbench', () => {
    const onSelectWorkbench = vi.fn();
    render(<Sidebar {...defaultProps} onSelectWorkbench={onSelectWorkbench} />);
    fireEvent.click(screen.getByText('Workbench'));
    expect(onSelectWorkbench).toHaveBeenCalledOnce();
  });

  it('marks the Workbench item active (aria-current) when route.name is workbench', () => {
    render(<Sidebar {...defaultProps} route={{ name: 'workbench' }} />);
    const btn = screen.getByText('Workbench').closest('button');
    expect(btn).toHaveAttribute('aria-current', 'page');
  });

  it('does not show a count pill when the Workbench is empty', async () => {
    apiMock.mockResolvedValue({ entries: [], computed: [] });
    render(<Sidebar {...defaultProps} />);
    await waitFor(() => expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench'));
    expect(screen.queryByTestId('workbench-badge-count')).toBeNull();
  });

  it('shows entries.length + computed.length as the badge count', async () => {
    apiMock.mockResolvedValue({
      entries: [{ id: 'e1' }, { id: 'e2' }],
      computed: [{ key: 'c1' }],
    });
    render(<Sidebar {...defaultProps} />);
    await waitFor(() =>
      expect(screen.getByTestId('workbench-badge-count')).toHaveTextContent('3'),
    );
  });
});
