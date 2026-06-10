// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
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
  onNewProject: vi.fn(),
  onSettings: vi.fn(),
};

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
