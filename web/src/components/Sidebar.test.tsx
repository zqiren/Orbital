// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, waitFor, cleanup, fireEvent, within } from '@testing-library/react';
import type { Project } from '../types';

afterEach(() => cleanup());

// The Sidebar no longer renders BlockedBadge, so useBlockedCount is not in its
// module graph and needs no mock here. BlockedBadge.test.tsx still covers the
// component itself.

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
  onReorderProjects: vi.fn().mockResolvedValue([]),
};

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockResolvedValue({ entries: [] });
});

describe('Sidebar — base render', () => {
  it('renders the Sidebar without crashing', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByText('Orbital')).toBeInTheDocument();
  });

  // Deliberate removal, not an oversight: the row was a non-clickable status
  // readout that greyed itself out at zero, so users read it as a state they
  // had to act on with nowhere to act. Pending approvals still surface as an
  // amber dot on the project row.
  it('does NOT render the global "Blocked" row', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.queryByText('Blocked')).toBeNull();
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

describe('Sidebar — Workspace zone (two-zone IA)', () => {
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

  it('pins the is_scratch project at the top of the Projects list, above the rest', () => {
    const { container } = render(
      <Sidebar {...defaultProps} projects={[mockProject, scratchProject]} />,
    );
    // Quick Tasks renders exactly once (pinned, not duplicated).
    expect(screen.getAllByText('Quick Tasks')).toHaveLength(1);

    // It now lives inside the Projects zone (<nav>) rather than the Workspace
    // block above it, and sorts first even though the prop order puts it last.
    const nav = container.querySelector('nav') as HTMLElement;
    expect(within(nav).getByText('Quick Tasks')).toBeInTheDocument();
    expect(within(nav).getByText('Test Project')).toBeInTheDocument();

    const rows = within(nav).getAllByRole('button');
    expect(rows[0]).toHaveTextContent('Quick Tasks');

    // Pinning changes placement only — it still routes like any project row.
    fireEvent.click(screen.getByText('Quick Tasks'));
    expect(defaultProps.onSelectProject).toHaveBeenCalledWith('scratch-1');
  });

  it('drops the pinned hairline when there are no regular projects', () => {
    const { rerender } = render(
      <Sidebar {...defaultProps} projects={[scratchProject]} />,
    );
    expect(screen.getByTestId('pinned-projects').className).not.toMatch(/border-b/);

    rerender(<Sidebar {...defaultProps} projects={[scratchProject, mockProject]} />);
    expect(screen.getByTestId('pinned-projects').className).toMatch(/border-b/);
  });

  it('the Projects count reflects non-scratch projects only', () => {
    render(<Sidebar {...defaultProps} projects={[scratchProject, mockProject]} />);
    // Two projects total, one scratch → Projects header count is 1.
    const header = screen.getByText('Projects').parentElement as HTMLElement;
    expect(within(header).getByText('1')).toBeInTheDocument();
  });
});

describe('Sidebar — manual project order (spec 056)', () => {
  const scratchProject: Project = {
    ...mockProject,
    project_id: 'scratch-1',
    name: 'Quick Tasks',
    is_scratch: true,
  };
  const alpha: Project = { ...mockProject, project_id: 'p-a', name: 'Alpha' };
  const beta: Project = { ...mockProject, project_id: 'p-b', name: 'Beta' };
  const gamma: Project = { ...mockProject, project_id: 'p-c', name: 'Gamma' };

  function navRowNames(container: HTMLElement): string[] {
    const nav = container.querySelector('nav') as HTMLElement;
    return within(nav)
      .getAllByRole('button')
      .map((b) => b.textContent ?? '');
  }

  it('renders regular projects in prop order — the API order is the render order', () => {
    const { container } = render(
      <Sidebar {...defaultProps} projects={[gamma, alpha, beta]} />,
    );
    expect(navRowNames(container)).toEqual(['Gamma', 'Alpha', 'Beta']);
  });

  it('re-renders in the new order when the reordered list arrives as props', () => {
    const { container, rerender } = render(
      <Sidebar {...defaultProps} projects={[alpha, beta, gamma]} />,
    );
    expect(navRowNames(container)).toEqual(['Alpha', 'Beta', 'Gamma']);

    rerender(<Sidebar {...defaultProps} projects={[gamma, alpha, beta]} />);
    expect(navRowNames(container)).toEqual(['Gamma', 'Alpha', 'Beta']);
  });

  it('gives every regular project a drag handle', () => {
    render(<Sidebar {...defaultProps} projects={[alpha, beta]} />);
    expect(screen.getByTestId('project-drag-handle-p-a')).toBeInTheDocument();
    expect(screen.getByTestId('project-drag-handle-p-b')).toBeInTheDocument();
  });

  it('does NOT give the pinned Quick Tasks row a handle', () => {
    // Scratch-first is an invariant of the list, not a position the user owns.
    render(<Sidebar {...defaultProps} projects={[scratchProject, alpha]} />);
    expect(screen.queryByTestId('project-drag-handle-scratch-1')).toBeNull();
    expect(screen.getByTestId('project-drag-handle-p-a')).toBeInTheDocument();
  });

  it('labels the handle with the translated, project-named string', () => {
    render(<Sidebar {...defaultProps} projects={[alpha]} />);
    expect(screen.getByTestId('project-drag-handle-p-a')).toHaveAttribute(
      'title',
      'Drag to reorder Alpha',
    );
  });

  it('keeps the handle out of the tab order and out of the a11y tree', () => {
    // §6 decision 5 dropped keyboard reordering on purpose, so the handle must
    // not advertise itself as a control that keyboard/AT users can operate —
    // and the list's own tab order must not change.
    const { container } = render(
      <Sidebar {...defaultProps} projects={[scratchProject, alpha, beta]} />,
    );
    const handle = screen.getByTestId('project-drag-handle-p-a');
    expect(handle).not.toHaveAttribute('tabindex');
    expect(handle).toHaveAttribute('aria-hidden', 'true');
    // One focusable row per project, exactly as before the handles existed.
    expect(navRowNames(container)).toEqual(['Quick Tasks', 'Alpha', 'Beta']);
  });
});

describe('Sidebar — Workbench nav item', () => {
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
    apiMock.mockResolvedValue({ entries: [] });
    render(<Sidebar {...defaultProps} />);
    await waitFor(() => expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench'));
    expect(screen.queryByTestId('workbench-badge-count')).toBeNull();
  });

  it('shows entries.length as the badge count', async () => {
    apiMock.mockResolvedValue({
      entries: [{ id: 'e1' }, { id: 'e2' }, { id: 'e3' }],
    });
    render(<Sidebar {...defaultProps} />);
    await waitFor(() =>
      expect(screen.getByTestId('workbench-badge-count')).toHaveTextContent('3'),
    );
  });

  it('refetches the badge count when orbital:workbench-changed fires', async () => {
    apiMock.mockResolvedValueOnce({ entries: [{ id: 'e1' }] });
    render(<Sidebar {...defaultProps} />);
    await waitFor(() =>
      expect(screen.getByTestId('workbench-badge-count')).toHaveTextContent('1'),
    );

    apiMock.mockResolvedValueOnce({ entries: [{ id: 'e1' }, { id: 'e2' }] });
    fireEvent(window, new CustomEvent('orbital:workbench-changed'));

    await waitFor(() =>
      expect(screen.getByTestId('workbench-badge-count')).toHaveTextContent('2'),
    );
  });

  it('cleans up the orbital:workbench-changed listener on unmount', async () => {
    apiMock.mockResolvedValue({ entries: [] });
    const { unmount } = render(<Sidebar {...defaultProps} />);
    await waitFor(() => expect(apiMock).toHaveBeenCalledTimes(1));

    unmount();
    fireEvent(window, new CustomEvent('orbital:workbench-changed'));
    await new Promise((r) => setTimeout(r, 0));
    expect(apiMock).toHaveBeenCalledTimes(1); // no refetch after unmount
  });
});
