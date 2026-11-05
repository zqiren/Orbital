// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, within } from '@testing-library/react';
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
  onSelectHome: vi.fn(),
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

  it('renders the Workspace zone label and a Workbench row, and no Calendar row', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByText('Workspace')).toBeInTheDocument();
    expect(screen.getByTestId('sidebar-workbench')).toHaveTextContent('Workbench');
    expect(screen.queryByText('Calendar')).toBeNull();
  });

  it('clicking the Workbench row goes home', () => {
    const onSelectHome = vi.fn();
    render(<Sidebar {...defaultProps} onSelectHome={onSelectHome} />);
    fireEvent.click(screen.getByTestId('sidebar-workbench'));
    expect(onSelectHome).toHaveBeenCalledOnce();
  });

  it('marks the Workbench row selected on home only', () => {
    const { rerender } = render(<Sidebar {...defaultProps} route={{ name: 'list' }} />);
    const row = screen.getByTestId('sidebar-workbench');
    expect(row).toHaveAttribute('aria-current', 'page');
    expect(row.className).toContain('bg-nav-selected');
    rerender(<Sidebar {...defaultProps} route={{ name: 'settings' }} />);
    expect(screen.getByTestId('sidebar-workbench')).not.toHaveAttribute('aria-current');
    expect(screen.getByTestId('sidebar-workbench').className).not.toContain('bg-nav-selected');
  });

  it('the Workbench row carries the Workbench glyph as its icon', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getByTestId('sidebar-workbench').querySelector('svg.lucide-layout-grid')).not.toBeNull();
  });

  it('the row glyph shows the blue dot while something waits, except on home', () => {
    const { rerender } = render(<Sidebar {...defaultProps} route={{ name: 'settings' }} waiting={false} />);
    expect(screen.queryByTestId('sidebar-workbench-dot')).toBeNull();
    rerender(<Sidebar {...defaultProps} route={{ name: 'settings' }} waiting />);
    expect(screen.getByTestId('sidebar-workbench-dot').className).toContain('bg-accent');
    rerender(<Sidebar {...defaultProps} route={{ name: 'list' }} waiting />);
    expect(screen.queryByTestId('sidebar-workbench-dot')).toBeNull();
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

describe('Sidebar — the brand row is plain again (2026-11-04)', () => {
  it('the Orbital wordmark is not a control', () => {
    const onSelectHome = vi.fn();
    render(<Sidebar {...defaultProps} onSelectHome={onSelectHome} />);
    expect(screen.queryByTestId('sidebar-brand-home')).toBeNull();
    const word = screen.getByText('Orbital');
    expect(word.closest('button')).toBeNull();
    fireEvent.click(word);
    expect(onSelectHome).not.toHaveBeenCalled();
  });

  it('there is exactly one way home in the list: the Workbench row', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.getAllByRole('button', { name: /^Workbench/ })).toHaveLength(1);
    expect(screen.queryByTestId('workbench-badge-count')).toBeNull();
    expect(apiMock).not.toHaveBeenCalledWith('/api/v2/workbench');
  });
});

describe('Sidebar — unread count per project (2026-11-05; spec 151 folded into 154)', () => {
  it('shows nothing at zero', () => {
    render(<Sidebar {...defaultProps} unreadCounts={{}} />);
    expect(screen.queryByTestId('project-unread-count-proj-1')).toBeNull();
  });

  it('shows the number of unread sessions with a labelled pill', () => {
    render(<Sidebar {...defaultProps} unreadCounts={{ 'proj-1': 3 }} />);
    const pill = screen.getByTestId('project-unread-count-proj-1');
    expect(pill).toHaveTextContent('3');
    expect(pill).toHaveAttribute('aria-label', '3 unread');
  });

  it('caps at 99+', () => {
    render(<Sidebar {...defaultProps} unreadCounts={{ 'proj-1': 120 }} />);
    expect(screen.getByTestId('project-unread-count-proj-1')).toHaveTextContent('99+');
  });
});

// Spec 095: a pinned dispatch runs no management turn, so the manager status
// stays idle for the worker's whole run. The row reads the worker flag too.
describe('Sidebar — project dot while a worker runs (spec 095)', () => {
  function rowDot() {
    const row = screen.getByText('Test Project').closest('button') as HTMLElement;
    return row.querySelector('.rounded-full') as HTMLElement;
  }

  it('is green with the manager idle and a worker running', () => {
    render(
      <Sidebar
        {...defaultProps}
        agentStatuses={{ 'proj-1': 'idle' }}
        subAgentsRunning={{ 'proj-1': true }}
      />,
    );
    expect(rowDot()).toHaveClass('bg-success-text');
  });

  it('is grey once the worker is done', () => {
    render(
      <Sidebar
        {...defaultProps}
        agentStatuses={{ 'proj-1': 'idle' }}
        subAgentsRunning={{ 'proj-1': false }}
      />,
    );
    expect(rowDot()).toHaveClass('bg-idle');
  });

  it('keeps a manager error red while a worker runs', () => {
    render(
      <Sidebar
        {...defaultProps}
        agentStatuses={{ 'proj-1': 'error' }}
        subAgentsRunning={{ 'proj-1': true }}
      />,
    );
    expect(rowDot()).toHaveClass('bg-error-text');
  });
});

// Spec 117 D10: one dismissible notice for upgraded users, on the project
// list. A brand-new install (no project in safe mode) never sees it.
describe('Sidebar — safe-mode upgrade notice (spec 117)', () => {
  const NOTICE =
    'New projects now run without the sandbox. Your existing projects keep Safe mode on; you can turn it off in Project settings → Safe mode.';

  beforeEach(() => {
    localStorage.clear();
  });

  it('shows when at least one project is in safe mode', () => {
    render(<Sidebar {...defaultProps} projects={[{ ...mockProject, safe_mode: true }]} />);
    expect(screen.getByTestId('safe-mode-notice')).toHaveTextContent(NOTICE);
  });

  it('never shows on a fresh install: no projects, projects created open, or only Quick Tasks', () => {
    const { unmount } = render(<Sidebar {...defaultProps} projects={[]} />);
    expect(screen.queryByTestId('safe-mode-notice')).toBeNull();
    unmount();

    const { unmount: unmount2 } = render(
      <Sidebar {...defaultProps} projects={[{ ...mockProject, safe_mode: false }]} />,
    );
    expect(screen.queryByTestId('safe-mode-notice')).toBeNull();
    unmount2();

    // A daemon that predates the field sends no safe_mode at all: no notice.
    const { unmount: unmount3 } = render(<Sidebar {...defaultProps} projects={[mockProject]} />);
    expect(screen.queryByTestId('safe-mode-notice')).toBeNull();
    unmount3();

    render(
      <Sidebar
        {...defaultProps}
        projects={[{ ...mockProject, project_id: 'scratch', is_scratch: true, safe_mode: true }]}
      />,
    );
    expect(screen.queryByTestId('safe-mode-notice')).toBeNull();
  });

  it('dismisses for good on this device', () => {
    const projects = [{ ...mockProject, safe_mode: true }];
    const { unmount } = render(<Sidebar {...defaultProps} projects={projects} />);
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByTestId('safe-mode-notice')).toBeNull();
    expect(localStorage.getItem('orbital.safeModeNotice.dismissed')).toBe('1');
    unmount();

    render(<Sidebar {...defaultProps} projects={projects} />);
    expect(screen.queryByTestId('safe-mode-notice')).toBeNull();
  });
});

describe('Sidebar — import entry on the New Project split button (batch 1026)', () => {
  it('shows no menu toggle when import is not offered', () => {
    render(<Sidebar {...defaultProps} />);
    expect(screen.queryByTestId('new-project-more')).toBeNull();
  });

  it('opens a one-item menu above the button and imports from it', () => {
    const onImportProjects = vi.fn();
    const onNewProject = vi.fn();
    render(<Sidebar {...defaultProps} onNewProject={onNewProject} onImportProjects={onImportProjects} />);
    const more = screen.getByTestId('new-project-more');
    expect(more.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByTestId('new-project-menu')).toBeNull();

    fireEvent.click(more);
    expect(more.getAttribute('aria-expanded')).toBe('true');
    const menu = screen.getByTestId('new-project-menu');
    expect(within(menu).getAllByRole('menuitem')).toHaveLength(1);

    fireEvent.click(screen.getByTestId('sidebar-import-projects'));
    expect(onImportProjects).toHaveBeenCalledTimes(1);
    expect(onNewProject).not.toHaveBeenCalled();
    expect(screen.queryByTestId('new-project-menu')).toBeNull();
  });

  it('keeps New Project a single click', () => {
    const onNewProject = vi.fn();
    render(<Sidebar {...defaultProps} onNewProject={onNewProject} onImportProjects={vi.fn()} />);
    fireEvent.click(screen.getByText('+ New Project'));
    expect(onNewProject).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId('new-project-menu')).toBeNull();
  });

  it('closes on Escape and on a click outside', () => {
    render(<Sidebar {...defaultProps} onImportProjects={vi.fn()} />);
    fireEvent.click(screen.getByTestId('new-project-more'));
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByTestId('new-project-menu')).toBeNull();

    fireEvent.click(screen.getByTestId('new-project-more'));
    fireEvent.pointerDown(document.body);
    expect(screen.queryByTestId('new-project-menu')).toBeNull();
  });
});
