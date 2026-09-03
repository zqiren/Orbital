// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ProjectDetail from './ProjectDetail';
import type { Project } from '../types';
import type { Route } from '../route';

// ProjectDetail now calls useSessions and useQueue (both need WebSocketProvider).
// Mock them to avoid the provider requirement in these unit tests.
vi.mock('../hooks/useSessions', () => ({
  useSessions: () => ({ sessions: [], loading: false, error: null, refresh: vi.fn() }),
}));
vi.mock('../hooks/useQueue', () => ({
  useQueue: () => ({
    snapshot: null,
    loading: false,
    error: null,
    refresh: vi.fn(),
    addItem: vi.fn(),
    removeItem: vi.fn(),
    editItem: vi.fn(),
    stopQueue: vi.fn(),
    resumeQueue: vi.fn(),
  }),
}));
// The Automations pane mounts useTriggers (→ useWebSocket). These tests only
// care about which pane renders, so stub it empty.
vi.mock('../hooks/useTriggers', () => ({
  useTriggers: () => ({
    triggers: [],
    loading: false,
    fetchTriggers: vi.fn(async () => []),
    createTrigger: vi.fn(),
    updateTrigger: vi.fn(),
    toggleTrigger: vi.fn(),
    deleteTrigger: vi.fn(),
  }),
}));
// BudgetCorner (in the header) calls useCost → useWebSocket. Mock it to a
// no-spend response so the corner renders nothing and the header label tests
// stay focused on the model label.
vi.mock('../hooks/useCost', () => ({
  useCost: () => ({ cost: null, loading: false, error: null, refresh: vi.fn() }),
}));

const mockProject: Project = {
  project_id: 'proj-42',
  name: 'Alpha Project',
  workspace: '/tmp/alpha',
  model: 'claude-3-5-sonnet',
  api_key: '',
  base_url: null,
  autonomy: 'hands_off',
  instructions: '',
};

const baseRoute: Extract<Route, { name: 'project' }> = {
  name: 'project',
  projectId: 'proj-42',
  tab: 'chat',
  sessionId: 'sess-xyz',
  settings: false,
};

function renderProjectDetail(
  routeOverride?: Partial<Extract<Route, { name: 'project' }>>,
  setRoute = vi.fn(),
) {
  const route = { ...baseRoute, ...routeOverride };
  return render(
    <ProjectDetail
      project={mockProject}
      agentStatus="idle"
      route={route}
      setRoute={setRoute}
    />,
  );
}

describe('ProjectDetail — gear icon in header', () => {
  it('renders the gear icon button in the project header', () => {
    renderProjectDetail();
    expect(screen.getByRole('button', { name: /project settings/i })).toBeInTheDocument();
  });

  it('clicking the gear icon calls setRoute with settings:true, preserving projectId, tab, and sessionId', () => {
    const setRoute = vi.fn();
    renderProjectDetail({}, setRoute);
    fireEvent.click(screen.getByRole('button', { name: /project settings/i }));
    expect(setRoute).toHaveBeenCalledTimes(1);
    expect(setRoute).toHaveBeenCalledWith({
      name: 'project',
      projectId: 'proj-42',
      tab: 'chat',
      sessionId: 'sess-xyz',
      settings: true,
    });
  });

  it('still renders queue/chat/files tab buttons (not a Settings tab)', () => {
    renderProjectDetail();
    // The queue tab is labelled "Tasks" (it parents the Queue and Automations
    // panes); its route key is still 'queue'.
    expect(screen.getByRole('button', { name: /^Tasks$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Chat$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Files$/i })).toBeInTheDocument();
    // Must NOT have a standalone "Settings" tab button
    expect(screen.queryByRole('button', { name: /^Settings$/i })).toBeNull();
  });

  it('tab click calls setRoute with settings:false to clear settings overlay', () => {
    const setRoute = vi.fn();
    renderProjectDetail({ tab: 'chat', settings: false }, setRoute);
    fireEvent.click(screen.getByRole('button', { name: /^Tasks$/i }));
    expect(setRoute).toHaveBeenCalledWith(
      expect.objectContaining({ tab: 'queue', settings: false }),
    );
  });
});

describe('ProjectDetail — Tasks tab panes', () => {
  it('shows the Queue│Automations switch only on the Tasks tab', () => {
    const { unmount } = renderProjectDetail({ tab: 'chat' });
    expect(screen.queryByTestId('queue-pane-automations')).toBeNull();
    unmount();

    renderProjectDetail({ tab: 'queue' });
    expect(screen.getByTestId('queue-pane-queue')).toBeInTheDocument();
    expect(screen.getByTestId('queue-pane-automations')).toBeInTheDocument();
  });

  it('defaults to the queue pane and renders the tab children there', () => {
    const route = { ...baseRoute, tab: 'queue' as const };
    render(
      <ProjectDetail
        project={mockProject}
        agentStatus="idle"
        route={route}
        setRoute={vi.fn()}
      >
        <div data-testid="queue-children" />
      </ProjectDetail>,
    );
    expect(screen.getByTestId('queue-children')).toBeInTheDocument();
    expect(screen.getByTestId('queue-pane-queue')).toHaveAttribute('aria-selected', 'true');
    expect(screen.queryByTestId('automations-pane')).toBeNull();
  });

  it('renders the automations pane (not the queue children) when queuePane=automations', () => {
    const route = { ...baseRoute, tab: 'queue' as const, queuePane: 'automations' as const };
    render(
      <ProjectDetail
        project={mockProject}
        agentStatus="idle"
        route={route}
        setRoute={vi.fn()}
      >
        <div data-testid="queue-children" />
      </ProjectDetail>,
    );
    expect(screen.getByTestId('automations-pane')).toBeInTheDocument();
    expect(screen.queryByTestId('queue-children')).toBeNull();
  });

  it('clicking the Automations pane sets route.queuePane', () => {
    const setRoute = vi.fn();
    renderProjectDetail({ tab: 'queue' }, setRoute);
    fireEvent.click(screen.getByTestId('queue-pane-automations'));
    expect(setRoute).toHaveBeenCalledWith(
      expect.objectContaining({ tab: 'queue', queuePane: 'automations' }),
    );
  });

  it('the trigger strip is a status line that deep-links into the automations pane', () => {
    const setRoute = vi.fn();
    render(
      <ProjectDetail
        project={mockProject}
        agentStatus="idle"
        route={baseRoute}
        setRoute={setRoute}
        triggers={[
          {
            id: 'trg-1',
            name: 'Daily build',
            enabled: false,
            type: 'schedule',
            schedule: { cron: '0 9 * * *', human: 'Every day at 09:00', timezone: 'UTC' },
            task: 'Build',
            autonomy: null,
            last_triggered: null,
            trigger_count: 0,
            created_at: '2026-01-01T00:00:00Z',
          },
        ]}
      />,
    );
    const summary = screen.getByTestId('trigger-summary');
    // The off-count stays glanceable from every tab.
    expect(summary.textContent).toContain('1 off');
    // No management controls left on the strip — one home for that.
    expect(screen.queryByRole('switch')).toBeNull();

    fireEvent.click(summary);
    expect(setRoute).toHaveBeenCalledWith(
      expect.objectContaining({ tab: 'queue', queuePane: 'automations' }),
    );
  });
});

describe('ProjectDetail — model header label', () => {
  // P3-F coupled removal: the header spend segment (the old budget_spent_usd
  // "$X.XX" suffix) is GONE. The GET /cost-backed spend corner ships in P3-G.
  // The header now renders the model label alone.
  it('Test 4: model label uses the global default (not agent_name) when project.model is empty', () => {
    const project: Project = {
      ...mockProject,
      model: '',
      agent_name: 'Research Bot',
    };
    render(
      <ProjectDetail
        project={project}
        agentStatus="idle"
        route={baseRoute}
        setRoute={vi.fn()}
        globalDefaultModel="deepseek-chat"
      />,
    );
    expect(screen.getByText('deepseek-chat')).toBeInTheDocument();
    expect(screen.queryByText(/Research Bot/)).not.toBeInTheDocument();
    // No fabricated $0.00 cost suffix.
    expect(screen.queryByText(/\$0\.00/)).not.toBeInTheDocument();
  });

  it('Test 5: with no model anywhere, the header renders no model/cost label', () => {
    const project: Project = {
      ...mockProject,
      model: '',
      agent_name: 'Research Bot',
    };
    render(
      <ProjectDetail
        project={project}
        agentStatus="idle"
        route={baseRoute}
        setRoute={vi.fn()}
        globalDefaultModel=""
      />,
    );
    expect(screen.queryByText(/Research Bot/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/·/)).not.toBeInTheDocument();
  });

  it('prefers the project-pinned model over the global default', () => {
    render(
      <ProjectDetail
        project={{ ...mockProject, model: 'gpt-4o' }}
        agentStatus="idle"
        route={baseRoute}
        setRoute={vi.fn()}
        globalDefaultModel="deepseek-chat"
      />,
    );
    expect(screen.getByText('gpt-4o')).toBeInTheDocument();
    expect(screen.queryByText(/\$/)).not.toBeInTheDocument();
  });
});

describe('ProjectDetail — calendar lens tab visibility', () => {
  it('never shows Calendar or Workbench tabs — they are global workspace surfaces', () => {
    // User decision 2026-07-24: per-project lenses removed; aggregation
    // surfaces live only in the Workspace section of the sidebar.
    const project = { ...mockProject, enabled_connectors: ['google-calendar'] } as Project;
    render(
      <ProjectDetail
        project={project}
        agentStatus="idle"
        route={baseRoute}
        setRoute={vi.fn()}
      />,
    );
    expect(screen.queryByRole('button', { name: /^Calendar/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /^Workbench/i })).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Spec 078 §5.1/§9.10 — who owns route.previewPath
//
// On the Chat tab, above the push threshold, the docked workspace panel
// (rendered by ChatTab) shows the preview. ProjectDetail must suppress its
// overlay drawer there, or a chat path click opens both at once. Everywhere
// else — Queue, Files, settings, mobile, narrow windows — the overlay stays.
// ---------------------------------------------------------------------------

function setViewportWidth(width: number) {
  Object.defineProperty(window, 'innerWidth', {
    value: width,
    configurable: true,
    writable: true,
  });
}

describe('ProjectDetail — overlay drawer vs. the docked panel', () => {
  afterEach(() => setViewportWidth(1024));

  // The drawer is always mounted; `inert` is what says "closed" (it is off
  // screen and out of the a11y tree, so getByRole cannot see it).
  function drawer(container: HTMLElement) {
    const el = container.querySelector('[role="dialog"]');
    expect(el).not.toBeNull();
    return el as HTMLElement;
  }

  it('keeps the overlay closed on a wide Chat tab — the panel has the preview', () => {
    setViewportWidth(1440);
    const { container } = renderProjectDetail({ tab: 'chat', previewPath: 'docs/plan.md' });
    expect(drawer(container)).toHaveAttribute('inert');
  });

  it('still opens the overlay on the Chat tab below the push threshold', () => {
    setViewportWidth(1024);
    const { container } = renderProjectDetail({ tab: 'chat', previewPath: 'docs/plan.md' });
    expect(drawer(container)).not.toHaveAttribute('inert');
  });

  it('still opens the overlay on the Queue and Files tabs at any width', () => {
    setViewportWidth(1440);
    const queue = renderProjectDetail({ tab: 'queue', previewPath: 'docs/plan.md' });
    expect(drawer(queue.container)).not.toHaveAttribute('inert');
    queue.unmount();

    const files = renderProjectDetail({ tab: 'files', previewPath: 'docs/plan.md' });
    expect(drawer(files.container)).not.toHaveAttribute('inert');
  });

  it('still opens the overlay over the settings surface', () => {
    setViewportWidth(1440);
    const { container } = renderProjectDetail({
      tab: 'chat',
      settings: true,
      previewPath: 'docs/plan.md',
    });
    expect(drawer(container)).not.toHaveAttribute('inert');
  });
});
