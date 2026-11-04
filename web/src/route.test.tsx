// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Tests for the Route type and Sidebar + ProjectDetail routing integration.
 *
 * Requirements verified:
 *   1. Selecting a project produces a project route on the chat home (no page).
 *   2. Spec 137: `page` replaces the old tab + settings + pricing trio. Opening
 *      a page (status line → Tasks, gear → Settings) keeps projectId and
 *      sessionId; Back goes up one level; the crumb walks the parent chain.
 *   3. The pure helpers: parseProjectPage, pageTrail, routeUp, routeKey.
 *   4. Sidebar produces the correct project route when a project is clicked.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { useState, type SetStateAction } from 'react';
import { describe, expect, it, vi } from 'vitest';
import type { Route } from './route';
import { PROJECT_PAGE_PARENT, pageTrail, parseProjectPage, routeKey, routeUp } from './route';

// Sidebar renders <BlockedBadge> which calls useBlockedCount → useWebSocket.
// These tests render Sidebar without a WebSocketProvider, so mock the hook to
// avoid the "must be used within a WebSocketProvider" throw.
vi.mock('./hooks/useBlockedCount', () => ({
  useBlockedCount: () => ({ blockedCount: 0, blockedSessions: [], loading: false }),
}));

// ProjectDetail now calls useSessions and useQueue (both need WebSocketProvider).
// Mock them to avoid the provider requirement in these unit tests.
vi.mock('./hooks/useSessions', () => ({
  useSessions: () => ({ sessions: [], loading: false, error: null, refresh: vi.fn() }),
}));
vi.mock('./hooks/useQueue', () => ({
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
// ProjectDetail's header BudgetCorner calls useCost → useWebSocket. Mock it.
vi.mock('./hooks/useCost', () => ({
  useCost: () => ({ cost: null, loading: false, error: null, refresh: vi.fn() }),
}));

import Sidebar from './components/Sidebar';
import ProjectDetail from './components/ProjectDetail';
import type { Project, AgentRunStatus } from './types';

// Minimal Project fixture
function makeProject(id: string): Project {
  return {
    project_id: id,
    name: `Project ${id}`,
    workspace: `/tmp/${id}`,
    model: 'claude-sonnet-4-5',
    api_key: '',
    base_url: null,
    autonomy: 'supervised',
    instructions: '',
    is_scratch: false,
  } as Project;
}

// ─── Sidebar tests ───────────────────────────────────────────────────────────

describe('Sidebar routing', () => {
  it('clicking a project invokes onSelectProject with the project id', () => {
    const projects = [makeProject('proj-1'), makeProject('proj-2')];
    const onSelectProject = vi.fn<(value: string) => void>();

    render(
      <Sidebar
        projects={projects}
        agentStatuses={{}}
        statusSummaries={{}}
        pendingApprovals={{}}
        route={{ name: 'list' }}
        connectionState="connected"
        onSelectProject={onSelectProject}
        onSelectHome={vi.fn()}
        onNewProject={vi.fn()}
        onSettings={vi.fn()}
        onReorderProjects={vi.fn().mockResolvedValue([])}
      />,
    );

    fireEvent.click(screen.getByText('Project proj-1'));

    expect(onSelectProject).toHaveBeenCalledOnce();
    expect(onSelectProject).toHaveBeenCalledWith('proj-1');
  });

  it('clicking a different project invokes onSelectProject with that id', () => {
    const projects = [makeProject('alpha'), makeProject('beta')];
    const onSelectProject = vi.fn<(value: string) => void>();

    render(
      <Sidebar
        projects={projects}
        agentStatuses={{}}
        statusSummaries={{}}
        pendingApprovals={{}}
        route={{ name: 'project', projectId: 'alpha' }}
        connectionState="connected"
        onSelectProject={onSelectProject}
        onSelectHome={vi.fn()}
        onNewProject={vi.fn()}
        onSettings={vi.fn()}
        onReorderProjects={vi.fn().mockResolvedValue([])}
      />,
    );

    fireEvent.click(screen.getByText('Project beta'));

    expect(onSelectProject).toHaveBeenCalledWith('beta');
  });

  it('highlights the active project from route.projectId', () => {
    const projects = [makeProject('active'), makeProject('inactive')];

    const { container } = render(
      <Sidebar
        projects={projects}
        agentStatuses={{}}
        statusSummaries={{}}
        pendingApprovals={{}}
        route={{ name: 'project', projectId: 'active' }}
        connectionState="connected"
        onSelectProject={vi.fn()}
        onSelectHome={vi.fn()}
        onNewProject={vi.fn()}
        onSettings={vi.fn()}
        onReorderProjects={vi.fn().mockResolvedValue([])}
      />,
    );

    // The active button carries the selection tint at full strength; the
    // inactive one only carries it as a /50 hover. The nav column uses its own
    // `nav-hover` tint rather than `card-hover`, because card-hover sits only
    // ~3 levels off the nav surface and would be invisible against it.
    const buttons = container.querySelectorAll('nav button');
    expect(buttons[0].className).toContain('bg-nav-selected');
    expect(buttons[1].className).not.toContain('bg-nav-selected');
  });
});

// ─── Mobile select-project regression ─────────────────────────────────────────
//
// Regression guard: on mobile, selecting a project from the Sidebar must drive
// mobileView to 'content', otherwise the content pane stays hidden and the user
// sees a blank screen. The mobile transition lives in App's onSelectProject
// handler (alongside setRoute). This harness replicates that wiring — a small
// host owns both `route` and `mobileView`, threads a select handler that mutates
// both into the Sidebar, and asserts the click flips mobileView to 'content'.

describe('Sidebar mobile select-project regression', () => {
  function Harness({ onMobileView }: { onMobileView: (v: string) => void }) {
    const [route, setRoute] = useState<Route>({ name: 'list' });
    const [, setMobileView] = useState<'sidebar' | 'content'>('sidebar');

    // Mirrors App.handleSelectProject: set the route AND transition mobile view.
    function handleSelectProject(id: string) {
      setRoute({ name: 'project', projectId: id, sessionId: undefined });
      setMobileView('content');
      onMobileView('content');
    }

    return (
      <Sidebar
        projects={[makeProject('p-mob')]}
        agentStatuses={{}}
        statusSummaries={{}}
        pendingApprovals={{}}
        route={route}
        connectionState="connected"
        onSelectProject={handleSelectProject}
        onSelectHome={vi.fn()}
        onNewProject={vi.fn()}
        onSettings={vi.fn()}
        onReorderProjects={vi.fn().mockResolvedValue([])}
      />
    );
  }

  it('selecting a project transitions mobileView to "content"', () => {
    const onMobileView = vi.fn<(value: string) => void>();
    render(<Harness onMobileView={onMobileView} />);

    fireEvent.click(screen.getByText('Project p-mob'));

    expect(onMobileView).toHaveBeenCalledWith('content');
  });
});

// ─── ProjectDetail routing tests ─────────────────────────────────────────────

describe('ProjectDetail routing (spec 137: pages, no tabs)', () => {
  const project = makeProject('proj-42');
  const baseRoute: Extract<Route, { name: 'project' }> = {
    name: 'project',
    projectId: 'proj-42',
  };

  function renderAt(route: Extract<Route, { name: 'project' }>) {
    const setRoute = vi.fn<(value: SetStateAction<Route>) => void>();
    render(
      <ProjectDetail
        project={project}
        agentStatus={'idle' as AgentRunStatus}
        route={route}
        setRoute={setRoute}
      />,
    );
    return setRoute;
  }

  it('the status line opens the Tasks page, preserving projectId and sessionId', () => {
    const setRoute = renderAt({ ...baseRoute, sessionId: 'sess-abc' });
    fireEvent.click(screen.getByTestId('trigger-summary'));
    expect(setRoute.mock.calls[0][0]).toMatchObject({
      name: 'project',
      projectId: 'proj-42',
      sessionId: 'sess-abc',
      page: 'tasks',
    });
  });

  it('the gear opens the Settings page, preserving the session', () => {
    const setRoute = renderAt({ ...baseRoute, sessionId: 'sess-abc' });
    fireEvent.click(screen.getByRole('button', { name: /project settings/i }));
    expect(setRoute.mock.calls[0][0]).toMatchObject({
      name: 'project',
      projectId: 'proj-42',
      sessionId: 'sess-abc',
      page: 'settings',
    });
  });

  // Spec 137 item 10: Back exists on the phone only — desktop and narrow
  // windows go back through the project column (a session row).
  it('Back from Tasks (phone) returns to the chat home', () => {
    Object.defineProperty(window, 'innerWidth', { value: 390, configurable: true, writable: true });
    try {
      const setRoute = renderAt({ ...baseRoute, page: 'tasks', sessionId: 'sess-abc' });
      fireEvent.click(screen.getByTestId('tasks-back-button'));
      const produced = setRoute.mock.calls[0][0] as Extract<Route, { name: 'project' }>;
      expect(produced).toMatchObject({ name: 'project', projectId: 'proj-42', sessionId: 'sess-abc' });
      expect(produced.page).toBeUndefined();
    } finally {
      Object.defineProperty(window, 'innerWidth', { value: 1024, configurable: true, writable: true });
    }
  });

  it('desktop pages have no Back: the column is the way home', () => {
    renderAt({ ...baseRoute, page: 'tasks', sessionId: 'sess-abc' });
    expect(screen.queryByTestId('tasks-back-button')).toBeNull();
    expect(screen.getByTestId('project-column')).toBeInTheDocument();
  });

  it('renders no tab bar on any page', () => {
    for (const page of [undefined, 'tasks', 'settings'] as const) {
      const { unmount } = render(
        <ProjectDetail project={project} agentStatus="idle" route={{ ...baseRoute, page }} setRoute={vi.fn()} />,
      );
      expect(screen.queryByRole('button', { name: /^Chat$/ })).toBeNull();
      expect(screen.queryByRole('button', { name: /^Files$/ })).toBeNull();
      unmount();
    }
  });
});

// ─── Global-settings nav regression (settings folded into Route union) ────────
//
// Bug being guarded: with global settings open, clicking a project in the
// sidebar must navigate to the project view — the settings surface must not
// linger. Previously App kept a `showGlobalSettings` boolean overlay separate
// from `route`, and the project-select handler only updated `route`, so the
// overlay "won". Folding settings into the Route union ({ name: 'settings' })
// makes `route` the single source of truth: selecting a project replaces the
// settings route, so the project view renders.
//
// This harness mirrors App's panel wiring: a host owns `route`, threads the
// real Sidebar's onSelectProject / onNewProject / onSettings into a route-only
// content switch, and asserts the settings surface yields to navigation.

describe('Global settings nav (settings in Route union)', () => {
  function AppLikeHarness() {
    const [route, setRoute] = useState<Route>({ name: 'settings' });

    // Mirrors App.handleSelectProject / handleNewProject / onSettings / onBack —
    // all of which drive `route` alone (no separate overlay boolean).
    function handleSelectProject(id: string) {
      setRoute({ name: 'project', projectId: id, sessionId: undefined });
    }
    function handleNewProject() {
      setRoute({ name: 'create' });
    }

    return (
      <div>
        <Sidebar
          projects={[makeProject('proj-1')]}
          agentStatuses={{}}
          statusSummaries={{}}
          pendingApprovals={{}}
          route={route}
          connectionState="connected"
          onSelectProject={handleSelectProject}
          onSelectHome={vi.fn()}
          onNewProject={handleNewProject}
          onSettings={() => setRoute({ name: 'settings' })}
          onReorderProjects={vi.fn().mockResolvedValue([])}
        />
        {/* Content panel driven SOLELY by route — same gating App now uses. */}
        {route.name === 'settings' && <div data-testid="settings-view">Settings</div>}
        {route.name === 'project' && (
          <div data-testid="project-view">Project {route.projectId}</div>
        )}
        {route.name === 'create' && <div data-testid="create-view">New Project</div>}
      </div>
    );
  }

  it('selecting a project from settings navigates to the project view', () => {
    render(<AppLikeHarness />);

    // Settings is showing initially.
    expect(screen.getByTestId('settings-view')).toBeInTheDocument();

    // Click the project in the sidebar (there are two "Project proj-1"-ish
    // labels? no — only the sidebar entry. Use the nav button text).
    fireEvent.click(screen.getByText('Project proj-1'));

    // Settings surface must be gone; project view must render.
    expect(screen.queryByTestId('settings-view')).toBeNull();
    expect(screen.getByTestId('project-view')).toHaveTextContent('Project proj-1');
  });

  it('starting a new project from settings navigates to the create view', () => {
    render(<AppLikeHarness />);

    expect(screen.getByTestId('settings-view')).toBeInTheDocument();

    fireEvent.click(screen.getByText('+ New Project'));

    expect(screen.queryByTestId('settings-view')).toBeNull();
    expect(screen.getByTestId('create-view')).toBeInTheDocument();
  });
});

// ─── Route type shape + page helpers ─────────────────────────────────────────

describe('Route type', () => {
  it('project route carries sessionId that survives a page change when spread', () => {
    const route: Route = { name: 'project', projectId: 'p1', sessionId: 'sess-99' };
    const after: Route = { ...(route as Extract<Route, { name: 'project' }>), page: 'tasks' };
    expect(after).toMatchObject({ name: 'project', projectId: 'p1', page: 'tasks', sessionId: 'sess-99' });
  });

  it('project route sessionId and page default to undefined (the chat home)', () => {
    const route: Route = { name: 'project', projectId: 'p1' };
    expect((route as Extract<Route, { name: 'project' }>).sessionId).toBeUndefined();
    expect((route as Extract<Route, { name: 'project' }>).page).toBeUndefined();
  });

  it('project route accepts an optional draft (Workbench prefill doorway) and defaults to undefined', () => {
    const withDraft: Route = { name: 'project', projectId: 'p1', draft: 'hello', draftSkill: 'automate' };
    expect((withDraft as Extract<Route, { name: 'project' }>).draft).toBe('hello');
    expect((withDraft as Extract<Route, { name: 'project' }>).draftSkill).toBe('automate');

    const withoutDraft: Route = { name: 'project', projectId: 'p1' };
    expect((withoutDraft as Extract<Route, { name: 'project' }>).draft).toBeUndefined();
  });
});

describe('page helpers (spec 137)', () => {
  it('parseProjectPage accepts the registry keys and nothing else', () => {
    expect(parseProjectPage('tasks')).toBe('tasks');
    expect(parseProjectPage('settings')).toBe('settings');
    expect(parseProjectPage('pricing')).toBe('pricing');
    // The retired tab keys are not pages.
    expect(parseProjectPage('queue')).toBeUndefined();
    expect(parseProjectPage('files')).toBeUndefined();
    expect(parseProjectPage('chat')).toBeUndefined();
    expect(parseProjectPage(undefined)).toBeUndefined();
    expect(parseProjectPage(3)).toBeUndefined();
  });

  it('pageTrail walks the parent chain from the top down', () => {
    expect(pageTrail(undefined)).toEqual([]);
    expect(pageTrail('tasks')).toEqual(['tasks']);
    expect(pageTrail('settings')).toEqual(['settings']);
    expect(pageTrail('pricing')).toEqual(['settings', 'pricing']);
    expect(PROJECT_PAGE_PARENT.pricing).toBe('settings');
  });

  it('routeUp goes one level up and drops the one-shot preview path', () => {
    const base = { name: 'project' as const, projectId: 'p', sessionId: 's', previewPath: 'a.md' };
    expect(routeUp({ ...base, page: 'pricing' })).toMatchObject({ page: 'settings', sessionId: 's', previewPath: undefined });
    expect(routeUp({ ...base, page: 'settings' }).page).toBeUndefined();
    expect(routeUp({ ...base, page: 'tasks' }).page).toBeUndefined();
    expect(routeUp(base).page).toBeUndefined();
  });

  it('routeKey serialises which screen is showing, and only that', () => {
    expect(routeKey({ name: 'list' })).toBe('list');
    expect(routeKey({ name: 'project', projectId: 'p', sessionId: 's' })).toBe('project:p:chat:s');
    expect(routeKey({ name: 'project', projectId: 'p', page: 'tasks' })).toBe('project:p:tasks:');
    // A pane switch or a draft is not a different screen.
    expect(routeKey({ name: 'project', projectId: 'p', page: 'tasks', queuePane: 'automations', draft: 'x' })).toBe(
      'project:p:tasks:',
    );
  });
});
