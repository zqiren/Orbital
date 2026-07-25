// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Tests for the Route type and Sidebar + ProjectDetail routing integration.
 *
 * Requirements verified:
 *   1. Selecting a project produces a project route with tab 'chat' and the right projectId.
 *   2. Switching tabs updates route.tab and preserves projectId (and sessionId if set).
 *   3. Setting route.sessionId and then switching tabs preserves sessionId.
 *   4. The settings flag (route.settings) toggles the settings surface without losing
 *      the project/tab context, and "back" (re-clicking a tab) returns to the project route.
 *   5. Sidebar produces the correct project route when a project is clicked.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { useState, type SetStateAction } from 'react';
import { describe, expect, it, vi } from 'vitest';
import type { Route } from './route';

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
        onSelectCalendar={vi.fn()}
        onSelectWorkbench={vi.fn()}
        onNewProject={vi.fn()}
        onSettings={vi.fn()}
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
        route={{ name: 'project', projectId: 'alpha', tab: 'chat' }}
        connectionState="connected"
        onSelectProject={onSelectProject}
        onSelectCalendar={vi.fn()}
        onSelectWorkbench={vi.fn()}
        onNewProject={vi.fn()}
        onSettings={vi.fn()}
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
        route={{ name: 'project', projectId: 'active', tab: 'chat' }}
        connectionState="connected"
        onSelectProject={vi.fn()}
        onSelectCalendar={vi.fn()}
        onSelectWorkbench={vi.fn()}
        onNewProject={vi.fn()}
        onSettings={vi.fn()}
      />,
    );

    // The active button carries the selection tint at full strength; the
    // inactive one only carries it as a /50 hover. The nav column uses its own
    // `nav-hover` tint rather than `card-hover`, because card-hover sits only
    // ~3 levels off the nav surface and would be invisible against it.
    const buttons = container.querySelectorAll('nav button');
    expect(buttons[0].className).toContain('bg-nav-hover');
    expect(buttons[1].className).not.toMatch(/\bbg-nav-hover\b(?!\/)/);
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
      setRoute({ name: 'project', projectId: id, tab: 'chat', sessionId: undefined });
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
        onSelectCalendar={vi.fn()}
        onSelectWorkbench={vi.fn()}
        onNewProject={vi.fn()}
        onSettings={vi.fn()}
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

describe('ProjectDetail routing', () => {
  const project = makeProject('proj-42');
  const baseRoute: Extract<Route, { name: 'project' }> = {
    name: 'project',
    projectId: 'proj-42',
    tab: 'chat',
  };

  it('switching tabs updates route.tab and preserves projectId', () => {
    const setRoute = vi.fn<(value: SetStateAction<Route>) => void>();

    render(
      <ProjectDetail
        project={project}
        agentStatus={'idle' as AgentRunStatus}
        route={baseRoute}
        setRoute={setRoute}

      />,
    );

    fireEvent.click(screen.getByText('Queue'));
    expect(setRoute).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'project', projectId: 'proj-42', tab: 'queue' }),
    );
  });

  it('switching tabs preserves sessionId when set', () => {
    const setRoute = vi.fn<(value: SetStateAction<Route>) => void>();
    const routeWithSession: Extract<Route, { name: 'project' }> = {
      ...baseRoute,
      tab: 'chat',
      sessionId: 'sess-abc',
    };

    render(
      <ProjectDetail
        project={project}
        agentStatus={'idle' as AgentRunStatus}
        route={routeWithSession}
        setRoute={setRoute}

      />,
    );

    fireEvent.click(screen.getByText('Files'));
    const produced = setRoute.mock.calls[0][0];
    expect(produced).toMatchObject({
      name: 'project',
      projectId: 'proj-42',
      tab: 'files',
      sessionId: 'sess-abc',
    });
  });

  it('clicking the gear icon sets route.settings=true, preserving project/tab', () => {
    const setRoute = vi.fn<(value: SetStateAction<Route>) => void>();

    render(
      <ProjectDetail
        project={project}
        agentStatus={'idle' as AgentRunStatus}
        route={baseRoute}
        setRoute={setRoute}

      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /project settings/i }));
    const produced = setRoute.mock.calls[0][0];
    expect(produced).toMatchObject({
      name: 'project',
      projectId: 'proj-42',
      tab: 'chat',
      settings: true,
    });
  });

  it('clicking a tab from settings clears route.settings (back to project)', () => {
    const setRoute = vi.fn<(value: SetStateAction<Route>) => void>();
    const settingsRoute: Extract<Route, { name: 'project' }> = {
      ...baseRoute,
      tab: 'queue',
      settings: true,
    };

    render(
      <ProjectDetail
        project={project}
        agentStatus={'idle' as AgentRunStatus}
        route={settingsRoute}
        setRoute={setRoute}

      />,
    );

    fireEvent.click(screen.getByText('Chat'));
    const produced = setRoute.mock.calls[0][0];
    expect(produced).toMatchObject({
      name: 'project',
      projectId: 'proj-42',
      tab: 'chat',
      settings: false,
    });
  });

  it('gear icon is present in the header regardless of route.settings value', () => {
    render(
      <ProjectDetail
        project={project}
        agentStatus={'idle' as AgentRunStatus}
        route={{ ...baseRoute, settings: true }}
        setRoute={vi.fn()}

      />,
    );

    // The gear button lives in the header — it must always be accessible
    expect(screen.getByRole('button', { name: /project settings/i })).toBeInTheDocument();
  });

  it('no tab is highlighted when settings is active', () => {
    render(
      <ProjectDetail
        project={project}
        agentStatus={'idle' as AgentRunStatus}
        route={{ ...baseRoute, tab: 'chat', settings: true }}
        setRoute={vi.fn()}

      />,
    );

    // Chat tab should NOT be highlighted
    const chatBtn = screen.getByText('Chat');
    expect(chatBtn.className).not.toContain('border-b-2');
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
      setRoute({ name: 'project', projectId: id, tab: 'chat', sessionId: undefined });
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
          onSelectCalendar={vi.fn()}
          onSelectWorkbench={vi.fn()}
          onNewProject={handleNewProject}
          onSettings={() => setRoute({ name: 'settings' })}
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

// ─── Route type shape tests ───────────────────────────────────────────────────

describe('Route type', () => {
  it('project route carries sessionId that survives tab changes when spread', () => {
    const route: Route = { name: 'project', projectId: 'p1', tab: 'chat', sessionId: 'sess-99' };
    // Simulate what ProjectDetail.handleTabChange does
    const after: Route = { ...(route as Extract<Route, { name: 'project' }>), tab: 'files', settings: false };
    expect(after).toMatchObject({ name: 'project', projectId: 'p1', tab: 'files', sessionId: 'sess-99' });
  });

  it('project route sessionId defaults to undefined when not provided', () => {
    const route: Route = { name: 'project', projectId: 'p1', tab: 'queue' };
    expect((route as Extract<Route, { name: 'project' }>).sessionId).toBeUndefined();
  });

  it('project route accepts an optional draft (Workbench prefill doorway) and defaults to undefined', () => {
    const withDraft: Route = { name: 'project', projectId: 'p1', tab: 'chat', draft: 'hello' };
    expect((withDraft as Extract<Route, { name: 'project' }>).draft).toBe('hello');

    const withoutDraft: Route = { name: 'project', projectId: 'p1', tab: 'chat' };
    expect((withoutDraft as Extract<Route, { name: 'project' }>).draft).toBeUndefined();
  });
});
