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
    expect(screen.getByRole('button', { name: /^Queue$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Chat$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Files$/i })).toBeInTheDocument();
    // Must NOT have a standalone "Settings" tab button
    expect(screen.queryByRole('button', { name: /^Settings$/i })).toBeNull();
  });

  it('tab click calls setRoute with settings:false to clear settings overlay', () => {
    const setRoute = vi.fn();
    renderProjectDetail({ tab: 'chat', settings: false }, setRoute);
    fireEvent.click(screen.getByRole('button', { name: /^Queue$/i }));
    expect(setRoute).toHaveBeenCalledWith(
      expect.objectContaining({ tab: 'queue', settings: false }),
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
  it('hides the Calendar tab by default (no availability, no connector)', () => {
    renderProjectDetail();
    expect(screen.queryByRole('button', { name: /^Calendar$/i })).toBeNull();
  });

  it('shows the Calendar tab when calendarAvailable is true', () => {
    render(
      <ProjectDetail
        project={mockProject}
        agentStatus="idle"
        route={baseRoute}
        setRoute={vi.fn()}
        calendarAvailable
      />,
    );
    expect(screen.getByRole('button', { name: /^Calendar$/i })).toBeInTheDocument();
  });

  it('shows the Calendar tab when the project has the google-calendar connector enabled', () => {
    const project = { ...mockProject, enabled_connectors: ['google-calendar'] } as Project;
    render(
      <ProjectDetail
        project={project}
        agentStatus="idle"
        route={baseRoute}
        setRoute={vi.fn()}
      />,
    );
    expect(screen.getByRole('button', { name: /^Calendar$/i })).toBeInTheDocument();
  });

  it('keeps the Calendar tab hidden for an unrelated enabled connector', () => {
    const project = { ...mockProject, enabled_connectors: ['gmail'] } as Project;
    render(
      <ProjectDetail
        project={project}
        agentStatus="idle"
        route={baseRoute}
        setRoute={vi.fn()}
      />,
    );
    expect(screen.queryByRole('button', { name: /^Calendar$/i })).toBeNull();
  });

  it('clicking the Calendar tab routes to tab:calendar (clearing settings)', () => {
    const setRoute = vi.fn();
    render(
      <ProjectDetail
        project={mockProject}
        agentStatus="idle"
        route={baseRoute}
        setRoute={setRoute}
        calendarAvailable
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /^Calendar$/i }));
    expect(setRoute).toHaveBeenCalledWith(
      expect.objectContaining({ tab: 'calendar', settings: false }),
    );
  });
});
