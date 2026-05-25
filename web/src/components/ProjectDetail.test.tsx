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
