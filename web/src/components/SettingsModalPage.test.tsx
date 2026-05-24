// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import SettingsModalPage from './SettingsModalPage';
import type { Project } from '../types';
import type { Route } from '../route';

// Mock SettingsView to avoid its heavy API/data deps — the shell tests focus on
// SettingsModalPage's own layout and navigation behaviour, not settings form internals.
vi.mock('./SettingsView', () => ({
  default: () => <div data-testid="settings-view-stub">SettingsView</div>,
}));

const mockProject: Project = {
  project_id: 'proj-1',
  name: 'My Test Project',
  workspace: '/tmp/proj',
  model: 'claude-3-5-sonnet',
  api_key: '',
  base_url: null,
  autonomy: 'hands_off',
  instructions: '',
};

const baseRoute: Extract<Route, { name: 'project' }> = {
  name: 'project',
  projectId: 'proj-1',
  tab: 'chat',
  sessionId: 'sess-abc',
  settings: true,
};

describe('SettingsModalPage', () => {
  it('renders the back band with project name', () => {
    render(
      <SettingsModalPage
        project={mockProject}
        route={baseRoute}
        setRoute={vi.fn()}
        onSave={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    const backBtn = screen.getByTestId('settings-back-button');
    expect(backBtn).toBeInTheDocument();
    expect(backBtn.textContent).toContain('Back to My Test Project');
  });

  it('renders the uppercase mono label and page title with project name', () => {
    render(
      <SettingsModalPage
        project={mockProject}
        route={baseRoute}
        setRoute={vi.fn()}
        onSave={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    // Mono label
    expect(screen.getByText(/project · this project only/i)).toBeInTheDocument();
    // Title
    const title = screen.getByTestId('settings-modal-title');
    expect(title.textContent).toContain('Project settings — My Test Project');
  });

  it('renders the embedded SettingsView', () => {
    render(
      <SettingsModalPage
        project={mockProject}
        route={baseRoute}
        setRoute={vi.fn()}
        onSave={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    expect(screen.getByTestId('settings-view-stub')).toBeInTheDocument();
  });

  it('back button calls setRoute restoring project route with settings cleared and tab+sessionId preserved', () => {
    const setRoute = vi.fn();
    render(
      <SettingsModalPage
        project={mockProject}
        route={baseRoute}
        setRoute={setRoute}
        onSave={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId('settings-back-button'));
    expect(setRoute).toHaveBeenCalledTimes(1);
    expect(setRoute).toHaveBeenCalledWith({
      name: 'project',
      projectId: 'proj-1',
      tab: 'chat',
      sessionId: 'sess-abc',
      settings: false,
    });
  });

  it('back button preserves the queue tab when that was the active tab', () => {
    const setRoute = vi.fn();
    const queueRoute: Extract<Route, { name: 'project' }> = {
      ...baseRoute,
      tab: 'queue',
      sessionId: undefined,
    };
    render(
      <SettingsModalPage
        project={mockProject}
        route={queueRoute}
        setRoute={setRoute}
        onSave={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId('settings-back-button'));
    expect(setRoute).toHaveBeenCalledWith({
      name: 'project',
      projectId: 'proj-1',
      tab: 'queue',
      sessionId: undefined,
      settings: false,
    });
  });
});
