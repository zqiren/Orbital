// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Project } from '../types';
import SettingsView from './SettingsView';

const mockApi = vi.fn();

vi.mock('../config', () => ({
  api: (...args: unknown[]) => mockApi(...args),
  BASE_URL: '',
  isRelayMode: false,
  ApiError: class ApiError extends Error {
    detail = '';
  },
}));

vi.mock('./SettingsRail', () => ({
  default: () => null,
  scrollToSettingsSection: vi.fn(),
}));
vi.mock('./LLMProviderSettings', () => ({ default: () => null }));
vi.mock('./FallbackModelsEditor', () => ({ default: () => null }));
vi.mock('./BudgetSection', () => ({
  default: (p: { limit: string; onLimitChange: (v: string) => void }) => (
    <input
      aria-label="Budget Limit (USD)"
      value={p.limit}
      onChange={(e) => p.onLimitChange(e.target.value)}
    />
  ),
}));
vi.mock('./ProjectConnectorToggles', () => ({ default: () => null }));
vi.mock('./NetworkAccessSection', () => ({ NetworkAccessSection: () => null }));
vi.mock('./SubAgentCard', () => ({
  default: (p: { slug: string; ready: boolean }) => (
    <div data-testid={`sub-agent-card-${p.slug}`} data-ready={String(p.ready)} />
  ),
}));

// Daemon-global WS events. Keep the handlers so tests can emit at them.
type WsHandler = (event: unknown) => void;
const wsHandlers = new Map<string, Set<WsHandler>>();
vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    on: (type: string, fn: WsHandler) => {
      if (!wsHandlers.has(type)) wsHandlers.set(type, new Set());
      wsHandlers.get(type)!.add(fn);
    },
    off: (type: string, fn: WsHandler) => {
      wsHandlers.get(type)?.delete(fn);
    },
    connectionState: 'connected',
    subscribe: vi.fn(),
  }),
}));

const project: Project = {
  project_id: 'project-1',
  name: 'Orbital',
  workspace: '/tmp/orbital',
  model: 'test-model',
  api_key: '',
  base_url: null,
  autonomy: 'hands_off',
  instructions: '',
  sub_agent_deployment_instructions: 'Stale list value',
};

/**
 * Project settings → Skills: a project skill whose name is already in
 * Orbital's global list shows as shared (the check) on load, so the shared
 * state survives a reload; the others offer the share action.
 */
describe('SettingsView skills — shared with all projects', () => {
  beforeEach(() => {
    mockApi.mockReset();
    mockApi.mockImplementation((path: string) => {
      if (path === '/api/v2/providers') return Promise.resolve({});
      if (path === '/api/v2/projects/project-1') return Promise.resolve({ ...project });
      if (path === '/api/v2/projects/project-1/skills') {
        return Promise.resolve([
          { name: 'Weekly-Report', description: 'Steps', path: '/w/a/SKILL.md', dir_name: 'weekly-report' },
          { name: 'invoice-flow', description: 'Bills', path: '/w/b/SKILL.md', dir_name: 'invoice-flow' },
        ]);
      }
      if (path === '/api/v2/skills') {
        return Promise.resolve({
          groups: [
            { source: 'global', path: '/d/skills', editable: true, exists: true, skills: [
              { name: 'weekly-report', description: 'x', source: 'global', path: '/d/skills/weekly-report/SKILL.md', dir_name: 'weekly-report', editable: true },
            ] },
            // A home-folder skill of the same name is not "shared by Orbital".
            { source: 'claude', path: '/u/.claude/skills', editable: false, exists: true, skills: [
              { name: 'invoice-flow', description: 'x', source: 'claude', path: '/u/.claude/skills/invoice-flow/SKILL.md', dir_name: 'invoice-flow', editable: false },
            ] },
          ],
        });
      }
      return Promise.resolve([]);
    });
  });

  it('marks a project skill already in the global list as shared on load', async () => {
    render(<SettingsView project={project} onSave={vi.fn()} onDelete={vi.fn()} />);
    expect(await screen.findByTestId('make-global-weekly-report-done')).toBeInTheDocument();
    expect(screen.getByTestId('make-global-invoice-flow')).toBeInTheDocument();
    expect(screen.queryByTestId('make-global-invoice-flow-done')).not.toBeInTheDocument();
  });
});
