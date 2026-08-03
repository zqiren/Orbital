// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Project } from '../types';
import { LocaleProvider } from '../i18n/LocaleContext';
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
vi.mock('./BudgetSection', () => ({ default: () => null }));
vi.mock('./ProjectConnectorToggles', () => ({ default: () => null }));
vi.mock('./NetworkAccessSection', () => ({ NetworkAccessSection: () => null }));
vi.mock('./SubAgentCard', () => ({ default: () => null }));

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

describe('SettingsView sub-agent deployment instructions', () => {
  beforeEach(() => {
    localStorage.clear();
    mockApi.mockReset();
    mockApi.mockImplementation((path: string) => {
      if (path === '/api/v2/providers') return Promise.resolve({});
      if (path === '/api/v2/projects/project-1') {
        return Promise.resolve({
          ...project,
          sub_agent_deployment_instructions: 'Use Codex for implementation.',
        });
      }
      return Promise.resolve([]);
    });
  });

  it('hydrates from project detail and saves both populated and cleared values', async () => {
    const onSave = vi.fn();
    render(<SettingsView project={project} onSave={onSave} onDelete={vi.fn()} />);

    const textarea = screen.getByRole('textbox', {
      name: 'Sub-agent deployment instructions',
    });
    expect(textarea).toHaveAttribute('rows', '4');
    expect(textarea).toHaveAttribute('maxlength', '4000');

    await waitFor(() => {
      expect(textarea).toHaveValue('Use Codex for implementation.');
    });

    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(onSave).toHaveBeenLastCalledWith(
      expect.objectContaining({
        sub_agent_deployment_instructions: 'Use Codex for implementation.',
      }),
    );

    fireEvent.change(textarea, { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(onSave).toHaveBeenLastCalledWith(
      expect.objectContaining({ sub_agent_deployment_instructions: '' }),
    );
  });

  it('renders the deployment field in Simplified Chinese', async () => {
    localStorage.setItem('orbital.locale', 'zh');
    render(
      <LocaleProvider>
        <SettingsView project={project} onSave={vi.fn()} onDelete={vi.fn()} />
      </LocaleProvider>,
    );

    const textarea = screen.getByRole('textbox', { name: '子Agent派发指示' });
    await waitFor(() => expect(textarea).not.toBeDisabled());
    expect(textarea).toHaveAttribute(
      'placeholder',
      '示例：使用 Gemini 进行调研，Claude Code 负责规划和审查，Codex 负责实现。除非任务无关，否则继续使用现有会话。',
    );
    expect(screen.getByText(/留空则由管理Agent自行决定/)).toBeInTheDocument();
  });
});

// Bug #36: GET /api/v2/settings/sub-agents took a measured 7.28 s on a cold
// packaged daemon, and for all 7 s the section claimed the user had no
// sub-agents installed. The empty state must not stand in for "still checking".
describe('SettingsView sub-agents loading state', () => {
  const INSTALL_HINT = "Install an agent's CLI on your machine to use it here.";
  const CHECKING = 'Checking installed sub-agents…';

  let settleSubAgents: {
    resolve: (value: unknown) => void;
    reject: (reason: unknown) => void;
  };

  beforeEach(() => {
    localStorage.clear();
    mockApi.mockReset();
    mockApi.mockImplementation((path: string) => {
      if (path === '/api/v2/settings/sub-agents') {
        return new Promise((resolve, reject) => {
          settleSubAgents = { resolve, reject };
        });
      }
      if (path === '/api/v2/providers') return Promise.resolve({});
      if (path === '/api/v2/projects/project-1') return Promise.resolve({ ...project });
      return Promise.resolve([]);
    });
  });

  it('shows a checking message instead of the install hint while the probe is in flight', async () => {
    render(<SettingsView project={project} onSave={vi.fn()} onDelete={vi.fn()} />);

    expect(screen.getByText(CHECKING)).toBeInTheDocument();
    expect(screen.queryByText(INSTALL_HINT)).toBeNull();

    await act(async () => {
      settleSubAgents.resolve([]);
    });

    await waitFor(() => expect(screen.getByText(INSTALL_HINT)).toBeInTheDocument());
    expect(screen.queryByText(CHECKING)).toBeNull();
  });

  it('clears the checking message when the probe fails, so the section never wedges', async () => {
    render(<SettingsView project={project} onSave={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.getByText(CHECKING)).toBeInTheDocument();

    await act(async () => {
      settleSubAgents.reject(new Error('probe failed'));
    });

    await waitFor(() => expect(screen.getByText(INSTALL_HINT)).toBeInTheDocument());
    expect(screen.queryByText(CHECKING)).toBeNull();
  });

  it('renders the installed list, not the checking message, once the probe resolves', async () => {
    render(<SettingsView project={project} onSave={vi.fn()} onDelete={vi.fn()} />);

    await act(async () => {
      settleSubAgents.resolve([
        { slug: 'codex', name: 'Codex', installed: true, ready: true },
      ]);
    });

    await waitFor(() => expect(screen.queryByText(CHECKING)).toBeNull());
    expect(screen.getByText(/Remember to Save enablement/)).toBeInTheDocument();
  });
});
