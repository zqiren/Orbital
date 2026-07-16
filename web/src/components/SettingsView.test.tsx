// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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
