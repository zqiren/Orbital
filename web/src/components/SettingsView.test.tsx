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

  it('hydrates from project detail, then saves what is typed — cleared included', async () => {
    const onSave = vi.fn(() => Promise.resolve());
    render(<SettingsView project={project} onSave={onSave} onDelete={vi.fn()} />);

    const textarea = screen.getByRole('textbox', {
      name: 'Sub-agent deployment instructions',
    });
    expect(textarea).toHaveAttribute('rows', '4');
    expect(textarea).toHaveAttribute('maxlength', '4000');

    await waitFor(() => {
      expect(textarea).toHaveValue('Use Codex for implementation.');
    });
    // Loading the page is not an edit.
    expect(onSave).not.toHaveBeenCalled();

    fireEvent.change(textarea, { target: { value: 'Use Gemini for research.' } });
    fireEvent.blur(textarea);
    await waitFor(() =>
      expect(onSave).toHaveBeenLastCalledWith({
        sub_agent_deployment_instructions: 'Use Gemini for research.',
      }),
    );

    fireEvent.change(textarea, { target: { value: '' } });
    fireEvent.blur(textarea);
    await waitFor(() =>
      expect(onSave).toHaveBeenLastCalledWith({ sub_agent_deployment_instructions: '' }),
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
    // The hint is behind the label's ⓘ now, not printed under it.
    expect(screen.queryByText(/留空则由管理Agent自行决定/)).not.toBeInTheDocument();
    // The ⓘ that belongs to THIS label — a positional index would silently
    // follow whichever other section happens to gain or lose a description.
    const hintToggle = screen
      .getByText('子Agent派发指示')
      .parentElement!.querySelector('button');
    fireEvent.click(hintToggle!);
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
    // Toggles save themselves now; there is no Save below to remind about.
    expect(screen.queryByText(/Remember to Save/)).toBeNull();
  });
});

// 2026-09-19: the page had one Save button at the bottom, so picking a model
// card (or anything else) looked applied and was not — the user re-picked a
// card several times. Every edit now saves itself.
describe('SettingsView autosave', () => {
  const CARD = {
    id: 'card_glm',
    name: 'OpenCode Go · glm-5.3-flash',
    provider: 'opencode-go',
    region: 'global',
    base_url: null,
    sdk: null,
    model: 'glm-5.3-flash',
    created_at: '2026-09-19T00:00:00+00:00',
    verified_at: null,
    last_used_at: null,
    last_error: null,
    key_set: true,
    key_masked: 'sk-o...wxyz',
    key_source: 'keychain',
    is_default: false,
    read_only: false,
  };
  const DEFAULT = { ...CARD, id: 'card_default', name: 'OpenCode Go · deepseek-v4-flash', model: 'deepseek-v4-flash', is_default: true };

  beforeEach(() => {
    localStorage.clear();
    mockApi.mockReset();
    mockApi.mockImplementation((path: string) => {
      if (path === '/api/v2/providers') return Promise.resolve({});
      if (path === '/api/v2/projects/project-1') return Promise.resolve({ ...project });
      if (path === '/api/v2/settings') {
        return Promise.resolve({ credential_cards: [DEFAULT, CARD], default_card_id: 'card_default' });
      }
      return Promise.resolve([]);
    });
  });

  it('has no Save button and says changes save automatically', () => {
    render(<SettingsView project={project} onSave={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.queryByRole('button', { name: 'Save' })).toBeNull();
    expect(screen.getByTestId('settings-autosave-hint')).toHaveTextContent(
      'Changes save automatically.',
    );
  });

  it('picking a model card saves card_id at once, and "Global default" saves null', async () => {
    const onSave = vi.fn(() => Promise.resolve());
    render(<SettingsView project={project} onSave={onSave} onDelete={vi.fn()} />);

    fireEvent.click(await screen.findByTestId('card-select-card_glm'));
    expect(onSave).toHaveBeenCalledWith({ card_id: 'card_glm' });
    expect(screen.getByTestId('card-selected-card_glm')).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('project-card-picker-global-default'));
    await waitFor(() => expect(onSave).toHaveBeenLastCalledWith({ card_id: null }));
    // Same race as the Retry test below: the pill exists as "Saving…" first.
    await waitFor(() =>
      expect(screen.getByTestId('settings-autosave-status')).toHaveTextContent('Saved'),
    );
  });

  it('an autonomy click saves at once', () => {
    const onSave = vi.fn(() => Promise.resolve());
    render(<SettingsView project={project} onSave={onSave} onDelete={vi.fn()} />);
    fireEvent.click(screen.getByText('Supervised'));
    expect(onSave).toHaveBeenCalledWith({ autonomy: 'supervised' });
  });

  it('typing saves by itself once the user pauses', async () => {
    const onSave = vi.fn(() => Promise.resolve());
    render(<SettingsView project={project} onSave={onSave} onDelete={vi.fn()} />);
    const name = screen.getByPlaceholderText('Display name for this agent');
    fireEvent.change(name, { target: { value: 'Scout' } });
    expect(onSave).not.toHaveBeenCalled();
    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ agent_name: 'Scout' }), {
      timeout: 2000,
    });
  });

  it('a budget limit is saved with the currency the form shows', async () => {
    const onSave = vi.fn(() => Promise.resolve());
    render(<SettingsView project={project} onSave={onSave} onDelete={vi.fn()} />);
    const limit = screen.getByLabelText('Budget Limit (USD)');
    fireEvent.change(limit, { target: { value: '12.5' } });
    fireEvent.blur(limit);
    await waitFor(() =>
      expect(onSave).toHaveBeenLastCalledWith({ budget_limit_usd: 12.5, budget_currency: 'USD' }),
    );
  });

  it('a failed save says so and Retry sends it again', async () => {
    const onSave = vi
      .fn()
      .mockRejectedValueOnce(new Error('Project not found'))
      .mockResolvedValue(undefined);
    render(<SettingsView project={project} onSave={onSave} onDelete={vi.fn()} />);

    fireEvent.click(screen.getByText('Supervised'));
    // waitFor, not findByTestId + a synchronous assert: the pill mounts as
    // "Saving…" first, so finding it says nothing about the rejection having
    // landed. On a slow CI runner it had not, and this flaked (same SHA red on
    // the main push, green on the tag push).
    await waitFor(() =>
      expect(screen.getByTestId('settings-autosave-status')).toHaveTextContent(
        "Couldn't save: Project not found",
      ),
    );
    fireEvent.click(screen.getByTestId('settings-autosave-retry'));
    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(2));
    expect(onSave).toHaveBeenLastCalledWith({ autonomy: 'supervised' });
    await waitFor(() =>
      expect(screen.getByTestId('settings-autosave-status')).toHaveTextContent('Saved'),
    );
  });
});
