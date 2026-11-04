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
// Spec 136 C.3: the section is built from the one sub-agent-memory reply,
// which now carries `ready`; /settings/sub-agents is not asked at all.
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
      if (path === '/api/v2/projects/project-1/sub-agent-memory') {
        return new Promise((resolve, reject) => {
          settleSubAgents = { resolve, reject };
        });
      }
      if (path === '/api/v2/providers') return Promise.resolve({});
      if (path === '/api/v2/projects/project-1') return Promise.resolve({ ...project });
      return Promise.resolve([]);
    });
  });

  const memoryEntry = (slug: string, name: string, ready?: boolean) => ({
    agent_slug: slug, agent_name: name, exists: false, content: '',
    last_modified: null, size_bytes: 0, ...(ready === undefined ? {} : { ready }),
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
      settleSubAgents.resolve([memoryEntry('codex', 'Codex', true)]);
    });

    await waitFor(() => expect(screen.queryByText(CHECKING)).toBeNull());
    expect(screen.getByTestId('sub-agent-card-codex')).toBeInTheDocument();
    // Toggles save themselves now; there is no Save below to remind about.
    expect(screen.queryByText(/Remember to Save/)).toBeNull();
  });

  it('builds the cards from the memory reply alone, with its ready flag', async () => {
    render(<SettingsView project={project} onSave={vi.fn()} onDelete={vi.fn()} />);
    await act(async () => {
      settleSubAgents.resolve([
        memoryEntry('codex', 'Codex', false),
        memoryEntry('cursor', 'Cursor'), // an older daemon sends no `ready`
      ]);
    });
    await waitFor(() => expect(screen.getByTestId('sub-agent-card-codex')).toBeInTheDocument());
    expect(screen.getByTestId('sub-agent-card-codex')).toHaveAttribute('data-ready', 'false');
    expect(screen.getByTestId('sub-agent-card-cursor')).toHaveAttribute('data-ready', 'true');
    const paths = mockApi.mock.calls.map((c) => c[0]);
    expect(paths).not.toContain('/api/v2/settings/sub-agents');
  });

  it('refetches the list when the daemon says the sub-agents changed', async () => {
    render(<SettingsView project={project} onSave={vi.fn()} onDelete={vi.fn()} />);
    await act(async () => {
      settleSubAgents.resolve([memoryEntry('codex', 'Codex', false)]);
    });
    await waitFor(() => expect(screen.getByTestId('sub-agent-card-codex')).toBeInTheDocument());

    act(() => {
      wsHandlers.get('sub_agents.updated')?.forEach((fn) => fn({ type: 'sub_agents.updated' }));
    });
    // The refetch must not drop the cards back to "checking…" meanwhile.
    expect(screen.queryByText(CHECKING)).toBeNull();
    await act(async () => {
      settleSubAgents.resolve([memoryEntry('codex', 'Codex', true)]);
    });
    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-card-codex')).toHaveAttribute('data-ready', 'true'));
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

  it('autonomy tiles are Float static with no border; the selected one carries the accent ring', () => {
    render(<SettingsView project={project} onSave={vi.fn(() => Promise.resolve())} onDelete={vi.fn()} />);
    expectFloatTile('Hands-off', true);
    expectFloatTile('Check-in', false);
    expectFloatTile('Supervised', false);
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

// Spec 117 §3.7: the Safe mode row, the sections that only make sense inside
// the sandbox, and the "runs outside the sandbox" list with its remove.
describe('SettingsView Safe mode (spec 117)', () => {
  const safeProject: Project = {
    ...project,
    safe_mode: true,
    approved_domains: ['example.com'],
    unsandboxed_executables: ['git', 'gh'],
    pending_unsandboxed_runs: [
      {
        id: 'run-1',
        command: 'git push origin main',
        reason: 'Seatbelt denied file-read-data ~/.ssh/config',
        created_at: '2026-10-18T00:00:00+00:00',
        session_id: 'sess-1',
      },
    ],
    granted_folders: [{ path: '/Users/me/Shared', mode: 'read_only' }],
  };

  beforeEach(() => {
    localStorage.clear();
    mockApi.mockReset();
    mockApi.mockImplementation((path: string) => {
      if (path === '/api/v2/providers') return Promise.resolve({});
      if (path === '/api/v2/projects/project-1') return Promise.resolve({ ...safeProject });
      if (path.includes('/unsandboxed-runs/')) return Promise.resolve({ status: 'ok' });
      if (path === '/api/v2/platform/folders/revoke') return Promise.resolve({ status: 'ok' });
      return Promise.resolve([]);
    });
  });

  function safeModeSwitch() {
    return screen.getByRole('switch', { name: 'Safe mode' });
  }

  it('reflects the project: on, off, and absent (treated as on)', () => {
    const { unmount } = render(
      <SettingsView project={safeProject} onSave={vi.fn()} onDelete={vi.fn()} />,
    );
    expect(safeModeSwitch().getAttribute('aria-checked')).toBe('true');
    unmount();

    const { unmount: unmount2 } = render(
      <SettingsView project={{ ...safeProject, safe_mode: false }} onSave={vi.fn()} onDelete={vi.fn()} />,
    );
    expect(safeModeSwitch().getAttribute('aria-checked')).toBe('false');
    unmount2();

    render(<SettingsView project={{ ...safeProject, safe_mode: undefined }} onSave={vi.fn()} onDelete={vi.fn()} />);
    expect(safeModeSwitch().getAttribute('aria-checked')).toBe('true');
  });

  it('a flip saves safe_mode at once', async () => {
    const onSave = vi.fn(() => Promise.resolve());
    render(<SettingsView project={safeProject} onSave={onSave} onDelete={vi.fn()} />);
    // Let the mount-time detail GET land first: it hydrates the same value
    // the row was rendered with, and a flip raced against it would be
    // overwritten by the hydration, not by the user.
    await act(async () => {});
    fireEvent.click(safeModeSwitch());
    expect(onSave).toHaveBeenCalledWith({ safe_mode: false });
    expect(safeModeSwitch().getAttribute('aria-checked')).toBe('false');
    fireEvent.click(safeModeSwitch());
    // The second save queues behind the first one's promise.
    await waitFor(() => expect(onSave).toHaveBeenLastCalledWith({ safe_mode: true }));
  });

  it('hides Folder access and Network access when safe mode is off, keeps them when on', () => {
    const { container, unmount } = render(
      <SettingsView project={safeProject} onSave={vi.fn()} onDelete={vi.fn()} />,
    );
    expect(container.querySelector('[data-settings-section="network"]')).not.toBeNull();
    expect(container.querySelector('[data-settings-section="folders"]')).not.toBeNull();
    expect(container.querySelector('[data-settings-section="unsandboxed"]')).not.toBeNull();
    fireEvent.click(safeModeSwitch());
    expect(container.querySelector('[data-settings-section="network"]')).toBeNull();
    expect(container.querySelector('[data-settings-section="folders"]')).toBeNull();
    unmount();

    const { container: c2 } = render(
      <SettingsView project={{ ...safeProject, safe_mode: false }} onSave={vi.fn()} onDelete={vi.fn()} />,
    );
    expect(c2.querySelector('[data-settings-section="network"]')).toBeNull();
    expect(c2.querySelector('[data-settings-section="folders"]')).toBeNull();
  });

  it('lists the executables allowed outside the sandbox and saves the list without a removed one', () => {
    const onSave = vi.fn(() => Promise.resolve());
    render(<SettingsView project={safeProject} onSave={onSave} onDelete={vi.fn()} />);
    expect(screen.getByText('git')).toBeInTheDocument();
    expect(screen.getByText('gh')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Remove git' }));
    expect(onSave).toHaveBeenCalledWith({ unsandboxed_executables: ['gh'] });
    expect(screen.queryByText('git')).toBeNull();
  });

  it('answers a parked request through the resolve route and applies the answer locally', async () => {
    const onSave = vi.fn(() => Promise.resolve());
    render(<SettingsView project={safeProject} onSave={onSave} onDelete={vi.fn()} />);
    expect(screen.getByText('git push origin main')).toBeInTheDocument();
    expect(screen.getByText('Seatbelt denied file-read-data ~/.ssh/config')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Always allow git in this project' }));
    await waitFor(() =>
      expect(mockApi).toHaveBeenCalledWith(
        '/api/v2/projects/project-1/unsandboxed-runs/run-1/resolve',
        { method: 'POST', body: JSON.stringify({ decision: 'always' }) },
      ),
    );
    await waitFor(() => expect(screen.queryByText('git push origin main')).toBeNull());
    // The executable joins the remembered list locally; the daemon wrote it
    // server-side, so no PUT is sent for it.
    expect(screen.getByText('git')).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
  });

  it('offers all four answers on a parked request, and "turn off safe mode" flips the row', async () => {
    render(<SettingsView project={safeProject} onSave={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Allow once' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Always allow git in this project' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Turn off Safe mode for this project' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Deny' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Turn off Safe mode for this project' }));
    await waitFor(() =>
      expect(mockApi).toHaveBeenCalledWith(
        '/api/v2/projects/project-1/unsandboxed-runs/run-1/resolve',
        { method: 'POST', body: JSON.stringify({ decision: 'disable_safe_mode' }) },
      ),
    );
    await waitFor(() => expect(safeModeSwitch().getAttribute('aria-checked')).toBe('false'));
  });

  it('removes a granted folder through the platform revoke route with the project id', async () => {
    render(<SettingsView project={safeProject} onSave={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.getByText('/Users/me/Shared')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Remove /Users/me/Shared' }));
    await waitFor(() =>
      expect(mockApi).toHaveBeenCalledWith('/api/v2/platform/folders/revoke', {
        method: 'POST',
        body: JSON.stringify({ path: '/Users/me/Shared', project_id: 'project-1' }),
      }),
    );
    await waitFor(() => expect(screen.queryByText('/Users/me/Shared')).toBeNull());
  });

  it('renders the row in Simplified Chinese with the spec copy', () => {
    localStorage.setItem('orbital.locale', 'zh');
    render(
      <LocaleProvider>
        <SettingsView project={safeProject} onSave={vi.fn()} onDelete={vi.fn()} />
      </LocaleProvider>,
    );
    expect(screen.getByRole('switch', { name: '安全模式' })).toBeInTheDocument();
    expect(screen.getByText('沙盒外运行')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '允许这一次' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '本项目始终允许 git' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '关闭本项目的安全模式' })).toBeInTheDocument();
  });
});

// Float pass (1104): a radio choice tile is a Float static layer with no
// border; the selected one adds the accent ring (and keeps its accent tint).
function expectFloatTile(title: string, selected: boolean) {
  const tile = screen.getByText(title).closest('button') as HTMLButtonElement;
  const cls = tile.className.split(/\s+/);
  expect(cls).toContain('surface-float-static');
  expect(tile.className).not.toMatch(/(^|\s)border(-|\s|$)/);
  if (selected) {
    expect(cls).toContain('ring-1');
    expect(tile.className).toContain('ring-accent');
  } else {
    expect(tile.className).not.toMatch(/(^|\s)ring-/);
  }
}
