// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import { LocaleProvider } from '../i18n/LocaleContext';

// Mock the config module's `api` helper so the component never touches the
// network. The mock returns whatever `mockEntries` is set to for the initial
// GET /api/v2/settings/sub-agents load.
const api = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => api(...args),
}));

// Install progress arrives over the daemon-global WS. Keep a registry of the
// handlers the card subscribes so tests can emit events at it.
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

import SubAgentSettings from './SubAgentSettings';

function emit(type: string, payload: Record<string, unknown>) {
  act(() => {
    wsHandlers.get(type)?.forEach(fn => fn({ type, ...payload }));
  });
}

interface InstallInfo {
  supported: boolean;
  state: 'installed' | 'installing' | 'failed' | 'not_installed';
  job_id?: string | null;
  platforms?: string[];
}

interface CredentialSpec {
  key: string;
  label?: string | null;
  type?: string | null;
  required?: boolean;
  configured?: boolean;
  has_setup_command?: boolean;
}

interface Entry {
  slug: string;
  name: string;
  installed: boolean;
  binary_path: string | null;
  version: string | null;
  ready: boolean;
  dependencies_met: boolean;
  missing_dependencies: string[];
  credentials_configured: boolean;
  missing_credentials: string[];
  supports_login?: boolean;
  setup_actions: unknown[];
  config: Record<string, string>;
  param_schema: Record<string, {
    allowed: string[] | null;
    default?: string | null;
  }>;
  install?: InstallInfo;
  emits_tool_activity?: boolean;
  credentials?: CredentialSpec[];
}

function makeEntry(overrides: Partial<Entry> = {}): Entry {
  return {
    slug: 'claude-code',
    name: 'Claude Code',
    installed: true,
    binary_path: '/usr/bin/claude',
    version: '1.0.0',
    ready: true,
    dependencies_met: true,
    missing_dependencies: [],
    credentials_configured: false,
    missing_credentials: ['claude_auth'],
    supports_login: true,
    setup_actions: [],
    config: {},
    param_schema: {},
    ...overrides,
  };
}

/** A not-installed dsh entry: Orbital-installable, API-key-only, silent. */
function makeDshEntry(overrides: Partial<Entry> = {}): Entry {
  return makeEntry({
    slug: 'dsh',
    name: 'DeepSeek Harness',
    installed: false,
    binary_path: null,
    version: null,
    ready: false,
    credentials_configured: false,
    missing_credentials: ['DEEPSEEK_API_KEY'],
    supports_login: false,
    install: { supported: true, state: 'not_installed', platforms: ['macos', 'linux'] },
    emits_tool_activity: false,
    credentials: [
      { key: 'DEEPSEEK_API_KEY', label: 'DeepSeek API Key', type: 'secret', required: true, configured: false, has_setup_command: false },
      { key: 'DEEPSEEK_BASE_URL', label: 'DeepSeek API Base URL', type: 'secret', required: false, configured: false, has_setup_command: false },
    ],
    ...overrides,
  });
}

afterEach(() => {
  cleanup();
  api.mockReset();
  wsHandlers.clear();
  localStorage.clear();
});

describe('SubAgentSettings', () => {
  beforeEach(() => {
    api.mockReset();
  });

  it('lists only installed agents (filters out not-installed)', async () => {
    api.mockResolvedValueOnce([
      makeEntry({ slug: 'claude-code', name: 'Claude Code', installed: true }),
      makeEntry({
        slug: 'gemini-cli',
        name: 'Gemini CLI',
        installed: false,
        binary_path: null,
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => {
      expect(screen.getByText('Claude Code')).toBeInTheDocument();
    });
    // The not-installed agent must not appear.
    expect(screen.queryByText('Gemini CLI')).not.toBeInTheDocument();
  });

  it('shows the install helper line near Refresh', async () => {
    api.mockResolvedValueOnce([makeEntry()]);

    render(<SubAgentSettings />);

    await waitFor(() => {
      expect(
        screen.getByText("Don't see an agent? Install its CLI, then Refresh."),
      ).toBeInTheDocument();
    });
    // Refresh button remains.
    expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
  });

  it('hides the Login button when supports_login is false', async () => {
    api.mockResolvedValueOnce([
      makeEntry({
        slug: 'apikey-only',
        name: 'API Key Only',
        installed: true,
        credentials_configured: false,
        supports_login: false,
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => {
      expect(screen.getByText('API Key Only')).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: /^login$/i })).not.toBeInTheDocument();
  });

  it('shows the Login button when supports_login is true and not logged in', async () => {
    api.mockResolvedValueOnce([
      makeEntry({
        slug: 'claude-code',
        name: 'Claude Code',
        installed: true,
        credentials_configured: false,
        supports_login: true,
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => {
      expect(screen.getByText('Claude Code')).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: /^login$/i })).toBeInTheDocument();
  });

  it('makes Cursor auto permission policy legible without a persisted override', async () => {
    api.mockResolvedValueOnce([
      makeEntry({
        slug: 'cursor',
        name: 'Cursor',
        config: {},
        param_schema: {
          model: { allowed: null, default: null },
          'permission-mode': {
            allowed: ['auto', 'ask'],
            default: 'auto',
          },
        },
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('Cursor')).toBeInTheDocument());
    expect(screen.getByRole('combobox')).toHaveValue('auto');
    expect(screen.getByRole('option', { name: 'Auto (default)' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Ask via permission card' })).toBeInTheDocument();
    expect(screen.getByText(/routes Cursor permission requests/)).toBeInTheDocument();
  });

  it('renders Cursor settings in Simplified Chinese without translating model IDs', async () => {
    localStorage.setItem('orbital.locale', 'zh');
    api.mockResolvedValueOnce([
      makeEntry({
        slug: 'cursor',
        name: 'Cursor',
        config: { model: 'cursor-grok-4.5-low' },
        param_schema: {
          model: { allowed: null, default: null },
          'permission-mode': {
            allowed: ['auto', 'ask'],
            default: 'auto',
          },
        },
      }),
    ]);

    render(
      <LocaleProvider>
        <SubAgentSettings />
      </LocaleProvider>,
    );

    await waitFor(() => expect(screen.getByText('Cursor')).toBeInTheDocument());
    expect(screen.getByDisplayValue('cursor-grok-4.5-low')).toBeInTheDocument();
    expect(screen.getByRole('option', { name: '自动执行（默认）' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: '通过权限卡询问' })).toBeInTheDocument();
    expect(screen.getByText(/转到 Orbital 权限卡/)).toBeInTheDocument();
  });

  it('keeps a stale saved model visible in the dropdown', async () => {
    // The codex model schema is live-populated from the account's model
    // list; a previously saved override may no longer be in it (that's how
    // the invalid `gpt-5.6` save was discovered). A <select> whose value has
    // no matching <option> renders BLANK — the user couldn't even see what
    // broken value is currently saved. The stale value must stay visible as
    // an option until they pick a valid one.
    api.mockResolvedValueOnce([
      makeEntry({
        slug: 'codex',
        name: 'Codex',
        config: { model: 'gpt-5.6' },
        param_schema: {
          model: { allowed: ['gpt-5.5', 'gpt-5.4-mini'], default: null },
        },
      }),
    ]);

    render(
      <LocaleProvider>
        <SubAgentSettings />
      </LocaleProvider>,
    );

    await waitFor(() => expect(screen.getByText('Codex')).toBeInTheDocument());
    const select = screen.getByRole('combobox') as HTMLSelectElement;
    expect(select.value).toBe('gpt-5.6');
    expect(screen.getByRole('option', { name: 'gpt-5.6' })).toBeInTheDocument();
    // The live options are still offered alongside it.
    expect(screen.getByRole('option', { name: 'gpt-5.5' })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Orbital-managed install
// ---------------------------------------------------------------------------

const LIST_PATH = '/api/v2/settings/sub-agents';
const listCallCount = () => api.mock.calls.filter(c => c[0] === LIST_PATH).length;

describe('SubAgentSettings — Orbital-managed install', () => {
  beforeEach(() => {
    api.mockReset();
    wsHandlers.clear();
  });

  /** List responses are consumed in order; the last one repeats. */
  function mockList(...responses: Entry[][]) {
    const queue = [...responses];
    api.mockImplementation(async (path: string) => {
      // The managed-credentials form probes the global LLM settings to decide
      // whether the provider-key checkbox can work at all.
      if (path === '/api/v2/settings') return { llm: { provider: 'deepseek', api_key_set: true } };
      if (path === LIST_PATH) return queue.length > 1 ? queue.shift()! : queue[0];
      if (path.endsWith('/install')) return { job_id: 'job-1', slug: 'dsh', state: 'installing' };
      throw new Error(`unexpected api call: ${path}`);
    });
  }

  it('lists a not-installed agent Orbital can install, with an Install button', async () => {
    mockList([makeDshEntry()]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('DeepSeek Harness')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Install' })).toBeInTheDocument();
  });

  it('still hides a not-installed agent Orbital cannot install', async () => {
    mockList([
      makeEntry({ slug: 'claude-code', name: 'Claude Code', installed: true }),
      makeDshEntry({
        slug: 'gemini-cli',
        name: 'Gemini CLI',
        // Bring-your-own everywhere: no platform list to be excluded from.
        install: { supported: false, state: 'not_installed', platforms: [] },
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(screen.queryByText('Gemini CLI')).not.toBeInTheDocument();
  });

  it('explains an unsupported platform instead of offering the button', async () => {
    mockList([
      makeDshEntry({
        install: { supported: false, state: 'not_installed', platforms: ['macos', 'linux'] },
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('DeepSeek Harness')).toBeInTheDocument());
    expect(screen.getByTestId('sub-agent-install-unsupported-dsh')).toHaveTextContent(
      'Not yet supported on this platform.',
    );
    expect(screen.queryByRole('button', { name: 'Install' })).not.toBeInTheDocument();
  });

  it('POSTs the install route and enters the installing state', async () => {
    mockList([makeDshEntry()]);

    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByRole('button', { name: 'Install' }));

    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-install-spinner-dsh')).toBeInTheDocument(),
    );
    expect(api).toHaveBeenCalledWith(`${LIST_PATH}/dsh/install`, { method: 'POST' });
    expect(screen.getByText('Installing...')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Install' })).not.toBeInTheDocument();
  });

  it('shows the latest progress line from the install job', async () => {
    mockList([makeDshEntry()]);

    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByRole('button', { name: 'Install' }));
    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-install-spinner-dsh')).toBeInTheDocument(),
    );

    emit('sub_agent_install_progress', { slug: 'dsh', job_id: 'job-1', line: 'added 12 packages' });
    expect(screen.getByTestId('sub-agent-install-progress-dsh')).toHaveTextContent(
      'added 12 packages',
    );

    emit('sub_agent_install_progress', { slug: 'dsh', job_id: 'job-1', line: 'added 480 packages' });
    expect(screen.getByTestId('sub-agent-install-progress-dsh')).toHaveTextContent(
      'added 480 packages',
    );
  });

  it('ignores install events for a different agent', async () => {
    mockList([makeDshEntry()]);

    render(<SubAgentSettings />);
    await screen.findByRole('button', { name: 'Install' });

    emit('sub_agent_install_progress', { slug: 'other', job_id: 'job-9', line: 'noise' });
    expect(screen.queryByTestId('sub-agent-install-progress-dsh')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Install' })).toBeInTheDocument();
  });

  it('refetches the list when the install job completes', async () => {
    mockList(
      [makeDshEntry()],
      [makeDshEntry({
        installed: true,
        binary_path: '/data/agents/dsh/node_modules/.bin/dsh-acp-demo',
        version: '1.0.0',
        install: { supported: true, state: 'installed', platforms: ['macos', 'linux'] },
      })],
    );

    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByRole('button', { name: 'Install' }));
    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-install-spinner-dsh')).toBeInTheDocument(),
    );

    emit('sub_agent_install_done', { slug: 'dsh', job_id: 'job-1', version: '1.0.0' });

    await waitFor(() => expect(listCallCount()).toBe(2));
    await waitFor(() => expect(screen.getByText('Installed')).toBeInTheDocument());
    expect(screen.queryByTestId('sub-agent-install-dsh')).not.toBeInTheDocument();
  });

  it('surfaces a failed install with a retry', async () => {
    mockList([makeDshEntry()]);

    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByRole('button', { name: 'Install' }));
    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-install-spinner-dsh')).toBeInTheDocument(),
    );

    emit('sub_agent_install_failed', {
      slug: 'dsh', job_id: 'job-1', error: 'node 18.20.0 is older than the required 20.12',
    });

    expect(screen.getByTestId('sub-agent-install-error-dsh')).toHaveTextContent(
      'node 18.20.0 is older than the required 20.12',
    );
    expect(screen.getByRole('button', { name: 'Retry install' })).toBeInTheDocument();
  });

  it('explains a failure that happened before the page loaded', async () => {
    mockList([
      makeDshEntry({
        install: { supported: true, state: 'failed', job_id: 'job-1', platforms: ['macos', 'linux'] },
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByTestId('sub-agent-install-error-dsh')).toHaveTextContent(
      'The last install attempt failed.',
    ));
    expect(screen.getByRole('button', { name: 'Retry install' })).toBeInTheDocument();
  });

  it('resumes the progress display for a job already running at mount', async () => {
    // Page refresh mid-install: the entry carries the state, so the card must
    // not wait for the next WS line to look busy.
    mockList([
      makeDshEntry({
        install: {
          supported: true, state: 'installing', job_id: 'job-1',
          platforms: ['macos', 'linux'],
        },
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-install-spinner-dsh')).toBeInTheDocument(),
    );
    expect(screen.queryByRole('button', { name: 'Install' })).not.toBeInTheDocument();
    expect(api).not.toHaveBeenCalledWith(`${LIST_PATH}/dsh/install`, { method: 'POST' });
  });

  it('surfaces a rejected install request inline', async () => {
    api.mockImplementation(async (path: string) => {
      // The managed-credentials form probes the global LLM settings to decide
      // whether the provider-key checkbox can work at all.
      if (path === '/api/v2/settings') return { llm: { provider: 'deepseek', api_key_set: true } };
      if (path === LIST_PATH) return [makeDshEntry()];
      throw new Error('install already running for dsh');
    });

    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByRole('button', { name: 'Install' }));

    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-install-error-dsh')).toHaveTextContent(
        'install already running for dsh',
      ),
    );
    expect(screen.getByRole('button', { name: 'Retry install' })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Orbital-held API credentials + silent-until-done copy
// ---------------------------------------------------------------------------

describe('SubAgentSettings — managed credentials', () => {
  beforeEach(() => {
    api.mockReset();
    wsHandlers.clear();
  });

  const installedDsh = (overrides: Partial<Entry> = {}) => makeDshEntry({
    installed: true,
    binary_path: '/data/agents/dsh/node_modules/.bin/dsh-acp-demo',
    version: '1.0.0',
    install: { supported: true, state: 'installed', platforms: ['macos', 'linux'] },
    ...overrides,
  });

  it('renders a key field per declared credential for an agent with no login flow', async () => {
    api.mockResolvedValue([installedDsh()]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByTestId('sub-agent-credentials-dsh')).toBeInTheDocument());
    expect(screen.getByTestId('sub-agent-cred-input-DEEPSEEK_API_KEY')).toHaveAttribute(
      'type', 'password',
    );
    expect(screen.getByTestId('sub-agent-cred-input-DEEPSEEK_BASE_URL')).toBeInTheDocument();
    // Required vs optional comes from the manifest, and the region-lock hint
    // rides the base-URL key.
    expect(screen.getByText(/DeepSeek API Base URL/)).toHaveTextContent('(optional)');
    expect(screen.getByText(/region-locked/)).toBeInTheDocument();
  });

  it('stays hidden for an agent with an interactive login flow', async () => {
    api.mockResolvedValue([
      makeEntry({
        slug: 'claude-code',
        name: 'Claude Code',
        supports_login: true,
        credentials: [{
          key: 'claude_auth',
          label: 'Claude subscription',
          type: 'oauth_cli',
          required: true,
          configured: false,
          has_setup_command: true,
        }],
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(screen.queryByTestId('sub-agent-credentials-claude-code')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^login$/i })).toBeInTheDocument();
  });

  it('leaves out a credential the agent ingests through its own CLI', async () => {
    // `has_setup_command` is the manifest's own signal that the key goes in
    // via the agent's login command, not a form Orbital renders.
    api.mockResolvedValue([installedDsh({
      credentials: [
        { key: 'DEEPSEEK_API_KEY', label: 'DeepSeek API Key', type: 'secret', required: true, configured: false },
        { key: 'CLI_OWNED_KEY', label: 'CLI-owned key', type: 'secret', required: false, configured: false, has_setup_command: true },
      ],
    })]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByTestId('sub-agent-credentials-dsh')).toBeInTheDocument());
    expect(screen.getByTestId('sub-agent-cred-input-DEEPSEEK_API_KEY')).toBeInTheDocument();
    expect(screen.queryByTestId('sub-agent-cred-input-CLI_OWNED_KEY')).not.toBeInTheDocument();
  });

  it('falls back to the missing-credential keys when the daemon declares none', async () => {
    api.mockResolvedValue([installedDsh({
      credentials: undefined,
      missing_credentials: ['DEEPSEEK_API_KEY'],
    })]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByTestId('sub-agent-credentials-dsh')).toBeInTheDocument());
    expect(screen.getByTestId('sub-agent-cred-input-DEEPSEEK_API_KEY')).toBeInTheDocument();
    expect(screen.queryByTestId('sub-agent-cred-input-DEEPSEEK_BASE_URL')).not.toBeInTheDocument();
  });

  it('POSTs a typed key as a write-only value', async () => {
    api.mockImplementation(async (path: string) => {
      // The managed-credentials form probes the global LLM settings to decide
      // whether the provider-key checkbox can work at all.
      if (path === '/api/v2/settings') return { llm: { provider: 'deepseek', api_key_set: true } };
      if (path === LIST_PATH) return [installedDsh()];
      return { slug: 'dsh', key: 'DEEPSEEK_API_KEY', set: true, masked: 'sk-1...cdef' };
    });

    render(<SubAgentSettings />);
    const input = await screen.findByTestId('sub-agent-cred-input-DEEPSEEK_API_KEY');
    fireEvent.change(input, { target: { value: 'sk-12345678abcdef' } });
    fireEvent.click(screen.getByTestId('sub-agent-cred-save-DEEPSEEK_API_KEY'));

    await waitFor(() => expect(api).toHaveBeenCalledWith(`${LIST_PATH}/dsh/credential`, {
      method: 'POST',
      body: JSON.stringify({ key: 'DEEPSEEK_API_KEY', value: 'sk-12345678abcdef' }),
    }));
    // The field is cleared: the daemon never gives key text back.
    await waitFor(() => expect(
      screen.getByTestId('sub-agent-cred-input-DEEPSEEK_API_KEY'),
    ).toHaveValue(''));
  });

  it('sends use_llm_provider_key instead of a value when the box is checked', async () => {
    api.mockImplementation(async (path: string) => {
      // The managed-credentials form probes the global LLM settings to decide
      // whether the provider-key checkbox can work at all.
      if (path === '/api/v2/settings') return { llm: { provider: 'deepseek', api_key_set: true } };
      if (path === LIST_PATH) return [installedDsh()];
      return { slug: 'dsh', key: 'DEEPSEEK_API_KEY', set: true, masked: 'sk-1...cdef' };
    });

    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByTestId('sub-agent-cred-provider-dsh'));
    // With the provider key selected there is nothing to type.
    expect(screen.getByTestId('sub-agent-cred-input-DEEPSEEK_API_KEY')).toBeDisabled();
    fireEvent.click(screen.getByTestId('sub-agent-cred-save-DEEPSEEK_API_KEY'));

    await waitFor(() => expect(api).toHaveBeenCalledWith(`${LIST_PATH}/dsh/credential`, {
      method: 'POST',
      body: JSON.stringify({ key: 'DEEPSEEK_API_KEY', use_llm_provider_key: true }),
    }));
  });

  it('surfaces a 409 from the provider-key path inline', async () => {
    api.mockImplementation(async (path: string) => {
      // The managed-credentials form probes the global LLM settings to decide
      // whether the provider-key checkbox can work at all.
      if (path === '/api/v2/settings') return { llm: { provider: 'deepseek', api_key_set: true } };
      if (path === LIST_PATH) return [installedDsh()];
      throw new Error("the global LLM key belongs to provider 'kimi', not deepseek");
    });

    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByTestId('sub-agent-cred-provider-dsh'));
    fireEvent.click(screen.getByTestId('sub-agent-cred-save-DEEPSEEK_API_KEY'));

    await waitFor(() => expect(
      screen.getByTestId('sub-agent-cred-error-DEEPSEEK_API_KEY'),
    ).toHaveTextContent("belongs to provider 'kimi'"));
  });

  it('offers the provider-key box only to agents declaring that credential', async () => {
    api.mockResolvedValue([installedDsh({
      slug: 'other-agent',
      name: 'Other Agent',
      credentials: [
        { key: 'OTHER_API_KEY', label: 'Other API Key', type: 'secret', required: true, configured: false },
      ],
      missing_credentials: ['OTHER_API_KEY'],
    })]);

    render(<SubAgentSettings />);

    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-credentials-other-agent')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('sub-agent-cred-input-OTHER_API_KEY')).toBeInTheDocument();
    expect(screen.queryByTestId('sub-agent-cred-provider-other-agent')).not.toBeInTheDocument();
  });

  it('offers removal of a stored credential and no CLI logout', async () => {
    api.mockImplementation(async (path: string) => {
      // The managed-credentials form probes the global LLM settings to decide
      // whether the provider-key checkbox can work at all.
      if (path === '/api/v2/settings') return { llm: { provider: 'deepseek', api_key_set: true } };
      if (path === LIST_PATH) {
        return [installedDsh({
          credentials_configured: true,
          missing_credentials: [],
          credentials: [
            { key: 'DEEPSEEK_API_KEY', label: 'DeepSeek API Key', type: 'secret', required: true, configured: true },
          ],
        })];
      }
      return { slug: 'dsh', key: 'DEEPSEEK_API_KEY', set: false };
    });

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('Stored')).toBeInTheDocument());
    // Logout drives the agent's own CLI; dsh has none.
    expect(screen.queryByRole('button', { name: /logout/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId('sub-agent-cred-remove-DEEPSEEK_API_KEY'));
    await waitFor(() => expect(api).toHaveBeenCalledWith(
      `${LIST_PATH}/dsh/credential/DEEPSEEK_API_KEY`,
      { method: 'DELETE' },
    ));
  });

  it('says so when the agent reports no tool activity', async () => {
    api.mockResolvedValue([installedDsh()]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByTestId('sub-agent-silent-note-dsh')).toHaveTextContent(
      'Runs silently — activity appears only in the final answer.',
    ));
  });

  it('leaves the silent note off agents that do report tool activity', async () => {
    api.mockResolvedValue([installedDsh({ emits_tool_activity: true })]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('DeepSeek Harness')).toBeInTheDocument());
    expect(screen.queryByTestId('sub-agent-silent-note-dsh')).not.toBeInTheDocument();
  });
});
