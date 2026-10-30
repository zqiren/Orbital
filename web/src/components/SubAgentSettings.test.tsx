// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react';
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
    labels?: Record<string, string> | null;
  }>;
  install?: InstallInfo;
  emits_tool_activity?: boolean;
  credentials?: CredentialSpec[];
  credential_state?: 'configured' | 'missing' | 'unknown' | null;
  update_needed?: boolean;
  min_version?: string | null;
  supports_update?: boolean;
  auth_expired?: boolean;
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

  it('renders a not-installed bring-your-own agent as a Not-detected row (spec 115 D4)', async () => {
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
    // The not-installed agent used to vanish; now it is the locate card:
    // the paste flow, with no login / config controls.
    const row = screen.getByTestId('sub-agent-not-detected-gemini-cli');
    expect(row).toHaveTextContent('Gemini CLI');
    expect(row).toHaveTextContent('Not detected');
    expect(row).toHaveTextContent('Copy a prompt into Gemini CLI and paste its answer below.');
    expect(within(row).queryByRole('combobox')).not.toBeInTheDocument();
    expect(within(row).queryByText('Login')).not.toBeInTheDocument();
    expect(within(row).getByTestId('sub-agent-locate-answer-gemini-cli')).toBeInTheDocument();
  });

  it('keeps the install helper out of the embedded view, and on the standalone page', async () => {
    // Embedded in Global Settings the helper notes ride behind the section's
    // ⓘ (GlobalSettings passes them as the description); only the standalone
    // page, which has no section heading, still prints them.
    api.mockResolvedValueOnce([makeEntry()]);
    const { unmount } = render(<SubAgentSettings />);
    await waitFor(() => {
      expect(screen.getByText('Claude Code')).toBeInTheDocument();
    });
    expect(
      screen.queryByText("Don't see an agent? Install its CLI, then Refresh."),
    ).not.toBeInTheDocument();
    unmount();

    api.mockResolvedValueOnce([makeEntry()]);
    render(<SubAgentSettings standalone />);
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

  // The CLI finishes the OAuth handshake itself (the result travels
  // server-side), so the UI's whole job is: offer the URL if the browser did
  // not open, and be honest about an unconfirmed outcome.
  async function renderClaudeCodeLoginCard() {
    // mockResolvedValue, not ...Once: the login.complete handler fires
    // `void onChanged()`, which re-fetches the list. With a one-shot mock that
    // refetch resolves to undefined, the card unmounts and takes the notice
    // under test with it — a race the assertion happened to win on a warm
    // machine and lost on CI. A persistent mock returns the same entry, so the
    // card survives the refetch whenever it lands.
    api.mockResolvedValue([
      makeEntry({
        slug: 'claude-code',
        name: 'Claude Code',
        installed: true,
        credentials_configured: false,
        supports_login: true,
      }),
    ]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    // The card subscribes to login.* in an effect that runs after the text
    // is on screen. An event emitted before that effect has run is simply
    // dropped, which is how `reports a failed login` went red on CI while
    // passing locally. The list itself holds one login.failed handler (the
    // CLI-missing watcher), so two means the card is listening.
    await waitFor(() => expect(wsHandlers.get('login.failed')?.size ?? 0).toBeGreaterThanOrEqual(2));
  }

  it('renders the streamed sign-in URL as a link', async () => {
    await renderClaudeCodeLoginCard();

    emit('login.progress', {
      slug: 'claude-code',
      job_id: 'job-1',
      line: "If the browser didn't open, visit: https://claude.com/cai/oauth/authorize?code=true",
    });

    expect(screen.getByTestId('sub-agent-login-url-claude-code')).toHaveAttribute(
      'href',
      'https://claude.com/cai/oauth/authorize?code=true',
    );
  });

  it('ignores login events for a different agent', async () => {
    await renderClaudeCodeLoginCard();

    emit('login.progress', { slug: 'other', job_id: 'job-9', line: 'visit: https://evil.test/x' });
    expect(screen.queryByTestId('sub-agent-login-url-claude-code')).not.toBeInTheDocument();
  });

  // Exit code 0 is NOT proof of a login — that is the whole bug this guards.
  // An unconfirmed result must read as "couldn't confirm", never as a failure.
  it('reports an unverified completion distinctly from a failure', async () => {
    await renderClaudeCodeLoginCard();

    emit('login.complete', {
      slug: 'claude-code',
      job_id: 'job-1',
      return_code: 0,
      verified: false,
    });

    const notice = screen.getByTestId('sub-agent-login-notice-claude-code');
    expect(notice).toHaveTextContent(/couldn't confirm/i);
    expect(notice).not.toHaveTextContent(/did not complete/i);
  });

  it('shows no notice when a completion is verified', async () => {
    await renderClaudeCodeLoginCard();

    emit('login.complete', {
      slug: 'claude-code',
      job_id: 'job-1',
      return_code: 0,
      verified: true,
    });

    expect(
      screen.queryByTestId('sub-agent-login-notice-claude-code'),
    ).not.toBeInTheDocument();
  });

  it('reports a failed login', async () => {
    await renderClaudeCodeLoginCard();

    emit('login.failed', { slug: 'claude-code', job_id: 'job-1', return_code: 1 });

    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-login-notice-claude-code')).toHaveTextContent(
        /did not complete/i,
      ),
    );
  });

  it('shows the MiniMax Code sign-in note and internal-model-id hint', async () => {
    // MiniMax Code's CLI has no sign-in probe, so the daemon reports the
    // `unknown` credential state; the card must explain where to sign in
    // and how its model ids look rather than claiming a login.
    api.mockResolvedValueOnce([
      makeEntry({
        slug: 'minimax-code',
        name: 'MiniMax Code',
        credentials_configured: true,
        missing_credentials: [],
        supports_login: false,
        credential_state: 'unknown',
        param_schema: {
          model: { allowed: null, default: null },
          'permission-mode': { allowed: ['auto', 'ask'], default: 'auto' },
        },
        credentials: [
          { key: 'minimax_auth', label: 'MiniMax account', type: 'oauth_cli', required: true, configured: true, has_setup_command: false },
        ],
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('MiniMax Code')).toBeInTheDocument());
    expect(screen.getByText(/Verify sign-in in MiniMax Code/)).toBeInTheDocument();
    expect(screen.getByTestId('sub-agent-host-note-minimax-code')).toHaveTextContent(/mcode login/);
    expect(screen.getByTestId('sub-agent-model-hint-minimax-code')).toHaveTextContent(/m:minimax:MiniMax-M3:v:thinking/);
    expect(screen.queryByRole('button', { name: /^login$/i })).toBeNull();
  });

  it('shows the Kimi Code model-alias hint and keeps the login flow', async () => {
    api.mockResolvedValueOnce([
      makeEntry({
        slug: 'kimi-code',
        name: 'Kimi Code',
        credentials_configured: false,
        missing_credentials: ['kimi_auth'],
        supports_login: true,
        param_schema: {
          model: { allowed: null, default: null },
          'permission-mode': { allowed: ['auto', 'ask'], default: 'auto' },
        },
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('Kimi Code')).toBeInTheDocument());
    expect(screen.getByTestId('sub-agent-model-hint-kimi-code')).toHaveTextContent(/kimi-code\/k3/);
    expect(screen.queryByTestId('sub-agent-host-note-kimi-code')).toBeNull();
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

  it('shows the CLI display names for live Claude Code models', async () => {
    // The daemon forwards claude's own model list with displayName labels:
    // the alias `opus` must read as "Opus 5.5", not the raw value, while an
    // option with no label (a saved legacy pin) falls back to its value.
    api.mockResolvedValueOnce([
      makeEntry({
        slug: 'claude-code',
        name: 'Claude Code',
        config: { model: 'claude-opus-4-8' },
        param_schema: {
          model: {
            allowed: ['opus', 'claude-fable-5-1[1m]', 'claude-opus-4-8'],
            default: null,
            labels: { opus: 'Opus 5.5', 'claude-fable-5-1[1m]': 'Fable 5.1' },
          },
        },
      }),
    ]);

    render(
      <LocaleProvider>
        <SubAgentSettings />
      </LocaleProvider>,
    );

    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    const opus = screen.getByRole('option', { name: 'Opus 5.5' }) as HTMLOptionElement;
    expect(opus.value).toBe('opus');
    expect(screen.getByRole('option', { name: 'Fable 5.1' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'claude-opus-4-8' })).toBeInTheDocument();
    expect((screen.getByRole('combobox') as HTMLSelectElement).value).toBe('claude-opus-4-8');
  });
});

// ---------------------------------------------------------------------------
// Pi — credential state + host-permission disclosure
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Spec 116 — a Claude Code older than the floor needs an UPDATE, not a login
// ---------------------------------------------------------------------------

describe('SubAgentSettings — version floor / Update (spec 116)', () => {
  function oldClaude(overrides: Partial<Entry> = {}): Entry {
    return makeEntry({
      slug: 'claude-code',
      name: 'Claude Code',
      installed: true,
      version: '2.1.100 (Claude Code)',
      ready: false,
      // An old CLI has no `auth status` subcommand, so it reads signed-out too.
      credentials_configured: false,
      missing_credentials: ['claude_auth'],
      supports_login: true,
      update_needed: true,
      min_version: '2.1.114',
      supports_update: true,
      ...overrides,
    });
  }

  async function renderOldClaude(overrides: Partial<Entry> = {}) {
    api.mockResolvedValue([oldClaude(overrides)]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
  }

  it('shows the needs-update notice and an Update button, and no Login button', async () => {
    await renderOldClaude();
    const block = screen.getByTestId('sub-agent-update-claude-code');
    expect(block).toHaveTextContent('Claude Code v2.1.100 is too old; Orbital needs 2.1.114 or newer.');
    expect(block).not.toHaveTextContent('(Claude Code) is too old');
    expect(screen.getByRole('button', { name: /^update$/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^login$/i })).not.toBeInTheDocument();
  });

  it('clicking Update POSTs the update route and shows the updating state', async () => {
    await renderOldClaude();
    api.mockResolvedValueOnce({ job_id: 'upd1', slug: 'claude-code', status: 'running' });

    fireEvent.click(screen.getByRole('button', { name: /^update$/i }));

    await waitFor(() => {
      expect(api).toHaveBeenCalledWith(
        '/api/v2/settings/sub-agents/claude-code/update',
        expect.objectContaining({ method: 'POST' }),
      );
    });
    await waitFor(() => {
      expect(screen.getByTestId('sub-agent-update-spinner-claude-code')).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: /updating/i })).toBeDisabled();
  });

  it('an update that leaves the CLI below the floor shows the manual npm command', async () => {
    await renderOldClaude();

    emit('login.complete', {
      slug: 'claude-code', job_id: 'upd1', kind: 'update', return_code: 0, verified: false,
    });

    // findBy: the emitted event's re-render can land after this line on a
    // loaded CI runner (batch 1026 rc: red once, green 8/8 locally).
    const notice = await screen.findByTestId('sub-agent-update-notice-claude-code');
    expect(notice).toHaveTextContent(/did not complete/i);
    expect(notice).toHaveTextContent('npm install -g @anthropic-ai/claude-code@latest');
    // The login notice is a different surface and stays untouched.
    expect(screen.queryByTestId('sub-agent-login-notice-claude-code')).not.toBeInTheDocument();
  });

  it('a verified update clears the notice and refetches the entry', async () => {
    await renderOldClaude();
    api.mockClear();

    emit('login.complete', {
      slug: 'claude-code', job_id: 'upd1', kind: 'update', return_code: 0, verified: true,
    });

    expect(screen.queryByTestId('sub-agent-update-notice-claude-code')).not.toBeInTheDocument();
    await waitFor(() => expect(api).toHaveBeenCalledWith('/api/v2/settings/sub-agents'));
  });

  it('an updater that failed outright shows the failed notice', async () => {
    await renderOldClaude();
    emit('login.failed', { slug: 'claude-code', job_id: 'upd1', kind: 'update', return_code: 1 });
    expect(await screen.findByTestId('sub-agent-update-notice-claude-code')).toHaveTextContent(/did not complete/i);
  });

  it('a 409 (worker running) shows the busy hint instead of an error', async () => {
    await renderOldClaude();
    api.mockRejectedValueOnce(Object.assign(new Error('a claude-code task is running'), { status: 409 }));

    fireEvent.click(screen.getByRole('button', { name: /^update$/i }));

    await waitFor(() => {
      expect(screen.getByTestId('sub-agent-update-notice-claude-code')).toHaveTextContent(
        'A Claude Code task is running; try again when it finishes.',
      );
    });
    expect(screen.getByRole('button', { name: /^update$/i })).toBeEnabled();
  });

  it('login events of kind update never touch the login notice or URL', async () => {
    await renderOldClaude();
    emit('login.progress', {
      slug: 'claude-code', job_id: 'upd1', kind: 'update',
      line: 'see https://example.test/release-notes',
    });
    expect(screen.queryByTestId('sub-agent-login-url-claude-code')).not.toBeInTheDocument();
  });

  it('renders the notice in Simplified Chinese', async () => {
    localStorage.setItem('orbital.locale', 'zh');
    api.mockResolvedValue([oldClaude({ version: '2.1.100' })]);
    render(
      <LocaleProvider>
        <SubAgentSettings />
      </LocaleProvider>,
    );
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(screen.getByTestId('sub-agent-update-claude-code')).toHaveTextContent(
      'Claude Code 版本过旧（v2.1.100），需要 2.1.114 或更新。',
    );
    expect(screen.getByRole('button', { name: '更新' })).toBeInTheDocument();
  });

  it('a CLI at the floor keeps the Login button and no update block', async () => {
    api.mockResolvedValue([oldClaude({ version: '2.1.302', update_needed: false })]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(screen.queryByTestId('sub-agent-update-claude-code')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^login$/i })).toBeInTheDocument();
  });
});

describe('SubAgentSettings — update removed the CLI (1018 bench item 5)', () => {
  const goneClaude = (): Entry => makeEntry({
    installed: false, binary_path: null, version: null, ready: false,
    update_needed: false, supports_update: true,
  });

  async function renderAndFailUpdate() {
    api.mockResolvedValue([makeEntry({
      version: '2.1.100 (Claude Code)', ready: false, update_needed: true,
      min_version: '2.1.114', supports_update: true,
    })]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    // The daemon dropped its cached detection; the refetch reads "not installed".
    api.mockClear();
    api.mockResolvedValue([goneClaude()]);
    emit('login.failed', {
      slug: 'claude-code', job_id: 'upd1', kind: 'update', return_code: 1,
      reason: 'cli_missing_after_update', verified: false,
      manual_command: 'npm install -g @anthropic-ai/claude-code@latest',
    });
  }

  it('says the CLI is gone, names the npm command, and refetches at once', async () => {
    await renderAndFailUpdate();
    await waitFor(() => expect(api).toHaveBeenCalledWith('/api/v2/settings/sub-agents'));
    // The refetched entry is not installed, so the update block is gone, but
    // the notice must stay: it is the only thing telling the user why.
    await waitFor(() => expect(screen.getByTestId('sub-agent-not-detected-claude-code')).toBeInTheDocument());
    const notice = screen.getByTestId('sub-agent-update-missing-claude-code');
    expect(notice).toHaveTextContent(/no longer installed/i);
    expect(notice).toHaveTextContent('npm install -g @anthropic-ai/claude-code@latest');
    expect(screen.queryByTestId('sub-agent-update-claude-code')).not.toBeInTheDocument();
  });

  it('the notice stays through a still-missing refetch and clears once the CLI is back', async () => {
    await renderAndFailUpdate();
    await screen.findByTestId('sub-agent-not-detected-claude-code');
    expect(screen.getByTestId('sub-agent-update-missing-claude-code')).toBeInTheDocument();
    // The user reinstalls by hand and clicks Refresh.
    api.mockResolvedValue([makeEntry({ version: '2.1.302', update_needed: false })]);
    fireEvent.click(screen.getByRole('button', { name: /^refresh$/i }));
    await waitFor(() => {
      expect(screen.queryByTestId('sub-agent-update-missing-claude-code')).not.toBeInTheDocument();
    });
  });

  it('renders the missing-CLI notice in Simplified Chinese', async () => {
    localStorage.setItem('orbital.locale', 'zh');
    api.mockResolvedValue([makeEntry({
      version: '2.1.100', ready: false, update_needed: true,
      min_version: '2.1.114', supports_update: true,
    })]);
    render(
      <LocaleProvider>
        <SubAgentSettings />
      </LocaleProvider>,
    );
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    api.mockResolvedValue([goneClaude()]);
    emit('login.failed', {
      slug: 'claude-code', job_id: 'upd1', kind: 'update', return_code: 1,
      reason: 'cli_missing_after_update', verified: false,
    });
    const notice = await screen.findByTestId('sub-agent-update-missing-claude-code');
    expect(notice).toHaveTextContent('已不在本机');
    expect(notice).toHaveTextContent('npm install -g @anthropic-ai/claude-code@latest');
  });

  it('an ordinary update failure still shows the did-not-complete notice', async () => {
    api.mockResolvedValue([makeEntry({
      version: '2.1.100', ready: false, update_needed: true,
      min_version: '2.1.114', supports_update: true,
    })]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    emit('login.failed', {
      slug: 'claude-code', job_id: 'upd1', kind: 'update', return_code: 1, reason: 'nonzero_exit',
    });
    expect(screen.getByTestId('sub-agent-update-notice-claude-code')).toHaveTextContent(/did not complete/i);
    expect(screen.queryByTestId('sub-agent-update-missing-claude-code')).not.toBeInTheDocument();
  });
});

describe('SubAgentSettings — expired sign-in (1018 bench item 9)', () => {
  const expired = (overrides: Partial<Entry> = {}): Entry => makeEntry({
    version: '2.1.302', credentials_configured: true, missing_credentials: [],
    auth_expired: true, ...overrides,
  });

  it('shows "Sign-in expired", a note and the Login button instead of Logged in', async () => {
    api.mockResolvedValue([expired()]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(screen.getByText('Sign-in expired')).toBeInTheDocument();
    expect(screen.queryByText('Logged in')).not.toBeInTheDocument();
    expect(screen.getByTestId('sub-agent-auth-expired-claude-code')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^login$/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^logout$/i })).not.toBeInTheDocument();
  });

  it('renders in Simplified Chinese', async () => {
    localStorage.setItem('orbital.locale', 'zh');
    api.mockResolvedValue([expired()]);
    render(
      <LocaleProvider>
        <SubAgentSettings />
      </LocaleProvider>,
    );
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(screen.getByText('登录已过期')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '登录' })).toBeInTheDocument();
  });

  it('a signed-in agent without the record keeps Logged in and Logout', async () => {
    api.mockResolvedValue([expired({ auth_expired: false })]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(screen.getByText('Logged in')).toBeInTheDocument();
    expect(screen.queryByTestId('sub-agent-auth-expired-claude-code')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^logout$/i })).toBeInTheDocument();
  });

  it('a logout the CLI refused says so (GitHub #50)', async () => {
    api.mockImplementation(async (path: string, opts?: { method?: string }) => (
      opts?.method === 'POST' && path.endsWith('/logout')
        ? { slug: 'claude-code', return_code: 1, stdout: '', stderr: 'not logged in here' }
        : [expired({ auth_expired: false })]
    ));
    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByRole('button', { name: /^logout$/i }));
    expect(await screen.findByText('not logged in here')).toBeInTheDocument();
  });
});

describe('SubAgentSettings — Pi', () => {
  function makePiEntry(overrides: Partial<Entry> = {}): Entry {
    return makeEntry({
      slug: 'pi',
      name: 'Pi',
      binary_path: '/opt/homebrew/bin/pi',
      version: '0.85.1',
      credentials_configured: true,
      missing_credentials: [],
      supports_login: false,
      config: {},
      param_schema: {
        model: { allowed: null, default: null },
        effort: { allowed: ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'], default: null },
      },
      credentials: [
        { key: 'PI_MODEL_PROVIDER_KEY', label: "Key for the configured model's provider", type: 'model_provider', required: true, configured: true, has_setup_command: false },
      ],
      credential_state: 'unknown',
      ...overrides,
    });
  }

  it('discloses host permissions and asks to verify sign-in in Pi', async () => {
    api.mockResolvedValueOnce([makePiEntry()]);
    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('Pi')).toBeInTheDocument());
    expect(screen.getByTestId('sub-agent-host-note-pi')).toHaveTextContent(/user account's permissions/);
    expect(screen.getByText('Verify sign-in in Pi')).toBeInTheDocument();
    expect(screen.getByText(/provider\/model/)).toBeInTheDocument();
    // Orbital owns neither a Pi login nor a Pi key field.
    expect(screen.queryByRole('button', { name: 'Login' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Logout' })).toBeNull();
    expect(screen.queryByTestId('sub-agent-credentials-pi')).toBeNull();
  });

  it.each([
    ['configured', 'Credentials ready'],
    ['missing', 'No credentials for this model'],
  ] as const)('labels the %s credential state', async (state, label) => {
    api.mockResolvedValueOnce([
      makePiEntry({ credential_state: state, credentials_configured: state === 'configured' }),
    ]);
    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText(label)).toBeInTheDocument());
    expect(screen.queryByRole('button', { name: 'Login' })).toBeNull();
  });

  it('renders the Pi notes in Simplified Chinese', async () => {
    localStorage.setItem('orbital.locale', 'zh');
    api.mockResolvedValueOnce([makePiEntry()]);
    render(
      <LocaleProvider>
        <SubAgentSettings />
      </LocaleProvider>,
    );

    await waitFor(() => expect(screen.getByText('Pi')).toBeInTheDocument());
    expect(screen.getByTestId('sub-agent-host-note-pi')).toHaveTextContent(/用户账户权限/);
    expect(screen.getByText('请在 Pi 中确认登录状态')).toBeInTheDocument();
    expect(screen.getByText('思考强度')).toBeInTheDocument();
  });

  it('leaves agents without a credential state on their login pill', async () => {
    api.mockResolvedValueOnce([makeEntry({ credentials_configured: true, missing_credentials: [] })]);
    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(screen.getByText('Logged in')).toBeInTheDocument();
    expect(screen.queryByTestId('sub-agent-host-note-claude-code')).toBeNull();
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

  it('shows a not-installed agent Orbital cannot install as Not detected, without an Install button', async () => {
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
    // Spec 115 D4: the compact row, not the full card — so no Install.
    expect(screen.getByTestId('sub-agent-not-detected-gemini-cli')).toHaveTextContent('Not detected');
    expect(screen.queryByRole('button', { name: 'Install' })).not.toBeInTheDocument();
    expect(screen.queryByTestId('sub-agent-install-unsupported-gemini-cli')).not.toBeInTheDocument();
  });

  it('explains an unsupported platform instead of offering the button', async () => {
    mockList([
      makeDshEntry({
        install: { supported: false, state: 'not_installed', platforms: ['macos', 'linux'] },
      }),
    ]);

    render(<SubAgentSettings />);

    // Spec 127 D3 hangs the locate card under this card, so the name is on
    // the page twice; wait on the card's own line.
    await waitFor(() => expect(screen.getByTestId('sub-agent-install-unsupported-dsh')).toHaveTextContent(
      'Not yet supported on this platform.',
    ));
    expect(screen.queryByRole('button', { name: 'Install' })).not.toBeInTheDocument();
  });

  // Spec 127 D3: an Orbital-installed agent that is not installed and cannot
  // be (this host, or a failed run) keeps its card AND gets the locate card
  // beneath it, so a hand-installed launcher can still be wired up.
  it('adds the locate card under an unsupported-platform card (spec 127 D3)', async () => {
    mockList([
      makeDshEntry({
        install: { supported: false, state: 'not_installed', platforms: ['macos', 'linux'] },
      }),
    ]);

    render(<SubAgentSettings />);

    // The card stays: its explanation line is still there.
    await waitFor(() => expect(screen.getByTestId('sub-agent-install-unsupported-dsh')).toBeInTheDocument());
    const row = screen.getByTestId('sub-agent-not-detected-dsh');
    expect(row).toHaveTextContent(
      'Or point Orbital at a DeepSeek Harness you installed yourself: paste the launcher path (or the YAML an agent returns for it) below.',
    );
    expect(row).not.toHaveTextContent('Copy a prompt into DeepSeek Harness');
    expect(within(row).getByTestId('sub-agent-locate-answer-dsh')).toBeInTheDocument();
  });

  it('adds the locate card under a card whose Orbital install failed (spec 127 D3)', async () => {
    mockList([
      makeDshEntry({
        install: { supported: true, state: 'failed', job_id: 'job-1', platforms: ['macos', 'linux'] },
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByTestId('sub-agent-install-error-dsh')).toBeInTheDocument());
    // Retry stays on the card; the row is the fallback, not a replacement.
    expect(screen.getByRole('button', { name: 'Retry install' })).toBeInTheDocument();
    expect(screen.getByTestId('sub-agent-not-detected-dsh')).toBeInTheDocument();
  });

  it('shows no locate card while an Orbital install is still on offer (spec 127 D3)', async () => {
    mockList([makeDshEntry()]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByRole('button', { name: 'Install' })).toBeInTheDocument());
    expect(screen.queryByTestId('sub-agent-not-detected-dsh')).not.toBeInTheDocument();
  });

  it('shows no locate card for an installed Orbital-managed agent (spec 127 D3)', async () => {
    mockList([
      makeDshEntry({
        installed: true,
        binary_path: '/data/agents/dsh/node_modules/.bin/dsh-acp-demo',
        install: { supported: true, state: 'installed', platforms: ['macos', 'linux'] },
      }),
    ]);

    render(<SubAgentSettings />);

    await waitFor(() => expect(screen.getByText('DeepSeek Harness')).toBeInTheDocument());
    expect(screen.queryByTestId('sub-agent-not-detected-dsh')).not.toBeInTheDocument();
  });

  it('refetches after a failed install so the locate card appears (spec 127 D3)', async () => {
    mockList(
      [makeDshEntry()],
      [makeDshEntry({
        install: { supported: true, state: 'failed', job_id: 'job-1', platforms: ['macos', 'linux'] },
      })],
    );

    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByRole('button', { name: 'Install' }));
    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-install-spinner-dsh')).toBeInTheDocument(),
    );
    expect(screen.queryByTestId('sub-agent-not-detected-dsh')).not.toBeInTheDocument();

    emit('sub_agent_install_failed', { slug: 'dsh', job_id: 'job-1', error: 'npm ci failed with exit code 1' });

    await waitFor(() => expect(screen.getByTestId('sub-agent-not-detected-dsh')).toBeInTheDocument());
    // The live error text is kept over the generic "failed earlier" line.
    expect(screen.getByTestId('sub-agent-install-error-dsh')).toHaveTextContent(
      'npm ci failed with exit code 1',
    );
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

  it('does not re-enter the installing state when the job fails before the POST resolves', async () => {
    // Spec 127 V4 on Windows (2026-10-22): the preflight ("no node on PATH")
    // fails within milliseconds, so `sub_agent_install_failed` reaches the
    // card over the WS before the 202 does. The spinner must not come back
    // beside the error.
    let resolvePost: (v: unknown) => void = () => {};
    api.mockImplementation(async (path: string) => {
      if (path === '/api/v2/settings') return { llm: { provider: 'deepseek', api_key_set: true } };
      if (path === LIST_PATH) return [makeDshEntry()];
      if (path.endsWith('/install')) return new Promise(r => { resolvePost = r; });
      throw new Error(`unexpected api call: ${path}`);
    });

    render(<SubAgentSettings />);
    fireEvent.click(await screen.findByRole('button', { name: 'Install' }));

    emit('sub_agent_install_failed', { slug: 'dsh', job_id: 'job-1', error: 'needs Node.js 20.12+' });
    await act(async () => {
      resolvePost({ job_id: 'job-1', slug: 'dsh', state: 'installing' });
    });

    expect(screen.getByTestId('sub-agent-install-error-dsh')).toHaveTextContent('needs Node.js 20.12+');
    expect(screen.queryByTestId('sub-agent-install-spinner-dsh')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry install' })).toBeInTheDocument();
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

// Codex used to be API-key-only in the UI: `codex login` (ChatGPT sign-in, the
// way most people use Codex) was advertised by the daemon but no button ever
// called it. Found on Windows v0.16.0: a signed-out codex could only be fixed
// from a terminal.
describe('SubAgentSettings — Codex sign-in', () => {
  async function renderSignedOutCodex() {
    api.mockResolvedValue([
      makeEntry({
        slug: 'codex',
        name: 'Codex',
        installed: true,
        credentials_configured: false,
        supports_login: true,
      }),
    ]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Codex')).toBeInTheDocument());
  }

  it('offers ChatGPT login alongside the API key', async () => {
    await renderSignedOutCodex();
    expect(screen.getByRole('button', { name: 'Login' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Set API Key' })).toBeInTheDocument();
  });

  it('links the sign-in page, not the local callback server', async () => {
    await renderSignedOutCodex();
    // The card subscribes to login.progress in an effect; on a slow runner the
    // first paint can land before that effect, and an event emitted then is
    // simply dropped (CI run 38146434946). Wait for the subscription, and read
    // the link back through waitFor like the other WS-driven assertions.
    await waitFor(() => expect(wsHandlers.get('login.progress')?.size ?? 0).toBeGreaterThan(0));
    emit('login.progress', {
      slug: 'codex', job_id: 'job-c',
      line: 'Starting local login server on http://localhost:1455.',
    });
    expect(screen.queryByTestId('sub-agent-login-url-codex')).not.toBeInTheDocument();
    emit('login.progress', {
      slug: 'codex', job_id: 'job-c',
      line: 'https://auth.openai.com/oauth/authorize?response_type=code&client_id=app_X.',
    });
    await waitFor(() =>
      expect(screen.getByTestId('sub-agent-login-url-codex')).toHaveAttribute(
        'href',
        'https://auth.openai.com/oauth/authorize?response_type=code&client_id=app_X',
      ),
    );
  });

  it('does not show Login once codex is signed in', async () => {
    api.mockResolvedValue([
      makeEntry({ slug: 'codex', name: 'Codex', installed: true,
                  credentials_configured: true, supports_login: true }),
    ]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Codex')).toBeInTheDocument());
    expect(screen.queryByRole('button', { name: 'Login' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Set API Key' })).not.toBeInTheDocument();
  });
});

describe('SubAgentSettings — desktop app sources', () => {
  beforeEach(() => {
    api.mockReset();
    wsHandlers.clear();
  });

  const DSH_URL = 'https://download.deepseek.com/desktop/dsh-latest-macos-arm64.dmg';

  function mockEntries(entries: Entry[]) {
    api.mockImplementation(async (path: string) => {
      if (path === '/api/v2/settings') return { llm: { provider: 'opencode', api_key_set: true } };
      if (path === LIST_PATH) return entries;
      throw new Error(`unexpected api call: ${path}`);
    });
  }

  /** dsh as the daemon reports it now: found inside DSH Desktop, ACP. */
  function desktopDsh(overrides: Record<string, unknown> = {}): Entry {
    return makeDshEntry({
      installed: true,
      binary_path: '/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh',
      version: '0.2.1-rc.3',
      ready: true,
      credentials_configured: true,
      missing_credentials: [],
      install: { supported: false, state: 'installed', platforms: [] },
      emits_tool_activity: true,
      ...overrides,
      // Fields the shared Entry type predates.
      ...({
        source: { kind: 'desktop_app', label: 'DeepSeek Harness',
                  app_path: '/Applications/DeepSeek Harness.app', app_version: '0.2.1-rc.3' },
        handshake: { state: 'ok', detail: null },
        desktop_app: { label: 'DeepSeek Harness', download_url: DSH_URL },
        ...overrides,
      } as Partial<Entry>),
    });
  }

  it('a not-detected desktop-app agent leads with the vendor download link', async () => {
    mockEntries([makeDshEntry({
      install: { supported: false, state: 'not_installed', platforms: [] },
      ...({ desktop_app: { label: 'DeepSeek Harness', download_url: DSH_URL } } as Partial<Entry>),
    })]);
    render(<SubAgentSettings />);
    const row = await screen.findByTestId('sub-agent-not-detected-dsh');
    const link = within(row).getByRole('link', { name: 'Install DeepSeek Harness' });
    expect(link).toHaveAttribute('href', DSH_URL);
    expect(link).toHaveAttribute('target', '_blank');
    expect(row).toHaveTextContent('Orbital finds it on its own once it is installed. Then click Refresh.');
    // The path fallback is still there, worded for an app.
    expect(row).toHaveTextContent('Installed somewhere else? Paste the path to DeepSeek Harness');
    expect(within(row).getByTestId('sub-agent-locate-answer-dsh')).toBeInTheDocument();
  });

  it('names the desktop app a detected agent comes from', async () => {
    mockEntries([desktopDsh()]);
    render(<SubAgentSettings />);
    const chip = await screen.findByTestId('sub-agent-source-dsh');
    expect(chip).toHaveTextContent('From the DeepSeek Harness app');
    expect(chip).toHaveAttribute('title', '/Applications/DeepSeek Harness.app · 0.2.1-rc.3');
    expect(screen.queryByTestId('sub-agent-handshake-failed-dsh')).not.toBeInTheDocument();
    expect(screen.queryByTestId('sub-agent-silent-note-dsh')).not.toBeInTheDocument();
  });

  it('a key the desktop app holds is named, not offered as a field', async () => {
    mockEntries([desktopDsh({
      credentials: [
        { key: 'DEEPSEEK_API_KEY', label: 'DeepSeek API Key', type: 'secret', required: true,
          configured: true, has_setup_command: false, stored_by: 'DeepSeek Harness' },
        { key: 'DEEPSEEK_BASE_URL', label: 'DeepSeek API Base URL', type: 'secret', required: false,
          configured: false, has_setup_command: false, stored_by: null },
      ],
    })]);
    render(<SubAgentSettings />);
    const owned = await screen.findByTestId('sub-agent-cred-stored-by-DEEPSEEK_API_KEY');
    expect(owned).toHaveTextContent('Saved in DeepSeek Harness');
    expect(owned).toHaveTextContent('DeepSeek Harness uses its own copy of this key, so change it there.');
    expect(screen.queryByTestId('sub-agent-cred-input-DEEPSEEK_API_KEY')).not.toBeInTheDocument();
    // A key nobody else holds keeps Orbital's field.
    expect(screen.getByTestId('sub-agent-cred-input-DEEPSEEK_BASE_URL')).toBeInTheDocument();
  });

  it('a failed handshake says so in the agent\'s words and opens the locate row', async () => {
    mockEntries([desktopDsh({
      ready: false,
      handshake: { state: 'failed', detail: 'error: --profile <name> is required' },
    })]);
    render(<SubAgentSettings />);
    const note = await screen.findByTestId('sub-agent-handshake-failed-dsh');
    expect(note).toHaveTextContent('DeepSeek Harness was found, but it did not answer Orbital, so it cannot be used yet.');
    expect(note).toHaveTextContent('error: --profile <name> is required');
    const row = screen.getByTestId('sub-agent-not-detected-dsh');
    expect(row).toHaveTextContent('Point Orbital at a different copy of DeepSeek Harness');
  });

  it('Change location opens the locate row under a working card and closes it again', async () => {
    mockEntries([desktopDsh()]);
    render(<SubAgentSettings />);
    const change = await screen.findByTestId('sub-agent-locate-change-dsh');
    expect(screen.queryByTestId('sub-agent-not-detected-dsh')).not.toBeInTheDocument();
    fireEvent.click(change);
    expect(screen.getByTestId('sub-agent-not-detected-dsh')).toHaveTextContent(
      'Point Orbital at a different copy of DeepSeek Harness');
    fireEvent.click(change);
    expect(screen.queryByTestId('sub-agent-not-detected-dsh')).not.toBeInTheDocument();
  });

  it('dsh on its own providers: no key pill, a providers note, dsh-labelled models', async () => {
    mockEntries([desktopDsh({
      credentials: [
        { key: 'DEEPSEEK_API_KEY', label: 'DeepSeek API Key', type: 'secret', required: false,
          configured: false, has_setup_command: false, stored_by: null },
        { key: 'DEEPSEEK_BASE_URL', label: 'DeepSeek API Base URL', type: 'secret', required: false,
          configured: false, has_setup_command: false, stored_by: null },
      ],
      config: { model: 'opencode-go/deepseek-v4-pro' },
      param_schema: {
        model: {
          allowed: ['opencode-go/deepseek-v4-flash', 'opencode-go/deepseek-v4-pro'],
          default: null,
          labels: {
            'opencode-go/deepseek-v4-flash': 'OpenCode Go · DeepSeek V4 Flash',
            'opencode-go/deepseek-v4-pro': 'OpenCode Go · DeepSeek V4 Pro',
          },
        },
        effort: { allowed: ['off', 'high', 'max'], default: null,
                  labels: { off: 'Off', high: 'High', max: 'Max' } },
      },
    })]);
    render(<SubAgentSettings />);
    expect(await screen.findByTestId('sub-agent-host-note-dsh')).toHaveTextContent(
      'Providers, keys and sign-ins are set up in DeepSeek Harness');
    // Whether dsh can serve a task is dsh's call: no "API key needed" pill.
    expect(screen.queryByText('API key needed')).not.toBeInTheDocument();
    expect(screen.queryByText('API key saved')).not.toBeInTheDocument();
    const [model] = screen.getAllByRole('combobox') as HTMLSelectElement[];
    expect(model.value).toBe('opencode-go/deepseek-v4-pro');
    expect(within(model).getByRole('option', { name: 'OpenCode Go · DeepSeek V4 Flash' })).toBeInTheDocument();
    // dsh offers no "low" for this route, so neither does the card.
    expect(screen.queryByRole('option', { name: 'low' })).not.toBeInTheDocument();
  });

  it('renders the desktop-app strings in Simplified Chinese', async () => {
    localStorage.setItem('orbital.locale', 'zh');
    mockEntries([desktopDsh({
      credentials: [
        { key: 'DEEPSEEK_API_KEY', label: 'DeepSeek API Key', type: 'secret', required: true,
          configured: true, has_setup_command: false, stored_by: 'DeepSeek Harness' },
      ],
    })]);
    render(<LocaleProvider><SubAgentSettings /></LocaleProvider>);
    expect(await screen.findByTestId('sub-agent-source-dsh')).toHaveTextContent('来自 DeepSeek Harness 应用');
    expect(screen.getByTestId('sub-agent-cred-stored-by-DEEPSEEK_API_KEY')).toHaveTextContent(
      '已保存在 DeepSeek Harness 中');
    expect(screen.getByTestId('sub-agent-locate-change-dsh')).toHaveTextContent('更改位置');
  });
});

// Spec 136 A.2: Settings shows the last sweep at once; when the daemon's
// background sweep (or a late model list) changes it, `sub_agents.updated`
// arrives and the list refetches without dropping back to the spinner.
describe('SubAgentSettings — background status updates (spec 136)', () => {
  it('refetches on sub_agents.updated and keeps the cards up meanwhile', async () => {
    let settle: (value: unknown) => void = () => {};
    api.mockResolvedValueOnce([makeEntry({ credentials_configured: false })]);
    render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(screen.getByText('Not logged in')).toBeInTheDocument();

    api.mockImplementationOnce(() => new Promise((resolve) => { settle = resolve; }));
    emit('sub_agents.updated', { reason: 'status' });
    expect(api).toHaveBeenLastCalledWith('/api/v2/settings/sub-agents');
    expect(screen.getByText('Claude Code')).toBeInTheDocument();
    expect(screen.queryByText('Loading sub-agents...')).toBeNull();

    await act(async () => {
      settle([makeEntry({ credentials_configured: true, missing_credentials: [] })]);
    });
    await waitFor(() => expect(screen.getByText('Logged in')).toBeInTheDocument());
  });

  it('stops listening when unmounted', async () => {
    api.mockResolvedValueOnce([makeEntry()]);
    const { unmount } = render(<SubAgentSettings />);
    await waitFor(() => expect(screen.getByText('Claude Code')).toBeInTheDocument());
    expect(wsHandlers.get('sub_agents.updated')?.size ?? 0).toBe(1);
    unmount();
    expect(wsHandlers.get('sub_agents.updated')?.size ?? 0).toBe(0);
  });
});
