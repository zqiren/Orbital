// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * SetupWizard unit tests — Task I1 (Spec 011 §0.7), spec 138 step flow.
 *
 * Covers:
 *  - spec 138 flow: agents → projects → api_key → accounts; projects skipped
 *    when the scan finds nothing; the key step skipped when a key exists;
 *    zero agents → the key step is compulsory (no Skip, amber note, way back);
 *  - the Agents step: statuses with history counts, Sign in through the
 *    sub-agent login job, unchecked agents → PUT sub_agents_hidden, the
 *    not-found chips opening the locate card;
 *  - Finish: telemetry, the tour marked pending on a fresh install, and the
 *    landing project handed to App;
 *  - the sandbox step is gone; featured connectors; connect flow; language
 *    widget; agent browser sign-in; skipping the key (keyless onboarding).
 *
 * The api client (web/src/config.ts `api`) is mocked — no network. The
 * LLMProviderSettings form is stubbed out (it has its own coverage).
 */

import { render, screen, waitFor, cleanup, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Connector } from '../types';
import { LocaleProvider } from '../i18n/LocaleContext';
import { LOCALE_STORAGE_KEY } from '../i18n/locales';
import { TOUR_STORAGE_KEY } from '../tour/tourLogic';

const apiMock = vi.hoisted(() => vi.fn());
const MockApiError = vi.hoisted(() => {
  return class MockApiError extends Error {
    constructor(
      public status: number,
      public detail: string,
    ) {
      super(detail);
      this.name = 'ApiError';
    }
  };
});
vi.mock('../config', () => ({ api: apiMock, ApiError: MockApiError }));
const stubUnscanned = vi.hoisted(() => ({ n: 0, agent: null as string | null, silent: false }));
const stubSave = vi.hoisted(() => ({ fn: null as (() => Promise<boolean>) | null }));
vi.mock('./LLMProviderSettings', async () => {
  const { useEffect } = await import('react');
  return {
    // Spec 160: the real form reports the unscanned-imports count it read.
    default: ({ onUnscannedImports, saveRef }: {
      onUnscannedImports?: (n: number, agent: string | null) => void;
      saveRef?: { current: (() => Promise<boolean>) | null };
    }) => {
      useEffect(() => {
        if (stubUnscanned.silent) return;
        onUnscannedImports?.(stubUnscanned.n, stubUnscanned.agent);
      }, [onUnscannedImports]);
      // The real form's save: true only when its key passed and was stored.
      useEffect(() => {
        if (saveRef) saveRef.current = stubSave.fn;
      });
      return <div data-testid="llm-provider-settings-stub" />;
    },
  };
});

type WsHandler = (event: unknown) => void;
const wsHandlers = vi.hoisted(() => new Map<string, Set<(event: unknown) => void>>());
vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    on: (type: string, fn: WsHandler) => {
      if (!wsHandlers.has(type)) wsHandlers.set(type, new Set());
      wsHandlers.get(type)!.add(fn);
    },
    off: (type: string, fn: WsHandler) => { wsHandlers.get(type)?.delete(fn); },
    connectionState: 'connected',
    subscribe: vi.fn(),
  }),
}));
function emitWs(type: string, payload: Record<string, unknown>) {
  act(() => { for (const fn of wsHandlers.get(type) ?? []) fn({ type, ...payload }); });
}

import SetupWizard from './SetupWizard';

afterEach(() => cleanup());

function makeConnector(overrides: Partial<Connector> = {}): Connector {
  return {
    id: 'google-calendar',
    name: 'Google Calendar',
    icon: '📅',
    auth_provider: 'google',
    auth_type: 'oauth2',
    server_url: 'https://example.com/mcp',
    oauth_scopes: [],
    tool_overrides: {},
    featured: true,
    status: 'available',
    connected: false,
    account: null,
    enabled_error: null,
    ...overrides,
  };
}

interface AgentFixture {
  slug: string;
  name: string;
  installed: boolean;
  credentials_configured: boolean;
  supports_login?: boolean;
}

const AGENTS: AgentFixture[] = [
  { slug: 'claude-code', name: 'Claude Code', installed: true, credentials_configured: true },
  { slug: 'codex', name: 'Codex', installed: true, credentials_configured: false, supports_login: true },
  { slug: 'gemini-cli', name: 'Gemini CLI', installed: true, credentials_configured: true },
  { slug: 'kimi-code', name: 'Kimi Code', installed: false, credentials_configured: false },
  { slug: 'pi', name: 'Pi', installed: false, credentials_configured: false },
];

const NO_PROJECTS = { status: 'ready', candidates: [], groups: [] };
const SOME_PROJECTS = {
  status: 'ready',
  candidates: [{
    source: 'claude-code', name: 'orbital-test', path: '/Users/x/orbital-test', session_count: 40,
    last_activity: '2026-10-26T08:00:00+00:00', already_imported: false, kind: 'folder', minor: false,
    is_git: true, by_agent: { 'claude-code': 40 },
  }],
  groups: [],
};

interface MockRoutes {
  settingsExtra?: Record<string, unknown>;
  apiKeySet?: boolean;
  hidden?: string[];
  /** PUT /api/v2/settings. Default resolves. */
  settingsPut?: (body: Record<string, unknown>) => Promise<unknown>;
  agents?: AgentFixture[];
  importable?: unknown;
  projects?: unknown[];
  connectors?: Connector[] | Error;
  connect?: () => Promise<unknown>;
  importResult?: unknown;
}

const telemetry: Array<{ event: string; fields: Record<string, unknown> }> = [];

/** Route the api mock by path; unmatched paths resolve to {}. */
function mockApi(routes: MockRoutes) {
  apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
    if (path === '/api/v2/telemetry/event') {
      telemetry.push(JSON.parse(String(options?.body ?? '{}')));
      return undefined;
    }
    if (path === '/api/v2/settings' && options?.method === 'PUT') {
      const body = JSON.parse(String(options.body ?? '{}'));
      return routes.settingsPut ? routes.settingsPut(body) : {};
    }
    if (path === '/api/v2/settings') {
      return {
        llm: { api_key_set: routes.apiKeySet ?? true },
        sub_agents_hidden: routes.hidden ?? [],
        ...(routes.settingsExtra ?? {}),
      };
    }
    if (path === '/api/v2/settings/sub-agents' || path === '/api/v2/settings/sub-agents/refresh') {
      return routes.agents ?? AGENTS;
    }
    if (path === '/api/v2/onboarding/history') {
      return { status: 'ready', sessions_total: 296, by_agent: { 'claude-code': 214, codex: 71, pi: 0, 'kimi-code': 3, 'minimax-code': 8 } };
    }
    if (path === '/api/v2/onboarding/importable-projects') return routes.importable ?? NO_PROJECTS;
    if (path === '/api/v2/onboarding/import') return routes.importResult ?? { created: [], skipped: [] };
    if (path === '/api/v2/projects') return routes.projects ?? [{ project_id: 'scratch', is_scratch: true }];
    if (path === '/api/v2/connectors') {
      if (routes.connectors instanceof Error) throw routes.connectors;
      return { connectors: routes.connectors ?? [] };
    }
    if (/^\/api\/v2\/connectors\/[^/]+\/connect$/.test(path) && options?.method === 'POST') {
      return routes.connect ? routes.connect() : {};
    }
    if (path === '/api/v2/platform/browser/warmup/status') {
      return { active: true };
    }
    return {};
  });
}

function calledPaths(): string[] {
  return apiMock.mock.calls.map(([path]) => path as string);
}

function settingsPuts(): Record<string, unknown>[] {
  return apiMock.mock.calls
    .filter(([path, opts]) => path === '/api/v2/settings' && opts?.method === 'PUT')
    .map(([, opts]) => JSON.parse(opts.body));
}

/** Click Continue on the Agents step once the agent list is in. */
async function continueFromAgents() {
  const btn = await waitFor(() => {
    const b = screen.getByTestId('wizard-agents-continue') as HTMLButtonElement;
    expect(b.disabled).toBe(false);
    return b;
  });
  // The importable list must be in before Continue decides to skip Projects.
  await waitFor(() => expect(calledPaths()).toContain('/api/v2/onboarding/importable-projects'));
  await act(async () => { fireEvent.click(btn); });
}

/** Render and walk to the accounts step (key set, nothing to import). */
async function renderAtAccounts(ui = <SetupWizard onComplete={vi.fn()} />) {
  render(ui);
  await continueFromAgents();
  await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
}

/** Render and walk to the key step (no key, nothing to import). */
async function renderAtKey(ui = <SetupWizard onComplete={vi.fn()} />) {
  render(ui);
  await continueFromAgents();
  await waitFor(() => expect(screen.getByText('The Orbital agent')).toBeTruthy());
}

beforeEach(() => {
  apiMock.mockReset();
  wsHandlers.clear();
  telemetry.length = 0;
  localStorage.removeItem(TOUR_STORAGE_KEY);
});

describe('SetupWizard — spec 138 step flow', () => {
  it('opens on the Agents step with the step indicator', async () => {
    mockApi({ apiKeySet: false });
    render(<SetupWizard onComplete={vi.fn()} />);
    expect(screen.getByText('Welcome to Orbital')).toBeTruthy();
    const steps = screen.getByTestId('wizard-steps');
    expect(steps.textContent).toBe('1 Agents·2 Projects·3 Orbital agent·4 Accounts');
    expect(steps.querySelector('[aria-current="step"]')?.textContent).toBe('1 Agents');
    // The history scan starts with the wizard.
    await waitFor(() => expect(calledPaths()).toContain('/api/v2/onboarding/history'));
  });

  it('skips Projects when the scan found nothing, then goes to the key step', async () => {
    mockApi({ apiKeySet: false, importable: NO_PROJECTS });
    await renderAtKey();
    expect(screen.queryByText('Your projects')).toBeNull();
    expect(screen.getByTestId('wizard-steps').querySelector('[aria-current="step"]')?.textContent).toBe('3 Orbital agent');
  });

  it('shows Projects when there is something to import; Back returns, Skip moves on', async () => {
    mockApi({ apiKeySet: false, importable: SOME_PROJECTS });
    render(<SetupWizard onComplete={vi.fn()} />);
    await continueFromAgents();
    await waitFor(() => expect(screen.getByText('Your projects')).toBeTruthy());
    expect(screen.getByText(/^Found in 296 conversations with your agents\./)).toBeTruthy();

    fireEvent.click(screen.getByTestId('import-back'));
    expect(screen.getByText('Welcome to Orbital')).toBeTruthy();
    await continueFromAgents();
    await waitFor(() => expect(screen.getByText('Your projects')).toBeTruthy());
    fireEvent.click(screen.getByTestId('import-skip'));
    expect(screen.getByText('The Orbital agent')).toBeTruthy();
  });

  it('Back to Agents and forward again keeps what the user unchecked on Projects (F2)', async () => {
    mockApi({ apiKeySet: false, importable: SOME_PROJECTS });
    render(<SetupWizard onComplete={vi.fn()} />);
    await continueFromAgents();
    const box = () => screen.getByTestId('import-check-/Users/x/orbital-test') as HTMLInputElement;
    await waitFor(() => expect(box().checked).toBe(true));
    fireEvent.click(box());
    expect(box().checked).toBe(false);

    fireEvent.click(screen.getByTestId('import-back'));
    await continueFromAgents();
    await waitFor(() => expect(screen.getByText('Your projects')).toBeTruthy());
    expect(box().checked).toBe(false);
  });

  it('the key step has Back: to Projects when it had rows, keeping the unchecks (I2)', async () => {
    mockApi({ apiKeySet: false, importable: SOME_PROJECTS });
    render(<SetupWizard onComplete={vi.fn()} />);
    await continueFromAgents();
    const box = () => screen.getByTestId('import-check-/Users/x/orbital-test') as HTMLInputElement;
    await waitFor(() => expect(box().checked).toBe(true));
    fireEvent.click(box());
    fireEvent.click(screen.getByTestId('import-skip'));
    await waitFor(() => expect(screen.getByText('The Orbital agent')).toBeTruthy());

    // Same wording as the Projects step's Back.
    expect(screen.getByTestId('wizard-key-back').textContent).toBe('Back');
    fireEvent.click(screen.getByTestId('wizard-key-back'));
    await waitFor(() => expect(screen.getByText('Your projects')).toBeTruthy());
    expect(box().checked).toBe(false);
  });

  it('the key step\'s Back goes to Agents when there was nothing to import', async () => {
    mockApi({ apiKeySet: false });
    await renderAtKey();
    fireEvent.click(screen.getByTestId('wizard-key-back'));
    expect(screen.getByText('Welcome to Orbital')).toBeTruthy();
  });

  it('after an import, the list is re-read so Back shows the imported folder as imported', async () => {
    mockApi({ apiKeySet: false, importable: SOME_PROJECTS });
    render(<SetupWizard onComplete={vi.fn()} />);
    await continueFromAgents();
    await waitFor(() => expect(screen.getByTestId('import-submit')).toBeTruthy());
    const before = calledPaths().filter((p) => p === '/api/v2/onboarding/importable-projects').length;
    await act(async () => { fireEvent.click(screen.getByTestId('import-submit')); });
    await waitFor(() => expect(screen.getByText('The Orbital agent')).toBeTruthy());
    await waitFor(() =>
      expect(calledPaths().filter((p) => p === '/api/v2/onboarding/importable-projects').length).toBeGreaterThan(before));
  });

  it('skips the key step when a key already exists', async () => {
    mockApi({ apiKeySet: true });
    await renderAtAccounts();
    expect(screen.queryByText('The Orbital agent')).toBeNull();
  });

  it('Next on the key step advances to accounts once the key is saved', async () => {
    mockApi({ apiKeySet: false });
    await renderAtKey();
    mockApi({ apiKeySet: true });
    fireEvent.click(screen.getByText('Next →'));
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
  });

  it('Next stays on the key step when the form refuses the key (its test failed), with no second line', async () => {
    mockApi({ apiKeySet: false });
    const save = vi.fn(async () => false);
    stubSave.fn = save;
    await renderAtKey();
    try {
      fireEvent.click(screen.getByText('Next →'));
      await waitFor(() => expect(save).toHaveBeenCalled());
      await new Promise((r) => setTimeout(r, 20));
      // The form draws the test's verdict; the wizard adds nothing over it.
      expect(screen.getByTestId('llm-provider-settings-stub')).toBeTruthy();
      expect(screen.queryByText('Connect Your Accounts')).toBeNull();
      expect(screen.queryByTestId('wizard-next-error')).toBeNull();
      expect(screen.getByText('Next →')).toBeTruthy();
    } finally {
      stubSave.fn = null;
    }
  });

  it('one Next during a running test waits for the form, then advances on a pass', async () => {
    mockApi({ apiKeySet: false });
    let pass: (ok: boolean) => void = () => {};
    stubSave.fn = () => new Promise<boolean>((r) => { pass = r; });
    await renderAtKey();
    try {
      fireEvent.click(screen.getByText('Next →'));
      await screen.findByText('Checking...');
      mockApi({ apiKeySet: true });
      pass(true);
      await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
    } finally {
      stubSave.fn = null;
    }
  });

  it('reports each step reached once, and done at Finish', async () => {
    mockApi({ apiKeySet: false, importable: SOME_PROJECTS });
    render(<SetupWizard onComplete={vi.fn()} />);
    await continueFromAgents();
    await waitFor(() => expect(screen.getByText('Your projects')).toBeTruthy());
    fireEvent.click(screen.getByTestId('import-back'));
    await continueFromAgents();
    await waitFor(() => expect(screen.getByText('Your projects')).toBeTruthy());
    fireEvent.click(screen.getByTestId('import-skip'));
    mockApi({ apiKeySet: false });
    await act(async () => { fireEvent.click(screen.getByTestId('wizard-skip-key')); });
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
    fireEvent.click(screen.getByTestId('wizard-get-started'));
    await waitFor(() => expect(telemetry.filter((e) => e.event === 'wizard_step_reached').map((e) => e.fields.step))
      .toEqual(['agents', 'projects', 'key', 'accounts', 'done']));
  });
});

describe('SetupWizard — Agents step', () => {
  it('lists installed agents checked, with sign-in status and history counts', async () => {
    mockApi({ apiKeySet: false });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('wizard-agent-claude-code')).toBeTruthy());
    expect(screen.getByText(/Orbital found these coding agents on this computer\./)).toBeTruthy();
    await waitFor(() =>
      expect(screen.getByTestId('wizard-agent-status-claude-code').textContent).toBe('Signed in · 214 conversations'),
    );
    expect(screen.getByTestId('wizard-agent-status-codex').textContent).toBe('Installed · not signed in');
    expect(screen.getByTestId('wizard-agent-login-codex').textContent).toBe('Sign in');
    // No history reader for gemini-cli → no count.
    expect(screen.getByTestId('wizard-agent-status-gemini-cli').textContent).toBe('Signed in');
    for (const slug of ['claude-code', 'codex', 'gemini-cli']) {
      expect((screen.getByTestId(`wizard-agent-check-${slug}`) as HTMLInputElement).checked).toBe(true);
    }
    expect(screen.getByTestId('wizard-agents-continue').textContent).toBe('Continue with 3 agents');
  });

  it('unchecked agents go into sub_agents_hidden on Continue (keeping hidden ones not on the list)', async () => {
    mockApi({ apiKeySet: false, hidden: ['aider'] });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('wizard-agent-check-codex')).toBeTruthy());
    fireEvent.click(screen.getByTestId('wizard-agent-check-codex'));
    expect(screen.getByTestId('wizard-agents-continue').textContent).toBe('Continue with 2 agents');
    await continueFromAgents();
    expect(settingsPuts()).toEqual([{ sub_agents_hidden: ['aider', 'codex'] }]);
  });

  it('ticks are locked while Continue saves, so an uncheck cannot be dropped (pass 2b)', async () => {
    mockApi({ apiKeySet: false });
    const base = apiMock.getMockImplementation()!;
    let release: (v: unknown) => void = () => {};
    apiMock.mockImplementation((path: string, opts?: { method?: string }) =>
      path === '/api/v2/settings' && opts?.method === 'PUT'
        ? new Promise((r) => { release = r; })
        : base(path, opts));
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('wizard-agent-check-codex')).toBeTruthy());
    await continueFromAgents();
    const codex = screen.getByTestId('wizard-agent-check-codex') as HTMLInputElement;
    expect(codex.disabled).toBe(true);
    await act(async () => { release({}); });
  });

  it('an agent with no login flow says where its key goes instead of offering Sign in', async () => {
    mockApi({
      apiKeySet: false,
      agents: [{ slug: 'gemini-cli', name: 'Gemini CLI', installed: true, credentials_configured: false, supports_login: false }],
    });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() =>
      expect(screen.getByTestId('wizard-agent-status-gemini-cli').textContent).toBe('Installed · add its key later in Settings'),
    );
    expect(screen.queryByTestId('wizard-agent-login-gemini-cli')).toBeNull();
  });

  it('an agent already hidden starts unchecked', async () => {
    mockApi({ apiKeySet: false, hidden: ['gemini-cli'] });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() =>
      expect((screen.getByTestId('wizard-agent-check-gemini-cli') as HTMLInputElement).checked).toBe(false),
    );
  });

  it('Sign in starts the agent\'s login job and refetches when it completes', async () => {
    mockApi({ apiKeySet: false });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('wizard-agent-login-codex')).toBeTruthy());
    await act(async () => { fireEvent.click(screen.getByTestId('wizard-agent-login-codex')); });
    expect(apiMock).toHaveBeenCalledWith('/api/v2/settings/sub-agents/codex/login', { method: 'POST' });

    const before = calledPaths().filter((p) => p === '/api/v2/settings/sub-agents').length;
    mockApi({ apiKeySet: false, agents: AGENTS.map((a) => (a.slug === 'codex' ? { ...a, credentials_configured: true } : a)) });
    emitWs('login.complete', { slug: 'codex' });
    await waitFor(() => expect(screen.getByTestId('wizard-agent-status-codex').textContent).toBe('Signed in · 71 conversations'));
    expect(calledPaths().filter((p) => p === '/api/v2/settings/sub-agents').length).toBe(before + 1);
  });

  it('agents not found are chips that open the existing locate card', async () => {
    mockApi({ apiKeySet: false });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('wizard-not-found')).toBeTruthy());
    expect(screen.getByText('Local Agents Not Found?')).toBeTruthy();
    expect(screen.queryByTestId('sub-agent-not-detected-kimi-code')).toBeNull();
    fireEvent.click(screen.getByTestId('wizard-locate-chip-kimi-code'));
    expect(screen.getByTestId('sub-agent-not-detected-kimi-code')).toBeTruthy();
    expect(screen.getByText('Copy a prompt into Kimi Code and paste its answer below.')).toBeTruthy();
  });
});

describe('SetupWizard — no coding agent found', () => {
  const NONE = AGENTS.map((a) => ({ ...a, installed: false, credentials_configured: false }));

  it('leads with the locate card and install names; Continue makes the key compulsory', async () => {
    mockApi({ apiKeySet: false, agents: NONE });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByText('Installed but not found?')).toBeTruthy());
    expect(screen.getByText(/We didn't find a coding agent on this computer\./)).toBeTruthy();
    // The first chip's card is open by default.
    expect(screen.getByTestId('sub-agent-not-detected-claude-code')).toBeTruthy();
    expect(screen.getByTestId('wizard-install-line').textContent).toContain('Claude Code · Codex · Gemini CLI · Kimi Code · Pi');
    expect(screen.getByTestId('wizard-agents-continue').textContent).toBe('Continue');

    await continueFromAgents();
    await waitFor(() => expect(screen.getByText('The Orbital agent')).toBeTruthy());
    expect(screen.queryByTestId('wizard-skip-key')).toBeNull();
    expect(screen.getByTestId('wizard-key-compulsory').textContent).toContain(
      'Without a coding agent, Orbital needs this key to do anything.',
    );
    fireEvent.click(screen.getByTestId('wizard-key-back-to-agents'));
    expect(screen.getByText('Installed but not found?')).toBeTruthy();
  });

  it('unchecking every found agent also makes the key compulsory', async () => {
    mockApi({ apiKeySet: false, agents: [AGENTS[0]] });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('wizard-agent-check-claude-code')).toBeTruthy());
    fireEvent.click(screen.getByTestId('wizard-agent-check-claude-code'));
    await continueFromAgents();
    await waitFor(() => expect(screen.getByTestId('wizard-key-compulsory')).toBeTruthy());
  });
});

describe('SetupWizard — Finish', () => {
  it('lands in the most recently active imported project and marks the tour pending on a fresh install', async () => {
    mockApi({
      apiKeySet: true,
      importable: SOME_PROJECTS,
      importResult: { created: [{ project_id: 'p_orb', name: 'orbital-test', workspace: '/Users/x/orbital-test', from: 'folder' }], skipped: [] },
    });
    const onComplete = vi.fn();
    render(<SetupWizard onComplete={onComplete} />);
    await continueFromAgents();
    await waitFor(() => expect(screen.getByText('Your projects')).toBeTruthy());
    await act(async () => { fireEvent.click(screen.getByTestId('import-submit')); });
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());

    fireEvent.click(screen.getByTestId('wizard-get-started'));
    expect(onComplete).toHaveBeenCalledWith({ landingProjectId: 'p_orb' });
    expect(localStorage.getItem(TOUR_STORAGE_KEY)).toBe('pending');
  });

  it('spec 160 round 3: with no key the import defers the read to the key step; with a key it does not', async () => {
    for (const apiKeySet of [false, true]) {
      apiMock.mockClear();
      mockApi({ apiKeySet, importable: SOME_PROJECTS });
      const { unmount } = render(<SetupWizard onComplete={vi.fn()} />);
      await continueFromAgents();
      await waitFor(() => expect(screen.getByText('Your projects')).toBeTruthy());
      await act(async () => { fireEvent.click(screen.getByTestId('import-submit')); });
      const post = apiMock.mock.calls.find(([p]) => p === '/api/v2/onboarding/import');
      expect(JSON.parse(String((post?.[1] as RequestInit).body)).defer_scan).toBe(apiKeySet ? undefined : true);
      unmount();
    }
  });

  it('without an import, hands App no landing project (Create Project follows)', async () => {
    mockApi({ apiKeySet: true });
    const onComplete = vi.fn();
    await renderAtAccounts(<SetupWizard onComplete={onComplete} />);
    fireEvent.click(screen.getByTestId('wizard-skip'));
    expect(onComplete).toHaveBeenCalledWith({ landingProjectId: null });
  });

  it('does not reopen a tour this device already finished', async () => {
    localStorage.setItem(TOUR_STORAGE_KEY, 'done');
    mockApi({ apiKeySet: true });
    await renderAtAccounts();
    fireEvent.click(screen.getByTestId('wizard-get-started'));
    expect(localStorage.getItem(TOUR_STORAGE_KEY)).toBe('done');
  });

  it('an install that already has its own projects is not marked for the tour', async () => {
    mockApi({ apiKeySet: true, projects: [{ project_id: 'scratch', is_scratch: true }, { project_id: 'mine', is_scratch: false }] });
    await renderAtAccounts();
    await waitFor(() => expect(calledPaths()).toContain('/api/v2/projects'));
    fireEvent.click(screen.getByTestId('wizard-get-started'));
    expect(localStorage.getItem(TOUR_STORAGE_KEY)).toBeNull();
  });
});

describe('SetupWizard — sandbox step is gone', () => {
  it('never touches the platform sandbox endpoints while advancing', async () => {
    mockApi({ apiKeySet: true, connectors: [makeConnector()] });
    await renderAtAccounts();
    await waitFor(() => expect(calledPaths()).toContain('/api/v2/connectors'));

    expect(calledPaths()).not.toContain('/api/v2/platform/status');
    expect(calledPaths()).not.toContain('/api/v2/platform/skip');
    expect(calledPaths()).not.toContain('/api/v2/platform/setup');
    // No sandbox copy anywhere on the merged screen.
    expect(screen.queryByText('Your Agents Are Sandboxed')).toBeNull();
    expect(screen.queryByText('Sandbox Not Configured')).toBeNull();
  });
});

describe('SetupWizard — featured connector cards', () => {
  it('renders only featured && available connectors', async () => {
    mockApi({
      apiKeySet: true,
      connectors: [
        makeConnector({ id: 'google-calendar', name: 'Google Calendar' }),
        makeConnector({ id: 'gmail', name: 'Gmail', status: 'pending_verification' }),
        makeConnector({ id: 'custom-x', name: 'Custom X', featured: false }),
      ],
    });
    await renderAtAccounts();

    await waitFor(() =>
      expect(screen.getByTestId('wizard-connector-google-calendar')).toBeTruthy(),
    );
    expect(screen.getByTestId('wizard-connectors-group')).toBeTruthy();
    expect(screen.queryByTestId('wizard-connector-gmail')).toBeNull();
    expect(screen.queryByTestId('wizard-connector-custom-x')).toBeNull();
  });

  it('shows the browser group alone when zero connectors qualify', async () => {
    mockApi({
      apiKeySet: true,
      connectors: [makeConnector({ id: 'gmail', status: 'pending_verification' })],
    });
    await renderAtAccounts();
    await waitFor(() => expect(calledPaths()).toContain('/api/v2/connectors'));

    expect(screen.queryByTestId('wizard-connectors-group')).toBeNull();
    expect(screen.getByTestId('wizard-browser-group')).toBeTruthy();
    expect(screen.getByText('Agent browser sign-in')).toBeTruthy();
  });

  it('shows the browser group alone when the connectors fetch fails', async () => {
    mockApi({ apiKeySet: true, connectors: new Error('boom') });
    await renderAtAccounts();
    await waitFor(() => expect(calledPaths()).toContain('/api/v2/connectors'));

    expect(screen.queryByTestId('wizard-connectors-group')).toBeNull();
    expect(screen.getByTestId('wizard-browser-group')).toBeTruthy();
    // Skip / Get Started still work — never a broken screen.
    expect(screen.getByTestId('wizard-skip')).toBeTruthy();
    expect(screen.getByTestId('wizard-get-started')).toBeTruthy();
  });
});

describe('SetupWizard — connect flow', () => {
  it('shows the waiting-for-consent busy state while the connect call blocks', async () => {
    let resolveConnect!: (v: unknown) => void;
    mockApi({
      apiKeySet: true,
      connectors: [makeConnector()],
      connect: () => new Promise((resolve) => { resolveConnect = resolve; }),
    });
    await renderAtAccounts();
    await waitFor(() =>
      expect(screen.getByTestId('wizard-connector-connect-google-calendar')).toBeTruthy(),
    );

    fireEvent.click(screen.getByTestId('wizard-connector-connect-google-calendar'));
    await waitFor(() =>
      expect(screen.getByText('Waiting for browser consent…')).toBeTruthy(),
    );
    expect(
      (screen.getByTestId('wizard-connector-connect-google-calendar') as HTMLButtonElement)
        .disabled,
    ).toBe(true);

    await act(async () => { resolveConnect({}); });
  });

  it('shows the OAuth-client hint on a 400 connect failure', async () => {
    mockApi({
      apiKeySet: true,
      connectors: [makeConnector()],
      connect: () => Promise.reject(new MockApiError(400, 'Google OAuth client not configured')),
    });
    await renderAtAccounts();
    await waitFor(() =>
      expect(screen.getByTestId('wizard-connector-connect-google-calendar')).toBeTruthy(),
    );

    await act(async () => {
      fireEvent.click(screen.getByTestId('wizard-connector-connect-google-calendar'));
    });

    const error = screen.getByTestId('wizard-connector-error-google-calendar');
    expect(error.getAttribute('role')).toBe('alert');
    expect(error.textContent).toBe(
      "A Google OAuth client isn't configured yet — you can set one up later in Settings → Connectors.",
    );
    // Button is usable again for a retry.
    expect(
      (screen.getByTestId('wizard-connector-connect-google-calendar') as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });

  it('refetches the list on success and shows Connected as {account}', async () => {
    const disconnected = makeConnector();
    const connected = makeConnector({ connected: true, account: 'alice@example.com' });
    mockApi({ apiKeySet: true, connectors: [disconnected] });
    await renderAtAccounts();
    await waitFor(() =>
      expect(screen.getByTestId('wizard-connector-connect-google-calendar')).toBeTruthy(),
    );

    mockApi({ apiKeySet: true, connectors: [connected] });
    const before = calledPaths().filter((p) => p === '/api/v2/connectors').length;
    await act(async () => {
      fireEvent.click(screen.getByTestId('wizard-connector-connect-google-calendar'));
    });

    expect(calledPaths().filter((p) => p === '/api/v2/connectors').length).toBe(before + 1);
    await waitFor(() =>
      expect(
        screen.getByTestId('wizard-connector-status-google-calendar').textContent,
      ).toBe('Connected as alice@example.com'),
    );
    expect(screen.queryByTestId('wizard-connector-connect-google-calendar')).toBeNull();
  });
});

describe('SetupWizard — language widget (Spec 008 A2)', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('renders on the first step, switches locale on change, and persists to localStorage', async () => {
    mockApi({ apiKeySet: false });
    render(
      <LocaleProvider>
        <SetupWizard onComplete={vi.fn()} />
      </LocaleProvider>,
    );

    await waitFor(() => expect(screen.getByText('Welcome to Orbital')).toBeTruthy());
    const select = screen.getByTestId('wizard-locale-select') as HTMLSelectElement;
    expect(select.value).toBe('en');

    fireEvent.change(select, { target: { value: 'zh' } });

    await waitFor(() => expect(screen.getByText('欢迎使用 Orbital')).toBeTruthy());
    expect(localStorage.getItem(LOCALE_STORAGE_KEY)).toBe('zh');
  });

  it('renders on the accounts step too — ambient on every step', async () => {
    mockApi({ apiKeySet: true });
    await renderAtAccounts(
      <LocaleProvider>
        <SetupWizard onComplete={vi.fn()} />
      </LocaleProvider>,
    );
    expect(screen.getByTestId('wizard-locale-select')).toBeTruthy();
  });
});

describe('SetupWizard — agent browser sign-in group', () => {
  it('POSTs the warmup endpoint and shows the opening state', async () => {
    mockApi({ apiKeySet: true });
    await renderAtAccounts();
    await waitFor(() => expect(screen.getByTestId('wizard-open-browser')).toBeTruthy());

    await act(async () => {
      fireEvent.click(screen.getByTestId('wizard-open-browser'));
    });

    expect(apiMock).toHaveBeenCalledWith('/api/v2/platform/browser/warmup', {
      method: 'POST',
      body: JSON.stringify({ url: 'https://accounts.google.com' }),
    });
    expect(screen.getByText('Browser open — sign in and close it...')).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// Keyless onboarding: the key step is skippable (contract §1, §8)
// ---------------------------------------------------------------------------

describe('SetupWizard — skip the key step', () => {
  it('says what the key is for, and offers Skip with its hint under Next', async () => {
    mockApi({ apiKeySet: false });
    await renderAtKey();
    expect(
      screen.getByText(/Add an API key for the Orbital agent/),
    ).toBeTruthy();
    const skip = screen.getByTestId('wizard-skip-key');
    expect(skip.textContent).toBe('Skip for now — use my existing agents');
    expect(screen.getByTestId('wizard-skip-key-hint').textContent).toMatch(
      /Add a key later in Settings\./,
    );
    // Next is still there and still the primary action.
    expect(screen.getByText('Next →')).toBeTruthy();
  });

  it('shows the token sign beside Next only when imported projects wait (spec 160)', async () => {
    mockApi({ apiKeySet: false });
    stubUnscanned.n = 0;
    await renderAtKey();
    expect(screen.queryByTestId('token-cost-sign')).toBeNull();
    cleanup();
    stubUnscanned.n = 3;
    mockApi({ apiKeySet: false });
    await renderAtKey();
    const sign = await screen.findByTestId('token-cost-sign');
    expect(sign.textContent).toBe('3');
    expect(sign.nextElementSibling?.textContent).toBe('Next →');
    expect(screen.getAllByTestId('token-cost-sign')).toHaveLength(1);
    // No worker ready: nothing beside Skip (it would read nothing).
    expect(screen.getByTestId('wizard-skip-key').previousElementSibling).toBeNull();
    cleanup();
    // Spec 167: with a worker ready, the worker sign sits beside Skip too.
    stubUnscanned.agent = 'Claude Code';
    mockApi({ apiKeySet: false });
    await renderAtKey();
    await screen.findAllByTestId('token-cost-sign');
    const signs = screen.getAllByTestId('token-cost-sign');
    expect(signs).toHaveLength(2);
    const beside = screen.getByTestId('wizard-skip-key').previousElementSibling as HTMLElement;
    expect(beside?.getAttribute('data-testid')).toBe('token-cost-sign');
    expect(beside.getAttribute('data-kind')).toBe('importAgent');
    expect(beside.textContent).toBe('3');
    stubUnscanned.n = 0;
    stubUnscanned.agent = null;
  });

  it('Skip persists setup_api_key_skipped and advances to the accounts step', async () => {
    mockApi({ apiKeySet: false });
    await renderAtKey();

    fireEvent.click(screen.getByTestId('wizard-skip-key'));
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());

    // [0] is the Agents step's hidden list. Nothing to read: no read.
    expect(settingsPuts().slice(1)).toEqual([{ setup_api_key_skipped: true, import_scan: false }]);
  });

  it('Skip with imports waiting and a worker ready reads them with the switch on (spec 167)', async () => {
    stubUnscanned.n = 2;
    stubUnscanned.agent = 'Claude Code';
    mockApi({ apiKeySet: false });
    await renderAtKey();
    fireEvent.click(screen.getByTestId('wizard-skip-key'));
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
    expect(settingsPuts().slice(1)).toEqual([{ setup_api_key_skipped: true, import_scan: true }]);
    stubUnscanned.n = 0;
    stubUnscanned.agent = null;
  });

  it('the switch in the sign beside Skip turns the read off', async () => {
    stubUnscanned.n = 2;
    stubUnscanned.agent = 'Claude Code';
    mockApi({ apiKeySet: false });
    await renderAtKey();
    const beside = screen.getByTestId('wizard-skip-key').previousElementSibling as HTMLElement;
    fireEvent.click(beside);
    fireEvent.click(await screen.findByTestId('token-cost-switch'));
    fireEvent.click(screen.getByTestId('wizard-skip-key'));
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
    expect(settingsPuts().slice(1)).toEqual([{ setup_api_key_skipped: true, import_scan: false }]);
    stubUnscanned.n = 0;
    stubUnscanned.agent = null;
  });

  it('Skip before the key form has reported still reads with the switch on (the sign inputs come from the settings read)', async () => {
    // The race: the form's own GET /settings has not landed when Skip is
    // clicked, so the wizard knows no count and no worker yet. It must not
    // send import_scan:false (which turns every waiting import off for good).
    stubUnscanned.silent = true;
    mockApi({
      apiKeySet: false,
      settingsExtra: { unscanned_imports: 2, import_scan_agent: { slug: 'claude-code', name: 'Claude Code' } },
    });
    await renderAtKey();
    fireEvent.click(screen.getByTestId('wizard-skip-key'));
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
    expect(settingsPuts().slice(1)).toEqual([{ setup_api_key_skipped: true, import_scan: true }]);
    stubUnscanned.silent = false;
  });

  it('Skip before the key form has reported, with nothing to read, sends import_scan:false', async () => {
    stubUnscanned.silent = true;
    mockApi({ apiKeySet: false, settingsExtra: { unscanned_imports: 0, import_scan_agent: null } });
    await renderAtKey();
    fireEvent.click(screen.getByTestId('wizard-skip-key'));
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
    expect(settingsPuts().slice(1)).toEqual([{ setup_api_key_skipped: true, import_scan: false }]);
    stubUnscanned.silent = false;
  });

  it('a failed Skip shows an error line and stays on the key step', async () => {
    mockApi({
      apiKeySet: false,
      settingsPut: (body) => ('setup_api_key_skipped' in body ? Promise.reject(new MockApiError(500, 'boom')) : Promise.resolve({})),
    });
    await renderAtKey();

    fireEvent.click(screen.getByTestId('wizard-skip-key'));
    await waitFor(() => expect(screen.getByTestId('wizard-skip-error')).toBeTruthy());
    expect(screen.getByTestId('wizard-skip-error').textContent).toMatch(/Couldn’t save\. Try again\./);
    expect(screen.queryByText('Connect Your Accounts')).toBeNull();
    expect(screen.getByText('The Orbital agent')).toBeTruthy();
  });

  it('zh: the Skip button and hint follow the contract copy', async () => {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'zh');
    try {
      mockApi({ apiKeySet: false });
      render(
        <LocaleProvider>
          <SetupWizard onComplete={vi.fn()} />
        </LocaleProvider>,
      );
      await continueFromAgents();
      await waitFor(() => expect(screen.getByTestId('wizard-skip-key')).toBeTruthy());
      expect(screen.getByTestId('wizard-skip-key').textContent).toBe('暂时跳过，使用已有 Agent');
      expect(screen.getByTestId('wizard-skip-key-hint').textContent).toBe(
        '之后可在设置中添加。',
      );
    } finally {
      window.localStorage.removeItem(LOCALE_STORAGE_KEY);
    }
  });
});
