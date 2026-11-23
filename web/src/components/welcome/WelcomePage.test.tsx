// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * WelcomePage — the one-page first run (batch 1123).
 *
 * Covers the Start gate (a signed-out agent does not count), Start's writes
 * (the key-skipped flag without a key, the passing key saved and made the
 * default with one, the import of what is ticked, never `sub_agents_hidden`),
 * the import line's checkbox and label, the review list and Esc, and the two
 * error states that must not read as "nothing found" (the agent list fetch,
 * the history scan).
 *
 * The api client is mocked; the connect panel is a stub whose footer context
 * the test controls (ConnectPanel.test.tsx covers the panel itself).
 */

import { render, screen, waitFor, cleanup, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../../config', () => ({ api: apiMock, ApiError: class extends Error {} }));

const panel = vi.hoisted(() => ({
  connected: false,
  canSubmit: false,
  busy: false,
  error: null as string | null,
  submit: null as null | ((done: (card: Record<string, unknown>) => void, setBusy: (b: boolean) => void) => void),
}));
const STORED_CARD = { id: 'card_new', provider: 'deepseek', model: 'deepseek-v4-pro', is_default: false };
vi.mock('../connect/ConnectPanel', async () => {
  const React = await import('react');
  return {
    default: function Stub({ footer, onDone }: {
      footer: (ctx: Record<string, unknown>) => React.ReactNode;
      onDone: (card: Record<string, unknown>) => void;
    }) {
      const [busy, setBusy] = React.useState(false);
      return (
        <div data-testid="connect-panel-stub">
          {footer({
            connected: panel.connected,
            canSubmit: panel.canSubmit,
            busy: panel.busy || busy,
            error: panel.error,
            provider: 'deepseek',
            model: 'deepseek-v4-pro',
            submit: () => (panel.submit ? panel.submit(onDone, setBusy) : onDone(STORED_CARD)),
          })}
        </div>
      );
    },
  };
});
vi.mock('../../hooks/useWebSocket', () => ({
  useWebSocket: () => ({ on: vi.fn(), off: vi.fn(), connectionState: 'connected', subscribe: vi.fn() }),
}));

import WelcomePage from './WelcomePage';
import { getBackgroundImport, resetBackgroundImportForTests } from '../../utils/backgroundImport';

interface Agent { slug: string; name: string; installed: boolean; credentials_configured: boolean; auth_expired?: boolean; supports_login?: boolean }
const READY: Agent = { slug: 'claude-code', name: 'Claude Code', installed: true, credentials_configured: true };
const OUT: Agent = { slug: 'codex', name: 'Codex', installed: true, credentials_configured: true, auth_expired: true };
const MISSING: Agent = { slug: 'pi', name: 'Pi', installed: false, credentials_configured: false };

const ONE_PROJECT = {
  status: 'ready',
  candidates: [{
    source: 'claude-code', name: 'orbital-test', path: '/Users/x/orbital-test', session_count: 40,
    last_activity: '2026-10-26T08:00:00+00:00', by_agent: { 'claude-code': 40 },
  }],
  groups: [],
};
const NO_PROJECTS = { status: 'ready', candidates: [], groups: [] };

interface Routes {
  agents?: Agent[] | Error;
  importable?: unknown;
  apiKeySet?: boolean;
}

const telemetry: Array<Record<string, unknown>> = [];

function mockApi(r: Routes) {
  apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
    if (path === '/api/v2/telemetry/event') {
      telemetry.push(JSON.parse(String(options?.body ?? '{}')));
      return undefined;
    }
    if (path === '/api/v2/settings' && options?.method === 'PUT') return {};
    if (path === '/api/v2/settings') return { llm: { api_key_set: r.apiKeySet ?? false } };
    if (path === '/api/v2/settings/sub-agents' || path === '/api/v2/settings/sub-agents/refresh') {
      if (r.agents instanceof Error) throw r.agents;
      return r.agents ?? [READY, MISSING];
    }
    if (path === '/api/v2/settings/cards') return { cards: [], default_card_id: null };
    if (path.startsWith('/api/v2/settings/cards/') && options?.method === 'PUT') return { default_card_id: 'card_new' };
    if (path === '/api/v2/onboarding/history') return { status: 'ready', sessions_total: 40, by_agent: { 'claude-code': 40 } };
    if (path === '/api/v2/onboarding/importable-projects') {
      if (r.importable instanceof Error) throw r.importable;
      return r.importable ?? ONE_PROJECT;
    }
    if (path === '/api/v2/onboarding/import') return { created: [], skipped: [] };
    return {};
  });
}

const calls = (path: string, method?: string) =>
  apiMock.mock.calls.filter(([p, o]) => p === path && (!method || o?.method === method));
const settingsPuts = () => calls('/api/v2/settings', 'PUT').map(([, o]) => JSON.parse(o.body));
const startBtn = () => screen.getByTestId('welcome-start') as HTMLButtonElement;

beforeEach(() => {
  apiMock.mockReset();
  resetBackgroundImportForTests();
  telemetry.length = 0;
  Object.assign(panel, { connected: false, canSubmit: false, busy: false, error: null, submit: null });
});
afterEach(() => cleanup());

describe('WelcomePage — Start gate', () => {
  it('a signed-out agent does not open Start', async () => {
    mockApi({ agents: [OUT, MISSING] });
    render(<WelcomePage onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('welcome-agent-login-codex')).toBeTruthy());
    expect(startBtn().disabled).toBe(true);
  });

  it('a ready agent opens it', async () => {
    mockApi({ agents: [READY, OUT] });
    render(<WelcomePage onComplete={vi.fn()} />);
    await waitFor(() => expect(startBtn().disabled).toBe(false));
    expect(screen.getByTestId('welcome-agent-ready-claude-code')).toBeTruthy();
  });

  it('a connected Orbital Agent opens it with no agent at all', async () => {
    panel.connected = true;
    panel.canSubmit = true;
    mockApi({ agents: [MISSING] });
    render(<WelcomePage onComplete={vi.fn()} />);
    await waitFor(() => expect(startBtn().disabled).toBe(false));
  });
});

describe('WelcomePage — Start', () => {
  it('without a key: persists the skip flag, fires the import of what is ticked, then hands over', async () => {
    mockApi({});
    const onComplete = vi.fn();
    render(<WelcomePage onComplete={onComplete} />);
    await waitFor(() => expect(startBtn().textContent).toBe('Import 1 project and start'));
    await act(async () => { fireEvent.click(startBtn()); });
    await waitFor(() => expect(onComplete).toHaveBeenCalled());
    expect(settingsPuts()).toEqual([{ setup_api_key_skipped: true }]);
    const imp = calls('/api/v2/onboarding/import', 'POST');
    expect(imp).toHaveLength(1);
    expect(JSON.parse(imp[0][1].body)).toEqual({ folders: ['/Users/x/orbital-test'], groups: [] });
    // never writes the agents denylist
    expect(settingsPuts().some((b) => 'sub_agents_hidden' in b)).toBe(false);
    const steps = telemetry.filter((e) => e.event === 'wizard_step_reached').map((e) => (e.fields as { step: string }).step);
    expect(steps).toEqual(['agents', 'done']);
  });

  it('the line checkbox clears the import: the button says Start and nothing is imported', async () => {
    mockApi({});
    const onComplete = vi.fn();
    render(<WelcomePage onComplete={onComplete} />);
    const check = await waitFor(() => screen.getByTestId('welcome-import-check'));
    expect(check.getAttribute('aria-checked')).toBe('true');
    fireEvent.click(check);
    expect(check.getAttribute('aria-checked')).toBe('false');
    expect(startBtn().textContent).toBe('Start');
    await act(async () => { fireEvent.click(startBtn()); });
    await waitFor(() => expect(onComplete).toHaveBeenCalled());
    expect(calls('/api/v2/onboarding/import')).toHaveLength(0);
  });

  it('with a passing key: saves it through the panel, makes it the default, no skip flag', async () => {
    panel.connected = true;
    panel.canSubmit = true;
    mockApi({ importable: NO_PROJECTS, apiKeySet: true });
    const onComplete = vi.fn();
    render(<WelcomePage onComplete={onComplete} />);
    await waitFor(() => expect(startBtn().disabled).toBe(false));
    await act(async () => { fireEvent.click(startBtn()); });
    await waitFor(() => expect(onComplete).toHaveBeenCalled());
    expect(calls('/api/v2/settings/cards/card_new/default', 'PUT')).toHaveLength(1);
    expect(settingsPuts()).toEqual([]);
    const steps = telemetry.filter((e) => e.event === 'wizard_step_reached').map((e) => (e.fields as { step: string }).step);
    expect(steps).toEqual(['agents', 'key', 'done']);
  });

  it('the panel goes busy, reports done, then idle before the settings check answers: Start still finishes', async () => {
    // The live order (ConnectPanel.save): submitting → onDone(card) → finally submitting=false,
    // while handleConnected is still awaiting GET /settings.
    panel.connected = true;
    panel.canSubmit = true;
    panel.submit = (done, setBusy) => {
      act(() => setBusy(true));
      setTimeout(() => {
        act(() => { done({ ...STORED_CARD, is_default: true }); setBusy(false); });
      }, 5);
    };
    mockApi({ importable: NO_PROJECTS, apiKeySet: true });
    const base = apiMock.getMockImplementation()!;
    apiMock.mockImplementation(async (path: string, o?: RequestInit) => {
      if (path === '/api/v2/settings' && !o?.method) await new Promise((r) => setTimeout(r, 40));
      return base(path, o);
    });
    const onComplete = vi.fn();
    render(<WelcomePage onComplete={onComplete} />);
    await waitFor(() => expect(startBtn().disabled).toBe(false));
    await act(async () => { fireEvent.click(startBtn()); });
    await waitFor(() => expect(onComplete).toHaveBeenCalled(), { timeout: 2000 });
  });

  it('a save that fails (busy, then idle, no done) gives Start back', async () => {
    panel.connected = true;
    panel.canSubmit = true;
    panel.submit = (_done, setBusy) => {
      act(() => setBusy(true));
      setTimeout(() => act(() => setBusy(false)), 5);
    };
    mockApi({ importable: NO_PROJECTS });
    const onComplete = vi.fn();
    render(<WelcomePage onComplete={onComplete} />);
    await waitFor(() => expect(startBtn().disabled).toBe(false));
    await act(async () => { fireEvent.click(startBtn()); });
    await waitFor(() => expect(startBtn().getAttribute('aria-busy')).toBeNull());
    expect(startBtn().disabled).toBe(false);
    expect(onComplete).not.toHaveBeenCalled();
  });

  it('hands over at once: the import runs in the background, Start never waits for it', async () => {
    mockApi({});
    const base = apiMock.getMockImplementation()!;
    // an import that never answers
    apiMock.mockImplementation(async (path: string, o?: RequestInit) => {
      if (path === '/api/v2/onboarding/import') return new Promise(() => {});
      return base(path, o);
    });
    const onComplete = vi.fn();
    render(<WelcomePage onComplete={onComplete} />);
    await waitFor(() => expect(startBtn().textContent).toBe('Import 1 project and start'));
    await act(async () => { fireEvent.click(startBtn()); });
    await waitFor(() => expect(onComplete).toHaveBeenCalled());
    expect(calls('/api/v2/onboarding/import', 'POST')).toHaveLength(1);
    expect(getBackgroundImport().status).toBe('pending');
  });

  it('a failed import does not hold the page: it hands over and the failure waits in the background', async () => {
    mockApi({});
    const base = apiMock.getMockImplementation()!;
    apiMock.mockImplementation(async (path: string, o?: RequestInit) => {
      if (path === '/api/v2/onboarding/import') throw new Error('boom');
      return base(path, o);
    });
    const onComplete = vi.fn();
    render(<WelcomePage onComplete={onComplete} />);
    await waitFor(() => expect(startBtn().textContent).toBe('Import 1 project and start'));
    await act(async () => { fireEvent.click(startBtn()); });
    await waitFor(() => expect(onComplete).toHaveBeenCalled());
    await waitFor(() => expect(getBackgroundImport().status).toBe('failed'));
    expect(getBackgroundImport().request).toEqual({ folders: ['/Users/x/orbital-test'], groups: [] });
  });

  it('a failed skip-flag write keeps the page and fires no import', async () => {
    mockApi({});
    const base = apiMock.getMockImplementation()!;
    apiMock.mockImplementation(async (path: string, o?: RequestInit) => {
      if (path === '/api/v2/settings' && o?.method === 'PUT') throw new Error('down');
      return base(path, o);
    });
    const onComplete = vi.fn();
    render(<WelcomePage onComplete={onComplete} />);
    await waitFor(() => expect(startBtn().textContent).toBe('Import 1 project and start'));
    await act(async () => { fireEvent.click(startBtn()); });
    await waitFor(() => expect(screen.getByTestId('welcome-start-error')).toBeTruthy());
    expect(onComplete).not.toHaveBeenCalled();
    expect(calls('/api/v2/onboarding/import')).toHaveLength(0);
    expect(startBtn().disabled).toBe(false);
  });
});

describe('WelcomePage — bands', () => {
  it('no history: the Projects band is absent', async () => {
    mockApi({ importable: NO_PROJECTS });
    render(<WelcomePage onComplete={vi.fn()} />);
    await waitFor(() => expect(startBtn().textContent).toBe('Start'));
    await waitFor(() => expect(screen.queryByTestId('welcome-band-projects')).toBeNull());
  });

  it('a scan error is shown in the band with Try again', async () => {
    mockApi({ importable: new Error('down') });
    render(<WelcomePage onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('welcome-projects-error')).toBeTruthy());
    expect(screen.getByTestId('welcome-projects-retry')).toBeTruthy();
  });

  it('a failed agent list is a retry, not "none found"', async () => {
    mockApi({ agents: new Error('down') });
    render(<WelcomePage onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('welcome-agents-error')).toBeTruthy());
    expect(screen.queryByTestId('welcome-agents-add')).toBeNull();
    expect(screen.queryByText('Installed but not found?')).toBeNull();
    expect(startBtn().disabled).toBe(true);
  });

  it('none installed: the quiet link opens the locate chips and the install line', async () => {
    mockApi({ agents: [MISSING] });
    render(<WelcomePage onComplete={vi.fn()} />);
    const link = await waitFor(() => screen.getByText('Installed but not found?'));
    fireEvent.click(link);
    expect(screen.getByTestId('welcome-locate-chip-pi')).toBeTruthy();
    expect(screen.getByTestId('welcome-install-line')).toBeTruthy();
  });

  it('View all opens the checklist and folds the other bands; Esc puts them back', async () => {
    mockApi({});
    render(<WelcomePage onComplete={vi.fn()} />);
    const review = await waitFor(() => screen.getByTestId('welcome-review'));
    await waitFor(() => expect(screen.getByTestId('welcome-agent-list')).toBeTruthy());
    fireEvent.click(review);
    expect(screen.getByTestId('welcome-review-list')).toBeTruthy();
    expect(screen.getByTestId('welcome-agents-summary')).toBeTruthy();
    expect(screen.queryByTestId('welcome-agent-list')).toBeNull();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByTestId('welcome-review-list')).toBeNull();
    expect(screen.getByTestId('welcome-agent-list')).toBeTruthy();
  });
});
