// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * SetupWizard unit tests — Task I1 (Spec 011 §0.7).
 *
 * Covers:
 *  - step flow api_key → accounts → complete (Get Started calls onComplete);
 *  - the sandbox step is gone: advancing never fetches /api/v2/platform/status
 *    (nor POSTs /platform/skip or /platform/setup);
 *  - featured filter: only `featured && status === 'available'` connectors
 *    render as cards;
 *  - connect: busy "waiting for browser consent" state, 400 → client-hint,
 *    success → list refetch showing "Connected as {account}";
 *  - zero-featured / fetch-failure fallback: browser sign-in group alone;
 *  - Skip calls onComplete.
 *
 * The api client (web/src/config.ts `api`) is mocked — no network. The
 * LLMProviderSettings form is stubbed out (it has its own coverage).
 */

import { render, screen, waitFor, cleanup, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Connector } from '../types';
import { LocaleProvider } from '../i18n/LocaleContext';
import { LOCALE_STORAGE_KEY } from '../i18n/locales';

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
vi.mock('./LLMProviderSettings', () => ({
  default: () => <div data-testid="llm-provider-settings-stub" />,
}));

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

interface MockRoutes {
  apiKeySet?: boolean;
  connectors?: Connector[] | Error;
  connect?: () => Promise<unknown>;
}

/** Route the api mock by path; unmatched paths resolve to {}. */
function mockApi(routes: MockRoutes) {
  apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
    if (path === '/api/v2/settings') {
      return { llm: { api_key_set: routes.apiKeySet ?? true } };
    }
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

beforeEach(() => {
  apiMock.mockReset();
});

describe('SetupWizard — step flow', () => {
  it('stays on api_key when no key is set, then advances to accounts on Next', async () => {
    mockApi({ apiKeySet: false });
    render(<SetupWizard onComplete={vi.fn()} />);

    // api_key step visible
    expect(screen.getByText('Welcome to Orbital')).toBeTruthy();
    await waitFor(() => expect(apiMock).toHaveBeenCalledWith('/api/v2/settings'));
    expect(screen.queryByText('Connect Your Accounts')).toBeNull();

    // Key now set — Next advances
    mockApi({ apiKeySet: true });
    fireEvent.click(screen.getByText('Next →'));
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
  });

  it('auto-advances to accounts when the key is already configured', async () => {
    mockApi({ apiKeySet: true });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
  });

  it('Get Started calls onComplete', async () => {
    mockApi({ apiKeySet: true });
    const onComplete = vi.fn();
    render(<SetupWizard onComplete={onComplete} />);
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());

    fireEvent.click(screen.getByTestId('wizard-get-started'));
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('Skip calls onComplete', async () => {
    mockApi({ apiKeySet: true });
    const onComplete = vi.fn();
    render(<SetupWizard onComplete={onComplete} />);
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());

    fireEvent.click(screen.getByTestId('wizard-skip'));
    expect(onComplete).toHaveBeenCalledTimes(1);
  });
});

describe('SetupWizard — sandbox step is gone', () => {
  it('never touches the platform sandbox endpoints while advancing', async () => {
    mockApi({ apiKeySet: true, connectors: [makeConnector()] });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
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
    render(<SetupWizard onComplete={vi.fn()} />);

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
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(calledPaths()).toContain('/api/v2/connectors'));

    expect(screen.queryByTestId('wizard-connectors-group')).toBeNull();
    expect(screen.getByTestId('wizard-browser-group')).toBeTruthy();
    expect(screen.getByText('Agent browser sign-in')).toBeTruthy();
  });

  it('shows the browser group alone when the connectors fetch fails', async () => {
    mockApi({ apiKeySet: true, connectors: new Error('boom') });
    render(<SetupWizard onComplete={vi.fn()} />);
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
    render(<SetupWizard onComplete={vi.fn()} />);
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
    render(<SetupWizard onComplete={vi.fn()} />);
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
    let listCalls = 0;
    apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === '/api/v2/settings') return { llm: { api_key_set: true } };
      if (path === '/api/v2/connectors') {
        listCalls += 1;
        return { connectors: [listCalls > 1 ? connected : disconnected] };
      }
      if (path === '/api/v2/connectors/google-calendar/connect' && options?.method === 'POST') {
        return {};
      }
      return {};
    });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() =>
      expect(screen.getByTestId('wizard-connector-connect-google-calendar')).toBeTruthy(),
    );

    await act(async () => {
      fireEvent.click(screen.getByTestId('wizard-connector-connect-google-calendar'));
    });

    expect(listCalls).toBe(2);
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

  it('renders on the api_key step, switches locale on change, and persists to localStorage', async () => {
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
    render(
      <LocaleProvider>
        <SetupWizard onComplete={vi.fn()} />
      </LocaleProvider>,
    );
    await waitFor(() => expect(screen.getByText('Connect Your Accounts')).toBeTruthy());
    expect(screen.getByTestId('wizard-locale-select')).toBeTruthy();
  });
});

describe('SetupWizard — agent browser sign-in group', () => {
  it('POSTs the warmup endpoint and shows the opening state', async () => {
    mockApi({ apiKeySet: true });
    render(<SetupWizard onComplete={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('wizard-open-browser')).toBeTruthy());

    await act(async () => {
      fireEvent.click(screen.getByTestId('wizard-open-browser'));
    });

    expect(apiMock).toHaveBeenCalledWith('/api/v2/platform/browser/warmup', { method: 'POST' });
    expect(screen.getByText('Browser open — sign in and close it...')).toBeTruthy();
  });
});
