// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * ConnectorSettings unit tests — Task E1 (Spec 011 §0.2/§0.6).
 *
 * Covers (per the plan's Vitest list — card states + actions):
 *  - card states: connected-as-account, connect button, pending-verification
 *    badge (not connectable);
 *  - connect: busy "waiting for browser consent…" state, refresh on success,
 *    BYO-client hint on the 400 client-not-configured error;
 *  - disconnect: PROVIDER-scoped confirm copy (names the auth_provider, not
 *    the single service), cancel path, POST + refresh on confirm;
 *  - custom MCP server: add-form POST payload, remove (DELETE) for custom
 *    entries only.
 *
 * The api client (web/src/config.ts `api`) is mocked — no network.
 */

import { render, screen, waitFor, cleanup, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Connector } from '../types';

const { apiMock, MockApiError } = vi.hoisted(() => {
  class MockApiError extends Error {
    constructor(
      public status: number,
      public detail: string,
    ) {
      super(detail);
      this.name = 'ApiError';
    }
  }
  return { apiMock: vi.fn(), MockApiError };
});
vi.mock('../config', () => ({ api: apiMock, ApiError: MockApiError }));

import ConnectorSettings from './ConnectorSettings';

afterEach(() => cleanup());

function makeConnector(overrides: Partial<Connector> = {}): Connector {
  return {
    id: 'google-calendar',
    name: 'Google Calendar',
    icon: '📅',
    auth_provider: 'google',
    auth_type: 'oauth2',
    server_url: 'https://mcp.example/calendar',
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

const CALENDAR = makeConnector();
const DRIVE = makeConnector({
  id: 'google-drive',
  name: 'Google Drive',
  icon: '📁',
  connected: true,
  account: 'zq@example.com',
});
const GMAIL = makeConnector({
  id: 'gmail',
  name: 'Gmail',
  icon: '✉️',
  status: 'pending_verification',
});
const CUSTOM = makeConnector({
  id: 'custom-my-server',
  name: 'My server',
  icon: '🔌',
  auth_provider: 'custom-my-server',
  auth_type: 'none',
  server_url: 'https://example.com/mcp',
});

/** Default api mock: GET list resolves; mutations resolve empty. */
function mockList(connectors: Connector[]) {
  apiMock.mockImplementation(async (path: string) => {
    if (path === '/api/v2/connectors') return { connectors };
    return {};
  });
}

/** All calls made to a given path (url, options). */
function callsTo(path: string) {
  return apiMock.mock.calls.filter(([url]) => url === path);
}

beforeEach(() => {
  apiMock.mockReset();
});

describe('ConnectorSettings — card states', () => {
  it('renders a connected card as "Connected as {account}" with a Disconnect action', async () => {
    mockList([DRIVE]);
    render(<ConnectorSettings />);
    await waitFor(() =>
      expect(screen.getByTestId('connector-status-google-drive').textContent).toBe(
        'Connected as zq@example.com',
      ),
    );
    expect(screen.getByTestId('connector-disconnect-google-drive')).toBeInTheDocument();
    expect(screen.queryByTestId('connector-connect-google-drive')).not.toBeInTheDocument();
  });

  it('renders an available disconnected card with a Connect button', async () => {
    mockList([CALENDAR]);
    render(<ConnectorSettings />);
    await waitFor(() =>
      expect(screen.getByTestId('connector-connect-google-calendar')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('connector-status-google-calendar').textContent).toBe(
      'Not connected',
    );
    expect(
      screen.queryByTestId('connector-disconnect-google-calendar'),
    ).not.toBeInTheDocument();
  });

  it('renders a pending_verification card with a badge and NO connect button (gmail)', async () => {
    mockList([GMAIL]);
    render(<ConnectorSettings />);
    await waitFor(() =>
      expect(screen.getByTestId('connector-pending-gmail').textContent).toBe(
        'Pending verification',
      ),
    );
    expect(screen.queryByTestId('connector-connect-gmail')).not.toBeInTheDocument();
    expect(screen.queryByTestId('connector-disconnect-gmail')).not.toBeInTheDocument();
  });

  it('shows enabled_error backend text verbatim on the card', async () => {
    mockList([makeConnector({ id: 'google-drive', name: 'Google Drive', connected: true, account: 'a@b.c', enabled_error: 'MCP session failed: boom' })]);
    render(<ConnectorSettings />);
    await waitFor(() =>
      expect(screen.getByText('MCP session failed: boom')).toBeInTheDocument(),
    );
  });

  it('only custom entries get a Remove action', async () => {
    mockList([CALENDAR, CUSTOM]);
    render(<ConnectorSettings />);
    await waitFor(() =>
      expect(screen.getByTestId('connector-remove-custom-my-server')).toBeInTheDocument(),
    );
    expect(screen.queryByTestId('connector-remove-google-calendar')).not.toBeInTheDocument();
  });
});

describe('ConnectorSettings — connect', () => {
  it('shows the "waiting for browser consent" busy state while the blocking connect is in flight, then refreshes', async () => {
    let resolveConnect!: (v: unknown) => void;
    let listCalls = 0;
    apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === '/api/v2/connectors') {
        listCalls += 1;
        // First list: disconnected; post-connect refresh: connected.
        return {
          connectors: [
            listCalls === 1
              ? CALENDAR
              : makeConnector({ connected: true, account: 'zq@example.com' }),
          ],
        };
      }
      if (path === '/api/v2/connectors/google-calendar/connect' && options?.method === 'POST') {
        return new Promise((resolve) => {
          resolveConnect = resolve;
        });
      }
      return {};
    });

    render(<ConnectorSettings />);
    const btn = await screen.findByTestId('connector-connect-google-calendar');
    fireEvent.click(btn);

    // Busy state: label flips and the button is disabled for the (up to 300s)
    // OAuth consent wait.
    await waitFor(() =>
      expect(screen.getByTestId('connector-connect-google-calendar').textContent).toBe(
        'Waiting for browser consent…',
      ),
    );
    expect(
      (screen.getByTestId('connector-connect-google-calendar') as HTMLButtonElement).disabled,
    ).toBe(true);

    await act(async () => {
      resolveConnect({ connected: true, account: 'zq@example.com' });
    });
    await waitFor(() =>
      expect(screen.getByTestId('connector-status-google-calendar').textContent).toBe(
        'Connected as zq@example.com',
      ),
    );
    expect(listCalls).toBe(2);
  });

  it('surfaces the BYO-client hint on the 400 client-not-configured error', async () => {
    apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === '/api/v2/connectors') return { connectors: [CALENDAR] };
      if (options?.method === 'POST') {
        throw new MockApiError(
          400,
          "OAuth client for provider 'google' is not configured (set connector_google_client_id / connector_google_client_secret)",
        );
      }
      return {};
    });
    render(<ConnectorSettings />);
    fireEvent.click(await screen.findByTestId('connector-connect-google-calendar'));

    const error = await screen.findByTestId('connector-error-google-calendar');
    // Backend detail verbatim + the translated hint naming the settings keys.
    expect(error.textContent).toContain('not configured');
    expect(error.textContent).toContain('connector_google_client_id');
    expect(error.textContent).toContain('connector_google_client_secret');
    expect(error.textContent).toContain("The OAuth client isn't configured");
  });

  it('shows the backend detail without the hint on a 409 (pending verification)', async () => {
    apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === '/api/v2/connectors') return { connectors: [CALENDAR] };
      if (options?.method === 'POST') {
        throw new MockApiError(409, 'connector gmail is pending verification');
      }
      return {};
    });
    render(<ConnectorSettings />);
    fireEvent.click(await screen.findByTestId('connector-connect-google-calendar'));

    const error = await screen.findByTestId('connector-error-google-calendar');
    expect(error.textContent).toContain('pending verification');
    expect(error.textContent).not.toContain('connector_google_client_id');
  });
});

describe('ConnectorSettings — provider-scoped disconnect', () => {
  it('confirm copy names the auth provider account, NOT the single service', async () => {
    mockList([DRIVE]);
    render(<ConnectorSettings />);
    fireEvent.click(await screen.findByTestId('connector-disconnect-google-drive'));

    const confirm = screen.getByTestId('connector-disconnect-confirm-google-drive');
    // Provider-scoped: one shared Google token — the copy must say "Google
    // account", never "Google Drive account".
    expect(confirm.textContent).toContain('Disconnect Google account?');
    expect(confirm.textContent).toContain('every Google connector goes offline');
    expect(confirm.textContent).not.toContain('Google Drive account');
  });

  it('cancel dismisses the confirm without calling the API', async () => {
    mockList([DRIVE]);
    render(<ConnectorSettings />);
    fireEvent.click(await screen.findByTestId('connector-disconnect-google-drive'));
    fireEvent.click(screen.getByText('Cancel'));

    expect(
      screen.queryByTestId('connector-disconnect-confirm-google-drive'),
    ).not.toBeInTheDocument();
    expect(callsTo('/api/v2/connectors/google-drive/disconnect')).toHaveLength(0);
  });

  it('confirm POSTs the disconnect and refreshes the list', async () => {
    mockList([DRIVE]);
    render(<ConnectorSettings />);
    fireEvent.click(await screen.findByTestId('connector-disconnect-google-drive'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('connector-disconnect-confirm-btn-google-drive'));
    });

    const calls = callsTo('/api/v2/connectors/google-drive/disconnect');
    expect(calls).toHaveLength(1);
    expect((calls[0][1] as RequestInit).method).toBe('POST');
    // Initial GET + post-disconnect refresh.
    expect(callsTo('/api/v2/connectors')).toHaveLength(2);
  });
});

describe('ConnectorSettings — custom MCP servers', () => {
  it('submits the add form as POST /custom with {name, url, auth_type} and clears it', async () => {
    mockList([CALENDAR]);
    render(<ConnectorSettings />);
    await screen.findByTestId('connector-connect-google-calendar');

    fireEvent.change(screen.getByTestId('connector-custom-name'), {
      target: { value: 'My server' },
    });
    fireEvent.change(screen.getByTestId('connector-custom-url'), {
      target: { value: 'https://example.com/mcp' },
    });
    fireEvent.change(screen.getByTestId('connector-custom-auth'), {
      target: { value: 'oauth2' },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('connector-custom-add'));
    });

    const calls = callsTo('/api/v2/connectors/custom');
    expect(calls).toHaveLength(1);
    const opts = calls[0][1] as RequestInit;
    expect(opts.method).toBe('POST');
    expect(JSON.parse(opts.body as string)).toEqual({
      name: 'My server',
      url: 'https://example.com/mcp',
      auth_type: 'oauth2',
    });
    // Form cleared + list refreshed.
    expect((screen.getByTestId('connector-custom-name') as HTMLInputElement).value).toBe('');
    expect((screen.getByTestId('connector-custom-url') as HTMLInputElement).value).toBe('');
    expect(callsTo('/api/v2/connectors')).toHaveLength(2);
  });

  it('shows the backend detail when adding fails', async () => {
    apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === '/api/v2/connectors') return { connectors: [] };
      if (options?.method === 'POST') throw new MockApiError(400, 'invalid server url');
      return {};
    });
    render(<ConnectorSettings />);
    fireEvent.change(screen.getByTestId('connector-custom-name'), {
      target: { value: 'x' },
    });
    fireEvent.change(screen.getByTestId('connector-custom-url'), {
      target: { value: 'nope' },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('connector-custom-add'));
    });
    expect(screen.getByTestId('connector-custom-error').textContent).toBe(
      'invalid server url',
    );
  });

  it('remove on a custom entry DELETEs it and refreshes', async () => {
    mockList([CUSTOM]);
    render(<ConnectorSettings />);
    const removeBtn = await screen.findByTestId('connector-remove-custom-my-server');
    await act(async () => {
      fireEvent.click(removeBtn);
    });

    const calls = callsTo('/api/v2/connectors/custom/custom-my-server');
    expect(calls).toHaveLength(1);
    expect((calls[0][1] as RequestInit).method).toBe('DELETE');
    expect(callsTo('/api/v2/connectors')).toHaveLength(2);
  });
});
