// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Who may use a connector (connector sharing, 2026-11-17): the access control
 * on each connected card in Global Settings → Connectors.
 *
 *  - default is "All agents"; a disconnected card has no control;
 *  - switching to "Selected agents" PUTs the right body;
 *  - ticking / unticking sends the full next selection;
 *  - unsupported agents are listed but cannot be ticked, the reason is behind
 *    a hover sign;
 *  - a failed save puts the last saved value back and shows the API's error.
 */

import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Connector, ConnectorAgent } from '../types';

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

import ConnectorSettings, { ConnectorOrigin } from './ConnectorSettings';
import { accessForMode, accessWithAgent, tickedAgents } from './ConnectorAccessControl';

afterEach(() => cleanup());

function makeConnector(overrides: Partial<Connector> = {}): Connector {
  return {
    id: 'gmail',
    name: 'Gmail',
    icon: '✉️',
    auth_provider: 'google',
    auth_type: 'oauth2',
    server_url: 'https://mcp.example/gmail',
    oauth_scopes: [],
    tool_overrides: {},
    featured: true,
    status: 'available',
    connected: true,
    account: 'me@example.com',
    enabled_error: null,
    access: { mode: 'all', agents: [] },
    ...overrides,
  };
}

const AGENTS: ConnectorAgent[] = [
  { id: 'orbital', name: 'Orbital', kind: 'manager', installed: true, supported: true, reason: null },
  { id: 'claude-code', name: 'Claude Code', kind: 'worker', installed: true, supported: true, reason: null },
  { id: 'codex', name: 'Codex', kind: 'worker', installed: true, supported: true, reason: null },
  { id: 'aider', name: 'Aider', kind: 'worker', installed: false, supported: false, reason: 'no_mcp_mount' },
];

type Handler = (path: string, options?: RequestInit) => unknown;

function mockApi(connectors: Connector[], onPut?: Handler) {
  apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
    if (path === '/api/v2/connectors') return { connectors };
    if (path === '/api/v2/connectors/agents') return { agents: AGENTS };
    if (options?.method === 'PUT' && onPut) return onPut(path, options);
    if (options?.method === 'PUT') {
      const body = JSON.parse(String(options.body));
      return { id: path.split('/')[4], access: body };
    }
    return {};
  });
}

function puts() {
  return apiMock.mock.calls
    .filter(([, o]) => o?.method === 'PUT')
    .map(([p, o]) => [p, JSON.parse(String(o.body))]);
}

async function openMenu(id = 'gmail') {
  const trigger = await screen.findByTestId(`connector-access-${id}`);
  // Wait for the agent list, so the menu is not opened on a loading list.
  await waitFor(() => expect(apiMock.mock.calls.some(([p]) => p === '/api/v2/connectors/agents')).toBe(true));
  fireEvent.click(trigger);
  return screen.getByTestId(`connector-access-menu-${id}`);
}

function check(agent: string, id = 'gmail') {
  return screen.getByTestId(`connector-access-check-${id}-${agent}`) as HTMLInputElement;
}

beforeEach(() => {
  apiMock.mockReset();
});

describe('connector access — display', () => {
  it('defaults to "All agents" on a connected card, every supported agent ticked and locked', async () => {
    mockApi([makeConnector({ access: undefined })]);
    render(<ConnectorSettings />);
    const trigger = await screen.findByTestId('connector-access-gmail');
    expect(trigger.textContent).toBe('All agents');
    await openMenu();
    expect(screen.getByTestId('connector-access-mode-all-gmail').getAttribute('aria-checked')).toBe('true');
    await waitFor(() => expect(check('orbital').checked).toBe(true));
    expect(check('orbital').disabled).toBe(true);
    expect(check('claude-code').checked).toBe(true);
  });

  it('shows no access control on a card that is not connected', async () => {
    mockApi([makeConnector({ connected: false, account: null })]);
    render(<ConnectorSettings />);
    await screen.findByTestId('connector-connect-gmail');
    expect(screen.queryByTestId('connector-access-gmail')).not.toBeInTheDocument();
  });

  it('shows the chosen agents by logo when the mode is selected', async () => {
    mockApi([makeConnector({ access: { mode: 'selected', agents: ['codex', 'orbital'] } })]);
    render(<ConnectorSettings />);
    const trigger = await screen.findByTestId('connector-access-gmail');
    await waitFor(() =>
      expect(Array.from(trigger.querySelectorAll('[data-agent]')).map((e) => e.getAttribute('data-agent'))).toEqual([
        'orbital',
        'codex',
      ]),
    );
  });

  it('reads "None" when selected has nobody', async () => {
    mockApi([makeConnector({ access: { mode: 'selected', agents: [] } })]);
    render(<ConnectorSettings />);
    const trigger = await screen.findByTestId('connector-access-gmail');
    expect(trigger.textContent).toBe('None');
  });
});

describe('connector access — order', () => {
  it('lists the agents that can be ticked first, unsupported ones after', async () => {
    const mixed = [AGENTS[0], AGENTS[3], AGENTS[1], AGENTS[2]];
    apiMock.mockImplementation(async (path: string) => {
      if (path === '/api/v2/connectors') return { connectors: [makeConnector()] };
      if (path === '/api/v2/connectors/agents') return { agents: mixed };
      return {};
    });
    render(<ConnectorSettings />);
    const menu = await openMenu();
    await waitFor(() => expect(check('aider')).toBeInTheDocument());
    const order = Array.from(menu.querySelectorAll('[data-testid^="connector-access-agent-gmail-"]')).map((e) =>
      e.getAttribute('data-testid')!.replace('connector-access-agent-gmail-', ''),
    );
    expect(order).toEqual(['orbital', 'claude-code', 'codex', 'aider']);
  });
});

describe('connector access — saving', () => {
  it('switching to selected PUTs every supported agent (nothing taken away by the switch)', async () => {
    mockApi([makeConnector()]);
    render(<ConnectorSettings />);
    await openMenu();
    await waitFor(() => expect(check('orbital')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('connector-access-mode-selected-gmail'));
    await waitFor(() =>
      expect(puts()).toEqual([
        ['/api/v2/connectors/gmail/access', { mode: 'selected', agents: ['orbital', 'claude-code', 'codex'] }],
      ]),
    );
    expect(check('orbital').disabled).toBe(false);
  });

  it('switching to selected restores the last selection the daemon kept', async () => {
    mockApi([makeConnector({ access: { mode: 'all', agents: ['codex'] } })]);
    render(<ConnectorSettings />);
    await openMenu();
    await waitFor(() => expect(check('orbital')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('connector-access-mode-selected-gmail'));
    await waitFor(() =>
      expect(puts()).toEqual([['/api/v2/connectors/gmail/access', { mode: 'selected', agents: ['codex'] }]]),
    );
  });

  it('unticking an agent sends the full next selection; back to all keeps it', async () => {
    mockApi([makeConnector({ access: { mode: 'selected', agents: ['orbital', 'codex'] } })]);
    render(<ConnectorSettings />);
    await openMenu();
    await waitFor(() => expect(check('codex').checked).toBe(true));
    fireEvent.click(check('codex'));
    await waitFor(() =>
      expect(puts()).toEqual([['/api/v2/connectors/gmail/access', { mode: 'selected', agents: ['orbital'] }]]),
    );
    expect(check('codex').checked).toBe(false);
    fireEvent.click(screen.getByTestId('connector-access-mode-all-gmail'));
    await waitFor(() =>
      expect(puts()[1]).toEqual(['/api/v2/connectors/gmail/access', { mode: 'all', agents: ['orbital'] }]),
    );
  });

  it('an unsupported agent is listed but cannot be ticked; why is behind a hover sign', async () => {
    mockApi([makeConnector({ access: { mode: 'selected', agents: ['orbital'] } })]);
    render(<ConnectorSettings />);
    const menu = await openMenu();
    await waitFor(() => expect(check('aider')).toBeInTheDocument());
    expect(check('aider').disabled).toBe(true);
    expect(check('aider').checked).toBe(false);
    // No visible reason text until the sign is hovered or tapped.
    expect(menu.textContent).not.toContain("can't use connectors");
    fireEvent.click(check('aider'));
    expect(puts()).toEqual([]);
    fireEvent.click(screen.getByTestId('connector-access-why-gmail-aider'));
    expect(screen.getByRole('tooltip').textContent).toBe("This agent can't use connectors.");
  });

  it('a failed save puts the saved value back and shows the API error on the card', async () => {
    mockApi([makeConnector({ access: { mode: 'selected', agents: ['orbital', 'codex'] } })], async () => {
      throw new MockApiError(400, 'unknown agent: codex');
    });
    render(<ConnectorSettings />);
    await openMenu();
    await waitFor(() => expect(check('codex').checked).toBe(true));
    fireEvent.click(check('codex'));
    await waitFor(() =>
      expect(screen.getByTestId('connector-error-gmail').textContent).toContain('unknown agent: codex'),
    );
    expect(check('codex').checked).toBe(true);
  });

  it('a failed switch to selected goes back to All agents', async () => {
    mockApi([makeConnector()], async () => {
      throw new MockApiError(503, 'Connector manager not available');
    });
    render(<ConnectorSettings />);
    await openMenu();
    await waitFor(() => expect(check('orbital')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('connector-access-mode-selected-gmail'));
    await waitFor(() => expect(screen.getByTestId('connector-error-gmail')).toBeInTheDocument());
    expect(screen.getByTestId('connector-access-mode-all-gmail').getAttribute('aria-checked')).toBe('true');
    expect(screen.getByTestId('connector-access-gmail').textContent).toBe('All agents');
  });
});

describe('connector access — pure helpers', () => {
  it('tickedAgents keeps list order and drops unsupported and unknown ids', () => {
    expect(tickedAgents({ mode: 'selected', agents: ['aider', 'codex', 'ghost', 'orbital'] }, AGENTS)).toEqual([
      'orbital',
      'codex',
    ]);
  });

  it('accessForMode(selected) with only unsupported ids starts from every supported agent', () => {
    expect(accessForMode('selected', { mode: 'all', agents: ['aider'] }, AGENTS)).toEqual({
      mode: 'selected',
      agents: ['orbital', 'claude-code', 'codex'],
    });
  });

  it('accessWithAgent adds once and removes', () => {
    const a = accessWithAgent({ mode: 'selected', agents: ['orbital'] }, 'codex', true);
    expect(a).toEqual({ mode: 'selected', agents: ['orbital', 'codex'] });
    expect(accessWithAgent(a, 'codex', true).agents).toEqual(['orbital', 'codex']);
    expect(accessWithAgent(a, 'orbital', false).agents).toEqual(['codex']);
  });
});

describe('connector origin slot', () => {
  it('renders nothing without an origin and the agent logo with one', () => {
    const { container, rerender } = render(<ConnectorOrigin />);
    expect(container.innerHTML).toBe('');
    rerender(<ConnectorOrigin agent="codex" />);
    expect(screen.getByTestId('connector-origin').querySelector('[data-agent="codex"]')).not.toBeNull();
  });
});
