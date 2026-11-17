// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Agents' own connectors (2026-11-17): below Orbital's connectors, each
 * agent's own MCP servers as it last reported them. Read-only: the agent's
 * logo and the names, no switch, no access control, no text on screen; one
 * hover sign carries the meaning. Nothing seen, or a failed load, shows
 * nothing.
 */

import { render, screen, waitFor, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import type { AgentHeldConnectors as Row } from '../types';

const { apiMock } = vi.hoisted(() => ({ apiMock: vi.fn() }));
vi.mock('../config', () => ({ api: apiMock, ApiError: Error }));

import AgentHeldConnectors from './AgentHeldConnectors';

afterEach(() => {
  cleanup();
  apiMock.mockReset();
});

const ROWS: Row[] = [
  {
    id: 'claude-code',
    name: 'Claude Code',
    last_seen: '2026-11-17T09:00:00Z',
    connectors: [
      { name: 'figma', raw_name: 'figma', status: 'needs-auth' },
      { name: 'Gmail', raw_name: 'claude.ai Gmail', status: 'pending' },
      { name: 'playwright', raw_name: 'playwright', status: 'connected' },
    ],
  },
  {
    id: 'codex',
    name: 'Codex',
    last_seen: '2026-11-17T09:01:00Z',
    connectors: [{ name: 'node_repl', raw_name: 'node_repl', status: 'connected' }],
  },
];

describe('AgentHeldConnectors', () => {
  it('lists each agent by its logo with its connector names', async () => {
    apiMock.mockResolvedValue({ agents: ROWS });
    render(<AgentHeldConnectors />);
    await screen.findByText('Gmail');
    expect(apiMock).toHaveBeenCalledWith('/api/v2/connectors/agent-held');
    const lists = screen.getAllByRole('list', { name: /own connectors/i });
    expect(lists).toHaveLength(2);
    expect(lists[0].querySelectorAll('li')).toHaveLength(3);
    expect(screen.getByRole('img', { name: 'Claude Code' })).toBeTruthy();
    expect(screen.getByRole('img', { name: 'Codex' })).toBeTruthy();
    // The agent's name is never written out as text, only the logo.
    expect(screen.queryByText('Claude Code')).toBeNull();
  });

  it('is read-only: no switch, checkbox or button but the one hint sign', async () => {
    apiMock.mockResolvedValue({ agents: ROWS });
    render(<AgentHeldConnectors />);
    await screen.findByText('node_repl');
    expect(screen.queryByRole('checkbox')).toBeNull();
    expect(screen.queryByRole('switch')).toBeNull();
    const buttons = screen.getAllByRole('button');
    expect(buttons).toHaveLength(1);
    expect(buttons[0].getAttribute('aria-label')).toMatch(/own connectors/i);
  });

  it('mutes a connector the agent is not signed in to', async () => {
    apiMock.mockResolvedValue({ agents: ROWS });
    render(<AgentHeldConnectors />);
    const figma = await screen.findByText('figma');
    const playwright = screen.getByText('playwright');
    expect(figma.closest('li')?.getAttribute('data-ready')).toBe('false');
    expect(playwright.closest('li')?.getAttribute('data-ready')).toBe('true');
    expect(screen.getByText('Gmail').closest('li')?.getAttribute('data-ready')).toBe('true');
  });

  it('shows nothing when no agent has been seen', async () => {
    apiMock.mockResolvedValue({ agents: [] });
    const { container } = render(<AgentHeldConnectors />);
    await waitFor(() => expect(apiMock).toHaveBeenCalled());
    expect(container.innerHTML).toBe('');
  });

  it('shows nothing when the load fails or an older daemon answers', async () => {
    apiMock.mockRejectedValue(new Error('404'));
    const { container } = render(<AgentHeldConnectors />);
    await waitFor(() => expect(apiMock).toHaveBeenCalled());
    expect(container.innerHTML).toBe('');
    cleanup();
    apiMock.mockResolvedValue({});
    const second = render(<AgentHeldConnectors />);
    await waitFor(() => expect(apiMock).toHaveBeenCalledTimes(2));
    expect(second.container.innerHTML).toBe('');
  });
});
