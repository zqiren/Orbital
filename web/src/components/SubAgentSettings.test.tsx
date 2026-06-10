// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';

// Mock the config module's `api` helper so the component never touches the
// network. The mock returns whatever `mockEntries` is set to for the initial
// GET /api/v2/settings/sub-agents load.
const api = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => api(...args),
}));

import SubAgentSettings from './SubAgentSettings';

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
  param_schema: Record<string, { allowed: string[] | null }>;
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

afterEach(() => {
  cleanup();
  api.mockReset();
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
});
