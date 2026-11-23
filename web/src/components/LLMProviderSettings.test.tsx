// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * The create-project heads-up (backlog #25): nothing while loading, nothing
 * once a key exists, the warning otherwise, and the keyless wording after the
 * wizard's Skip.
 *
 * Batch 1123 moved the provider form's rules to the connect panel; their
 * tests live in connect/ConnectPanel.test.tsx (when the key is tested, Next
 * tests first and saves only on a pass, region, model row, one-click
 * sign-ins, Custom, editing) and connect/providerMeta.test.ts (tile order,
 * region defaults by language, consoles, search aliases).
 *
 * The api client (web/src/config.ts `api`) is mocked — no network.
 */

import { render, screen, waitFor, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import LLMProviderSettings from './LLMProviderSettings';

afterEach(() => cleanup());
beforeEach(() => { apiMock.mockReset(); });

function mockSettings(llm: { api_key_set: boolean }, setup?: { api_key_skipped?: boolean }) {
  apiMock.mockImplementation(async (path: string) =>
    path === '/api/v2/settings' ? { llm, ...(setup ? { setup } : {}) } : {});
}

describe('LLMProviderSettings — the create-project heads-up (backlog #25)', () => {
  it('renders nothing while global settings are still loading', () => {
    apiMock.mockImplementation(() => new Promise(() => {}));
    const { container } = render(<LLMProviderSettings />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing (no happy-path card) once a global API key is configured', async () => {
    mockSettings({ api_key_set: true });
    const { container } = render(<LLMProviderSettings />);
    await waitFor(() => expect(apiMock).toHaveBeenCalledWith('/api/v2/settings'));
    await new Promise((r) => setTimeout(r, 0));
    expect(container).toBeEmptyDOMElement();
  });

  it('renders only the not-configured warning when no global API key is set', async () => {
    mockSettings({ api_key_set: false });
    render(<LLMProviderSettings />);
    await waitFor(() => expect(screen.getByText(/No LLM provider configured yet/)).toBeTruthy());
  });

  it('keyless mode says projects still work through their agent', async () => {
    mockSettings({ api_key_set: false }, { api_key_skipped: true });
    render(<LLMProviderSettings />);
    await waitFor(() => expect(screen.getByText(/talks to your local agent directly/)).toBeTruthy());
    expect(screen.queryByText(/No LLM provider configured yet/)).toBeNull();
  });
});
