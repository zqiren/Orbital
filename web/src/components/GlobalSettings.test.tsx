// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Global settings saves as you go, like project settings. It used to end in a
 * single Save button below thirteen sections — every one of which, except the
 * five fields this page owns, had been saving itself all along.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';

const api = vi.hoisted(() => vi.fn());
vi.mock('../config', async (orig) => ({ ...(await orig<typeof import('../config')>()), api }));
vi.mock('../hooks/useCredentialCards', () => ({
  useCredentialCards: () => ({ cards: [], defaultCardId: null }),
}));
// The self-saving sections are not under test; stub them out.
vi.mock('./CredentialCards', () => ({ default: () => null }));
vi.mock('./FallbackModelsEditor', () => ({ default: () => null }));
vi.mock('./CredentialStore', () => ({ default: () => null }));
vi.mock('./BrowserSignInCard', () => ({ default: () => null }));
vi.mock('./PairPhone', () => ({ default: () => null }));
vi.mock('./SubAgentSettings', () => ({ default: () => null }));
vi.mock('./ConnectorSettings', () => ({ default: () => null }));
vi.mock('./TelemetrySettings', () => ({ default: () => null }));
vi.mock('./AboutSection', () => ({ default: () => null }));
vi.mock('./SettingsRail', () => ({ default: () => null }));

import GlobalSettings from './GlobalSettings';

const SETTINGS = {
  user_preferences_content: 'I like short answers',
  user_memory_content: '',
  user_memory_enabled: true,
  scratch_workspace: '/tmp/scratch',
  llm: { fallback_models: [] },
};

async function renderLoaded() {
  render(<GlobalSettings onBack={vi.fn()} />);
  await act(async () => { await vi.advanceTimersByTimeAsync(0); });
}

function puts() {
  return api.mock.calls
    .filter(([, opts]) => opts?.method === 'PUT')
    .map(([path, opts]) => [path, JSON.parse(opts.body)]);
}

beforeEach(() => {
  vi.useFakeTimers();
  api.mockReset();
  api.mockResolvedValue({});
  vi.stubGlobal('fetch', vi.fn(async (url: string) => ({
    json: async () => (String(url).endsWith('/settings') ? SETTINGS : {}),
  })));
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('GlobalSettings — autosave', () => {
  it('has no Save button', async () => {
    await renderLoaded();
    expect(screen.queryByTestId('global-settings-save')).not.toBeInTheDocument();
    expect(screen.getByTestId('global-settings-autosave-hint')).toBeInTheDocument();
  });

  it('saves a typed field after a pause, as a patch of that field only', async () => {
    await renderLoaded();
    const aboutYou = screen.getByDisplayValue('I like short answers');
    fireEvent.change(aboutYou, { target: { value: 'I like long answers' } });
    expect(puts()).toEqual([]);
    await act(async () => { await vi.advanceTimersByTimeAsync(900); });
    expect(puts()).toEqual([
      ['/api/v2/settings', { user_preferences_content: 'I like long answers' }],
    ]);
  });

  it('saves on leaving the field without waiting out the pause', async () => {
    await renderLoaded();
    const aboutYou = screen.getByDisplayValue('I like short answers');
    fireEvent.change(aboutYou, { target: { value: 'x' } });
    fireEvent.blur(aboutYou);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(puts()).toEqual([['/api/v2/settings', { user_preferences_content: 'x' }]]);
  });

  it('saves the user-memory switch at once', async () => {
    await renderLoaded();
    fireEvent.click(screen.getByTestId('user-memory-toggle'));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(puts()).toEqual([['/api/v2/settings', { user_memory_enabled: false }]]);
  });

  it('never sends an emptied Quick Tasks workspace', async () => {
    await renderLoaded();
    fireEvent.change(screen.getByDisplayValue('/tmp/scratch'), { target: { value: '' } });
    await act(async () => { await vi.advanceTimersByTimeAsync(900); });
    expect(puts()).toEqual([]);
  });
});
