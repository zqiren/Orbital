// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 153 §4.2: the Agents step's private-notes segment. "N memories only it
 * can use" joins the status line when the agent keeps notes; clicking it
 * expands the per-folder breakdown (folder name, count; "folder not found"
 * with the agent's own dir name when no transcript names the folder). Hidden
 * at zero, and shown with one agent installed too.
 */

import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';

vi.mock('../config', () => ({ api: vi.fn(async () => ({})) }));
vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({ on: vi.fn(), off: vi.fn(), connectionState: 'connected', subscribe: vi.fn() }),
}));

import WizardAgentsStep from './WizardAgentsStep';
import { LocaleProvider } from '../i18n/LocaleContext';
import type { HistoryStatus } from '../utils/onboarding';
import type { SubAgentListEntry } from '../hooks/useSubAgentList';

afterEach(() => {
  cleanup();
  localStorage.removeItem('orbital.locale');
});

const CLAUDE: SubAgentListEntry = { slug: 'claude-code', name: 'Claude Code', installed: true, credentials_configured: true };
const CODEX: SubAgentListEntry = { slug: 'codex', name: 'Codex', installed: true, credentials_configured: true };

const HISTORY: HistoryStatus = {
  status: 'ready',
  sessions_total: 296,
  by_agent: { 'claude-code': 209, codex: 87 },
  notes_by_agent: { 'claude-code': 146, codex: 0 },
  notes_by_folder: [
    { agent: 'claude-code', folder: null, encoded: '-Users-k-old-repo', n: 3 },
    { agent: 'claude-code', folder: '/Users/k/Desktop/orbital-test', encoded: '-Users-k-Desktop-orbital-test', n: 143 },
  ],
};

function renderStep(entries: SubAgentListEntry[], history: HistoryStatus | null, zh = false) {
  if (zh) localStorage.setItem('orbital.locale', 'zh');
  return render(
    <LocaleProvider>
      <WizardAgentsStep entries={entries} history={history} ticks={{}} onToggle={vi.fn()} reload={vi.fn(async () => {})} />
    </LocaleProvider>,
  );
}

describe('WizardAgentsStep — private notes segment', () => {
  it('adds the notes count to the status line, only where there are notes', () => {
    renderStep([CLAUDE, CODEX], HISTORY);
    expect(screen.getByTestId('wizard-agent-status-claude-code').textContent).toBe(
      'Signed in · 209 conversations · 146 memories only it can use',
    );
    expect(screen.getByTestId('wizard-agent-status-codex').textContent).toBe('Signed in · 87 conversations');
    expect(screen.queryByTestId('wizard-agent-notes-codex')).toBeNull();
  });

  it('reads the zh copy', () => {
    renderStep([CLAUDE], HISTORY, true);
    expect(screen.getByTestId('wizard-agent-status-claude-code').textContent).toBe(
      '已登录 · 209 次对话 · 146 条只有它自己能用的记忆',
    );
  });

  it('a click expands the per-folder breakdown, most notes first', () => {
    renderStep([CLAUDE, CODEX], HISTORY);
    const toggle = screen.getByTestId('wizard-agent-notes-claude-code');
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByTestId('wizard-agent-notes-breakdown-claude-code')).toBeNull();
    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    const rows = Array.from(
      screen.getByTestId('wizard-agent-notes-breakdown-claude-code').querySelectorAll('li'),
    ).map((li) => li.textContent);
    expect(rows).toEqual(['orbital-test143', 'folder not found · -Users-k-old-repo3']);
    fireEvent.click(toggle);
    expect(screen.queryByTestId('wizard-agent-notes-breakdown-claude-code')).toBeNull();
  });

  it('shows with one agent installed (the sentence is still true)', () => {
    renderStep([CLAUDE], HISTORY);
    expect(screen.getByTestId('wizard-agent-notes-claude-code').textContent).toBe('146 memories only it can use');
  });

  it('hides at zero and on an older daemon without notes fields', () => {
    renderStep([CLAUDE], { ...HISTORY, notes_by_agent: { 'claude-code': 0 }, notes_by_folder: [] });
    expect(screen.getByTestId('wizard-agent-status-claude-code').textContent).toBe('Signed in · 209 conversations');
    cleanup();
    renderStep([CLAUDE], { status: 'ready', sessions_total: 209, by_agent: { 'claude-code': 209 } });
    expect(screen.queryByTestId('wizard-agent-notes-claude-code')).toBeNull();
  });

  it('uses the singular for one note', () => {
    renderStep([CLAUDE], { ...HISTORY, notes_by_agent: { 'claude-code': 1 } });
    expect(screen.getByTestId('wizard-agent-notes-claude-code').textContent).toBe('1 memory only it can use');
  });
});
