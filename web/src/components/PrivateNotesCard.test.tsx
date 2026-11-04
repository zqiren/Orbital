// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 153 §4.4: the imported project's private-notes card. Shown when the
 * project was imported, has notes and at least two available workers; the
 * sharing sentence only once the fold completed; a click opens the imported
 * index through the chat's path opener.
 */

import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import PrivateNotesCard from './PrivateNotesCard';
import { LocaleProvider } from '../i18n/LocaleContext';

let answer: unknown;
beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockImplementation(async (path: string) => {
    if (path === '/api/v2/projects/p1/imported-notes') {
      if (answer instanceof Error) throw answer;
      return answer;
    }
    return {};
  });
});
afterEach(() => {
  cleanup();
  localStorage.removeItem('orbital.locale');
});

const FOLDED = { agent: 'claude-code', count: 145, folded: true, index_path: 'orbital/imported/claude-code/INDEX.md' };
const TWO = { imported: true, available_workers: ['claude-code', 'codex'] };

function renderCard(project: { imported?: boolean; available_workers?: string[] } = TWO, onOpenPath = vi.fn(), zh = false) {
  if (zh) localStorage.setItem('orbital.locale', 'zh');
  render(
    <LocaleProvider>
      <PrivateNotesCard projectId="p1" project={project} agents={[{ slug: 'claude-code', name: 'Claude Code' }]} onOpenPath={onOpenPath} />
    </LocaleProvider>,
  );
  return onOpenPath;
}

describe('PrivateNotesCard', () => {
  it('states the fact and opens the imported index on click', async () => {
    answer = FOLDED;
    const open = renderCard();
    const card = await screen.findByTestId('private-notes-card');
    expect(card.textContent).toBe('Claude Code has 145 memories of this project. Every agent here can use them now.Open');
    fireEvent.click(card);
    expect(open).toHaveBeenCalledWith('orbital/imported/claude-code/INDEX.md');
  });

  it('reads the zh copy', async () => {
    answer = FOLDED;
    renderCard(TWO, vi.fn(), true);
    const card = await screen.findByTestId('private-notes-card');
    expect(card.textContent).toBe('Claude Code 对这个项目有 145 条记忆，现在这里的每个 Agent 都能用。打开');
  });

  it('before the fold completes, only the count sentence', async () => {
    answer = { ...FOLDED, folded: false };
    renderCard();
    const card = await screen.findByTestId('private-notes-card');
    expect(card.textContent).toBe('Claude Code has 145 memories of this project.Open');
    expect(card.getAttribute('data-folded')).toBe('false');
  });

  it('hides with fewer than two available workers', async () => {
    answer = FOLDED;
    renderCard({ imported: true, available_workers: ['claude-code'] });
    await waitFor(() => expect(apiMock).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.queryByTestId('private-notes-card')).toBeNull();
  });

  it('hides at zero, on an error and for a project that was not imported', async () => {
    answer = { ...FOLDED, count: 0 };
    renderCard();
    await waitFor(() => expect(apiMock).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.queryByTestId('private-notes-card')).toBeNull();
    cleanup();
    answer = new Error('boom');
    renderCard();
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.queryByTestId('private-notes-card')).toBeNull();
    cleanup();
    apiMock.mockClear();
    answer = FOLDED;
    renderCard({ available_workers: ['claude-code', 'codex'] });
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.queryByTestId('private-notes-card')).toBeNull();
    expect(apiMock).not.toHaveBeenCalled();
  });
});
