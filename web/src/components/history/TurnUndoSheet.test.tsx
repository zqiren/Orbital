// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { TurnFile } from '../../types';

const calls: Array<{ path: string; method?: string; body?: unknown }> = [];
let turnResponse: unknown = null;
let restoreResponse: unknown = null;

vi.mock('../../config', () => ({
  api: vi.fn(async (path: string, opts?: { method?: string; body?: string }) => {
    calls.push({ path, method: opts?.method, body: opts?.body ? JSON.parse(opts.body) : undefined });
    if (path.includes('/files/history/turn')) return turnResponse;
    if (path.includes('/files/history/restore')) return restoreResponse;
    if (path.includes('/files/history/status')) return { status: 'watching', has_changes: true };
    return {};
  }),
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
}));

import TurnUndoSheet from './TurnUndoSheet';
import { __resetFileHistoryStores } from '../../hooks/useFileHistory';

const editor = (agents: string[], kind = 'tool') => ({
  v: 9, ts: '', kind: kind as 'tool', agents, ambiguous: false, session_id: 'S', session_name: 'Release', neutral: agents.length === 0,
});

function tf(path: string, over: Partial<TurnFile> = {}): TurnFile {
  return {
    path, kind: 'modify', added: 3, removed: 3, lines: 6,
    before_sha: `b-${path}`, turn_sha: `t-${path}`, current_sha: `t-${path}`, before_size: 1, turn_size: 1,
    clean: true, later_editor: null, ambiguous: false, whole_file_only: false, first_v: 2, last_v: 3,
    editor: editor(['codex']),
    ...over,
  };
}

const TURN = { sessionId: 'S', fromTs: '2026-11-04T14:20:00', toTs: '2026-11-04T14:40:00' };

beforeEach(() => {
  __resetFileHistoryStores();
  calls.length = 0;
  turnResponse = {
    status: 'watching', session_id: 'S', session_name: 'Release v0.20.0',
    files: [
      tf('site/download.html'),
      tf('README.md', { lines: 4 }),
      tf('docs/notes.md', { clean: false, current_sha: 'later', later_editor: editor(['orbital']) }),
      tf('docs/new.md', { kind: 'create', before_sha: null }),
    ],
    no_history: 1,
  };
  restoreResponse = {
    results: [
      { path: 'site/download.html', status: 'ok', current_sha: 'b-site/download.html' },
      { path: 'README.md', status: 'ok', current_sha: 'b-README.md' },
    ],
  };
});

function rows() {
  return screen.getAllByTestId('turn-undo-row');
}
const box = (row: HTMLElement) => within(row).getByRole('checkbox');

describe('TurnUndoSheet (spec 065 §5.9, §7)', () => {
  it('checks untouched files; a file changed later and a created file start unchecked', async () => {
    render(<TurnUndoSheet projectId="p1" turn={TURN} onView={vi.fn()} onRestored={vi.fn()} />);
    await waitFor(() => expect(rows()).toHaveLength(4));
    expect(rows().map((r) => box(r).getAttribute('aria-checked'))).toEqual(['true', 'true', 'false', 'false']);
    expect(within(rows()[2]).getByTestId('turn-later')).toHaveTextContent('Changed again later by');
    expect(within(rows()[2]).getByTestId('turn-later').querySelector('[data-agent="orbital"]')).not.toBeNull();
    expect(rows()[3]).toHaveTextContent('Created this file');
    expect(screen.getByTestId('turn-undo-skipped')).toHaveTextContent('1 file has no history: binary or too large');
    expect(screen.getByTestId('turn-undo-restore')).toHaveTextContent('Restore 2 selected files');
  });

  it('memory files come last, unchecked, under a 记忆 label', async () => {
    turnResponse = {
      status: 'watching', session_id: 'S', session_name: 'x',
      files: [tf('orbital/PROJECT_STATE.md'), tf('README.md'), tf('orbital/INDEX.md', { kind: 'create', before_sha: null })],
      no_history: 0,
    };
    render(<TurnUndoSheet projectId="p1" turn={TURN} onView={vi.fn()} onRestored={vi.fn()} />);
    await waitFor(() => expect(rows()).toHaveLength(3));
    expect(rows().map((r) => r.getAttribute('data-path'))).toEqual(['README.md', 'orbital/PROJECT_STATE.md', 'orbital/INDEX.md']);
    expect(rows().map((r) => box(r).getAttribute('aria-checked'))).toEqual(['true', 'false', 'false']);
    const label = screen.getByTestId('turn-undo-memory');
    expect(label).toHaveTextContent('Memory');
    // The label sits between the work files and the memory files.
    expect(rows()[0].compareDocumentPosition(label) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(label.compareDocumentPosition(rows()[1]) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByTestId('turn-undo-restore')).toHaveTextContent('Restore 1 selected file');
  });

  it('the button counts as boxes change', async () => {
    render(<TurnUndoSheet projectId="p1" turn={TURN} onView={vi.fn()} onRestored={vi.fn()} />);
    await waitFor(() => expect(rows()).toHaveLength(4));
    fireEvent.click(box(rows()[0]));
    expect(screen.getByTestId('turn-undo-restore')).toHaveTextContent('Restore 1 selected file');
    fireEvent.click(box(rows()[1]));
    expect(screen.getByTestId('turn-undo-restore')).toBeDisabled();
  });

  it('restores each checked file to what the turn started from, against the hash the user reviewed', async () => {
    const onRestored = vi.fn();
    render(<TurnUndoSheet projectId="p1" turn={TURN} onView={vi.fn()} onRestored={onRestored} />);
    await waitFor(() => expect(rows()).toHaveLength(4));
    fireEvent.click(screen.getByTestId('turn-undo-restore'));
    await waitFor(() => expect(onRestored).toHaveBeenCalled());
    const body = calls.find((c) => c.path.endsWith('/restore'))!.body as { items: unknown[] };
    expect(body.items).toEqual([
      { path: 'site/download.html', sha: 'b-site/download.html', expect_sha: 't-site/download.html' },
      { path: 'README.md', sha: 'b-README.md', expect_sha: 't-README.md' },
    ]);
    expect(onRestored.mock.calls[0][0]).toBe(2);
    expect(onRestored.mock.calls[0][2]).toBe(true);

    // The toast's undo puts the turn's content back.
    calls.length = 0;
    await onRestored.mock.calls[0][1]();
    expect((calls.find((c) => c.path.endsWith('/restore'))!.body as { items: unknown[] }).items).toEqual([
      { path: 'site/download.html', sha: 't-site/download.html', expect_sha: 'b-site/download.html' },
      { path: 'README.md', sha: 't-README.md', expect_sha: 'b-README.md' },
    ]);
  });

  it('a checked created file restores to nothing (delete), only because the user checked it', async () => {
    render(<TurnUndoSheet projectId="p1" turn={TURN} onView={vi.fn()} onRestored={vi.fn()} />);
    await waitFor(() => expect(rows()).toHaveLength(4));
    fireEvent.click(box(rows()[0]));
    fireEvent.click(box(rows()[1]));
    fireEvent.click(box(rows()[3]));
    fireEvent.click(screen.getByTestId('turn-undo-restore'));
    await waitFor(() => expect(calls.some((c) => c.path.endsWith('/restore'))).toBe(true));
    expect((calls.find((c) => c.path.endsWith('/restore'))!.body as { items: unknown[] }).items).toEqual([
      { path: 'docs/new.md', sha: null, expect_sha: 't-docs/new.md' },
    ]);
  });

  it('a conflict keeps the sheet open, unchecks the row and re-reads the turn', async () => {
    restoreResponse = {
      results: [
        { path: 'site/download.html', status: 'ok', current_sha: 'b-site/download.html' },
        { path: 'README.md', status: 'conflict', current_sha: 'someone-else' },
      ],
    };
    const onRestored = vi.fn();
    render(<TurnUndoSheet projectId="p1" turn={TURN} onView={vi.fn()} onRestored={onRestored} />);
    await waitFor(() => expect(rows()).toHaveLength(4));
    const turnReads = calls.filter((c) => c.path.includes('/turn')).length;
    turnResponse = {
      ...(turnResponse as object),
      files: [
        tf('site/download.html', { clean: false, current_sha: 'b-site/download.html', later_editor: editor([], 'restore') }),
        tf('README.md', { clean: false, current_sha: 'someone-else', later_editor: editor(['claude-code']) }),
      ],
    };
    fireEvent.click(screen.getByTestId('turn-undo-restore'));
    await waitFor(() => expect(onRestored).toHaveBeenCalled());
    expect(onRestored.mock.calls[0][2]).toBe(false);
    await waitFor(() => expect(calls.filter((c) => c.path.includes('/turn')).length).toBeGreaterThan(turnReads));
    await waitFor(() => expect(rows()).toHaveLength(2));
    expect(box(rows()[1])).toHaveAttribute('aria-checked', 'false');
    expect(within(rows()[1]).getByTestId('turn-later').querySelector('[data-agent="claude-code"]')).not.toBeNull();
  });

  it('查看 opens the file history on the turn version', async () => {
    const onView = vi.fn();
    render(<TurnUndoSheet projectId="p1" turn={TURN} onView={onView} onRestored={vi.fn()} />);
    await waitFor(() => expect(rows()).toHaveLength(4));
    fireEvent.click(within(rows()[1]).getByRole('button', { name: 'View' }));
    expect(onView).toHaveBeenCalledWith('README.md', 3);
  });
});
