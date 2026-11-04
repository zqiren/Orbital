// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { FileContent, FileHistoryList, HistoryVersion, HistoryView } from '../types';

const calls: Array<{ path: string; method?: string; body?: unknown }> = [];
let statusResponse: unknown = { status: 'watching', has_changes: true };
let listResponse: FileHistoryList | null = null;
let viewResponse: HistoryView | null = null;
let restoreResponse: unknown = { results: [{ path: 'notes.md', status: 'ok', current_sha: 'sR' }] };

vi.mock('../config', () => ({
  api: vi.fn(async (path: string, opts?: { method?: string; body?: string }) => {
    calls.push({ path, method: opts?.method, body: opts?.body ? JSON.parse(opts.body) : undefined });
    if (path.includes('/files/history/status')) return statusResponse;
    if (path.includes('/files/history/view')) return viewResponse;
    if (path.includes('/files/history/restore')) return restoreResponse;
    if (path.includes('/files/history/name')) return { path: 'notes.md', sha: 's2', name: 'x' };
    if (path.includes('/files/history?')) return listResponse;
    if (path.includes('/files/content')) return { path: 'notes.md', content: 'restored', size: 8, truncated: false, type: 'text' };
    return {};
  }),
  apiBytes: vi.fn(),
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
}));

import FilePreview from './FilePreview';
import { __resetFileHistoryStores, requestHistoryFocus } from '../hooks/useFileHistory';

const NOW = new Date();
const today = (h: number, m: number) =>
  new Date(NOW.getFullYear(), NOW.getMonth(), NOW.getDate(), h, m).toISOString();

function ver(v: number, over: Partial<HistoryVersion> = {}): HistoryVersion {
  return {
    v, ts: today(10, v), kind: 'modify', sha: `s${v}`, prev_sha: `s${v - 1}`, size: 100,
    added: 1, removed: 1, lines: 2,
    attrib: { kind: 'tool', sessions: [{ session_id: 'S1', agents: ['codex'], name: 'Release notes' }] },
    ambiguous: false, name: null, turn: { session_id: 'S1', from_ts: 1, to_ts: null }, group: `g${v}`, current: false,
    ...over,
  };
}

const md = (over: Partial<FileContent> = {}): FileContent => ({
  path: 'notes.md', content: '# Notes\n\nnew line\n', size: 18, truncated: false, type: 'text', ...over,
});

function renderPreview(props: Partial<Parameters<typeof FilePreview>[0]> = {}) {
  return render(
    <FilePreview fileContent={md()} loading={false} selectedPath="notes.md" history={{ projectId: 'p1' }} {...props} />,
  );
}

beforeEach(() => {
  __resetFileHistoryStores();
  localStorage.clear();
  calls.length = 0;
  statusResponse = { status: 'watching', has_changes: true };
  listResponse = {
    path: 'notes.md', status: 'watching', whole_file_only: false,
    versions: [
      ver(4, { current: true, group: 'g3' }),
      ver(3, { group: 'g3' }),
      ver(2, { group: 'g2', attrib: { kind: 'tool', sessions: [{ session_id: 'S2', agents: ['claude-code'], name: 'Smoke fixes' }] } }),
      ver(1, { kind: 'baseline', attrib: { kind: 'baseline' }, turn: null, group: 'g1', prev_sha: null }),
    ],
  };
  viewResponse = {
    path: 'notes.md', v: 2, sha: 's2', from_v: 1, from_sha: 's1', whole_file_only: false, size: 10, from_size: 9,
    editors: [{ v: 2, ts: '', kind: 'tool', agents: ['claude-code'], ambiguous: false, session_id: 'S2', session_name: 'Smoke fixes', neutral: false }],
    lines: [
      { text: '# Notes', op: 'same', editor: null },
      { text: '', op: 'same', editor: null },
      { text: 'old line', op: 'del', editor: 0 },
      { text: 'new line', op: 'add', editor: 0 },
    ],
    first_change: 2,
  };
  restoreResponse = { results: [{ path: 'notes.md', status: 'ok', current_sha: 'sR' }] };
});

afterEach(() => {
  vi.useRealTimers();
});

describe('FilePreview history (spec 065 §5.8)', () => {
  it('without the history prop, or before the first recorded change, there is no 历史 glyph (D10)', async () => {
    const { unmount } = render(<FilePreview fileContent={md()} loading={false} selectedPath="notes.md" />);
    expect(screen.queryByTestId('history-open')).toBeNull();
    unmount();
    statusResponse = { status: 'watching', has_changes: false };
    renderPreview();
    await waitFor(() => expect(calls.some((c) => c.path.includes('/status'))).toBe(true));
    await act(async () => {});
    expect(screen.queryByTestId('history-open')).toBeNull();
    expect(screen.queryByTestId('history-layout')).toBeNull();
  });

  it('opens the column: turns folded with a count, 当前, a day label, restore muted on the current version', async () => {
    renderPreview();
    const glyph = await screen.findByTestId('history-open');
    expect(glyph).toHaveTextContent('History');
    fireEvent.click(glyph);
    const column = await screen.findByTestId('history-column');
    await waitFor(() => expect(within(column).getAllByTestId('history-row')).toHaveLength(3));
    const rows = within(column).getAllByTestId('history-row');
    expect(rows[0]).toHaveTextContent('Current');
    expect(rows[0]).toHaveTextContent('2 edits');
    expect(rows[0]).toHaveTextContent('Changed 4 lines');
    expect(rows[0].querySelector('[data-agent="codex"]')?.getAttribute('title')).toBe('Codex');
    expect(rows[0]).toHaveTextContent('Release notes');
    expect(rows[2]).toHaveTextContent('When history began');
    expect(within(column).getByText('Today')).toBeInTheDocument();
    expect(within(column).getByTestId('history-restore')).toBeDisabled();
    expect(glyph).toHaveAttribute('aria-pressed', 'true');
  });

  it('a folded turn expands into one sub-row per write', async () => {
    renderPreview();
    fireEvent.click(await screen.findByTestId('history-open'));
    const column = await screen.findByTestId('history-column');
    await waitFor(() => expect(within(column).getAllByTestId('history-row')).toHaveLength(3));
    fireEvent.click(within(column).getByRole('button', { name: 'Show each edit' }));
    expect(within(column).getAllByTestId('history-subrow').map((r) => r.getAttribute('data-version'))).toEqual(['4', '3']);
  });

  it('selecting a past version paints its changes in the editor colour, then restores it as a new version', async () => {
    renderPreview();
    fireEvent.click(await screen.findByTestId('history-open'));
    const column = await screen.findByTestId('history-column');
    await waitFor(() => expect(within(column).getAllByTestId('history-row')).toHaveLength(3));
    fireEvent.click(within(column).getAllByTestId('history-row')[1]);

    const view = await screen.findByTestId('history-version-view');
    const del = view.querySelector('[data-hist="del"]') as HTMLElement;
    const add = view.querySelector('[data-hist="add"]') as HTMLElement;
    expect(del).toHaveTextContent('old line');
    expect(del.getAttribute('style')).toContain('line-through');
    expect(add).toHaveTextContent('new line');
    expect(add.getAttribute('style')).toContain('--hist-c-claude-code');
    // The legend is the only key to the colours: the agents this view tints.
    expect(within(column).getByTestId('history-legend').querySelector('[data-agent="claude-code"]')).not.toBeNull();
    const viewCall = calls.find((c) => c.path.includes('/view'))!;
    expect(viewCall.path).toContain('version=2');

    const restore = within(column).getByTestId('history-restore');
    expect(restore).toBeEnabled();
    fireEvent.click(restore);
    await waitFor(() => expect(calls.some((c) => c.path.endsWith('/restore'))).toBe(true));
    const body = calls.find((c) => c.path.endsWith('/restore'))!.body as { items: unknown[] };
    // The target, and the hash of what the user saw as current.
    expect(body.items).toEqual([{ path: 'notes.md', sha: 's2', expect_sha: 's4', version: 2 }]);
    const toast = await screen.findByTestId('history-toast');
    expect(toast).toHaveTextContent(/Restored to \d\d:\d\d/);

    // The restore is undoable: put back what it replaced.
    calls.length = 0;
    fireEvent.click(within(toast).getByRole('button', { name: 'Undo' }));
    await waitFor(() => expect(calls.some((c) => c.path.endsWith('/restore'))).toBe(true));
    expect((calls.find((c) => c.path.endsWith('/restore'))!.body as { items: unknown[] }).items).toEqual([
      { path: 'notes.md', sha: 's4', expect_sha: 'sR' },
    ]);
  });

  it('a conflict (someone wrote after the review) restores nothing and says so', async () => {
    restoreResponse = { results: [{ path: 'notes.md', status: 'conflict', current_sha: 'sX' }] };
    renderPreview();
    fireEvent.click(await screen.findByTestId('history-open'));
    const column = await screen.findByTestId('history-column');
    await waitFor(() => expect(within(column).getAllByTestId('history-row')).toHaveLength(3));
    fireEvent.click(within(column).getAllByTestId('history-row')[1]);
    fireEvent.click(within(column).getByTestId('history-restore'));
    expect(await screen.findByTestId('history-toast')).toHaveTextContent('The file changed again');
  });

  it('closing the column returns the file to the current version', async () => {
    renderPreview();
    fireEvent.click(await screen.findByTestId('history-open'));
    const column = await screen.findByTestId('history-column');
    await waitFor(() => expect(within(column).getAllByTestId('history-row')).toHaveLength(3));
    fireEvent.click(within(column).getAllByTestId('history-row')[1]);
    await screen.findByTestId('history-version-view');
    fireEvent.click(within(column).getByTestId('history-close'));
    expect(screen.queryByTestId('history-version-view')).toBeNull();
    expect(screen.queryByTestId('history-column')).toBeNull();
    expect(screen.getByText('new line')).toBeInTheDocument();
  });

  it('names a version inline: the pencil, Enter saves', async () => {
    renderPreview();
    fireEvent.click(await screen.findByTestId('history-open'));
    const column = await screen.findByTestId('history-column');
    await waitFor(() => expect(within(column).getAllByTestId('history-row')).toHaveLength(3));
    fireEvent.click(within(within(column).getAllByTestId('history-row')[1]).getByTestId('history-name'));
    const input = within(column).getByRole('textbox', { name: 'Name this version' });
    fireEvent.change(input, { target: { value: 'Release draft' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(calls.some((c) => c.path.endsWith('/name'))).toBe(true));
    expect(calls.find((c) => c.path.endsWith('/name'))!.body).toEqual({ path: 'notes.md', sha: 's2', name: 'Release draft' });
  });

  it('a project whose watcher is down shows 历史不可用 and never the reason (it can hold a path)', async () => {
    statusResponse = { status: 'unavailable', reason: 'store_unreadable: /Users/x/secret', has_changes: true };
    listResponse = { path: 'notes.md', status: 'unavailable', reason: 'store_unreadable: /Users/x/secret', whole_file_only: false, versions: [] };
    renderPreview();
    fireEvent.click(await screen.findByTestId('history-open'));
    expect(await screen.findByTestId('history-unavailable')).toHaveTextContent('History unavailable');
    expect(document.body.textContent).not.toContain('/Users/x/secret');
  });

  it('a project that had history and went unavailable keeps the glyph and says 历史不可用 (the daemon reports no changes then)', async () => {
    // Seen once with changes on this device...
    const first = renderPreview();
    await screen.findByTestId('history-open');
    first.unmount();
    __resetFileHistoryStores();
    // ...then the store became unreadable: the daemon answers has_changes false.
    statusResponse = { status: 'unavailable', reason: 'store_unreadable: /x', has_changes: false };
    listResponse = { path: 'notes.md', status: 'unavailable', whole_file_only: false, versions: [] };
    renderPreview();
    fireEvent.click(await screen.findByTestId('history-open'));
    expect(await screen.findByTestId('history-unavailable')).toHaveTextContent('History unavailable');
  });

  it('an unavailable project that never had a change shows nothing (D10)', async () => {
    statusResponse = { status: 'unavailable', reason: 'x', has_changes: false };
    renderPreview({ history: { projectId: 'p-never' } });
    await waitFor(() => expect(calls.some((c) => c.path.includes('p-never') && c.path.includes('/status'))).toBe(true));
    await act(async () => {});
    expect(screen.queryByTestId('history-open')).toBeNull();
  });

  it('a skipped file says why in one line', async () => {
    listResponse = { path: 'big.log', status: 'watching', whole_file_only: false, versions: [], skipped_reason: 'size' };
    renderPreview();
    fireEvent.click(await screen.findByTestId('history-open'));
    expect(await screen.findByTestId('history-skipped')).toHaveTextContent('Files over 1 MB have no history');
  });

  it('Office files are restore-only: no tinted view is requested', async () => {
    listResponse = {
      path: 'deck.pptx', status: 'watching', whole_file_only: true,
      versions: [ver(2, { current: true, lines: null, added: null, removed: null, size: 48000 }), ver(1, { size: 4600, lines: null, added: null, removed: null })],
    };
    renderPreview();
    fireEvent.click(await screen.findByTestId('history-open'));
    const column = await screen.findByTestId('history-column');
    expect(await within(column).findByText('This file type can only be restored whole')).toBeInTheDocument();
    await waitFor(() => expect(within(column).getAllByTestId('history-row')).toHaveLength(2));
    fireEvent.click(within(column).getAllByTestId('history-row')[1]);
    expect(within(column).getByTestId('history-restore')).toBeEnabled();
    await act(async () => {});
    expect(calls.some((c) => c.path.includes('/view'))).toBe(false);
  });

  it('a focus request (查看 in the undo sheet) opens the column on that turn', async () => {
    requestHistoryFocus('p1', 'notes.md', 2);
    renderPreview();
    const column = await screen.findByTestId('history-column');
    await waitFor(() =>
      expect(within(column).getAllByTestId('history-row')[1]).toHaveAttribute('aria-pressed', 'true'),
    );
  });
});
