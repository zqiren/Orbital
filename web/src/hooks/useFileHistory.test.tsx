// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { act, render, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const turnCalls: string[] = [];
let release: (() => void) | null = null;
let filesNow: string[] = [];
let hasChanges = true;

vi.mock('../config', () => ({
  api: vi.fn(async (path: string) => {
    if (path.includes('/status')) return { status: 'watching', has_changes: hasChanges };
    if (path.includes('/turn')) {
      turnCalls.push(path);
      const snapshot = [...filesNow];
      if (turnCalls.length === 1) await new Promise<void>((r) => (release = r));
      return { status: 'watching', session_id: 'S', files: snapshot.map((p) => ({ path: p })), no_history: 0 };
    }
    return {};
  }),
  isRelayMode: false,
  BASE_URL: '',
}));

import {
  __resetFileHistoryStores,
  bumpFileHistory,
  noteAgentStatus,
  useHistoryStatus,
  useTurnFiles,
} from './useFileHistory';

function Probe() {
  const data = useTurnFiles('p1', 'S', '2026-11-04T10:00:00Z', null, true);
  return <span data-testid="n">{data ? data.files.length : -1}</span>;
}

function StatusProbe({ pid }: { pid: string }) {
  const s = useHistoryStatus(pid);
  return <span data-testid={`s-${pid}`}>{s ? String(s.has_changes) : 'null'}</span>;
}

beforeEach(() => {
  __resetFileHistoryStores();
  turnCalls.length = 0;
  release = null;
  filesNow = [];
  hasChanges = true;
});

describe('useFileHistory stores', () => {
  it('a bump that lands while the turn is being read triggers one more read (no lost refresh)', async () => {
    filesNow = ['a.md'];
    const { getByTestId } = render(<Probe />);
    await waitFor(() => expect(turnCalls).toHaveLength(1));
    // The watcher records another file and the run ends while the first read is in flight.
    filesNow = ['a.md', 'b.md'];
    act(() => bumpFileHistory('p1'));
    await act(async () => release?.());
    await waitFor(() => expect(turnCalls).toHaveLength(2));
    await waitFor(() => expect(getByTestId('n').textContent).toBe('2'));
  });

  it('one status request serves every surface of a project', async () => {
    const { api } = await import('../config');
    (api as ReturnType<typeof vi.fn>).mockClear();
    const { getAllByTestId } = render(
      <>
        <StatusProbe pid="p9" />
        <StatusProbe pid="p9" />
        <StatusProbe pid="p9" />
      </>,
    );
    await waitFor(() => expect(getAllByTestId('s-p9')[0].textContent).toBe('true'));
    const statusCalls = (api as ReturnType<typeof vi.fn>).mock.calls.filter((c) => String(c[0]).includes('/status'));
    expect(statusCalls).toHaveLength(1);
  });

  it('a run that ends anywhere re-reads a status this device already holds', async () => {
    // The project's first change lands in a run started from the Workbench:
    // no ChatTab is mounted to see it end, yet the cached "no changes" must
    // not outlive the run (the changed-files line would never show).
    hasChanges = false;
    const { getByTestId } = render(<StatusProbe pid="p3" />);
    await waitFor(() => expect(getByTestId('s-p3').textContent).toBe('false'));
    hasChanges = true;
    act(() => noteAgentStatus('p3', 'running'));
    act(() => noteAgentStatus('p3', 'idle'));
    await waitFor(() => expect(getByTestId('s-p3').textContent).toBe('true'));
  });

  it('a run end does not fetch the status of a project nothing has asked about', async () => {
    const { api } = await import('../config');
    (api as ReturnType<typeof vi.fn>).mockClear();
    noteAgentStatus('p4', 'running');
    noteAgentStatus('p4', 'idle');
    await Promise.resolve();
    expect((api as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);
  });
});
