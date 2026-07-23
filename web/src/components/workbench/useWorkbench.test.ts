// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * useWorkbench tests (Task 7). Covers the plan's Vitest list at the hook
 * level: sort order of the merged list, exit optimistic removal + revert,
 * and the 409-conflict refetch path. The api client is mocked — no network.
 */

import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { WorkbenchComputedCard, WorkbenchEntry } from './types';

const { apiMock, MockApiError } = vi.hoisted(() => {
  class MockApiError extends Error {
    status: number;
    detail: string;
    constructor(status: number, detail: string) {
      super(detail);
      this.status = status;
      this.detail = detail;
    }
  }
  return { apiMock: vi.fn(), MockApiError };
});
vi.mock('../../config', () => ({ api: apiMock, ApiError: MockApiError }));

import { useWorkbench, mergeAndSort } from './useWorkbench';

function entry(overrides: Partial<WorkbenchEntry> = {}): WorkbenchEntry {
  return {
    project_id: 'proj-a',
    id: 'e1',
    text: 'Ping Simon about the invoice',
    due: null,
    evidence: 'quote',
    from_session: 'sess-1',
    confidence: 'stated',
    created: '2026-07-01',
    touched: null,
    age_days: 5,
    overdue: false,
    days_late: null,
    ...overrides,
  };
}

function computedCard(overrides: Partial<WorkbenchComputedCard> = {}): WorkbenchComputedCard {
  return {
    type: 'broken_automation',
    project_id: 'proj-a',
    key: 'trigger-1',
    text: 'Automation "nightly sync" has not run on schedule.',
    since: '2026-07-10',
    ...overrides,
  };
}

beforeEach(() => {
  apiMock.mockReset();
});

describe('mergeAndSort', () => {
  it('sorts overdue first, then oldest-created/since first (the forgotten float up)', () => {
    const items = mergeAndSort(
      [
        entry({ id: 'newer', created: '2026-07-20', overdue: false }),
        entry({ id: 'older', created: '2026-06-01', overdue: false }),
        entry({ id: 'overdue-entry', created: '2026-07-15', overdue: true }),
      ],
      [
        computedCard({ type: 'overdue', key: 'overdue-computed', since: '2026-05-01' }),
        computedCard({ type: 'broken_automation', key: 'automation-1', since: '2026-07-05' }),
      ],
    );
    const ids = items.map((it) => (it.kind === 'entry' ? it.data.id : it.data.key));
    // Overdue bucket (entries + overdue-computed) sorted oldest-first, then
    // the not-yet-overdue bucket sorted oldest-first.
    expect(ids).toEqual([
      'overdue-computed', // overdue bucket: since 2026-05-01
      'overdue-entry', // overdue bucket: created 2026-07-15
      'older', // not-overdue bucket: created 2026-06-01
      'automation-1', // not-overdue bucket: since 2026-07-05
      'newer', // not-overdue bucket: created 2026-07-20
    ]);
  });
});

describe('useWorkbench — fetch', () => {
  it('fetches the global surface with no project_id', async () => {
    apiMock.mockResolvedValueOnce({ entries: [], computed: [] });
    renderHook(() => useWorkbench({}));
    await waitFor(() => expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench'));
  });

  it('appends project_id in lens mode', async () => {
    apiMock.mockResolvedValueOnce({ entries: [], computed: [] });
    renderHook(() => useWorkbench({ projectId: 'proj-a' }));
    await waitFor(() =>
      expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench?project_id=proj-a'),
    );
  });
});

describe('useWorkbench — exitEntry', () => {
  it('optimistically removes the entry, then confirms via the exit POST', async () => {
    apiMock.mockResolvedValueOnce({ entries: [entry()], computed: [] }); // initial GET
    const { result } = renderHook(() => useWorkbench({}));
    await waitFor(() => expect(result.current.entries).toHaveLength(1));

    apiMock.mockResolvedValueOnce({ status: 'ok' }); // exit POST
    await act(async () => {
      await result.current.exitEntry('proj-a', 'e1', 'fulfilled');
    });

    expect(result.current.entries).toHaveLength(0);
    const exitCall = apiMock.mock.calls.find(([p]) => (p as string).includes('/exit'));
    expect(exitCall?.[0]).toBe('/api/v2/workbench/proj-a/entries/e1/exit');
    expect(JSON.parse((exitCall?.[1] as { body: string }).body)).toEqual({
      kind: 'fulfilled',
      reason: '',
    });
  });

  it('reverts the optimistic removal when the exit POST fails (non-409)', async () => {
    apiMock.mockResolvedValueOnce({ entries: [entry()], computed: [] });
    const { result } = renderHook(() => useWorkbench({}));
    await waitFor(() => expect(result.current.entries).toHaveLength(1));

    apiMock.mockRejectedValueOnce(new MockApiError(500, 'boom'));
    await act(async () => {
      await result.current.exitEntry('proj-a', 'e1', 'irrelevant');
    });

    expect(result.current.entries).toHaveLength(1);
    expect(result.current.conflict).toBe(false);
  });

  it('on 409 conflict, reverts, refetches, and flags conflict', async () => {
    apiMock.mockResolvedValueOnce({ entries: [entry()], computed: [] }); // initial GET
    const { result } = renderHook(() => useWorkbench({}));
    await waitFor(() => expect(result.current.entries).toHaveLength(1));

    apiMock.mockRejectedValueOnce(new MockApiError(409, 'conflict'));
    apiMock.mockResolvedValueOnce({ entries: [entry({ text: 'refetched' })], computed: [] }); // refetch
    await act(async () => {
      await result.current.exitEntry('proj-a', 'e1', 'fulfilled');
    });

    expect(result.current.conflict).toBe(true);
    expect(result.current.entries[0].text).toBe('refetched');
  });
});

describe('useWorkbench — dismissComputed', () => {
  it('optimistically removes the computed card, POSTs dismiss', async () => {
    apiMock.mockResolvedValueOnce({ entries: [], computed: [computedCard()] });
    const { result } = renderHook(() => useWorkbench({}));
    await waitFor(() => expect(result.current.computed).toHaveLength(1));

    apiMock.mockResolvedValueOnce({ status: 'ok' });
    await act(async () => {
      await result.current.dismissComputed('proj-a', 'broken_automation', 'trigger-1');
    });

    expect(result.current.computed).toHaveLength(0);
    const dismissCall = apiMock.mock.calls.find(([p]) => (p as string).includes('/dismiss'));
    expect(dismissCall?.[0]).toBe(
      '/api/v2/workbench/proj-a/computed/broken_automation/trigger-1/dismiss',
    );
  });
});

describe('useWorkbench — doorway', () => {
  it('openEntry POSTs /open and returns the session id', async () => {
    apiMock.mockResolvedValueOnce({ entries: [], computed: [] });
    const { result } = renderHook(() => useWorkbench({}));
    await waitFor(() => expect(result.current.loading).toBe(false));

    apiMock.mockResolvedValueOnce({ session_id: 'sess-new' });
    let sessionId = '';
    await act(async () => {
      sessionId = await result.current.openEntry('proj-a', 'e1');
    });
    expect(sessionId).toBe('sess-new');
    expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench/proj-a/entries/e1/open', {
      method: 'POST',
    });
  });

  it('migrate POSTs /migrate and returns the session id', async () => {
    apiMock.mockResolvedValueOnce({ entries: [], computed: [] });
    const { result } = renderHook(() => useWorkbench({}));
    await waitFor(() => expect(result.current.loading).toBe(false));

    apiMock.mockResolvedValueOnce({ session_id: 'sess-migrate' });
    let sessionId = '';
    await act(async () => {
      sessionId = await result.current.migrate('proj-a');
    });
    expect(sessionId).toBe('sess-migrate');
    expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench/proj-a/migrate', { method: 'POST' });
  });
});
