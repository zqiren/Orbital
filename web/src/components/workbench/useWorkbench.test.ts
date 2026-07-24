// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * useWorkbench tests (Task 7). Covers the plan's Vitest list at the hook
 * level: exit optimistic removal + revert, and the 409-conflict refetch
 * path. The api client is mocked — no network.
 */

import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { WorkbenchEntry } from './types';

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

import { useWorkbench } from './useWorkbench';

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

beforeEach(() => {
  apiMock.mockReset();
});

describe('useWorkbench — fetch', () => {
  it('fetches the global surface with no project_id', async () => {
    apiMock.mockResolvedValueOnce({ entries: [] });
    renderHook(() => useWorkbench({}));
    await waitFor(() => expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench'));
  });

  it('appends project_id in lens mode', async () => {
    apiMock.mockResolvedValueOnce({ entries: [] });
    renderHook(() => useWorkbench({ projectId: 'proj-a' }));
    await waitFor(() =>
      expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench?project_id=proj-a'),
    );
  });
});

describe('useWorkbench — exitEntry', () => {
  it('optimistically removes the entry, then confirms via the exit POST', async () => {
    apiMock.mockResolvedValueOnce({ entries: [entry()] }); // initial GET
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
    apiMock.mockResolvedValueOnce({ entries: [entry()] });
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
    apiMock.mockResolvedValueOnce({ entries: [entry()] }); // initial GET
    const { result } = renderHook(() => useWorkbench({}));
    await waitFor(() => expect(result.current.entries).toHaveLength(1));

    apiMock.mockRejectedValueOnce(new MockApiError(409, 'conflict'));
    apiMock.mockResolvedValueOnce({ entries: [entry({ text: 'refetched' })] }); // refetch
    await act(async () => {
      await result.current.exitEntry('proj-a', 'e1', 'fulfilled');
    });

    expect(result.current.conflict).toBe(true);
    expect(result.current.entries[0].text).toBe('refetched');
  });
});

describe('useWorkbench — doorway', () => {
  it('openEntry POSTs /open and returns the session id', async () => {
    apiMock.mockResolvedValueOnce({ entries: [] });
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
    apiMock.mockResolvedValueOnce({ entries: [] });
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
