// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * useCalendar tests (Task H1). Covers the plan's Vitest list for the hook:
 * fetch params incl. project_id, refresh flow (POST then refetch), and
 * link/unlink payloads. The api client is mocked — no network.
 */

import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../../config', () => ({ api: apiMock }));

import { useCalendar } from './useCalendar';

const AVAIL = { available: true, sources: [] };

type Call = [string, (RequestInit & { method?: string; body?: string }) | undefined];

function setupMock(events: unknown[] = []) {
  apiMock.mockImplementation(async (path: string) => {
    if (path.startsWith('/api/v2/calendar/availability')) return AVAIL;
    if (path.startsWith('/api/v2/calendar/events')) return { events };
    if (path === '/api/v2/calendar/refresh') return { status: 'ok' };
    if (path.includes('/link')) return { status: 'ok' };
    return {};
  });
}

function eventsCalls(): Call[] {
  return apiMock.mock.calls.filter(([p]) => (p as string).startsWith('/api/v2/calendar/events')) as Call[];
}

beforeEach(() => {
  apiMock.mockReset();
});

describe('useCalendar — fetch params', () => {
  it('fetches availability and the range events with no project_id (global)', async () => {
    setupMock();
    renderHook(() => useCalendar({ start: 'S', end: 'E' }));
    await waitFor(() =>
      expect(apiMock).toHaveBeenCalledWith('/api/v2/calendar/availability'),
    );
    await waitFor(() => expect(eventsCalls().length).toBeGreaterThan(0));
    expect(eventsCalls()[0][0]).toBe('/api/v2/calendar/events?start=S&end=E');
  });

  it('appends project_id in lens mode', async () => {
    setupMock();
    renderHook(() => useCalendar({ projectId: 'proj-a', start: 'S', end: 'E' }));
    await waitFor(() => expect(eventsCalls().length).toBeGreaterThan(0));
    expect(eventsCalls()[0][0]).toBe('/api/v2/calendar/events?start=S&end=E&project_id=proj-a');
  });

  it('surfaces an events fetch error', async () => {
    apiMock.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/v2/calendar/availability')) return AVAIL;
      if (path.startsWith('/api/v2/calendar/events')) throw new Error('boom');
      return {};
    });
    const { result } = renderHook(() => useCalendar({ start: 'S', end: 'E' }));
    await waitFor(() => expect(result.current.error).toBe('boom'));
  });
});

describe('useCalendar — refresh flow', () => {
  it('POSTs /refresh then refetches events, in that order', async () => {
    setupMock();
    const { result } = renderHook(() => useCalendar({ start: 'S', end: 'E' }));
    await waitFor(() => expect(result.current.loading).toBe(false));

    apiMock.mockClear();
    setupMock();
    await act(async () => {
      await result.current.refresh();
    });

    const paths = apiMock.mock.calls.map(([p]) => p as string);
    const refreshCall = apiMock.mock.calls.find(([p]) => p === '/api/v2/calendar/refresh') as Call;
    expect(refreshCall).toBeTruthy();
    expect(refreshCall[1]?.method).toBe('POST');

    const refreshIdx = paths.indexOf('/api/v2/calendar/refresh');
    const eventsIdx = paths.findIndex((p) => p.startsWith('/api/v2/calendar/events'));
    expect(refreshIdx).toBeGreaterThanOrEqual(0);
    expect(eventsIdx).toBeGreaterThan(refreshIdx);
  });
});

describe('useCalendar — link/unlink payloads', () => {
  it('links with the composite path (slashes preserved) + project_id, then refetches', async () => {
    setupMock();
    const { result } = renderHook(() => useCalendar({ start: 'S', end: 'E' }));
    await waitFor(() => expect(result.current.loading).toBe(false));

    apiMock.mockClear();
    setupMock();
    await act(async () => {
      await result.current.linkEvent('eventkit', 'abc/123', 'proj-a');
    });

    const linkCall = apiMock.mock.calls.find(([p]) => (p as string).includes('/link')) as Call;
    expect(linkCall[0]).toBe('/api/v2/calendar/events/eventkit/abc/123/link');
    expect(linkCall[1]?.method).toBe('POST');
    expect(JSON.parse(linkCall[1]?.body as string)).toEqual({ project_id: 'proj-a' });
    // Refetched afterwards.
    expect(eventsCalls().length).toBeGreaterThan(0);
  });

  it('unlinks with project_id null', async () => {
    setupMock();
    const { result } = renderHook(() => useCalendar({ start: 'S', end: 'E' }));
    await waitFor(() => expect(result.current.loading).toBe(false));

    apiMock.mockClear();
    setupMock();
    await act(async () => {
      await result.current.linkEvent('eventkit', 'abc', null);
    });

    const linkCall = apiMock.mock.calls.find(([p]) => (p as string).includes('/link')) as Call;
    expect(JSON.parse(linkCall[1]?.body as string)).toEqual({ project_id: null });
  });
});
