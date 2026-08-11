// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

vi.mock('./useWebSocket', () => ({
  useWebSocket: () => ({
    on: vi.fn(),
    off: vi.fn(),
    connectionState: 'connected',
    subscribe: vi.fn(),
  }),
}));

let apiFn = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiFn(...args),
  ApiError: class ApiError extends Error {},
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000/ws',
}));

import { useQueue } from './useQueue';

/** The POST body of the last /queue/items call, parsed. */
function lastAddItemBody(): Record<string, unknown> {
  const call = apiFn.mock.calls.find(
    ([path, init]) =>
      typeof path === 'string' &&
      path.endsWith('/queue/items') &&
      (init as RequestInit | undefined)?.method === 'POST',
  );
  if (!call) throw new Error('no POST /queue/items call recorded');
  return JSON.parse((call[1] as RequestInit).body as string);
}

describe('useQueue.addItem', () => {
  beforeEach(() => {
    apiFn = vi.fn().mockResolvedValue({
      version: 1,
      state: 'idle',
      items: [],
      chat_session_id: null,
    });
  });

  it('sends file_refs from the composer through to the POST body', async () => {
    const { result } = renderHook(() => useQueue('p1'));
    await waitFor(() => expect(result.current.snapshot).not.toBeNull());

    await act(async () => {
      await result.current.addItem('crop this', {
        priority: 1,
        review: true,
        fileRefs: ['uploads/2026-08-11T101010-shot.png'],
      });
    });

    expect(lastAddItemBody()).toEqual({
      content: 'crop this',
      file_refs: ['uploads/2026-08-11T101010-shot.png'],
      priority: 1,
      review_before_advance: true,
    });
  });

  it('defaults file_refs to an empty list for a text-only item', async () => {
    const { result } = renderHook(() => useQueue('p1'));
    await waitFor(() => expect(result.current.snapshot).not.toBeNull());

    await act(async () => {
      await result.current.addItem('text only');
    });

    expect(lastAddItemBody()).toEqual({
      content: 'text only',
      file_refs: [],
      priority: 0,
      review_before_advance: false,
    });
  });
});
