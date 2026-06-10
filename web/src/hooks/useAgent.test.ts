// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';

// ---------------------------------------------------------------------------
// Mocks — must be declared before the module under test is imported
// ---------------------------------------------------------------------------

// Capture every api() call so we can assert the request body shape. The hook
// builds the URL + RequestInit; we proxy through and record both.
const apiMock = vi.fn(async (..._args: unknown[]) => ({ status: 'ok' }));
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiMock(...args),
  ApiError: class ApiError extends Error {},
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000/ws',
}));

import { useAgent } from './useAgent';

beforeEach(() => {
  apiMock.mockClear();
});

/** Decode the JSON body of the most recent api() call. */
function lastBody(): Record<string, unknown> {
  expect(apiMock).toHaveBeenCalled();
  const call = apiMock.mock.calls[apiMock.mock.calls.length - 1] as unknown[];
  const opts = call[1] as RequestInit;
  return JSON.parse(opts.body as string);
}

function lastUrl(): string {
  const call = apiMock.mock.calls[apiMock.mock.calls.length - 1] as unknown[];
  return call[0] as string;
}

describe('useAgent threads session_id into lifecycle calls', () => {
  it('new-session sends session_id in the body', async () => {
    const { result } = renderHook(() => useAgent());
    await result.current.newSession('p1', 'sess-9');
    expect(lastUrl()).toContain('/api/v2/agents/p1/new-session');
    expect(lastBody()).toEqual({ session_id: 'sess-9' });
  });

  it('cancel sends session_id in the body', async () => {
    const { result } = renderHook(() => useAgent());
    await result.current.cancelMessage('p1', 'sess-9');
    expect(lastUrl()).toContain('/api/v2/agents/p1/cancel');
    expect(lastBody()).toEqual({ session_id: 'sess-9' });
  });

  it('no longer exposes a stopAgent action (the /stop route was removed)', () => {
    const { result } = renderHook(() => useAgent());
    expect((result.current as Record<string, unknown>).stopAgent).toBeUndefined();
  });

  it('approve sends session_id alongside the existing body fields', async () => {
    const { result } = renderHook(() => useAgent());
    await result.current.approveToolCall('p1', 'tc-1', 'go ahead', true, 'sess-9');
    expect(lastUrl()).toContain('/api/v2/agents/p1/approve');
    expect(lastBody()).toEqual({
      tool_call_id: 'tc-1',
      reply_text: 'go ahead',
      approve_all: true,
      session_id: 'sess-9',
    });
  });

  it('deny sends session_id alongside the existing body fields', async () => {
    const { result } = renderHook(() => useAgent());
    await result.current.denyToolCall('p1', 'tc-1', 'nope', 'sess-9');
    expect(lastUrl()).toContain('/api/v2/agents/p1/deny');
    expect(lastBody()).toEqual({
      tool_call_id: 'tc-1',
      reason: 'nope',
      session_id: 'sess-9',
    });
  });

  it('omits session_id when none is provided (backend treats it as optional)', async () => {
    const { result } = renderHook(() => useAgent());
    await result.current.cancelMessage('p1');
    expect(lastBody()).toEqual({});
  });
});
