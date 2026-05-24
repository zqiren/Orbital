// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

let apiFn = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiFn(...args),
  ApiError: class ApiError extends Error {},
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000/ws',
}));

import { useChatHistory } from './useChatHistory';
import type { ChatMessage } from '../types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMsg(role: ChatMessage['role'] = 'assistant'): ChatMessage {
  return {
    role,
    content: 'hello',
    source: 'main',
    timestamp: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useChatHistory — sessionId param', () => {
  beforeEach(() => {
    apiFn = vi.fn();
  });

  it('fetches /chat without session_id param when sessionId is not provided', async () => {
    apiFn.mockResolvedValue([makeMsg()]);

    const { result } = renderHook(() => useChatHistory());

    await act(async () => {
      await result.current.loadHistory('proj-1');
    });

    expect(apiFn).toHaveBeenCalledWith('/api/v2/agents/proj-1/chat');
  });

  it('fetches /chat without session_id param when sessionId is null', async () => {
    apiFn.mockResolvedValue([makeMsg()]);

    const { result } = renderHook(() => useChatHistory({ sessionId: null }));

    await act(async () => {
      await result.current.loadHistory('proj-null');
    });

    expect(apiFn).toHaveBeenCalledWith('/api/v2/agents/proj-null/chat');
  });

  it('appends ?session_id=<id> when sessionId is provided', async () => {
    apiFn.mockResolvedValue([makeMsg()]);

    const { result } = renderHook(() => useChatHistory({ sessionId: 'sess-xyz' }));

    await act(async () => {
      await result.current.loadHistory('proj-2');
    });

    expect(apiFn).toHaveBeenCalledWith(
      '/api/v2/agents/proj-2/chat?session_id=sess-xyz',
    );
  });

  it('URL-encodes the session_id in the query param', async () => {
    apiFn.mockResolvedValue([]);

    const { result } = renderHook(() => useChatHistory({ sessionId: 'sess/with spaces' }));

    await act(async () => {
      await result.current.loadHistory('proj-3');
    });

    expect(apiFn).toHaveBeenCalledWith(
      '/api/v2/agents/proj-3/chat?session_id=sess%2Fwith%20spaces',
    );
  });

  it('returns the messages from the API and sets loading correctly', async () => {
    const msgs = [makeMsg('user'), makeMsg('assistant')];
    apiFn.mockResolvedValue(msgs);

    const { result } = renderHook(() => useChatHistory({ sessionId: 'sess-1' }));

    let returned: ChatMessage[] = [];
    await act(async () => {
      returned = await result.current.loadHistory('proj-4');
    });

    expect(returned).toEqual(msgs);
    expect(result.current.messages).toEqual(msgs);
    expect(result.current.loading).toBe(false);
  });

  it('returns [] and clears messages on API failure', async () => {
    apiFn.mockRejectedValue(new Error('500'));

    const { result } = renderHook(() => useChatHistory({ sessionId: 'sess-bad' }));

    let returned: ChatMessage[] = [{ role: 'user', content: 'x', source: 'u', timestamp: '' }];
    await act(async () => {
      returned = await result.current.loadHistory('proj-err');
    });

    expect(returned).toEqual([]);
    expect(result.current.messages).toEqual([]);
    expect(result.current.loading).toBe(false);
  });

  it('existing behaviour (no sessionId) is unchanged — mergeRealtimeEvent still works', async () => {
    apiFn.mockResolvedValue([]);

    const { result } = renderHook(() => useChatHistory());

    await act(async () => {
      await result.current.loadHistory('proj-5');
    });

    // Verify mergeRealtimeEvent is still exported and callable
    act(() => {
      result.current.mergeRealtimeEvent({
        type: 'chat.stream_delta',
        project_id: 'proj-5',
        text: 'hello',
        source: 'main',
        is_final: false,
      });
    });

    expect(result.current.messages.length).toBe(1);
    expect(result.current.messages[0].content).toBe('hello');
  });
});
