// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The status pushes itself: `sub_agent.status` is applied as is, with no
// fetch; the list is read on mount and again on a WebSocket reconnect; no
// poll, and unrelated events no longer trigger a refetch.

import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebSocketEvent } from '../types';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

type Listener = (e: WebSocketEvent) => void;
// Stable on/off, as the real provider's useCallbacks are.
const ws = vi.hoisted(() => {
  const listeners = new Map<string, Set<(e: unknown) => void>>();
  return {
    listeners,
    connectionState: 'connected' as 'connected' | 'connecting' | 'disconnected',
    on: (type: string, fn: (e: unknown) => void) => {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)!.add(fn);
    },
    off: (type: string, fn: (e: unknown) => void) => {
      listeners.get(type)?.delete(fn);
    },
  };
});
vi.mock('./useWebSocket', () => ({
  useWebSocket: () => ({
    connectionState: ws.connectionState,
    on: ws.on as (type: string, fn: Listener) => void,
    off: ws.off as (type: string, fn: Listener) => void,
  }),
}));

import { useSubAgentStatus } from './useSubAgentStatus';

function emit(type: string, payload: Record<string, unknown>) {
  act(() => {
    ws.listeners.get(type)?.forEach((fn) => fn({ type, ...payload }));
  });
}

function statusEvent(over: Record<string, unknown> = {}) {
  return {
    project_id: 'p1', session_id: 's1', handle: 'claude-code',
    display_name: 'Claude Code', status: 'running', awaiting_approval: false,
    background_commands: [], ...over,
  };
}

const idleList = {
  session_id: 's1',
  agents: [{ handle: 'claude-code', display_name: 'Claude Code', status: 'idle', background_commands: [] }],
};

beforeEach(() => {
  apiMock.mockReset();
  ws.listeners.clear();
  ws.connectionState = 'connected';
});

describe('useSubAgentStatus — pushed status', () => {
  it('applies a sub_agent.status event without a fetch', async () => {
    apiMock.mockResolvedValue(idleList);
    const { result } = renderHook(() => useSubAgentStatus('p1', 's1'));
    await waitFor(() => expect(result.current.agents[0]?.status).toBe('idle'));
    expect(apiMock).toHaveBeenCalledTimes(1);

    emit('sub_agent.status', statusEvent());
    expect(result.current.agents).toEqual([{
      handle: 'claude-code', display_name: 'Claude Code', status: 'running',
      awaiting_approval: false, background_commands: [],
    }]);
    emit('sub_agent.status', statusEvent({ awaiting_approval: true }));
    expect(result.current.agents[0].awaiting_approval).toBe(true);
    expect(apiMock).toHaveBeenCalledTimes(1);
  });

  it('adds a newly spawned worker and removes a stopped one', async () => {
    apiMock.mockResolvedValue({ session_id: 's1', agents: [] });
    const { result } = renderHook(() => useSubAgentStatus('p1', 's1'));
    await waitFor(() => expect(apiMock).toHaveBeenCalledTimes(1));
    emit('sub_agent.status', statusEvent({ status: 'idle' }));
    expect(result.current.agents.map((a) => a.handle)).toEqual(['claude-code']);
    emit('sub_agent.status', statusEvent({ status: 'stopped' }));
    expect(result.current.agents).toEqual([]);
  });

  it('ignores another session, another project, and fanout workers', async () => {
    apiMock.mockResolvedValue(idleList);
    const { result } = renderHook(() => useSubAgentStatus('p1', 's1'));
    await waitFor(() => expect(result.current.agents).toHaveLength(1));
    emit('sub_agent.status', statusEvent({ session_id: 's2' }));
    emit('sub_agent.status', statusEvent({ project_id: 'p2' }));
    emit('sub_agent.status', statusEvent({ handle: 'worker:abcd1234-0' }));
    expect(result.current.agents).toEqual(idleList.agents);
  });

  it('without a session prop, follows the session the daemon resolved', async () => {
    apiMock.mockResolvedValue(idleList);
    const { result } = renderHook(() => useSubAgentStatus('p1', undefined));
    await waitFor(() => expect(result.current.agents).toHaveLength(1));
    emit('sub_agent.status', statusEvent({ session_id: 's9' }));
    expect(result.current.agents[0].status).toBe('idle');
    emit('sub_agent.status', statusEvent());
    expect(result.current.agents[0].status).toBe('running');
  });

  it('an event that lands while the list is loading wins over the older answer', async () => {
    let answer: (v: unknown) => void = () => {};
    apiMock.mockReturnValue(new Promise((r) => { answer = r; }));
    const { result } = renderHook(() => useSubAgentStatus('p1', 's1'));
    emit('sub_agent.status', statusEvent());
    await act(async () => { answer(idleList); });
    expect(result.current.agents[0].status).toBe('running');
  });

  it('no longer refetches on unrelated events, and never polls', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      apiMock.mockResolvedValue({ ...idleList, agents: [{ ...idleList.agents[0], status: 'running' }] });
      renderHook(() => useSubAgentStatus('p1', 's1'));
      await waitFor(() => expect(apiMock).toHaveBeenCalledTimes(1));
      for (const type of ['chat.sub_agent_message', 'agent.status', 'sub_agent.started', 'sub_agent.completed']) {
        expect(ws.listeners.get(type)?.size ?? 0).toBe(0);
      }
      await act(async () => { vi.advanceTimersByTime(20000); });
      expect(apiMock).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('useSubAgentStatus — reconnect', () => {
  it('refetches when the WebSocket reconnects', async () => {
    apiMock.mockResolvedValue(idleList);
    const { rerender } = renderHook(() => useSubAgentStatus('p1', 's1'));
    await waitFor(() => expect(apiMock).toHaveBeenCalledTimes(1));

    ws.connectionState = 'disconnected';
    rerender();
    expect(apiMock).toHaveBeenCalledTimes(1);
    ws.connectionState = 'connected';
    apiMock.mockResolvedValue({ ...idleList, agents: [{ ...idleList.agents[0], status: 'running' }] });
    rerender();
    await waitFor(() => expect(apiMock).toHaveBeenCalledTimes(2));
  });

  it('does not refetch while staying connected', async () => {
    apiMock.mockResolvedValue(idleList);
    const { rerender } = renderHook(() => useSubAgentStatus('p1', 's1'));
    await waitFor(() => expect(apiMock).toHaveBeenCalledTimes(1));
    rerender();
    rerender();
    expect(apiMock).toHaveBeenCalledTimes(1);
  });
});
