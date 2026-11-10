// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// The live sub-agent status of one chat session. The daemon pushes every
// change as `sub_agent.status` (one event per worker, sent only when its
// status changed), and this hook applies it as is. It reads the full list
// once on mount and again whenever the WebSocket reconnects, since events
// sent while it was down are lost. There is no poll. Lifted out of
// SubAgentStatusBar (spec 099 §3.1) so the pin control's running dot reads
// the same pipeline. An agent with no live adapter is absent from the list,
// which means "not spawned" (idle).

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../config';
import { useWebSocket } from './useWebSocket';
import { isWorkerHandle } from '../utils/subAgentHandle';
import type { SubAgentRunStatus, SubAgentStatusEvent, WebSocketEvent } from '../types';

export interface SubAgentInfo {
  handle: string;
  display_name: string;
  status: SubAgentRunStatus;
  /** The open turn waits on the user's approval. Absent from an older
   *  daemon, which reads as false. */
  awaiting_approval?: boolean;
  background_commands: string[];
}

export interface SubAgentStatusState {
  agents: SubAgentInfo[];
  refresh: () => Promise<void>;
}

/** A pushed status, or null for a worker that left (`stopped`). */
function fromEvent(e: SubAgentStatusEvent): SubAgentInfo | null {
  if (e.status === 'stopped') return null;
  return {
    handle: e.handle,
    display_name: e.display_name || e.handle,
    status: e.status,
    awaiting_approval: e.awaiting_approval === true,
    background_commands: Array.isArray(e.background_commands) ? e.background_commands : [],
  };
}

function upsert(list: SubAgentInfo[], handle: string, next: SubAgentInfo | null): SubAgentInfo[] {
  const i = list.findIndex((a) => a.handle === handle);
  if (next === null) return i < 0 ? list : list.filter((a) => a.handle !== handle);
  if (i < 0) return [...list, next];
  if (JSON.stringify(list[i]) === JSON.stringify(next)) return list;
  const copy = list.slice();
  copy[i] = next;
  return copy;
}

export function useSubAgentStatus(
  projectId: string,
  sessionId: string | undefined,
  { enabled = true }: { enabled?: boolean } = {},
): SubAgentStatusState {
  const [agents, setAgents] = useState<SubAgentInfo[]>([]);
  const { on, off, connectionState } = useWebSocket();
  const alive = useRef(true);
  // The session the list belongs to: the prop, or (without one) the holder
  // session the daemon resolved for the last fetch.
  const resolvedSession = useRef<string | null>(sessionId ?? null);
  // Events that arrive while a fetch is in flight are newer than its
  // answer: they are laid back over it when it lands.
  const sinceFetch = useRef<Map<string, SubAgentInfo | null> | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const refresh = useCallback(async () => {
    // Bug #48 (fix C): a session switch re-fires this; abort the superseded
    // request instead of letting discarded responses pile up.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    sinceFetch.current = new Map();
    try {
      const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      const data = await api<{ session_id: string | null; agents: SubAgentInfo[] }>(
        `/api/v2/agents/${projectId}/sub-agents/status${qs}`,
        { signal: controller.signal },
      );
      if (alive.current && !controller.signal.aborted) {
        resolvedSession.current = sessionId ?? data?.session_id ?? null;
        // Fanout workers (spec 009 §0.5) get their own live surface —
        // FanoutCard — and must NOT also show up as chips here.
        let next = (data?.agents ?? []).filter((a) => !isWorkerHandle(a.handle));
        sinceFetch.current?.forEach((info, handle) => { next = upsert(next, handle, info); });
        // Identity-stable update: unchanged payloads keep the previous array
        // reference so effects keyed on state don't re-fire.
        setAgents((prev) =>
          JSON.stringify(prev) === JSON.stringify(next) ? prev : next,
        );
      }
    } catch {
      /* daemon may be restarting — the reconnect refetch catches up */
    } finally {
      if (abortRef.current === controller) sinceFetch.current = null;
    }
  }, [projectId, sessionId]);

  // Read the list on mount, then apply each pushed status.
  useEffect(() => {
    if (!enabled) return;
    alive.current = true;
    resolvedSession.current = sessionId ?? null;
    refresh();
    const handler = (e: WebSocketEvent) => {
      if (e.type !== 'sub_agent.status') return;
      if (e.project_id !== projectId || !e.handle || isWorkerHandle(e.handle)) return;
      const mine = sessionId ?? resolvedSession.current;
      if (!mine || (e.session_id && e.session_id !== mine)) return;
      const info = fromEvent(e);
      sinceFetch.current?.set(e.handle, info);
      setAgents((prev) => upsert(prev, e.handle, info));
    };
    on('sub_agent.status', handler);
    return () => {
      alive.current = false;
      abortRef.current?.abort();
      off('sub_agent.status', handler);
    };
  }, [enabled, projectId, sessionId, on, off, refresh]);

  // A reconnect: whatever was pushed while the socket was down is lost.
  const lastConnection = useRef(connectionState);
  useEffect(() => {
    const prev = lastConnection.current;
    lastConnection.current = connectionState;
    if (!enabled || connectionState !== 'connected' || prev === undefined || prev === 'connected') return;
    refresh();
  }, [enabled, connectionState, refresh]);

  return { agents, refresh };
}
