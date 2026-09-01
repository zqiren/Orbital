// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Live per-session sub-agent status (Piece 3 Part D vocabulary:
// running | background-running | idle), extracted from SubAgentStatusBar so
// the composer can read the same truth: while the session's PINNED worker
// (spec 074) has an open turn, a send queues behind it — the composer must
// say "Queue", exactly as it does for a busy management agent, instead of
// pretending the session is idle because the management loop is.
//
// Fanout worker handles are filtered here (they render in FanoutCard, never
// as chips or composer state).

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../config';
import { useWebSocket } from './useWebSocket';
import { isWorkerHandle } from '../utils/subAgentHandle';
import type { SubAgentRunStatus, WebSocketEvent } from '../types';

export interface SubAgentInfo {
  handle: string;
  display_name: string;
  status: SubAgentRunStatus;
  background_commands: string[];
}

/** WS signals that can change a sub-agent's run status. */
const REFRESH_EVENTS = [
  'sub_agent.started', 'sub_agent.completed', 'sub_agent.error',
  'sub_agent.failed', 'sub_agent.stopped', 'sub_agent.turn_interrupted',
  'chat.sub_agent_message', 'agent.status',
] as const;

export function useSubAgentStatus(projectId: string, sessionId?: string) {
  const [agents, setAgents] = useState<SubAgentInfo[]>([]);
  const { on, off } = useWebSocket();
  const alive = useRef(true);

  const abortRef = useRef<AbortController | null>(null);
  const refresh = useCallback(async () => {
    // Bug #48 (fix C): a session switch re-fires this; abort the superseded
    // request instead of letting discarded responses pile up.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      const data = await api<{ session_id: string | null; agents: SubAgentInfo[] }>(
        `/api/v2/agents/${projectId}/sub-agents/status${qs}`,
        { signal: controller.signal },
      );
      if (alive.current) {
        const next = (data?.agents ?? []).filter((a) => !isWorkerHandle(a.handle));
        // Identity-stable update: unchanged payloads keep the previous array
        // reference so effects keyed on state don't re-fire (and unstable
        // hook identities — e.g. test mocks recreating on/off per render —
        // cannot produce a render loop through this setState).
        setAgents((prev) =>
          JSON.stringify(prev) === JSON.stringify(next) ? prev : next,
        );
      }
    } catch {
      /* daemon may be restarting — status just goes quiet */
    }
  }, [projectId, sessionId]);

  // Refresh on every sub-agent lifecycle signal + light poll while non-idle.
  useEffect(() => {
    alive.current = true;
    refresh();
    const handler = (e: WebSocketEvent) => {
      if ('project_id' in e && e.project_id && e.project_id !== projectId) return;
      refresh();
    };
    REFRESH_EVENTS.forEach((ev) => on(ev, handler));
    return () => {
      alive.current = false;
      abortRef.current?.abort();
      REFRESH_EVENTS.forEach((ev) => off(ev, handler));
    };
  }, [projectId, on, off, refresh]);

  useEffect(() => {
    const anyActive = agents.some((a) => a.status !== 'idle');
    if (!anyActive) return;
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [agents, refresh]);

  return { agents, refresh };
}
