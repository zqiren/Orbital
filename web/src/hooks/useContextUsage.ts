// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../config';
import type { WebSocketEvent } from '../types';
import { useWebSocket } from './useWebSocket';

/**
 * Context in use for one chat session, feeding the composer's context line.
 *
 * Deliberately rides an event that already exists rather than adding one.
 * Every management LLM response appends to the token ledger, and that append
 * already emits a debounced `budget.spend_updated` — the same event `useCost`
 * listens to. Context changes on exactly those appends and no others, so
 * re-fetching here needs no new broadcaster, no new WS type, and no polling.
 * A sub-agent append will also wake us; re-reading a small endpoint is
 * cheaper than the plumbing to distinguish them.
 *
 * `usage.used === null` means the session has never made a management call.
 * The server distinguishes that from zero on purpose, and so does the UI: an
 * unmeasured session renders nothing rather than a confident empty meter.
 */
export interface ContextUsage {
  /** Prompt tokens on the last management call, or null if it never ran. */
  used: number | null;
  /** Raw context window of the model that served that call. */
  window: number | null;
  /** Prompt size at which the agent will summarize earlier messages. */
  threshold: number | null;
  provider: string | null;
  model: string | null;
}

export interface UseContextUsageResult {
  usage: ContextUsage | null;
  refresh: () => Promise<void>;
}

export function useContextUsage(
  projectId: string | null,
  sessionId: string | undefined,
): UseContextUsageResult {
  const [usage, setUsage] = useState<ContextUsage | null>(null);
  const ws = useWebSocket();

  // Clear synchronously when the session changes so the previous session's
  // fill never paints for a frame against the new one's transcript.
  const queryKey = `${projectId ?? ''}|${sessionId ?? ''}`;
  const [lastQueryKey, setLastQueryKey] = useState(queryKey);
  if (lastQueryKey !== queryKey) {
    setLastQueryKey(queryKey);
    setUsage(null);
  }

  const refresh = useCallback(async () => {
    if (!projectId || !sessionId) {
      setUsage(null);
      return;
    }
    try {
      const data = await api<ContextUsage>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/context`,
      );
      setUsage(data);
    } catch {
      // Non-fatal by design: this is ambient chrome. A failed read leaves the
      // last known fill rather than throwing an error state into the composer.
    }
  }, [projectId, sessionId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!projectId) return;
    const handler = (event: WebSocketEvent) => {
      if (event.type !== 'budget.spend_updated') return;
      // PRESENCE-guarded, matching useCost: `project_id` is the WS routing
      // key, not a promised payload field. An unguarded compare against
      // `undefined` would mute every event.
      if ('project_id' in event && event.project_id !== projectId) return;
      void refresh();
    };
    ws.on('budget.spend_updated', handler);
    return () => {
      ws.off('budget.spend_updated', handler);
    };
  }, [projectId, refresh, ws]);

  // Reconnect → re-read. Appends that landed while the socket was down never
  // reached us, and context only ever grows, so a stale fill under-reports.
  const prevConnRef = useRef(ws.connectionState);
  useEffect(() => {
    const prev = prevConnRef.current;
    prevConnRef.current = ws.connectionState;
    if (ws.connectionState === 'connected' && prev !== 'connected') {
      void refresh();
    }
  }, [ws.connectionState, refresh]);

  return { usage, refresh };
}
