// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../config';
import type { SessionListEntry, WebSocketEvent } from '../types';
import { useWebSocket } from './useWebSocket';

// Per-project cache: avoids refetch when the user switches between projects
// and returns to one they've already loaded. Keyed by projectId.
const sessionsCache = new Map<string, SessionListEntry[]>();

// useSessions
//
// Fetches the per-project session list (GET /api/v2/projects/{pid}/sessions)
// and exposes each entry — including the persistent `last_terminal_event`
// and `last_activity_at` fields — to the UI. Consumers render durable
// per-session indicators (warning glyph for errors, etc.) that survive WS
// disconnects and page reloads, since the backend stores these fields outside
// of WS event flow.
//
// The hook maintains a per-project in-memory cache so switching between
// projects does not trigger an unnecessary blank-loading state (the cached
// data is shown immediately while a background refresh occurs).
//
// WS subscription: any `agent.status` event for this project triggers a
// refresh, since every session transition (new session, idle, error, stopped,
// etc.) is reported via that event. This keeps the list live without polling.
//
// Phase 1B's SessionSidebar / SessionListItem are the primary consumers.
export function useSessions(projectId: string | null) {
  const [sessions, setSessions] = useState<SessionListEntry[]>(() => {
    if (projectId) return sessionsCache.get(projectId) ?? [];
    return [];
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ws = useWebSocket();

  // Keep a stable ref to the latest projectId for use in WS handler.
  const projectIdRef = useRef(projectId);
  projectIdRef.current = projectId;

  const refresh = useCallback(async () => {
    if (!projectId) {
      setSessions([]);
      return [];
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api<SessionListEntry[]>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/sessions`,
      );
      sessionsCache.set(projectId, data);
      setSessions(data);
      return data;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to list sessions';
      setError(msg);
      return [];
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  // On projectId change: load from cache immediately (no blank flash), then
  // kick off a background refresh.
  useEffect(() => {
    if (!projectId) {
      setSessions([]);
      return;
    }
    const cached = sessionsCache.get(projectId);
    if (cached) {
      setSessions(cached);
    }
    void refresh();
  }, [projectId, refresh]);

  // WS subscription: refresh session list when agent status transitions for
  // this project. agent.status covers: new_session, idle, running, waiting,
  // stopped, error, pending_approval — all the events that can change the
  // session list shape or a session's status field.
  useEffect(() => {
    if (!projectId) return;
    const handler = (event: WebSocketEvent) => {
      if (event.type !== 'agent.status') return;
      if (event.project_id !== projectIdRef.current) return;
      void refresh();
    };
    ws.on('agent.status', handler);
    return () => {
      ws.off('agent.status', handler);
    };
  }, [projectId, refresh, ws]);

  return { sessions, loading, error, refresh };
}

// Alias for consumers that prefer the more explicit name.
export const useSessionList = useSessions;
