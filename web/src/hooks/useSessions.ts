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
      const resp = await api<{ sessions?: SessionListEntry[] } | SessionListEntry[]>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/sessions`,
      );
      // The endpoint returns { project_id, sessions: [...] }. Unwrap it; also
      // tolerate a bare array defensively so a shape change can't crash the
      // SessionSidebar (sessions.filter) and blank the whole project view.
      const data = Array.isArray(resp) ? resp : (resp?.sessions ?? []);
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
  //
  // A pinned-worker dispatch (spec 074) never wakes the management loop, so
  // it emits NO agent.status at all — the fresh session materialized by the
  // first pinned message stayed invisible in the sidebar until something
  // unrelated refreshed it. The sub-agent lifecycle DOES speak on that path:
  // sub_agent.started fires when the spawn-on-demand lands (just after the
  // user row is persisted), and chat.sub_agent_message on every dispatch ack
  // and worker reply (which also moves last_activity_at).
  useEffect(() => {
    if (!projectId) return;
    const handler = (event: WebSocketEvent) => {
      if (!('project_id' in event) || event.project_id !== projectIdRef.current) return;
      void refresh();
    };
    const events = ['agent.status', 'sub_agent.started', 'chat.sub_agent_message'] as const;
    events.forEach((ev) => ws.on(ev, handler));
    return () => {
      events.forEach((ev) => ws.off(ev, handler));
    };
  }, [projectId, refresh, ws]);

  // Rename a session (display label only). Optimistically updates the local
  // list + cache so the new name shows immediately, then PATCHes the backend.
  // Resolves to the new name; rejects (and reverts) on API failure.
  const renameSession = useCallback(
    async (sessionId: string, name: string): Promise<string> => {
      if (!projectId) throw new Error('No project selected');
      const trimmed = name.trim();
      if (!trimmed) throw new Error('name must not be empty');
      // Optimistic update.
      setSessions((prev) => {
        const next = prev.map((s) =>
          s.session_id === sessionId ? { ...s, name: trimmed } : s,
        );
        sessionsCache.set(projectId, next);
        return next;
      });
      try {
        await api(
          `/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
          {
            method: 'PATCH',
            body: JSON.stringify({ name: trimmed }),
          },
        );
        return trimmed;
      } catch (e) {
        // Revert by refetching authoritative state.
        void refresh();
        throw e;
      }
    },
    [projectId, refresh],
  );

  // Pin/unpin a session to the top of the sidebar (spec 067). Same optimistic
  // shape as renameSession — the row jumps immediately, then the PATCH lands;
  // a failure reverts by refetching authoritative state rather than trying to
  // undo locally, so the list can never disagree with the file on disk.
  const pinSession = useCallback(
    async (sessionId: string, pinned: boolean): Promise<boolean> => {
      if (!projectId) throw new Error('No project selected');
      setSessions((prev) => {
        const next = prev.map((s) =>
          s.session_id === sessionId ? { ...s, pinned } : s,
        );
        sessionsCache.set(projectId, next);
        return next;
      });
      try {
        await api(
          `/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
          {
            method: 'PATCH',
            body: JSON.stringify({ pinned }),
          },
        );
        return pinned;
      } catch (e) {
        void refresh();
        throw e;
      }
    },
    [projectId, refresh],
  );

  // Pin the session to a sub-agent (spec 074): the composer's "Talking to"
  // dropdown. `slug` pins/retargets; `null` unpins (explicit null in the
  // PATCH body — the backend treats absence as "leave untouched"). Same
  // optimistic shape as pinSession; a failure refetches authoritative state.
  const pinAgent = useCallback(
    async (sessionId: string, slug: string | null): Promise<string | null> => {
      if (!projectId) throw new Error('No project selected');
      setSessions((prev) => {
        const next = prev.map((s) =>
          s.session_id === sessionId ? { ...s, pinned_target: slug } : s,
        );
        sessionsCache.set(projectId, next);
        return next;
      });
      try {
        await api(
          `/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
          {
            method: 'PATCH',
            body: JSON.stringify({ pinned_target: slug }),
          },
        );
        return slug;
      } catch (e) {
        void refresh();
        throw e;
      }
    },
    [projectId, refresh],
  );

  // Delete a session (removes its JSONL on disk). Removes the row from the
  // local list + cache on success. Rejects on failure (e.g. 409 for a running
  // session) so the caller can surface an error without mutating the list.
  const deleteSession = useCallback(
    async (sessionId: string): Promise<void> => {
      if (!projectId) throw new Error('No project selected');
      await api(
        `/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
        { method: 'DELETE' },
      );
      setSessions((prev) => {
        const next = prev.filter((s) => s.session_id !== sessionId);
        sessionsCache.set(projectId, next);
        return next;
      });
    },
    [projectId],
  );

  return { sessions, loading, error, refresh, renameSession, pinSession, pinAgent, deleteSession };
}

// Alias for consumers that prefer the more explicit name.
export const useSessionList = useSessions;
