// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import type { QueueSnapshot, WebSocketEvent } from '../types';
import { useWebSocket } from './useWebSocket';

/**
 * Per-project queue state: snapshot + mutation helpers + WS-driven refresh.
 * Mirrors the chat-history hook's pattern (fetch on mount, refetch on WS).
 */
export function useQueue(projectId: string | null) {
  const [snapshot, setSnapshot] = useState<QueueSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ws = useWebSocket();

  const refresh = useCallback(async () => {
    if (!projectId) {
      setSnapshot(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api<QueueSnapshot>(
        `/api/v2/projects/${projectId}/queue`,
      );
      setSnapshot(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  // Initial fetch + on project change
  useEffect(() => {
    refresh();
  }, [refresh]);

  // WebSocket subscription: any queue.* event for this project → refresh
  useEffect(() => {
    if (!projectId) return;
    const handler = (event: WebSocketEvent) => {
      if (!event.type.startsWith('queue.')) return;
      if ('project_id' in event && event.project_id !== projectId) return;
      void refresh();
    };
    const types = [
      'queue.item_added',
      'queue.item_edited',
      'queue.item_removed',
      'queue.item_advanced',
      'queue.state_changed',
      'queue.reordered',
    ] as const;
    types.forEach((t) => ws.on(t, handler));
    return () => {
      types.forEach((t) => ws.off(t, handler));
    };
  }, [projectId, refresh, ws]);

  const addItem = useCallback(
    async (content: string, opts?: { priority?: number; review?: boolean }) => {
      if (!projectId) return;
      await api(`/api/v2/projects/${projectId}/queue/items`, {
        method: 'POST',
        body: JSON.stringify({
          content,
          priority: opts?.priority ?? 0,
          review_before_advance: opts?.review ?? false,
        }),
      });
    },
    [projectId],
  );

  const removeItem = useCallback(
    async (itemId: string) => {
      if (!projectId) return;
      try {
        await api(`/api/v2/projects/${projectId}/queue/items/${itemId}`, {
          method: 'DELETE',
        });
      } catch (e) {
        // The server gates delete on live slot-holder state, which can disagree
        // with the client's item.state across the WS/REST lag window (e.g. a
        // 409 "cannot delete a running item"). Surface it and resync rather than
        // failing silently.
        setError(e instanceof Error ? e.message : String(e));
        void refresh();
        throw e;
      }
    },
    [projectId, refresh],
  );

  const editItem = useCallback(
    async (
      itemId: string,
      patch: { content?: string; priority?: number; review_before_advance?: boolean },
    ) => {
      if (!projectId) return;
      await api(`/api/v2/projects/${projectId}/queue/items/${itemId}`, {
        method: 'PATCH',
        body: JSON.stringify(patch),
      });
    },
    [projectId],
  );

  const stopQueue = useCallback(
    async (durationSeconds?: number) => {
      if (!projectId) return;
      await api(`/api/v2/projects/${projectId}/queue/stop`, {
        method: 'POST',
        ...(durationSeconds != null
          ? { body: JSON.stringify({ duration_seconds: durationSeconds }) }
          : {}),
      });
    },
    [projectId],
  );

  const resumeQueue = useCallback(async () => {
    if (!projectId) return;
    await api(`/api/v2/projects/${projectId}/queue/resume`, { method: 'POST' });
  }, [projectId]);

  return {
    snapshot,
    loading,
    error,
    refresh,
    addItem,
    removeItem,
    editItem,
    stopQueue,
    resumeQueue,
  };
}
