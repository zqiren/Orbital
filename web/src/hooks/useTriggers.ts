// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import type { Trigger, WebSocketEvent } from '../types';
import { useWebSocket } from './useWebSocket';

/** Body of POST /triggers — what the create form sends. */
export interface TriggerDraft {
  name: string;
  type: 'schedule' | 'file_watch';
  task: string;
  enabled?: boolean;
  schedule?: { cron: string; human?: string; timezone?: string };
  watch_path?: string;
  patterns?: string[];
  recursive?: boolean;
  debounce_seconds?: number;
}

/**
 * Body of PATCH /triggers/{id} — a partial update. `type` is absent on
 * purpose: a schedule automation cannot become a file-watch one (different
 * required fields, different scheduler arming); create a new one instead.
 */
export type TriggerPatch = Partial<Omit<TriggerDraft, 'type'>>;

export function useTriggers(projectId: string) {
  const [triggers, setTriggers] = useState<Trigger[]>([]);
  const [loading, setLoading] = useState(false);
  const { on, off } = useWebSocket();

  const fetchTriggers = useCallback(async () => {
    if (!projectId) return [];
    setLoading(true);
    try {
      const data = await api<Trigger[]>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/triggers`,
      );
      setTriggers(data);
      return data;
    } catch {
      return [];
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  const updateTrigger = useCallback(
    async (triggerId: string, patch: TriggerPatch) => {
      const updated = await api<Trigger>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/triggers/${encodeURIComponent(triggerId)}`,
        {
          method: 'PATCH',
          body: JSON.stringify(patch),
        },
      );
      // Replace in place — an edit must never reorder or drop the row.
      setTriggers((prev) =>
        prev.map((t) => (t.id === triggerId ? updated : t)),
      );
      return updated;
    },
    [projectId],
  );

  const toggleTrigger = useCallback(
    (triggerId: string, enabled: boolean) => updateTrigger(triggerId, { enabled }),
    [updateTrigger],
  );

  const createTrigger = useCallback(
    async (draft: TriggerDraft) => {
      const created = await api<Trigger>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/triggers`,
        {
          method: 'POST',
          body: JSON.stringify(draft),
        },
      );
      setTriggers((prev) =>
        prev.some((t) => t.id === created.id) ? prev : [...prev, created],
      );
      return created;
    },
    [projectId],
  );

  const deleteTrigger = useCallback(
    async (triggerId: string) => {
      await api(
        `/api/v2/projects/${encodeURIComponent(projectId)}/triggers/${encodeURIComponent(triggerId)}`,
        { method: 'DELETE' },
      );
      setTriggers((prev) => prev.filter((t) => t.id !== triggerId));
    },
    [projectId],
  );

  // Real-time trigger updates via WebSocket
  useEffect(() => {
    const handleCreated = (event: WebSocketEvent) => {
      if (event.type === 'trigger.created' && event.project_id === projectId) {
        setTriggers((prev) => {
          if (prev.some((t) => t.id === event.trigger.id)) return prev;
          return [...prev, event.trigger];
        });
      }
    };
    const handleDeleted = (event: WebSocketEvent) => {
      if (event.type === 'trigger.deleted' && event.project_id === projectId) {
        setTriggers((prev) => prev.filter((t) => t.id !== event.trigger_id));
      }
    };
    // An edit/toggle updates the row IN PLACE. Never append and never drop:
    // created/deleted are for records appearing and going away.
    const handleUpdated = (event: WebSocketEvent) => {
      if (event.type === 'trigger.updated' && event.project_id === projectId) {
        setTriggers((prev) =>
          prev.map((t) => (t.id === event.trigger.id ? event.trigger : t)),
        );
      }
    };
    const handleFired = (event: WebSocketEvent) => {
      if (event.type === 'trigger.fired' && event.project_id === projectId) {
        setTriggers((prev) =>
          prev.map((t) =>
            t.id === event.trigger_id
              ? { ...t, last_triggered: event.timestamp, trigger_count: t.trigger_count + 1 }
              : t,
          ),
        );
      }
    };
    on('trigger.created', handleCreated);
    on('trigger.deleted', handleDeleted);
    on('trigger.updated', handleUpdated);
    on('trigger.fired', handleFired);
    return () => {
      off('trigger.created', handleCreated);
      off('trigger.deleted', handleDeleted);
      off('trigger.updated', handleUpdated);
      off('trigger.fired', handleFired);
    };
  }, [projectId, on, off]);

  return {
    triggers,
    loading,
    fetchTriggers,
    createTrigger,
    updateTrigger,
    toggleTrigger,
    deleteTrigger,
  };
}
