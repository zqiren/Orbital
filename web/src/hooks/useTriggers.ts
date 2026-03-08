// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import type { Trigger, WebSocketEvent } from '../types';
import { useWebSocket } from './useWebSocket';

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

  const toggleTrigger = useCallback(
    async (triggerId: string, enabled: boolean) => {
      const updated = await api<Trigger>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/triggers/${encodeURIComponent(triggerId)}`,
        {
          method: 'PATCH',
          body: JSON.stringify({ enabled }),
        },
      );
      setTriggers((prev) =>
        prev.map((t) => (t.id === triggerId ? updated : t)),
      );
      return updated;
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
    on('trigger.fired', handleFired);
    return () => {
      off('trigger.created', handleCreated);
      off('trigger.deleted', handleDeleted);
      off('trigger.fired', handleFired);
    };
  }, [projectId, on, off]);

  return { triggers, loading, fetchTriggers, toggleTrigger, deleteTrigger };
}
