// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import { api } from '../config';
import type { BlockedSessionEntry, WebSocketEvent } from '../types';
import { useWebSocket } from './useWebSocket';

interface BlockedCountResponse {
  blocked_count: number;
  blocked_sessions: BlockedSessionEntry[];
}

/**
 * useBlockedCount — global hook for cross-project pending_approval tracking.
 *
 * Fetches GET /api/v2/blocked on mount for the initial blocked count, then
 * keeps the count live by subscribing to the global `blocked-count-changed`
 * WS event (which fires whenever any session enters or leaves pending_approval,
 * regardless of project).
 *
 * This is intentionally not scoped to a single project so the top-level UI
 * (e.g. a global badge) can reflect the total across all running agents.
 *
 * @returns `{ blockedCount, blockedSessions, loading }`
 */
export function useBlockedCount(): {
  blockedCount: number;
  blockedSessions: BlockedSessionEntry[];
  loading: boolean;
} {
  const [blockedCount, setBlockedCount] = useState(0);
  const [blockedSessions, setBlockedSessions] = useState<BlockedSessionEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const ws = useWebSocket();

  // Fetch initial state from the REST endpoint.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api<BlockedCountResponse>('/api/v2/blocked')
      .then((data) => {
        if (cancelled) return;
        setBlockedCount(data.blocked_count);
        setBlockedSessions(data.blocked_sessions);
      })
      .catch(() => {
        // Non-fatal: keep the default zero values.
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Subscribe to the global WS event so the count stays live.
  useEffect(() => {
    const handler = (event: WebSocketEvent) => {
      if (event.type !== 'blocked-count-changed') return;
      setBlockedCount(event.blocked_count);
      setBlockedSessions(event.blocked_sessions);
    };
    ws.on('blocked-count-changed', handler);
    return () => {
      ws.off('blocked-count-changed', handler);
    };
  }, [ws]);

  return { blockedCount, blockedSessions, loading };
}
