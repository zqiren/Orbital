// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import type {
  BlockedSessionEntry,
  BudgetPausedProjectEntry,
  WebSocketEvent,
} from '../types';
import { useWebSocket } from './useWebSocket';

interface BlockedCountResponse {
  blocked_count: number;
  blocked_sessions: BlockedSessionEntry[];
  budget_paused_projects?: BudgetPausedProjectEntry[];
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
  /**
   * Projects whose queue is paused for budget (P3-G). Distinct from
   * `blockedSessions` (which is pending_approval only). The Blocked surface
   * reason-codes these as a separate marker — approval semantics are untouched.
   */
  budgetPausedProjects: BudgetPausedProjectEntry[];
  loading: boolean;
} {
  const [blockedCount, setBlockedCount] = useState(0);
  const [blockedSessions, setBlockedSessions] = useState<BlockedSessionEntry[]>([]);
  const [budgetPausedProjects, setBudgetPausedProjects] = useState<
    BudgetPausedProjectEntry[]
  >([]);
  const [loading, setLoading] = useState(true);
  const ws = useWebSocket();

  const fetchBlocked = useCallback(async () => {
    try {
      const data = await api<BlockedCountResponse>('/api/v2/blocked');
      setBlockedCount(data.blocked_count);
      setBlockedSessions(data.blocked_sessions);
      setBudgetPausedProjects(data.budget_paused_projects ?? []);
    } catch {
      // Non-fatal: keep the current values.
    }
  }, []);

  // Fetch initial state from the REST endpoint.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchBlocked().finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [fetchBlocked]);

  // Subscribe to the global blocked-count-changed event so approvals AND
  // budget pauses stay live (the daemon carries budget_paused_projects on it).
  useEffect(() => {
    const handler = (event: WebSocketEvent) => {
      if (event.type !== 'blocked-count-changed') return;
      setBlockedCount(event.blocked_count);
      setBlockedSessions(event.blocked_sessions);
      setBudgetPausedProjects(event.budget_paused_projects ?? []);
    };
    ws.on('blocked-count-changed', handler);
    return () => {
      ws.off('blocked-count-changed', handler);
    };
  }, [ws]);

  // A budget pause/resume fires `queue.state_changed` (project-scoped), NOT
  // blocked-count-changed — re-fetch /blocked on it so the budget-paused list
  // stays current without polling. Cheap: one GET on a real state transition.
  useEffect(() => {
    const handler = (event: WebSocketEvent) => {
      if (event.type !== 'queue.state_changed') return;
      void fetchBlocked();
    };
    ws.on('queue.state_changed', handler);
    return () => {
      ws.off('queue.state_changed', handler);
    };
  }, [ws, fetchBlocked]);

  return { blockedCount, blockedSessions, budgetPausedProjects, loading };
}
