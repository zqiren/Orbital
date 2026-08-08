// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { api } from '../config';
import type { WebSocketEvent } from '../types';
import type { CostResponse, CostWindow } from '../budget/types';
import {
  getPricingVersion,
  subscribePricingVersion,
} from '../budget/pricingVersion';
import { useWebSocket } from './useWebSocket';

/**
 * Per-project derived-cost view. The SINGLE source of recorded spend for the
 * Budget surface: it reads GET /api/v2/projects/{pid}/cost EXCLUSIVELY and
 * re-fetches when a `budget.spend_updated` WS event arrives for this project.
 *
 * The window is the budget period the Budget section currently shows; passing
 * it explicitly (rather than letting the server default) means changing the
 * period select re-fetches the matching window immediately.
 *
 * Freshness contract (backlog #40):
 *  - `loading` is true from the FIRST render when a fetch is coming, so
 *    consumers never mistake "not fetched yet" for a measured $0.00.
 *  - `cost` is cleared synchronously when the query identity (project, window,
 *    pricing version) changes — a stale number from the previous identity is
 *    never rendered, not even for one frame. Same-identity re-fetches
 *    (spend_updated) keep the last known spend visible instead of blinking.
 *  - A pricing-table edit bumps the module-level pricing version, which every
 *    mounted useCost observes — no prop threading, so BudgetCorner/QueueTab
 *    recompute together with the settings meter.
 *  - On WS reconnect the cost re-fetches: spend events missed while
 *    disconnected can no longer leave a long-lived consumer stale.
 *
 * Mirrors useQueue's fetch-on-mount + refresh-on-WS pattern.
 */
export interface UseCostResult {
  cost: CostResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useCost(projectId: string | null, window: CostWindow): UseCostResult {
  const pricingVersion = useSyncExternalStore(subscribePricingVersion, getPricingVersion);

  const [cost, setCost] = useState<CostResponse | null>(null);
  const [loading, setLoading] = useState(projectId != null);
  const [error, setError] = useState<string | null>(null);
  const ws = useWebSocket();

  // Reset synchronously (adjust-state-during-render) when the query identity
  // changes, so a render between prop change and effect never shows the
  // previous project/window/rates' number.
  const queryKey = `${projectId ?? ''}|${window}|${pricingVersion}`;
  const [lastQueryKey, setLastQueryKey] = useState(queryKey);
  if (lastQueryKey !== queryKey) {
    setLastQueryKey(queryKey);
    setCost(null);
    setLoading(projectId != null);
    setError(null);
  }

  const refresh = useCallback(async () => {
    if (!projectId) {
      setCost(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api<CostResponse>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/cost?window=${encodeURIComponent(window)}`,
      );
      setCost(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [projectId, window]);

  // Fetch on mount + whenever the query identity changes. The pricingVersion
  // bump (post pricing-edit) re-runs this with the same `refresh` identity,
  // re-reading /cost under the new rates.
  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refresh, pricingVersion]);

  // WS: a debounced budget.spend_updated for THIS project → re-fetch /cost.
  useEffect(() => {
    if (!projectId) return;
    const handler = (event: WebSocketEvent) => {
      if (event.type !== 'budget.spend_updated') return;
      if (event.project_id !== projectId) return;
      void refresh();
    };
    ws.on('budget.spend_updated', handler);
    return () => {
      ws.off('budget.spend_updated', handler);
    };
  }, [projectId, refresh, ws]);

  // WS reconnect → re-fetch: any spend_updated that fired while the socket was
  // down never reached us, so the last-known number may be stale. The ref
  // starts at the current state so mount doesn't double-fetch.
  const prevConnRef = useRef(ws.connectionState);
  useEffect(() => {
    const prev = prevConnRef.current;
    prevConnRef.current = ws.connectionState;
    if (ws.connectionState === 'connected' && prev !== 'connected') {
      void refresh();
    }
  }, [ws.connectionState, refresh]);

  return { cost, loading, error, refresh };
}
