// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import type { WebSocketEvent } from '../types';
import type { CostResponse, CostWindow } from '../budget/types';
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
 * Mirrors useQueue's fetch-on-mount + refresh-on-WS pattern.
 */
export interface UseCostResult {
  cost: CostResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useCost(
  projectId: string | null,
  window: CostWindow,
  /**
   * Bump to force a re-fetch without a project/window change. Used after a
   * pricing-table edit (PUT /pricing/overrides) so the meter/breakdown
   * recompute historical cost against the NEW rates — the rate change fires no
   * WS event (that's ledger-append only), so the editor signals a recompute by
   * incrementing this. Event-driven, not polled.
   */
  refreshKey: number = 0,
): UseCostResult {
  const [cost, setCost] = useState<CostResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ws = useWebSocket();

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

  // Fetch on mount + whenever the project, window, or refreshKey changes. The
  // refreshKey bump (post pricing-edit) re-runs this with the same `refresh`
  // identity, re-reading /cost under the new rates.
  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refresh, refreshKey]);

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

  return { cost, loading, error, refresh };
}
