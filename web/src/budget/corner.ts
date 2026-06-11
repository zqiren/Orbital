// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { CostResponse } from './types';
import { deriveMeter } from './derive';

/**
 * Pure derivation for the project-header budget corner (P3-G). No React, no
 * i18n — takes the same GET /cost response the settings meter consumes plus the
 * configured limit and the queue's pause_reason, and produces the small shape
 * the corner renders. Kept separate so it unit-tests without jsdom.
 *
 * The corner reads the SAME useCost query as the Budget settings meter, so the
 * two can never disagree. It layers ONE extra input the meter doesn't have: the
 * queue's pause_reason, which forces the exhausted state even if recorded spend
 * is momentarily below 100% (e.g. the limit was just lowered, or a trip fired
 * on a boundary the ledger rounds under).
 */

/** Corner visual/semantic state. */
export type CornerState = 'normal' | 'warn' | 'exhausted';

export interface CornerModel {
  state: CornerState;
  /** Converted window spend (in `currency`). */
  spent: number;
  /** The configured limit, or null when no limit is set. */
  limit: number | null;
  /** Budget (converted-total) currency the amounts are denominated in. */
  currency: string;
}

/**
 * Derive the corner model.
 *
 * Precedence (highest first):
 *   1. EXHAUSTED — the queue is budget-paused (pause_reason === 'budget') OR
 *      recorded spend is ≥ 100% of a set limit. The pause_reason wins even when
 *      spend reads under the limit, because the daemon already tripped.
 *   2. WARN — 80% ≤ spend < 100% of a set limit.
 *   3. NORMAL — under 80%, or no limit set (spend shown alone).
 *
 * Reuses deriveMeter for the spend/limit/currency extraction so the numbers are
 * byte-identical to the settings meter; only the top-level state differs (the
 * meter has a 'none' state for no-limit; the corner folds that into 'normal'
 * and shows the bare spend).
 */
export function deriveCorner(
  cost: CostResponse | null,
  limit: number | null,
  pauseReason: 'user' | 'budget' | null | undefined,
): CornerModel {
  const meter = deriveMeter(cost, limit);
  const budgetPaused = pauseReason === 'budget';

  let state: CornerState;
  if (budgetPaused || meter.state === 'danger') {
    state = 'exhausted';
  } else if (meter.state === 'warn') {
    state = 'warn';
  } else {
    // 'normal' or 'none' (no limit) → normal corner with bare spend.
    state = 'normal';
  }

  return {
    state,
    spent: meter.spent,
    limit: meter.limit,
    currency: meter.currency,
  };
}
