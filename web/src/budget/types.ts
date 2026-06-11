// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Shapes for GET /api/v2/projects/{pid}/cost. The data layer is codes/enums/
 * ISO currency only (no display strings) — see the route docstring. The
 * frontend renders these via the i18n layer + the shared money formatter.
 */

/** Spend window enum — mirrors the backend WINDOWS tuple. */
export type CostWindow = 'daily' | 'weekly' | 'monthly' | 'total';

/** Per-source/model token usage (raw counts). */
export interface CostTokens {
  uncached_input: number;
  cache_read: number;
  cache_write: number;
  output: number;
}

/** Rate-derived cost for a management breakdown row, in the row's native ccy. */
export interface CostMoney {
  uncached_input: number;
  cache_read: number;
  cache_write: number;
  output: number;
  total: number;
  /** ISO currency the row's cost numbers are denominated in. */
  currency?: string;
}

/** One management breakdown row (per provider/model/source). */
export interface CostBreakdownRow {
  provider: string;
  model: string;
  source: string;
  /** ISO currency this row's cost is denominated in (its NATIVE currency). */
  currency: string;
  tokens: CostTokens;
  cost: CostMoney;
}

/** The window's total converted into the target (budget) currency. */
export interface ConvertedTotal {
  currency: string;
  amount: number;
  /** True when at least one row needed FX conversion (≈ estimate marker). */
  estimated: boolean;
  /** Currencies that had no FX rate and were dropped from the conversion. */
  unconverted?: string[];
}

/** Provider-reported sub-agent cost, verbatim (or null). */
export interface ReportedCost {
  amount: number;
  currency: string;
}

/** One sub-agent display row — display-only, never counted against the limit. */
export interface SubagentRow {
  provider: string;
  model: string;
  source: string;
  tokens: CostTokens;
  /** Provider-reported cost verbatim, or null for subscription-auth agents. */
  reported_cost: ReportedCost | null;
}

/** Full GET /cost response. */
export interface CostResponse {
  window: CostWindow;
  /** Spend grouped by native currency (management only). */
  by_currency: Record<string, number>;
  converted_total: ConvertedTotal;
  breakdown: CostBreakdownRow[];
  subagents: SubagentRow[];
  /**
   * The daemon's static FX pair table the conversion used, keyed
   * "{A}_per_{B}" (today only "CNY_per_USD" is meaningful). Codes/numbers
   * only — lets the footnote render the rate the total was converted at.
   */
  fx_rates?: Record<string, number>;
}
