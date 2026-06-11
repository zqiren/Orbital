// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type {
  CostResponse,
  CostWindow,
  SubagentRow,
  CostBreakdownRow,
} from './types';

/**
 * Pure derivation helpers for the Budget surface. No React, no i18n — these
 * take data + the limit and produce the small shapes the view renders. Kept
 * separate so they unit-test without jsdom.
 */

/** Spend-meter visual state. */
export type MeterState = 'none' | 'normal' | 'warn' | 'danger';

export interface MeterModel {
  state: MeterState;
  /** 0..1 fill fraction (clamped). 0 when there is no limit. */
  fraction: number;
  spent: number;
  /** null when no limit is set. */
  limit: number | null;
  currency: string;
}

/**
 * Derive the meter model from the cost response + the configured limit.
 *
 * - no limit            → state 'none', fraction 0 (spend shown alone)
 * - spend < 80% of limit → 'normal'
 * - 80% ≤ spend < 100%   → 'warn'
 * - spend ≥ 100%         → 'danger'
 *
 * `spent` is the management converted_total (the number enforced against the
 * limit). Sub-agent spend is excluded by construction (the route already
 * splits it out) — never fold it in here.
 */
export function deriveMeter(
  cost: CostResponse | null,
  limit: number | null,
): MeterModel {
  const spent = cost?.converted_total.amount ?? 0;
  const currency = cost?.converted_total.currency ?? 'USD';
  if (limit == null || !(limit > 0)) {
    return { state: 'none', fraction: 0, spent, limit: null, currency };
  }
  const ratio = spent / limit;
  const fraction = Math.max(0, Math.min(1, ratio));
  let state: MeterState;
  if (ratio >= 1) state = 'danger';
  else if (ratio >= 0.8) state = 'warn';
  else state = 'normal';
  return { state, fraction, spent, limit, currency };
}

/** A unified breakdown row for the table (management OR sub-agent). */
export interface DisplayRow {
  key: string;
  provider: string;
  model: string;
  source: string;
  /** Whether this is a sub-agent (display-only) row. */
  isSubagent: boolean;
  /** Input tokens = uncached_input (cached shown separately). */
  inputTokens: number;
  /** Cached tokens = cache_read + cache_write. */
  cachedTokens: number;
  outputTokens: number;
  /**
   * Cost cell. For management rows: { amount, currency } in the row's NATIVE
   * currency. For sub-agent rows with a reported_cost: that verbatim. For
   * sub-agent rows WITHOUT a reported_cost: null → the view renders the
   * "subscription" label.
   */
  cost: { amount: number; currency: string } | null;
}

function managementRow(row: CostBreakdownRow): DisplayRow {
  return {
    key: `mgmt:${row.source}:${row.provider}:${row.model}`,
    provider: row.provider,
    model: row.model,
    source: row.source,
    isSubagent: false,
    inputTokens: row.tokens.uncached_input,
    cachedTokens: row.tokens.cache_read + row.tokens.cache_write,
    outputTokens: row.tokens.output,
    cost: { amount: row.cost.total, currency: row.currency },
  };
}

function subagentRow(row: SubagentRow): DisplayRow {
  return {
    key: `sub:${row.source}:${row.provider}:${row.model}`,
    provider: row.provider,
    model: row.model,
    source: row.source,
    isSubagent: true,
    inputTokens: row.tokens.uncached_input,
    cachedTokens: row.tokens.cache_read + row.tokens.cache_write,
    outputTokens: row.tokens.output,
    // null reported_cost → subscription label in the cost column.
    cost: row.reported_cost
      ? { amount: row.reported_cost.amount, currency: row.reported_cost.currency }
      : null,
  };
}

/**
 * Assemble the breakdown table rows: management rows first (in native ccy),
 * then sub-agent rows (reported_cost verbatim, else subscription). Stable
 * order so the table doesn't reshuffle on refresh.
 */
export function assembleRows(cost: CostResponse | null): DisplayRow[] {
  if (!cost) return [];
  return [
    ...cost.breakdown.map(managementRow),
    ...cost.subagents.map(subagentRow),
  ];
}

/** Whether the table has any sub-agent rows (drives the footnote line). */
export function hasSubagents(cost: CostResponse | null): boolean {
  return !!cost && cost.subagents.length > 0;
}

/** Model for the single footnote line under the breakdown table. */
export interface FootnoteModel {
  /** Show the estimate marker (per converted_total.estimated). */
  estimated: boolean;
  /**
   * CNY-per-USD rate to render, or null. Non-null ONLY when a real
   * cross-currency conversion happened (some breakdown/subagent native
   * currency differs from converted_total.currency) AND the daemon surfaced
   * the rate. Never invented.
   */
  fxRate: number | null;
  /** Show the "sub-agents not counted against the limit" clause. */
  showSubagents: boolean;
}

/**
 * Derive the footnote segments. Rate is shown only on a REAL cross-currency
 * conversion; the sub-agents clause only when sub-agent rows exist; the
 * estimate marker follows converted_total.estimated. Null/empty cost (no rows
 * at all) → an all-off model (no footnote under an empty table).
 */
export function deriveFootnote(cost: CostResponse | null): FootnoteModel {
  const off: FootnoteModel = { estimated: false, fxRate: null, showSubagents: false };
  if (!cost) return off;
  const hasRows = cost.breakdown.length > 0 || cost.subagents.length > 0;
  if (!hasRows) return off;

  const target = cost.converted_total.currency;
  const nativeCurrencies = [
    ...cost.breakdown.map((r) => r.currency),
    ...cost.subagents
      .map((s) => s.reported_cost?.currency)
      .filter((c): c is string => !!c),
  ];
  const crossCurrency = nativeCurrencies.some((c) => c !== target);

  const rawRate = cost.fx_rates?.['CNY_per_USD'];
  const fxRate =
    crossCurrency && typeof rawRate === 'number' && rawRate > 0 ? rawRate : null;

  return {
    estimated: cost.converted_total.estimated,
    fxRate,
    showSubagents: cost.subagents.length > 0,
  };
}

/** Map a budget_period to the GET /cost window enum. They are identical. */
export function periodToWindow(period: string | undefined): CostWindow {
  switch (period) {
    case 'weekly':
      return 'weekly';
    case 'monthly':
      return 'monthly';
    case 'total':
      return 'total';
    case 'daily':
    default:
      return 'daily';
  }
}
