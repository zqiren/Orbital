// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import {
  assembleRows,
  deriveFootnote,
  deriveMeter,
  hasSubagents,
  periodToWindow,
} from './derive';
import type { CostResponse } from './types';

function makeCost(over: Partial<CostResponse> = {}): CostResponse {
  return {
    window: 'daily',
    by_currency: { USD: 0 },
    converted_total: { currency: 'USD', amount: 0, estimated: true },
    breakdown: [],
    subagents: [],
    ...over,
  };
}

describe('deriveMeter', () => {
  it('returns state "none" with no limit (spend shown alone)', () => {
    const cost = makeCost({ converted_total: { currency: 'USD', amount: 0.42, estimated: true } });
    const m = deriveMeter(cost, null);
    expect(m.state).toBe('none');
    expect(m.fraction).toBe(0);
    expect(m.spent).toBeCloseTo(0.42);
    expect(m.limit).toBeNull();
  });

  it('treats a non-positive limit as no limit', () => {
    const m = deriveMeter(makeCost(), 0);
    expect(m.state).toBe('none');
  });

  it('returns "normal" below 80%', () => {
    const cost = makeCost({ converted_total: { currency: 'USD', amount: 5, estimated: true } });
    const m = deriveMeter(cost, 10);
    expect(m.state).toBe('normal');
    expect(m.fraction).toBeCloseTo(0.5);
  });

  it('returns "warn" at exactly 80%', () => {
    const cost = makeCost({ converted_total: { currency: 'USD', amount: 8, estimated: true } });
    expect(deriveMeter(cost, 10).state).toBe('warn');
  });

  it('returns "warn" between 80% and 100%', () => {
    const cost = makeCost({ converted_total: { currency: 'USD', amount: 9.5, estimated: true } });
    expect(deriveMeter(cost, 10).state).toBe('warn');
  });

  it('returns "danger" at exactly 100%', () => {
    const cost = makeCost({ converted_total: { currency: 'USD', amount: 10, estimated: true } });
    const m = deriveMeter(cost, 10);
    expect(m.state).toBe('danger');
    expect(m.fraction).toBe(1);
  });

  it('clamps fraction to 1 when over budget', () => {
    const cost = makeCost({ converted_total: { currency: 'USD', amount: 25, estimated: true } });
    const m = deriveMeter(cost, 10);
    expect(m.state).toBe('danger');
    expect(m.fraction).toBe(1);
  });

  it('defaults spend to 0 when cost is null', () => {
    const m = deriveMeter(null, 10);
    expect(m.spent).toBe(0);
    expect(m.state).toBe('normal');
  });
});

describe('assembleRows', () => {
  it('returns [] for null cost', () => {
    expect(assembleRows(null)).toEqual([]);
  });

  it('maps management rows: input=uncached, cached=read+write, native currency cost', () => {
    const cost = makeCost({
      breakdown: [
        {
          provider: 'anthropic',
          model: 'claude-x',
          source: 'management',
          currency: 'USD',
          tokens: { uncached_input: 100, cache_read: 30, cache_write: 20, output: 50 },
          cost: { uncached_input: 0, cache_read: 0, cache_write: 0, output: 0, total: 1.25 },
        },
      ],
    });
    const rows = assembleRows(cost);
    expect(rows).toHaveLength(1);
    expect(rows[0].isSubagent).toBe(false);
    expect(rows[0].inputTokens).toBe(100);
    expect(rows[0].cachedTokens).toBe(50);
    expect(rows[0].outputTokens).toBe(50);
    expect(rows[0].cost).toEqual({ amount: 1.25, currency: 'USD' });
  });

  it('maps sub-agent rows with reported_cost verbatim', () => {
    const cost = makeCost({
      subagents: [
        {
          provider: 'anthropic',
          model: 'claude-code',
          source: 'subagent:claude-code',
          tokens: { uncached_input: 1000, cache_read: 0, cache_write: 0, output: 2000 },
          reported_cost: { amount: 0.42, currency: 'USD' },
        },
      ],
    });
    const rows = assembleRows(cost);
    expect(rows[0].isSubagent).toBe(true);
    expect(rows[0].cost).toEqual({ amount: 0.42, currency: 'USD' });
  });

  it('maps a subscription sub-agent (null reported_cost) to cost=null (subscription label)', () => {
    const cost = makeCost({
      subagents: [
        {
          provider: 'openai',
          model: 'codex',
          source: 'subagent:codex',
          tokens: { uncached_input: 0, cache_read: 0, cache_write: 0, output: 3000 },
          reported_cost: null,
        },
      ],
    });
    const rows = assembleRows(cost);
    expect(rows[0].isSubagent).toBe(true);
    expect(rows[0].cost).toBeNull();
  });

  it('orders management rows before sub-agent rows', () => {
    const cost = makeCost({
      breakdown: [
        {
          provider: 'anthropic', model: 'claude-x', source: 'management', currency: 'USD',
          tokens: { uncached_input: 1, cache_read: 0, cache_write: 0, output: 1 },
          cost: { uncached_input: 0, cache_read: 0, cache_write: 0, output: 0, total: 1 },
        },
      ],
      subagents: [
        {
          provider: 'openai', model: 'codex', source: 'subagent:codex',
          tokens: { uncached_input: 0, cache_read: 0, cache_write: 0, output: 1 },
          reported_cost: null,
        },
      ],
    });
    const rows = assembleRows(cost);
    expect(rows.map((r) => r.isSubagent)).toEqual([false, true]);
  });
});

describe('hasSubagents', () => {
  it('false for null or empty', () => {
    expect(hasSubagents(null)).toBe(false);
    expect(hasSubagents(makeCost())).toBe(false);
  });
  it('true when subagents present', () => {
    const cost = makeCost({
      subagents: [{
        provider: 'p', model: 'm', source: 'subagent:codex',
        tokens: { uncached_input: 0, cache_read: 0, cache_write: 0, output: 1 },
        reported_cost: null,
      }],
    });
    expect(hasSubagents(cost)).toBe(true);
  });
});

describe('periodToWindow', () => {
  it('maps each period to its window', () => {
    expect(periodToWindow('daily')).toBe('daily');
    expect(periodToWindow('weekly')).toBe('weekly');
    expect(periodToWindow('monthly')).toBe('monthly');
    expect(periodToWindow('total')).toBe('total');
  });
  it('defaults unknown/undefined to daily', () => {
    expect(periodToWindow(undefined)).toBe('daily');
    expect(periodToWindow('nonsense')).toBe('daily');
  });
});

describe('deriveFootnote', () => {
  const mgmtRow = (currency: string) => ({
    provider: 'p', model: 'm', source: 'management', currency,
    tokens: { uncached_input: 1, cache_read: 0, cache_write: 0, output: 1 },
    cost: { uncached_input: 0, cache_read: 0, cache_write: 0, output: 0, total: 1 },
  });
  const subRow = (currency: string | null) => ({
    provider: 'p', model: 'm', source: 'subagent:claude-code',
    tokens: { uncached_input: 0, cache_read: 0, cache_write: 0, output: 1 },
    reported_cost: currency ? { amount: 0.5, currency } : null,
  });

  it('is all-off for null cost or an empty table', () => {
    expect(deriveFootnote(null)).toEqual({ estimated: false, fxRate: null, showSubagents: false });
    expect(deriveFootnote(makeCost())).toEqual({ estimated: false, fxRate: null, showSubagents: false });
  });

  it('no rate when all native currencies match the target (no real conversion)', () => {
    const cost = makeCost({
      breakdown: [mgmtRow('USD')],
      fx_rates: { CNY_per_USD: 7.2 },
    });
    const f = deriveFootnote(cost);
    expect(f.fxRate).toBeNull();
    expect(f.estimated).toBe(true); // per converted_total.estimated
    expect(f.showSubagents).toBe(false);
  });

  it('shows the rate on a real cross-currency conversion (breakdown row)', () => {
    const cost = makeCost({
      breakdown: [mgmtRow('USD'), mgmtRow('CNY')],
      fx_rates: { CNY_per_USD: 7.2 },
    });
    expect(deriveFootnote(cost).fxRate).toBe(7.2);
  });

  it('shows the rate when only a subagent reported_cost currency differs', () => {
    const cost = makeCost({
      converted_total: { currency: 'CNY', amount: 1, estimated: true },
      subagents: [subRow('USD')],
      fx_rates: { CNY_per_USD: 7.2 },
    });
    const f = deriveFootnote(cost);
    expect(f.fxRate).toBe(7.2);
    expect(f.showSubagents).toBe(true);
  });

  it('null rate when cross-currency but the daemon surfaced no rate (never invented)', () => {
    const cost = makeCost({
      breakdown: [mgmtRow('CNY')],
      fx_rates: {},
    });
    expect(deriveFootnote(cost).fxRate).toBeNull();
  });

  it('null rate when fx_rates is absent entirely', () => {
    const cost = makeCost({ breakdown: [mgmtRow('CNY')] });
    expect(deriveFootnote(cost).fxRate).toBeNull();
  });

  it('ignores a non-positive rate', () => {
    const cost = makeCost({
      breakdown: [mgmtRow('CNY')],
      fx_rates: { CNY_per_USD: 0 },
    });
    expect(deriveFootnote(cost).fxRate).toBeNull();
  });

  it('subagents clause only when subagent rows exist; subscription rows (null reported_cost) do not trigger the rate', () => {
    const cost = makeCost({
      breakdown: [mgmtRow('USD')],
      subagents: [subRow(null)],
      fx_rates: { CNY_per_USD: 7.2 },
    });
    const f = deriveFootnote(cost);
    expect(f.showSubagents).toBe(true);
    // The subscription row carries no currency → no cross-currency signal.
    expect(f.fxRate).toBeNull();
  });
});
