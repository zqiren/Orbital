// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { deriveCorner } from './corner';
import type { CostResponse } from './types';

function cost(amount: number, currency = 'USD', estimated = false): CostResponse {
  return {
    window: 'daily',
    by_currency: { [currency]: amount },
    converted_total: { currency, amount, estimated },
    breakdown: [],
    subagents: [],
  };
}

describe('deriveCorner', () => {
  it('normal: no limit, spend shown alone', () => {
    const c = deriveCorner(cost(0.84), null, null);
    expect(c.state).toBe('normal');
    expect(c.spent).toBeCloseTo(0.84);
    expect(c.limit).toBeNull();
    expect(c.currency).toBe('USD');
  });

  it('normal: under 80% of limit', () => {
    const c = deriveCorner(cost(5), 10, null);
    expect(c.state).toBe('normal');
    expect(c.limit).toBe(10);
  });

  it('warn: at exactly 80% of limit', () => {
    const c = deriveCorner(cost(8), 10, null);
    expect(c.state).toBe('warn');
  });

  it('warn: between 80% and 100%', () => {
    const c = deriveCorner(cost(9.5), 10, null);
    expect(c.state).toBe('warn');
  });

  it('exhausted: at exactly 100% of limit', () => {
    const c = deriveCorner(cost(10), 10, null);
    expect(c.state).toBe('exhausted');
  });

  it('exhausted: over 100% of limit', () => {
    const c = deriveCorner(cost(12), 10, null);
    expect(c.state).toBe('exhausted');
  });

  it('precedence: pause_reason="budget" forces exhausted even when spend is under the limit', () => {
    // Spend is only 30% of limit, but the queue is budget-paused → exhausted.
    const c = deriveCorner(cost(3), 10, 'budget');
    expect(c.state).toBe('exhausted');
  });

  it('precedence: pause_reason="budget" forces exhausted even with NO limit set', () => {
    const c = deriveCorner(cost(3), null, 'budget');
    expect(c.state).toBe('exhausted');
  });

  it('pause_reason="user" does NOT force exhausted (stays per-spend)', () => {
    const c = deriveCorner(cost(3), 10, 'user');
    expect(c.state).toBe('normal');
  });

  it('null cost: normal with zero spend when no pause', () => {
    const c = deriveCorner(null, 10, null);
    expect(c.state).toBe('normal');
    expect(c.spent).toBe(0);
  });

  it('null cost + budget pause: still exhausted', () => {
    const c = deriveCorner(null, null, 'budget');
    expect(c.state).toBe('exhausted');
  });

  it('carries the converted-total currency (CNY)', () => {
    const c = deriveCorner(cost(50, 'CNY'), 100, null);
    expect(c.currency).toBe('CNY');
    expect(c.spent).toBe(50);
  });
});
