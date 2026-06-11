// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { formatMoney, formatTokens } from './format';

describe('formatMoney', () => {
  it('renders USD with a dollar sign in en', () => {
    expect(formatMoney(1.23, 'USD', 'en')).toBe('$1.23');
  });

  it('renders CNY with a yuan sign', () => {
    // narrowSymbol → ¥ in both locales.
    expect(formatMoney(8.9, 'CNY', 'en')).toBe('¥8.90');
    expect(formatMoney(8.9, 'CNY', 'zh')).toBe('¥8.90');
  });

  it('always shows two fraction digits', () => {
    expect(formatMoney(5, 'USD', 'en')).toBe('$5.00');
    expect(formatMoney(0, 'USD', 'en')).toBe('$0.00');
  });

  it('does not throw on an unknown ISO code (falls back to code-tagged decimal)', () => {
    const out = formatMoney(1.5, 'XۀZ', 'en'); // deliberately invalid
    expect(out).toContain('1.50');
  });

  it('handles large amounts with grouping', () => {
    // en-US groups thousands with commas.
    expect(formatMoney(1234.5, 'USD', 'en')).toBe('$1,234.50');
  });
});

describe('formatTokens', () => {
  it('renders sub-thousand counts verbatim', () => {
    expect(formatTokens(0, 'en')).toBe('0');
    expect(formatTokens(999, 'en')).toBe('999');
  });

  it('renders thousands as K', () => {
    expect(formatTokens(12_000, 'en')).toBe('12K');
    expect(formatTokens(1_500, 'en')).toBe('1.5K');
  });

  it('renders millions as M, trimming a trailing .0', () => {
    expect(formatTokens(1_000_000, 'en')).toBe('1M');
    expect(formatTokens(1_500_000, 'en')).toBe('1.5M');
    expect(formatTokens(2_000_000, 'en')).toBe('2M');
  });
});
