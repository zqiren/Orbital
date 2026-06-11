// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { Locale } from '../i18n/locales';

/**
 * Single shared money formatter for the budget surface.
 *
 * The data layer stays ISO currency codes (USD, CNY, …); the *symbol* is a
 * locale-layer concern (CNY → ¥, USD → $) handled here via Intl.NumberFormat.
 * Keeping every dollar/yuan figure on this util means the breakdown table, the
 * spend meter and the limit row all render currency identically.
 *
 * `Intl.NumberFormat` maps a BCP-47 locale: 'en' → 'en-US', 'zh' → 'zh-CN'.
 * USD under zh-CN renders "US$" by default; we force narrow symbols so the
 * common cases read "$1.23" / "¥8.90" the way the mockup specifies.
 */

const BCP47: Record<Locale, string> = { en: 'en-US', zh: 'zh-CN' };

// Memoize formatters — Intl.NumberFormat construction is comparatively heavy
// and the breakdown table can build many cells per render.
const cache = new Map<string, Intl.NumberFormat>();

function formatter(
  locale: Locale,
  currency: string,
  fractionDigits: number,
): Intl.NumberFormat {
  const key = `${locale}:${currency}:${fractionDigits}`;
  let fmt = cache.get(key);
  if (!fmt) {
    try {
      fmt = new Intl.NumberFormat(BCP47[locale], {
        style: 'currency',
        currency,
        currencyDisplay: 'narrowSymbol',
        minimumFractionDigits: fractionDigits,
        maximumFractionDigits: fractionDigits,
      });
    } catch {
      // Unknown/invalid ISO code (or an engine without narrowSymbol): fall
      // back to a plain decimal with the code appended so we never throw in
      // render. e.g. "1.23 XYZ".
      fmt = new Intl.NumberFormat(BCP47[locale], {
        minimumFractionDigits: fractionDigits,
        maximumFractionDigits: fractionDigits,
      });
      const decimal = fmt;
      const tagged: Intl.NumberFormat = {
        ...decimal,
        format: (n: number) => `${decimal.format(n)} ${currency}`,
      } as Intl.NumberFormat;
      cache.set(key, tagged);
      return tagged;
    }
  }
  cache.set(key, fmt);
  return fmt;
}

/**
 * Format an amount in `currency` (ISO code) for `locale`. `fractionDigits`
 * defaults to 2; pass 0 for whole-unit displays like the footnote's "$1".
 */
export function formatMoney(
  amount: number,
  currency: string,
  locale: Locale,
  fractionDigits = 2,
): string {
  return formatter(locale, currency, fractionDigits).format(amount);
}

/**
 * Format a token count compactly (e.g. 1_500_000 → "1.5M", 12_000 → "12K").
 * Locale-aware grouping via Intl; the K/M suffix is ASCII for both locales
 * (matches the mockup — no localized large-number units).
 */
export function formatTokens(n: number, locale: Locale): string {
  if (n >= 1_000_000) {
    return `${trimZero(n / 1_000_000, locale)}M`;
  }
  if (n >= 1_000) {
    return `${trimZero(n / 1_000, locale)}K`;
  }
  return new Intl.NumberFormat(BCP47[locale]).format(n);
}

function trimZero(value: number, locale: Locale): string {
  // One decimal place, but drop a trailing ".0" (1.0M → "1M").
  const rounded = Math.round(value * 10) / 10;
  return new Intl.NumberFormat(BCP47[locale], {
    minimumFractionDigits: 0,
    maximumFractionDigits: 1,
  }).format(rounded);
}
