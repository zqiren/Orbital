// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { translate } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';
import type { Locale } from '../i18n/locales';
import { formatMoney } from './format';

/**
 * Localized text for the budget-trip timeline row (P3-G). Pure — no React.
 * ChatView's `budget_event` branch and the unit tests share this single
 * implementation so the selection logic (pause/stop/no-limit × en/zh) can
 * never drift between the two.
 *
 * Per the repo's non-React i18n convention (see chatTransform.ts /
 * capsuleSummaryText): an optional translator param defaults to English so
 * callers/tests that omit it get byte-identical English output. `locale`
 * additionally drives the money formatting (currency symbol/grouping), so
 * pass BOTH bound to the same locale from React code.
 */

export type TimelineTr = (
  key: StringKey,
  vars?: Record<string, string | number>,
) => string;

const EN_TR: TimelineTr = (k, v) => translate('en', k, v);

const WINDOW_WORD: Record<'daily' | 'weekly' | 'monthly' | 'total', StringKey> = {
  daily: 'settings.budget.window.day',
  weekly: 'settings.budget.window.week',
  monthly: 'settings.budget.window.month',
  total: 'settings.budget.window.total',
};

/** Payload codes/numbers of a budget_blocked timeline item (codes only). */
export interface BudgetTimelineEvent {
  action: 'pause' | 'stop';
  window: 'daily' | 'weekly' | 'monthly' | 'total';
  spend: number;
  limit: number | null;
  currency: string;
}

/**
 * Compose the localized budget-trip row text. A null limit falls back to the
 * no-limit phrasing; action picks pause vs stop wording.
 */
export function budgetTimelineText(
  item: BudgetTimelineEvent,
  locale: Locale = 'en',
  tr: TimelineTr = EN_TR,
): string {
  const windowWord = tr(WINDOW_WORD[item.window]);
  const spentStr = formatMoney(item.spend, item.currency, locale);
  if (item.limit == null) {
    return tr('budget.timeline.blocked.noLimit', {
      spent: spentStr,
      window: windowWord,
    });
  }
  const key: StringKey =
    item.action === 'stop'
      ? 'budget.timeline.blocked.stop'
      : 'budget.timeline.blocked.pause';
  return tr(key, {
    spent: spentStr,
    limit: formatMoney(item.limit, item.currency, locale),
    window: windowWord,
  });
}
