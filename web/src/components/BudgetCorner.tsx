// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useMemo } from 'react';
import type { Project } from '../types';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import type { StringKey } from '../i18n/strings';
import { useCost } from '../hooks/useCost';
import { formatMoney } from '../budget/format';
import { periodToWindow } from '../budget/derive';
import { deriveCorner } from '../budget/corner';

/**
 * Project-header budget corner (P3-G). Renders next to the model name:
 *
 *  - NORMAL: converted window spend + window word, e.g. "≈ $0.84 today".
 *  - WARN (≥80% of limit): same shape with spend/limit, warning color.
 *  - EXHAUSTED (≥100% OR queue pause_reason === 'budget'): a red pill
 *    "Budget paused · raise limit to resume" that, on click, navigates to the
 *    settings Budget section.
 *
 * Uses the SAME useCost query as the settings Budget meter so the two can never
 * disagree, and is event-driven (useCost re-fetches on budget.spend_updated) —
 * NO polling. The pause_reason comes from the project's queue snapshot (owned by
 * the parent so we don't spin up a second queue subscription here).
 */

const WINDOW_WORD: Record<'daily' | 'weekly' | 'monthly' | 'total', StringKey> = {
  daily: 'settings.budget.window.day',
  weekly: 'settings.budget.window.week',
  monthly: 'settings.budget.window.month',
  total: 'settings.budget.window.total',
};

interface BudgetCornerProps {
  project: Project;
  /** Queue pause reason, when paused. Forces the exhausted pill if 'budget'. */
  pauseReason?: 'user' | 'budget' | null;
  /** Navigate to the settings Budget section (anchored when supported). */
  onOpenBudgetSettings: () => void;
}

export default function BudgetCorner({
  project,
  pauseReason,
  onOpenBudgetSettings,
}: BudgetCornerProps) {
  const t = useT();
  const { locale } = useLocale();

  const period = project.budget_period || 'daily';
  const window = periodToWindow(period);
  const { cost } = useCost(project.project_id, window);

  const limit =
    project.budget_limit_usd != null && project.budget_limit_usd > 0
      ? project.budget_limit_usd
      : null;

  const corner = useMemo(
    () => deriveCorner(cost, limit, pauseReason),
    [cost, limit, pauseReason],
  );

  // Nothing to show until /cost resolves AND there's something to convey: with
  // no limit and zero spend the corner would just read "≈ $0.00 today", which
  // is noise — suppress it. Once spend or a limit exists, render.
  const hasSignal = corner.spent > 0 || corner.limit != null || corner.state === 'exhausted';
  if (cost == null && corner.state !== 'exhausted') return null;
  if (!hasSignal) return null;

  const windowWord = t(WINDOW_WORD[window]);
  const spentStr = formatMoney(corner.spent, corner.currency, locale);
  const limitStr =
    corner.limit != null ? formatMoney(corner.limit, corner.currency, locale) : '';

  if (corner.state === 'exhausted') {
    return (
      <button
        type="button"
        onClick={onOpenBudgetSettings}
        data-testid="budget-corner"
        data-state="exhausted"
        aria-label={t('budget.corner.aria')}
        className="font-mono text-[11px] font-medium leading-none px-2 py-1 rounded-full text-error bg-error/10 hover:bg-error/15 transition-colors duration-150 max-md:min-h-[44px] max-md:flex max-md:items-center"
      >
        {t('budget.corner.exhausted')}
      </button>
    );
  }

  if (corner.state === 'warn') {
    return (
      <button
        type="button"
        onClick={onOpenBudgetSettings}
        data-testid="budget-corner"
        data-state="warn"
        aria-label={t('budget.corner.aria')}
        className="font-mono text-[11px] text-warning hover:underline transition-colors duration-150 max-md:min-h-[44px] max-md:flex max-md:items-center"
      >
        {t('budget.corner.warn', { spent: spentStr, limit: limitStr, window: windowWord })}
      </button>
    );
  }

  // normal
  return (
    <button
      type="button"
      onClick={onOpenBudgetSettings}
      data-testid="budget-corner"
      data-state="normal"
      aria-label={t('budget.corner.aria')}
      className="font-mono text-[11px] text-secondary hover:text-primary transition-colors duration-150 max-md:min-h-[44px] max-md:flex max-md:items-center"
    >
      {t('budget.corner.normal', { spent: spentStr, window: windowWord })}
    </button>
  );
}
