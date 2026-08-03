// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useMemo } from 'react';
import { api } from '../config';
import type { BudgetResetOutcome, Project } from '../types';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import type { StringKey } from '../i18n/strings';
import { useCost } from '../hooks/useCost';
import { formatMoney, formatTokens } from '../budget/format';
import {
  assembleRows,
  deriveFootnote,
  deriveMeter,
  periodToWindow,
} from '../budget/derive';

type BudgetPeriod = 'daily' | 'weekly' | 'monthly' | 'total';

interface BudgetSectionProps {
  project: Project;
  /** Live limit string (parent owns it so Save can read it). */
  limit: string;
  onLimitChange: (next: string) => void;
  currency: string;
  onCurrencyChange: (next: string) => void;
  period: BudgetPeriod;
  onPeriodChange: (next: BudgetPeriod) => void;
  action: 'pause' | 'stop';
  onActionChange: (next: 'pause' | 'stop') => void;
  /** Navigate to the pricing editor. */
  onEditPricing: () => void;
  /**
   * Bumped after a pricing-table save so useCost re-fetches /cost and the meter
   * + breakdown recompute historical spend against the new rates.
   */
  costRefreshKey?: number;
}

const PERIOD_OPTIONS: { value: BudgetPeriod; labelKey: StringKey; hintKey: StringKey }[] = [
  { value: 'daily', labelKey: 'settings.budget.period.day', hintKey: 'settings.budget.hint.daily' },
  { value: 'weekly', labelKey: 'settings.budget.period.week', hintKey: 'settings.budget.hint.weekly' },
  { value: 'monthly', labelKey: 'settings.budget.period.month', hintKey: 'settings.budget.hint.monthly' },
  { value: 'total', labelKey: 'settings.budget.period.total', hintKey: 'settings.budget.hint.total' },
];

const WINDOW_WORD: Record<BudgetPeriod, StringKey> = {
  daily: 'settings.budget.window.day',
  weekly: 'settings.budget.window.week',
  monthly: 'settings.budget.window.month',
  total: 'settings.budget.window.total',
};

// A small, common ISO set for the currency select. The data layer is ISO
// codes; the label renders the code (the symbol is applied by the formatter
// when amounts are shown).
const CURRENCIES = ['USD', 'CNY', 'EUR', 'GBP', 'JPY'];

/** YYYY-MM-DD for the anchor/reset labels (locale-neutral, ISO date only). */
function anchorDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toISOString().slice(0, 10);
}

export default function BudgetSection({
  project,
  limit,
  onLimitChange,
  currency,
  onCurrencyChange,
  period,
  onPeriodChange,
  action,
  onActionChange,
  onEditPricing,
  costRefreshKey,
}: BudgetSectionProps) {
  const t = useT();
  const { locale } = useLocale();
  const pid = project.project_id;

  const window = periodToWindow(period);
  const { cost, loading, error, refresh } = useCost(pid, window, costRefreshKey);

  // Until /cost has ever landed, a rendered number would be a fabricated zero
  // rather than a measurement (bug #36). "Pending" covers the mount frame
  // BEFORE useCost's effect flips `loading` as well as the in-flight window.
  // A refresh with `cost` already present keeps showing the last known spend
  // instead of blinking back to a placeholder.
  const pendingFirstLoad = !cost && (loading || error == null);
  const failedWithNoData = !cost && !loading && error != null;
  const spendUnknown = pendingFirstLoad || failedWithNoData;

  const limitNum = limit.trim() !== '' ? parseFloat(limit) : null;
  const validLimit = limitNum != null && !Number.isNaN(limitNum) ? limitNum : null;

  const meter = useMemo(() => deriveMeter(cost, validLimit), [cost, validLimit]);
  const rows = useMemo(() => assembleRows(cost), [cost]);

  // Currency the spend is denominated in for display (the converted total's
  // currency = the budget currency).
  const displayCcy = cost?.converted_total.currency ?? currency;

  // Footnote: estimate marker + FX rate (only on a real cross-currency
  // conversion) + the sub-agents clause (only when sub-agent rows exist).
  const footnote = useMemo(() => deriveFootnote(cost), [cost]);
  const footnoteSegments = useMemo(() => {
    const segs: string[] = [];
    if (footnote.estimated) segs.push(t('settings.budget.footnote.estimated'));
    if (footnote.fxRate != null) {
      segs.push(t('settings.budget.footnote.fxRate', {
        quote: formatMoney(footnote.fxRate, 'CNY', locale),
        base: formatMoney(1, 'USD', locale, 0),
      }));
    }
    if (footnote.showSubagents) segs.push(t('settings.budget.footnote.subagents'));
    return segs;
    // `t` is unstable; bind to locale per the i18n convention.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [footnote, locale]);

  const periodHintKey = PERIOD_OPTIONS.find((p) => p.value === period)!.hintKey;
  const periodHint =
    period === 'total'
      ? t('settings.budget.hint.total', { date: anchorDate(project.budget_anchor_ts) })
      : t(periodHintKey);

  async function handleReset() {
    if (!confirm(t('settings.budget.resetConfirm'))) return;
    try {
      const resp = await api<{ budget_reset?: BudgetResetOutcome }>(
        `/api/v2/projects/${encodeURIComponent(pid)}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reset_budget_anchor: true }),
        },
      );
      // Only a total-mode reset applies; refresh the meter to reflect the new
      // window. A non-total no-op (code: not_total_mode) leaves spend as-is.
      if (resp.budget_reset?.applied) {
        await refresh();
      }
    } catch {
      // silently ignore — the reset link is best-effort
    }
  }

  const meterText =
    validLimit == null
      ? t('settings.budget.meter.noLimit', {
          window: t(WINDOW_WORD[period]),
          spent: formatMoney(meter.spent, displayCcy, locale),
        })
      : t('settings.budget.meter.withLimit', {
          window: t(WINDOW_WORD[period]),
          spent: formatMoney(meter.spent, displayCcy, locale),
          limit: formatMoney(validLimit, displayCcy, locale),
        });

  const barColor =
    meter.state === 'danger'
      ? 'bg-error'
      : meter.state === 'warn'
        ? 'bg-warning'
        : 'bg-accent';
  const meterTextColor =
    meter.state === 'danger'
      ? 'text-error'
      : meter.state === 'warn'
        ? 'text-warning'
        : 'text-primary';

  return (
    <div>
      <label className="block text-sm font-medium text-primary mb-2">
        {t('settings.budget.label')}
      </label>

      <div className="space-y-4">
        {/* Limit-as-sentence row */}
        <div className="flex items-center gap-2 flex-wrap text-sm text-primary">
          <span>{t('settings.budget.sentence.prefix')}</span>
          <input
            type="number"
            step="0.01"
            min="0"
            value={limit}
            onChange={(e) => onLimitChange(e.target.value)}
            placeholder={t('createProject.budget.placeholder')}
            aria-label={t('createProject.budget.label')}
            className="w-24 text-sm bg-sidebar border border-border rounded-lg px-2 py-1.5 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
          />
          <select
            value={currency}
            onChange={(e) => onCurrencyChange(e.target.value)}
            aria-label={t('settings.budget.aria.currency')}
            className="text-sm bg-sidebar border border-border rounded-lg px-2 py-1.5 text-primary focus:outline-none focus:border-accent transition-all duration-150"
          >
            {/* Ensure the project's set-once currency is always an option even
                if it falls outside the common list. */}
            {(CURRENCIES.includes(currency) ? CURRENCIES : [currency, ...CURRENCIES]).map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <span>{t('settings.budget.sentence.suffix')}</span>
          <select
            value={period}
            onChange={(e) => onPeriodChange(e.target.value as BudgetPeriod)}
            aria-label={t('settings.budget.aria.period')}
            className="text-sm bg-sidebar border border-border rounded-lg px-2 py-1.5 text-primary focus:outline-none focus:border-accent transition-all duration-150"
          >
            {PERIOD_OPTIONS.map((p) => (
              <option key={p.value} value={p.value}>{t(p.labelKey)}</option>
            ))}
          </select>
        </div>

        {/* Period hint + reset (reset only in total mode) */}
        <div className="flex items-center gap-3 flex-wrap">
          <p className="text-xs text-secondary">{periodHint}</p>
          {period === 'total' && (
            <button
              type="button"
              onClick={handleReset}
              className="text-xs text-secondary hover:text-primary underline transition-colors duration-150"
            >
              {t('settings.budget.reset', { date: anchorDate(project.budget_anchor_ts) })}
            </button>
          )}
        </div>

        {/* Behavior cards */}
        <div>
          <p className="text-xs text-secondary mb-2">{t('settings.budget.behavior.label')}</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {([
              { value: 'pause', titleKey: 'settings.budget.behavior.pause.title', descKey: 'settings.budget.behavior.pause.desc', accent: true },
              { value: 'stop', titleKey: 'settings.budget.behavior.stop.title', descKey: 'settings.budget.behavior.stop.desc', accent: false },
            ] as const).map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => onActionChange(opt.value)}
                className={`text-left border rounded-lg p-3 transition-all duration-150 max-md:min-h-[44px] ${
                  action === opt.value
                    ? 'border-accent bg-accent/5'
                    : 'border-border hover:border-secondary/40'
                }`}
              >
                <span className="text-sm font-medium text-primary block">{t(opt.titleKey)}</span>
                <span className="text-xs text-secondary mt-1 block">{t(opt.descKey)}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Spend meter */}
        <div>
          {pendingFirstLoad ? (
            <p className="text-sm text-secondary mb-1.5">{t('settings.budget.meter.loading')}</p>
          ) : failedWithNoData ? (
            <p className="text-sm text-error mb-1.5">{t('settings.budget.meter.loadError')}</p>
          ) : (
            <p className={`text-sm font-medium mb-1.5 ${meterTextColor}`}>{meterText}</p>
          )}
          {!spendUnknown && validLimit != null && (
            <div className="h-2 w-full bg-sidebar border border-border rounded-full overflow-hidden">
              <div
                className={`h-full ${barColor} transition-all duration-300`}
                style={{ width: `${Math.round(meter.fraction * 100)}%` }}
                role="progressbar"
                aria-valuenow={Math.round(meter.fraction * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              />
            </div>
          )}
        </div>

        {/* Breakdown table */}
        <div>
          {rows.length === 0 ? (
            // "Nothing recorded" is a claim about the data, so it waits for the
            // data — while /cost is unresolved the meter line carries the state.
            spendUnknown ? null : (
              <p className="text-xs text-secondary/60 italic">{t('settings.budget.breakdown.empty')}</p>
            )
          ) : (
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="text-secondary text-left">
                  <th className="font-normal py-1 pr-2">{/* model name col */}</th>
                  <th className="font-normal py-1 px-2 text-right">{t('settings.budget.breakdown.input')}</th>
                  <th className="font-normal py-1 px-2 text-right">{t('settings.budget.breakdown.cached')}</th>
                  <th className="font-normal py-1 px-2 text-right">{t('settings.budget.breakdown.output')}</th>
                  <th className="font-normal py-1 pl-2 text-right">{t('settings.budget.breakdown.cost')}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.key} className="border-t border-border/50">
                    <td className="py-1 pr-2 text-primary truncate max-w-[160px]">
                      {r.model}
                      {r.isSubagent && (
                        <span className="text-secondary/60"> · {r.provider}</span>
                      )}
                    </td>
                    <td className="py-1 px-2 text-right text-secondary tabular-nums">{formatTokens(r.inputTokens, locale)}</td>
                    <td className="py-1 px-2 text-right text-secondary tabular-nums">{formatTokens(r.cachedTokens, locale)}</td>
                    <td className="py-1 px-2 text-right text-secondary tabular-nums">{formatTokens(r.outputTokens, locale)}</td>
                    <td className="py-1 pl-2 text-right text-primary tabular-nums">
                      {r.cost
                        ? formatMoney(r.cost.amount, r.cost.currency, locale)
                        : <span className="text-secondary/60 italic">{t('settings.budget.breakdown.subscription')}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* One footnote line: estimate marker (per converted_total.estimated)
              + the FX rate actually used (only on a real cross-currency
              conversion) + the sub-agents-not-counted clause (only when
              sub-agent rows exist). Hidden entirely under an empty table. */}
          {footnoteSegments.length > 0 && (
            <p className="text-[11px] text-secondary/60 mt-2">
              {footnoteSegments.join(' · ')}
            </p>
          )}
        </div>

        {/* Pricing editor link */}
        <button
          type="button"
          onClick={onEditPricing}
          data-pricing-editor-link
          className="text-xs text-accent hover:underline transition-colors duration-150"
        >
          {t('settings.budget.editPricing')}
        </button>
      </div>
    </div>
  );
}
