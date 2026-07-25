// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useMemo, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import { api, ApiError } from '../config';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import type { StringKey } from '../i18n/strings';
import {
  RATE_FIELDS,
  buildOverrideDoc,
  cellKey,
  effectiveCell,
  effectiveCurrency,
  emptyStaged,
  errorCodeToKey,
  hasStagedChanges,
  validateRateInput,
  validationReasonToKey,
  type PricingTable,
  type RateField,
  type StagedEdits,
} from '../budget/pricing';

interface PricingEditorProps {
  /**
   * Called after a successful PUT. The host bumps a cost-refresh nonce so the
   * Budget meter/breakdown re-fetch GET /cost against the NEW rates — that is
   * how "editing a rate visibly recomputes historical cost" is realized
   * (no polling). Optional so the component renders standalone in tests.
   */
  onSaved?: () => void;
}

const CURRENCY_OPTIONS = ['USD', 'CNY', 'EUR', 'GBP', 'JPY'];

// i18n labels for the four rate columns, in RATE_FIELDS order.
const FIELD_LABEL: Record<RateField, StringKey> = {
  input_per_1m: 'pricing.field.input',
  cached_input_per_1m: 'pricing.field.cachedInput',
  cache_write_per_1m: 'pricing.field.cacheWrite',
  output_per_1m: 'pricing.field.output',
};

/** Render `_default` with a clear label; otherwise the model id verbatim. */
function modelLabel(model: string): string | null {
  return model === '_default' ? null : model;
}

export default function PricingEditor({ onSaved }: PricingEditorProps) {
  const t = useT();
  const { locale } = useLocale();

  const [table, setTable] = useState<PricingTable | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [staged, setStaged] = useState<StagedEdits>(emptyStaged);
  // Per-cell client-side validation messages, keyed by cellKey.
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  // Raw text the user has typed into a cell, keyed by cellKey. The input is
  // controlled by this map (with a resolved-value fallback) so typing never
  // remounts/jumps the caret; `staged.rates` holds only the parsed numbers.
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const fetchTable = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const data = await api<PricingTable>('/api/v2/pricing');
      setTable(data);
    } catch (e) {
      setLoadError(e instanceof ApiError ? e.detail : t('pricing.loadError'));
    } finally {
      setLoading(false);
    }
    // `t` is unstable (rebinds per locale); the load message is resolved at
    // call time so binding to it would needlessly re-fetch on language switch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    void fetchTable();
  }, [fetchTable]);

  const providerNames = useMemo(
    () => (table ? Object.keys(table.providers).sort() : []),
    [table],
  );

  const dirty = hasStagedChanges(staged);
  const hasFieldErrors = Object.keys(fieldErrors).length > 0;

  // --- staged-edit mutators -------------------------------------------------

  function stageRate(provider: string, model: string, field: RateField, raw: string) {
    const k = cellKey(provider, model, field);
    setSaved(false);
    // Always record the raw text so the controlled input reflects keystrokes.
    setDrafts((prev) => ({ ...prev, [k]: raw }));

    // Empty input clears any staged edit for this cell (back to resolved value)
    // and clears the per-cell error.
    if (raw.trim() === '') {
      setStaged((prev) => {
        const rates = { ...prev.rates };
        delete rates[k];
        return { ...prev, rates };
      });
      clearFieldError(k);
      return;
    }

    const result = validateRateInput(raw);
    if (!result.ok) {
      setFieldErrors((prev) => ({
        ...prev,
        [k]: t(validationReasonToKey(result.reason) as StringKey),
      }));
      return;
    }
    clearFieldError(k);
    setStaged((prev) => {
      const rates = { ...prev.rates, [k]: result.value };
      const reverts = { ...prev.reverts };
      delete reverts[k]; // typing a value cancels a pending revert
      return { ...prev, rates, reverts };
    });
  }

  function stageRevert(provider: string, model: string, field: RateField) {
    const k = cellKey(provider, model, field);
    setSaved(false);
    setStaged((prev) => {
      const rates = { ...prev.rates };
      delete rates[k];
      return { ...prev, rates, reverts: { ...prev.reverts, [k]: true } };
    });
    // Drop any raw draft so the displayed value falls back to the resolved one.
    setDrafts((prev) => {
      const next = { ...prev };
      delete next[k];
      return next;
    });
    clearFieldError(k);
  }

  function clearFieldError(k: string) {
    setFieldErrors((prev) => {
      if (!(k in prev)) return prev;
      const next = { ...prev };
      delete next[k];
      return next;
    });
  }

  function stageCurrency(provider: string, value: string) {
    setSaved(false);
    setStaged((prev) => ({
      ...prev,
      currencies: { ...prev.currencies, [provider]: value },
    }));
  }

  // --- save -----------------------------------------------------------------

  async function handleSave() {
    if (!table || hasFieldErrors) return;
    setSaving(true);
    setSaveError(null);
    setSaved(false);
    const body = buildOverrideDoc(table, staged);
    try {
      // PUT returns the freshly-resolved table (same shape as GET); adopt it so
      // origins/values reflect the persisted state without a second round-trip.
      const next = await api<PricingTable>('/api/v2/pricing/overrides', {
        method: 'PUT',
        body: JSON.stringify(body),
      });
      setTable(next);
      setStaged(emptyStaged());
      setDrafts({});
      setFieldErrors({});
      setSaved(true);
      // Recompute trigger: the host refetches GET /cost so the meter/breakdown
      // reflect the new rates against historical token usage.
      onSaved?.();
    } catch (e) {
      if (e instanceof ApiError) {
        // The 400 body is codes-only ({"code": "..."}); api() surfaces it as
        // `detail`. Map the code to a localized message.
        const code = parseErrorCode(e.detail);
        setSaveError(t(errorCodeToKey(code) as StringKey));
      } else {
        setSaveError(t('pricing.error.generic'));
      }
    } finally {
      setSaving(false);
    }
  }

  // --- render ---------------------------------------------------------------

  if (loading) {
    return <p className="text-sm text-secondary px-6 py-8">{t('pricing.loading')}</p>;
  }
  if (loadError || !table) {
    return (
      <div className="px-6 py-8">
        <p className="text-sm text-error mb-3">{loadError ?? t('pricing.loadError')}</p>
        <button
          type="button"
          onClick={() => void fetchTable()}
          className="text-sm font-medium text-accent hover:underline"
        >
          {t('pricing.retry')}
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-[860px] mx-auto py-6 px-6 max-md:px-4">
      <p className="text-sm text-secondary mb-5">{t('pricing.intro')}</p>

      <div className="space-y-8">
        {providerNames.map((provider) => {
          const ptable = table.providers[provider];
          const curr = effectiveCurrency(table, staged, provider);
          const models = Object.keys(ptable.models).sort(sortModels);
          return (
            <section key={provider} data-pricing-provider={provider}>
              {/* Provider header + currency select */}
              <div className="flex items-center justify-between gap-3 mb-2 flex-wrap">
                <h3 className="text-sm font-semibold text-primary font-mono">{provider}</h3>
                <label className="flex items-center gap-2 text-xs text-secondary">
                  {t('pricing.currency.label')}
                  <select
                    value={curr.value}
                    onChange={(e) => stageCurrency(provider, e.target.value)}
                    aria-label={t('pricing.currency.aria', { provider })}
                    className={`text-xs bg-sidebar border rounded-lg px-2 py-1 text-primary focus:outline-none focus:border-accent transition-all duration-150 ${
                      curr.origin === 'override' ? 'border-accent' : 'border-border'
                    }`}
                  >
                    {(CURRENCY_OPTIONS.includes(curr.value)
                      ? CURRENCY_OPTIONS
                      : [curr.value, ...CURRENCY_OPTIONS]
                    ).map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                  {curr.origin === 'override' && (
                    <span className="text-2xs text-accent uppercase tracking-wide">
                      {t('pricing.badge.override')}
                    </span>
                  )}
                </label>
              </div>

              {/* Per-model 4-tier grid */}
              <div className="overflow-x-auto">
                <table className="w-full text-xs border-collapse min-w-[560px]">
                  <thead>
                    <tr className="text-secondary text-left">
                      <th className="font-normal py-1.5 pr-3">{t('pricing.col.model')}</th>
                      {RATE_FIELDS.map((f) => (
                        <th key={f} className="font-normal py-1.5 px-2 text-right">
                          {t(FIELD_LABEL[f])}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((model) => (
                      <tr key={model} className="border-t border-border/50 align-top">
                        <td className="py-1.5 pr-3 text-primary font-mono truncate max-w-[180px]">
                          {modelLabel(model) ?? (
                            <span className="italic text-secondary">
                              {t('pricing.model.default')}
                            </span>
                          )}
                        </td>
                        {RATE_FIELDS.map((field) => {
                          const cell = effectiveCell(table, staged, provider, model, field);
                          const k = cellKey(provider, model, field);
                          const isOverride = cell.origin === 'override';
                          const err = fieldErrors[k];
                          return (
                            <td key={field} className="py-1.5 px-2">
                              <div className="flex items-center justify-end gap-1">
                                <input
                                  type="number"
                                  inputMode="decimal"
                                  step="0.01"
                                  min="0"
                                  // Controlled by the raw-draft map (caret-stable),
                                  // falling back to the resolved/staged value. A
                                  // reverted cell shows its current value muted.
                                  value={k in drafts ? drafts[k] : String(cell.value)}
                                  onChange={(e) => stageRate(provider, model, field, e.target.value)}
                                  aria-label={t('pricing.cell.aria', {
                                    provider,
                                    model: modelLabel(model) ?? t('pricing.model.default'),
                                    field: t(FIELD_LABEL[field]),
                                  })}
                                  aria-invalid={err ? true : undefined}
                                  data-origin={cell.origin}
                                  className={`w-20 text-xs text-right tabular-nums bg-sidebar border rounded-md px-1.5 py-1 focus:outline-none focus:border-accent transition-all duration-150 ${
                                    err
                                      ? 'border-error text-error'
                                      : isOverride
                                        ? 'border-accent text-primary'
                                        : 'border-border text-secondary'
                                  }`}
                                />
                                {/* Revert control — only when this cell is an
                                    override (persisted or staged), not reverted. */}
                                {isOverride && (
                                  <button
                                    type="button"
                                    onClick={() => stageRevert(provider, model, field)}
                                    title={t('pricing.revert.title')}
                                    aria-label={t('pricing.revert.aria', {
                                      provider,
                                      field: t(FIELD_LABEL[field]),
                                    })}
                                    className="shrink-0 text-secondary hover:text-accent transition-colors p-0.5"
                                  >
                                    <RotateCcw size={12} />
                                  </button>
                                )}
                              </div>
                              {err && (
                                <p className="text-2xs text-error mt-0.5 text-right">{err}</p>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          );
        })}
      </div>

      {/* Legend */}
      <p className="text-[11px] text-secondary/70 mt-6">
        <span className="inline-block w-2 h-2 rounded-sm border border-accent align-middle mr-1" />
        {t('pricing.legend')}
      </p>

      {/* Override file path (read-only debug aid) */}
      <p className="text-[11px] text-secondary/50 mt-1 font-mono break-all">
        {t('pricing.overridePath', { path: table.override_path })}
      </p>

      {/* Save bar */}
      <div className="flex items-center gap-3 mt-6 pt-4 border-t border-border">
        <button
          type="button"
          onClick={() => void handleSave()}
          disabled={saving || !dirty || hasFieldErrors}
          className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px]"
        >
          {saving ? t('pricing.saving') : t('pricing.save')}
        </button>
        {saved && <span className="text-sm text-success">{t('pricing.saved')}</span>}
        {hasFieldErrors && !saveError && (
          <span className="text-sm text-error">{t('pricing.fixErrors')}</span>
        )}
        {saveError && <span className="text-sm text-error">{saveError}</span>}
      </div>

      {/* Locale-layer currency-symbol hint (the data layer stays ISO codes). */}
      <p className="text-[11px] text-secondary/50 mt-2">
        {t('pricing.currencyHint', { sample: sampleSymbol(curr0(table, staged, providerNames), locale) })}
      </p>
    </div>
  );
}

// Default-currency sample for the symbol hint footnote (first provider's
// effective currency, or USD). Pure helper kept module-scope.
function curr0(
  table: PricingTable,
  staged: StagedEdits,
  providerNames: string[],
): string {
  const first = providerNames[0];
  if (!first) return 'USD';
  return effectiveCurrency(table, staged, first).value;
}

function sampleSymbol(currency: string, locale: 'en' | 'zh'): string {
  try {
    const parts = new Intl.NumberFormat(locale === 'zh' ? 'zh-CN' : 'en-US', {
      style: 'currency',
      currency,
      currencyDisplay: 'narrowSymbol',
    }).formatToParts(0);
    return parts.find((p) => p.type === 'currency')?.value ?? currency;
  } catch {
    return currency;
  }
}

/** `_default` sorts last; everything else alphabetical (matches GET ordering). */
function sortModels(a: string, b: string): number {
  if (a === '_default') return 1;
  if (b === '_default') return -1;
  return a.localeCompare(b);
}

/**
 * Extract the backend error code from the api() detail string. The 400 body is
 * `{"code": "invalid_rate"}`; api() JSON-stringifies a non-string detail, so we
 * try to parse it back out, falling through to the raw string.
 */
function parseErrorCode(detail: string): string {
  try {
    const parsed = JSON.parse(detail);
    if (parsed && typeof parsed.code === 'string') return parsed.code;
  } catch {
    /* not JSON — fall through */
  }
  return detail;
}
