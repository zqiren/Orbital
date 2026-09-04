// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Picks one credential card (spec 082 §3.9).
 *
 * The single control that replaced project settings' four-field LLM override
 * and each fallback row's provider/model/key trio: a project (or a fallback
 * rung) no longer stores a provider, a model, an endpoint and a key that the
 * daemon has to pair back together — it stores one card id.
 *
 * The first option, "Global default — <card name>", is `card_id: null`, which
 * is a *reference* to whatever the default is at run time, not a copy of the
 * default's id. Cards carrying a `last_error` stay selectable and are flagged
 * instead (spec 072 D2: errors are shown, never acted on).
 */
import { useMemo } from 'react';
import type { CredentialCard } from '../types';
import { cardHealth, cardOptionLabel } from '../utils/cardHealth';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import { translate } from '../i18n/useT';
import Select from './Select';

/** Sentinel for the "follow the global default" option — `null` cannot be an
 *  <option> value, and '' would collide with a card whose id is missing. */
export const GLOBAL_DEFAULT_VALUE = '__global_default__';

interface CardPickerProps {
  cards: CredentialCard[];
  /** The EFFECTIVE default card id, used to name the first option. */
  defaultCardId: string | null;
  /** Currently referenced card; `null`/undefined = follow the global default. */
  value: string | null | undefined;
  onChange: (cardId: string | null) => void;
  /** Offer "Global default" (project settings). Fallback rungs name a card. */
  allowGlobalDefault?: boolean;
  disabled?: boolean;
  /** Hide the health line under the control (compact fallback rows). */
  hideHealth?: boolean;
  label?: string;
  'data-testid'?: string;
}

export default function CardPicker({
  cards,
  defaultCardId,
  value,
  onChange,
  allowGlobalDefault = true,
  disabled,
  hideHealth,
  label,
  'data-testid': testId,
}: CardPickerProps) {
  const t = useT();
  const { locale } = useLocale();
  // Bound to `locale`, not to the unstable `t` — the CLAUDE.md rule for
  // threading a translator into module-level helpers.
  const tr = useMemo(
    () => (key: Parameters<typeof t>[0], vars?: Parameters<typeof t>[1]) =>
      translate(locale, key, vars),
    [locale],
  );

  const defaultCard = cards.find((c) => c.id === defaultCardId) ?? cards.find((c) => c.is_default);
  const selected = value ? cards.find((c) => c.id === value) : undefined;
  // A pinned id with no matching card is a card that was deleted out from
  // under this referrer; the daemon repoints on delete, but a stale render
  // must not silently look like "global default".
  const dangling = !!value && !selected;

  // The card actually used at run time: the pinned one, else the default.
  const effective = selected ?? (value ? undefined : defaultCard);
  const health = effective ? cardHealth(effective, tr) : null;

  const selectValue = value ?? (allowGlobalDefault ? GLOBAL_DEFAULT_VALUE : '');

  return (
    <div>
      {label && (
        <label className="block text-sm font-medium text-primary mb-1.5">{label}</label>
      )}
      <Select
        value={dangling ? '' : selectValue}
        disabled={disabled}
        data-testid={testId}
        aria-label={label ?? t('cards.picker.aria')}
        onChange={(e) => {
          const v = e.target.value;
          onChange(v === GLOBAL_DEFAULT_VALUE || v === '' ? null : v);
        }}
        className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150 disabled:opacity-50"
      >
        {allowGlobalDefault && (
          <option value={GLOBAL_DEFAULT_VALUE}>
            {defaultCard
              ? t('cards.picker.globalDefaultNamed', { name: defaultCard.name })
              : t('cards.picker.globalDefaultNone')}
          </option>
        )}
        {!allowGlobalDefault && !value && (
          <option value="">{t('cards.picker.choose')}</option>
        )}
        {dangling && <option value="">{t('cards.picker.missing')}</option>}
        {cards.map((card) => (
          <option key={card.id} value={card.id}>
            {cardOptionLabel(card, tr)}
          </option>
        ))}
      </Select>

      {!hideHealth && health && (
        <p
          data-testid={testId ? `${testId}-health` : undefined}
          className={`text-xs mt-1 ${
            health.kind === 'error'
              ? 'text-error'
              : health.kind === 'verified'
                ? 'text-secondary'
                : 'text-secondary/70'
          }`}
        >
          {health.text}
        </p>
      )}
      {!hideHealth && cards.length === 0 && (
        <p className="text-xs text-secondary/70 mt-1">{t('cards.picker.empty')}</p>
      )}
    </div>
  );
}
