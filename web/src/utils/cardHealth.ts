// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The one-line health summary a credential card shows wherever it is listed
 * (spec 082 §3.5): "Verified 2h ago", "402 Insufficient credits · 22:53",
 * "Unverified".
 *
 * Health is displayed, never acted on — a red card is still selectable, and
 * nothing stored changes because of an error (spec 072 D2).
 *
 * Module-level, so it takes an optional translator that defaults to English
 * (the CLAUDE.md rule for non-React code): a caller that omits it — a test, a
 * label built outside a component — gets byte-identical English output.
 */
import { translate } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';
import type { TVars } from '../i18n/useT';
import type { CredentialCard } from '../types';

export type CardTranslate = (key: StringKey, vars?: TVars) => string;

const EN: CardTranslate = (key, vars) => translate('en', key, vars);

export type CardHealthKind = 'verified' | 'error' | 'unverified';

export interface CardHealth {
  kind: CardHealthKind;
  text: string;
}

/** Compact "how long ago" — no plural forms needed: m/h/d are unit symbols. */
export function relativeTime(
  iso: string,
  t: CardTranslate = EN,
  now: number = Date.now(),
): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return iso;
  const seconds = Math.max(0, Math.floor((now - then) / 1000));
  if (seconds < 60) return t('cards.time.justNow');
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return t('cards.time.minutes', { n: minutes });
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return t('cards.time.hours', { n: hours });
  const days = Math.floor(hours / 24);
  if (days < 30) return t('cards.time.days', { n: days });
  return new Date(then).toLocaleDateString();
}

function clockTime(iso: string): string {
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return iso;
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * `last_error` wins over `verified_at`: a card that worked this morning and
 * 401'd this afternoon is a broken card, and the error is what the user came
 * to the list to see. The provider's own sentence is shown verbatim (backend
 * text is never translated); only the frame around it goes through t().
 */
export function cardHealth(card: CredentialCard, t: CardTranslate = EN): CardHealth {
  const err = card.last_error;
  if (err) {
    const message = err.message || t('cards.health.genericError');
    return {
      kind: 'error',
      text:
        err.status != null
          ? t('cards.health.errorWithStatus', {
              status: err.status,
              message,
              time: clockTime(err.at),
            })
          : t('cards.health.error', { message, time: clockTime(err.at) }),
    };
  }
  if (card.verified_at) {
    return {
      kind: 'verified',
      text: t('cards.health.verified', { when: relativeTime(card.verified_at, t) }),
    };
  }
  return { kind: 'unverified', text: t('cards.health.unverified') };
}

/**
 * A card's one-line identity for a picker option: name, model and key suffix.
 * A card with no model is the migrated "needs a model" shape — say so, since
 * such a card cannot run until it is edited.
 */
export function cardOptionLabel(card: CredentialCard, t: CardTranslate = EN): string {
  const parts = [card.name];
  parts.push(card.model || t('cards.needsModel'));
  if (card.key_masked) parts.push(card.key_masked);
  const label = parts.join(' · ');
  return card.last_error ? t('cards.option.flagged', { label }) : label;
}
