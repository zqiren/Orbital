// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The global credential-card list (spec 082).
 *
 * Cards live in one flat, globally-ordered list served by `GET /settings`
 * alongside `default_card_id` — there is no separate list route, so this hook
 * reads the settings document and keeps only the two card fields. Every
 * surface that offers a card (the Credentials list, the project picker, the
 * fallback editor) mounts this; the daemon already serves settings to several
 * components per screen, so a small extra read is in keeping.
 *
 * Ordering is the daemon's (env card first, then last-used desc, then created
 * asc) and is never re-sorted here — the recency order is what keeps a long
 * list usable.
 */
import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import type { CredentialCard } from '../types';

interface CardsSettingsResponse {
  credential_cards?: CredentialCard[];
  default_card_id?: string | null;
}

export interface UseCredentialCards {
  cards: CredentialCard[];
  /** The EFFECTIVE default ('env' while AGENT_OS_API_KEY is set). */
  defaultCardId: string | null;
  loading: boolean;
  /** Re-read the list from the daemon. Every mutation ends with this. */
  refresh: () => Promise<void>;
  /** Splice one card in place after a PUT/POST/test, without a round trip. */
  applyCard: (card: CredentialCard) => void;
}

export function useCredentialCards(): UseCredentialCards {
  const [cards, setCards] = useState<CredentialCard[]>([]);
  const [defaultCardId, setDefaultCardId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const data = await api<CardsSettingsResponse>('/api/v2/settings');
      setCards(data.credential_cards ?? []);
      setDefaultCardId(data.default_card_id ?? null);
    } catch {
      // A pre-082 daemon (or an unreachable one) simply has no cards; the
      // surfaces above render their empty state rather than an error wall.
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    api<CardsSettingsResponse>('/api/v2/settings')
      .then((data) => {
        if (cancelled) return;
        setCards(data.credential_cards ?? []);
        setDefaultCardId(data.default_card_id ?? null);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const applyCard = useCallback((card: CredentialCard) => {
    setCards((prev) => {
      const idx = prev.findIndex((c) => c.id === card.id);
      if (idx === -1) return [...prev, card];
      const next = [...prev];
      next[idx] = card;
      return next;
    });
  }, []);

  return { cards, defaultCardId, loading, refresh, applyCard };
}

/** The card a `card_id` resolves to, following `null` to the default. */
export function resolveCard(
  cards: CredentialCard[],
  cardId: string | null | undefined,
  defaultCardId: string | null,
): CredentialCard | undefined {
  if (cardId) return cards.find((c) => c.id === cardId);
  return cards.find((c) => c.id === defaultCardId) ?? cards.find((c) => c.is_default);
}
