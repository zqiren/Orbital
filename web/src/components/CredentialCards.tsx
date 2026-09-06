// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Global Settings → LLM providers (spec 082 §3.9).
 *
 * One flat list, in the daemon's order (env card first, then last-used desc).
 * Deliberately NOT grouped by provider: a card is a *working setup*, and the
 * same provider legitimately appears many times (one card per model), so the
 * provider is a property of a row rather than a chapter heading (D10).
 *
 * The list itself is `CardList`, shared verbatim with project settings — same
 * rows, same highlight, same three actions. This file owns only what is
 * ABOVE the list: the intro and "Add provider". Selecting a provider here
 * sets the global default; `CardList` makes that call.
 *
 * TokenDance one-click is deliberately NOT here. It is a provider like the
 * seventeen others, so it belongs inside the Add flow next to the key field
 * it replaces (`LLMProviderSettings`, provider === 'tokendance'), where it is
 * invisible to everyone who cannot use it instead of being a top-level button
 * they must first understand in order to ignore. Prominence for the market
 * that wants it is a RANKING decision, not a visibility one: `orderProviders`
 * sorts TokenDance first for the zh locale. A wrong locale guess then costs a
 * scroll, not a dead end.
 *
 * "Add provider" opens the existing provider form as a modal. Its Save posts the
 * card and the daemon runs the connection test as part of that — a failed
 * test still saves, shown red on the row, so a provider outage never blocks
 * saving (D9). The modal therefore does not close itself: the verdict is the
 * thing the user opened it for.
 */
import { useState } from 'react';
import { Plus } from 'lucide-react';
import type { ProviderRegistry } from '../types';
import { useCredentialCards } from '../hooks/useCredentialCards';
import { useT } from '../i18n/useT';
import CardList, { CardFormModal } from './CardList';

interface CredentialCardsProps {
  providers: ProviderRegistry;
}

export default function CredentialCards({ providers }: CredentialCardsProps) {
  const t = useT();
  const { cards, defaultCardId, loading, refresh, applyCard } = useCredentialCards();

  const [adding, setAdding] = useState(false);
  return (
    <div data-testid="credential-cards">
      <p className="text-[13px] leading-relaxed text-secondary mb-4">{t('cards.intro')}</p>

      {/* List-level actions. Per-card actions live on the card, and there are
          only ever three of those. */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <button
          type="button"
          data-testid="cards-add"
          onClick={() => setAdding(true)}
          className="inline-flex items-center gap-1.5 bg-accent text-white text-sm font-medium rounded-lg px-4 py-2 hover:bg-accent/90 transition-all duration-150 max-md:min-h-[44px]"
        >
          <Plus className="w-4 h-4" />
          {t('cards.add')}
        </button>
      </div>
      <CardList
        mode="global"
        cards={cards}
        defaultCardId={defaultCardId}
        providers={providers}
        loading={loading}
        onRefresh={refresh}
        onCardUpdated={applyCard}
        data-testid="cards-list"
      />

      {adding && (
        <CardFormModal
          card={null}
          onClose={() => setAdding(false)}
          onSaved={() => refresh()}
        />
      )}
    </div>
  );
}
