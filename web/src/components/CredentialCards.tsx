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
 * One-click sign-ins (TokenDance, ChatGPT) are deliberately NOT here. They
 * are providers like the others, so they live inside the Add flow as tiles
 * (`connect/ConnectPanel`), next to the key field they replace, instead of
 * top-level buttons everyone must first understand in order to ignore.
 *
 * "Add provider" opens the connect panel as a dialog. A key is saved only when
 * its connection test passes; a one-click sign-in stores its card itself.
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
      {/* The intro prose is the enclosing SettingsSection's ⓘ description
          (GlobalSettings) — hidden until asked for, like project settings. */}
      {/* List-level actions. Per-card actions live on the card, and there are
          only ever three of those. */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <button
          type="button"
          data-testid="cards-add"
          onClick={() => setAdding(true)}
          className="inline-flex items-center gap-1.5 bg-accent text-white text-body font-medium rounded-lg px-4 py-2 hover:bg-accent/90 transition-all duration-150 max-md:min-h-[44px]"
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
          onRemoved={() => { void refresh(); }}
        />
      )}
    </div>
  );
}
