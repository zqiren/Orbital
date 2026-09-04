// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Global Settings → Credential cards (spec 082 §3.9).
 *
 * One flat list, in the daemon's order (env card first, then last-used desc).
 * Deliberately NOT grouped by provider: a card is a *working setup*, and the
 * same provider legitimately appears many times (one card per model), so the
 * provider is a property of a row rather than a chapter heading (D10).
 *
 * The list itself is `CardList`, shared verbatim with project settings — same
 * rows, same highlight, same three actions. This file owns only what is
 * ABOVE the list: the intro, "Add card", and one-click sign-in. Selecting a
 * card here sets the global default; `CardList` makes that call.
 *
 * "Add card" opens the existing provider form as a modal. Its Save posts the
 * card and the daemon runs the connection test as part of that — a failed
 * test still saves, shown red on the row, so a provider outage never blocks
 * saving (D9). The modal therefore does not close itself: the verdict is the
 * thing the user opened it for.
 */
import { useState } from 'react';
import { Loader2, Plus } from 'lucide-react';
import type { ProviderRegistry, TokendanceSigninResponse } from '../types';
import { api } from '../config';
import { useCredentialCards } from '../hooks/useCredentialCards';
import { useT } from '../i18n/useT';
import watchaLogo from '../assets/watcha-logo.png';
import CardList, { CardFormModal } from './CardList';

interface CredentialCardsProps {
  providers: ProviderRegistry;
}

export default function CredentialCards({ providers }: CredentialCardsProps) {
  const t = useT();
  const { cards, defaultCardId, loading, refresh, applyCard } = useCredentialCards();

  const [adding, setAdding] = useState(false);
  const [tokendanceBusy, setTokendanceBusy] = useState(false);
  const [tokendanceMsg, setTokendanceMsg] = useState<{ kind: 'ok' | 'err'; text: string } | null>(
    null,
  );

  function errorText(err: unknown): string {
    return err instanceof Error && err.message ? err.message : t('cards.action.failed');
  }

  async function handleTokendance() {
    setTokendanceBusy(true);
    setTokendanceMsg(null);
    try {
      // No card_id: one-click always provisions a NEW TokenDance card here
      // and never touches another card's key (§3.6). Re-keying an existing
      // TokenDance card is the reconnect link inside that card's Edit form.
      const res = await api<TokendanceSigninResponse>(
        '/api/v2/providers/tokendance/signin',
        { method: 'POST', body: JSON.stringify({}) },
      );
      await refresh();
      setTokendanceMsg({
        kind: res.test && !res.test.ok ? 'err' : 'ok',
        text:
          res.test && !res.test.ok
            ? t('cards.save.savedWithError', { message: res.test.message })
            : t('llm.tokendance.signin.success'),
      });
    } catch (err: unknown) {
      setTokendanceMsg({ kind: 'err', text: errorText(err) });
    } finally {
      setTokendanceBusy(false);
    }
  }

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
        <button
          type="button"
          data-testid="cards-tokendance"
          onClick={handleTokendance}
          disabled={tokendanceBusy}
          className="inline-flex items-center gap-2 text-sm font-medium rounded-lg px-4 py-2 bg-card text-primary border border-border hover:border-[#8bdc7e] hover:shadow-sm disabled:opacity-60 disabled:cursor-not-allowed transition-all duration-150 max-md:min-h-[44px]"
        >
          {tokendanceBusy ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <img src={watchaLogo} alt="" className="w-5 h-5 rounded-full" aria-hidden="true" />
          )}
          {tokendanceBusy
            ? t('llm.tokendance.signin.waiting')
            : t('llm.tokendance.signin.button')}
        </button>
      </div>
      {tokendanceMsg && (
        <p
          data-testid="cards-tokendance-msg"
          className={`text-xs mb-3 ${tokendanceMsg.kind === 'ok' ? 'text-success' : 'text-error'}`}
        >
          {tokendanceMsg.text}
        </p>
      )}

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
