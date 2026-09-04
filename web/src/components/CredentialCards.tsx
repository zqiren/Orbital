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
 * "Add card" opens the existing provider form as a modal. Its Save posts the
 * card and the daemon runs the connection test as part of that — a failed
 * test still saves, shown red on the row, so a provider outage never blocks
 * saving (D9). The modal therefore does not close itself: the verdict is the
 * thing the user opened it for.
 */
import { useMemo, useState } from 'react';
import { Check, Loader2, Plus, Star, Trash2, X, Zap } from 'lucide-react';
import type {
  CardDeleteResponse,
  CardMutationResponse,
  CredentialCard,
  ProviderRegistry,
  TokendanceSigninResponse,
} from '../types';
import { api } from '../config';
import { useCredentialCards } from '../hooks/useCredentialCards';
import { cardHealth } from '../utils/cardHealth';
import { useT, translate } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import watchaLogo from '../assets/watcha-logo.png';
import LLMProviderSettings from './LLMProviderSettings';

interface CredentialCardsProps {
  providers: ProviderRegistry;
}

/** A row's in-flight action, so only that row's control shows a spinner. */
type BusyKind = 'default' | 'test' | 'delete' | 'rename';

export default function CredentialCards({ providers }: CredentialCardsProps) {
  const t = useT();
  const { locale } = useLocale();
  const tr = useMemo(
    () => (key: Parameters<typeof t>[0], vars?: Parameters<typeof t>[1]) =>
      translate(locale, key, vars),
    [locale],
  );

  const { cards, defaultCardId, loading, refresh, applyCard } = useCredentialCards();

  // null = closed; { card: null } = Add; { card } = Edit.
  const [modal, setModal] = useState<{ card: CredentialCard | null } | null>(null);
  const [busy, setBusy] = useState<{ id: string; kind: BusyKind } | null>(null);
  const [rowError, setRowError] = useState<{ id: string; message: string } | null>(null);
  const [rowNote, setRowNote] = useState<{ id: string; message: string } | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; value: string } | null>(null);
  const [tokendanceBusy, setTokendanceBusy] = useState(false);
  const [tokendanceMsg, setTokendanceMsg] = useState<{ kind: 'ok' | 'err'; text: string } | null>(
    null,
  );

  function errorText(err: unknown): string {
    return err instanceof Error && err.message ? err.message : t('cards.action.failed');
  }

  async function setDefault(card: CredentialCard) {
    setBusy({ id: card.id, kind: 'default' });
    setRowError(null);
    setRowNote(null);
    try {
      const res = await api<{ default_card_id: string | null; applied: boolean }>(
        `/api/v2/settings/cards/${encodeURIComponent(card.id)}/default`,
        { method: 'PUT' },
      );
      // The env card outranks a stored default while AGENT_OS_API_KEY is set:
      // the choice IS recorded, it just is not what runs yet. Saying so beats
      // a click that silently appears to do nothing.
      if (!res.applied) setRowNote({ id: card.id, message: t('cards.default.envWins') });
      await refresh();
    } catch (err: unknown) {
      setRowError({ id: card.id, message: errorText(err) });
    } finally {
      setBusy(null);
    }
  }

  async function testCard(card: CredentialCard) {
    setBusy({ id: card.id, kind: 'test' });
    setRowError(null);
    setRowNote(null);
    try {
      const res = await api<CardMutationResponse>(
        `/api/v2/settings/cards/${encodeURIComponent(card.id)}/test`,
        { method: 'POST' },
      );
      applyCard(res.card);
      // HTTP 200 even for a failed test — the verdict is in the body, and the
      // refreshed card already carries verified_at / last_error for the line
      // under the name. The note is only the immediate "I just clicked that".
      if (res.test?.ok) {
        setRowNote({ id: card.id, message: res.test.message || t('cards.save.verified') });
      }
    } catch (err: unknown) {
      setRowError({ id: card.id, message: errorText(err) });
    } finally {
      setBusy(null);
    }
  }

  async function deleteCard(card: CredentialCard) {
    if (!confirm(t('cards.delete.confirm', { name: card.name }))) return;
    setBusy({ id: card.id, kind: 'delete' });
    setRowError(null);
    setRowNote(null);
    try {
      const res = await api<CardDeleteResponse>(
        `/api/v2/settings/cards/${encodeURIComponent(card.id)}`,
        { method: 'DELETE' },
      );
      // Deleting a referenced card moves those projects onto the default —
      // the one behaviour worth naming out loud (D5), so the affected
      // projects are listed rather than silently repointed.
      if (res.reassigned_projects?.length) {
        setRowNote({
          id: card.id,
          message: t('cards.delete.reassigned', {
            projects: res.reassigned_projects.map((p) => p.name).join(', '),
          }),
        });
      }
      await refresh();
    } catch (err: unknown) {
      setRowError({ id: card.id, message: errorText(err) });
    } finally {
      setBusy(null);
    }
  }

  async function commitRename(card: CredentialCard, name: string) {
    setRenaming(null);
    const next = name.trim();
    if (!next || next === card.name) return;
    setBusy({ id: card.id, kind: 'rename' });
    setRowError(null);
    try {
      // A name-only edit does not re-test (contract §3), so the health line
      // under a renamed card stays exactly as it was.
      const res = await api<CardMutationResponse>(
        `/api/v2/settings/cards/${encodeURIComponent(card.id)}`,
        { method: 'PUT', body: JSON.stringify({ name: next }) },
      );
      applyCard(res.card);
    } catch (err: unknown) {
      setRowError({ id: card.id, message: errorText(err) });
    } finally {
      setBusy(null);
    }
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

      {/* Actions above the list */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <button
          type="button"
          data-testid="cards-add"
          onClick={() => setModal({ card: null })}
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

      {/* The list */}
      {loading ? (
        <div className="flex items-center gap-2 text-sm text-secondary py-4">
          <Loader2 className="w-4 h-4 animate-spin" />
          {t('llm.loading')}
        </div>
      ) : cards.length === 0 ? (
        <p className="text-sm text-secondary/70 italic py-2" data-testid="cards-empty">
          {t('cards.empty')}
        </p>
      ) : (
        <ul className="space-y-2">
          {cards.map((card) => {
            const health = cardHealth(card, tr);
            const providerName = providers[card.provider]?.display_name || card.provider;
            const isDefault = card.is_default || card.id === defaultCardId;
            const rowBusy = busy?.id === card.id ? busy.kind : null;
            return (
              <li
                key={card.id}
                data-testid={`card-row-${card.id}`}
                className="border border-border rounded-lg px-3 py-2.5 bg-sidebar"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      {/* Provider mark: the registry ships no per-provider
                          icons, so the monogram stands in — it still tells two
                          rows apart at a glance, which is the job. */}
                      <span
                        aria-hidden="true"
                        className="shrink-0 w-5 h-5 rounded grid place-items-center bg-card border border-border text-[10px] font-semibold text-secondary"
                      >
                        {providerName.slice(0, 1).toUpperCase()}
                      </span>
                      {renaming?.id === card.id ? (
                        <input
                          autoFocus
                          value={renaming.value}
                          data-testid={`card-name-input-${card.id}`}
                          onChange={(e) => setRenaming({ id: card.id, value: e.target.value })}
                          onBlur={() => commitRename(card, renaming.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') commitRename(card, renaming.value);
                            if (e.key === 'Escape') setRenaming(null);
                          }}
                          className="text-sm font-medium bg-card border border-border rounded px-2 py-0.5 text-primary focus:outline-none focus:border-accent min-w-0 flex-1"
                        />
                      ) : (
                        <button
                          type="button"
                          disabled={card.read_only}
                          data-testid={`card-name-${card.id}`}
                          onClick={() => setRenaming({ id: card.id, value: card.name })}
                          title={card.read_only ? undefined : t('cards.rename.title')}
                          className="text-sm font-medium text-primary truncate min-w-0 text-left hover:text-accent disabled:hover:text-primary disabled:cursor-default transition-colors"
                        >
                          {card.name}
                        </button>
                      )}
                      {isDefault && (
                        <span
                          data-testid={`card-default-${card.id}`}
                          className="shrink-0 inline-flex items-center gap-1 text-[11px] rounded-full px-2 py-0.5 bg-accent/10 text-accent"
                        >
                          <Star className="w-3 h-3" />
                          {t('cards.badge.default')}
                        </span>
                      )}
                      {card.read_only && (
                        <span className="shrink-0 text-[11px] rounded-full px-2 py-0.5 bg-card border border-border text-secondary">
                          {t('cards.badge.env')}
                        </span>
                      )}
                      {card.region === 'china' && (
                        <span className="shrink-0 text-[11px] rounded-full px-2 py-0.5 bg-card border border-border text-secondary">
                          {t('llm.region.china')}
                        </span>
                      )}
                    </div>

                    <p className="text-xs text-secondary mt-1 truncate">
                      {providerName}
                      {' · '}
                      <span className="font-mono">{card.model || t('cards.needsModel')}</span>
                      {card.key_masked ? (
                        <>
                          {' · '}
                          <span className="font-mono">{card.key_masked}</span>
                        </>
                      ) : (
                        <> {' · '}{t('cards.noKey')}</>
                      )}
                    </p>

                    <p
                      data-testid={`card-health-${card.id}`}
                      className={`text-xs mt-0.5 ${
                        health.kind === 'error'
                          ? 'text-error'
                          : health.kind === 'verified'
                            ? 'text-success'
                            : 'text-secondary/70'
                      }`}
                    >
                      {health.text}
                    </p>

                    {rowNote?.id === card.id && (
                      <p className="text-xs text-secondary mt-1" data-testid={`card-note-${card.id}`}>
                        {rowNote.message}
                      </p>
                    )}
                    {rowError?.id === card.id && (
                      <p className="text-xs text-error mt-1" data-testid={`card-error-${card.id}`}>
                        {rowError.message}
                      </p>
                    )}
                  </div>

                  {/* Row actions */}
                  <div className="flex items-center gap-1 shrink-0">
                    {!isDefault && (
                      <button
                        type="button"
                        data-testid={`card-set-default-${card.id}`}
                        onClick={() => setDefault(card)}
                        disabled={rowBusy === 'default'}
                        title={t('cards.action.setDefault')}
                        className="text-xs text-secondary hover:text-accent transition-colors px-2 py-1 disabled:opacity-50 max-md:min-h-[44px]"
                      >
                        {rowBusy === 'default' ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          t('cards.action.setDefault')
                        )}
                      </button>
                    )}
                    <button
                      type="button"
                      data-testid={`card-test-${card.id}`}
                      onClick={() => testCard(card)}
                      disabled={rowBusy === 'test'}
                      title={t('cards.action.test')}
                      className="text-xs text-secondary hover:text-accent transition-colors px-2 py-1 disabled:opacity-50 max-md:min-h-[44px]"
                    >
                      {rowBusy === 'test' ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Zap className="w-3.5 h-3.5" />
                      )}
                    </button>
                    <button
                      type="button"
                      data-testid={`card-edit-${card.id}`}
                      onClick={() => setModal({ card })}
                      disabled={card.read_only}
                      title={t('cards.action.edit')}
                      className="text-xs text-secondary hover:text-accent transition-colors px-2 py-1 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px]"
                    >
                      {t('cards.action.edit')}
                    </button>
                    <button
                      type="button"
                      data-testid={`card-delete-${card.id}`}
                      onClick={() => deleteCard(card)}
                      disabled={card.read_only || rowBusy === 'delete'}
                      title={t('cards.action.delete')}
                      className="text-secondary hover:text-error transition-colors px-2 py-1 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px]"
                    >
                      {rowBusy === 'delete' ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {modal && (
        <CardFormModal
          card={modal.card}
          onClose={() => setModal(null)}
          onSaved={() => refresh()}
        />
      )}
    </div>
  );
}

/**
 * The Add/Edit dialog. It wraps the existing provider form unchanged (D10) and
 * deliberately stays open after a save so the connection-test verdict the form
 * renders is readable; the user closes it when they have read it.
 */
function CardFormModal({
  card,
  onClose,
  onSaved,
}: {
  card: CredentialCard | null;
  onClose: () => void;
  onSaved: (result: CardMutationResponse) => void;
}) {
  const t = useT();
  const [savedOnce, setSavedOnce] = useState(false);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={card ? t('cards.modal.editTitle') : t('cards.modal.addTitle')}
        data-testid="card-modal"
        className="bg-background rounded-xl shadow-xl border border-border w-full max-w-[560px] max-h-[85vh] flex flex-col mx-4 max-md:max-w-full max-md:max-h-full max-md:h-full max-md:mx-0 max-md:rounded-none"
      >
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border shrink-0">
          <h2 className="text-sm font-semibold text-primary">
            {card ? t('cards.modal.editTitle') : t('cards.modal.addTitle')}
          </h2>
          <button
            type="button"
            data-testid="card-modal-close"
            onClick={onClose}
            aria-label={t('cards.modal.close')}
            className="text-secondary hover:text-primary transition-all duration-150 p-1 max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
          >
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto min-h-0 px-5 py-4">
          <LLMProviderSettings
            mode="global"
            card={card}
            onCardSaved={(result) => {
              setSavedOnce(true);
              onSaved(result);
            }}
          />
        </div>

        {savedOnce && (
          <div className="px-5 py-3 border-t border-border shrink-0 flex items-center justify-between gap-3">
            <span className="flex items-center gap-1.5 text-xs text-secondary">
              <Check className="w-3.5 h-3.5 text-success" />
              {t('cards.modal.savedHint')}
            </span>
            <button
              type="button"
              data-testid="card-modal-done"
              onClick={onClose}
              className="text-sm font-medium text-secondary border border-border rounded-lg px-4 py-1.5 hover:text-primary transition-all duration-150"
            >
              {t('cards.modal.done')}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
