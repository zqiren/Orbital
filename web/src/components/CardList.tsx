// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The credential-card list. ONE component for both places a card is chosen.
 *
 * Spec 082 shipped two different designs for the same object: Global Settings
 * got a rich list with four per-row actions, project settings got a plain
 * <select>. Same card, two mental models, and from the project side you could
 * not test, edit or delete anything.
 *
 * This is the single design, used by both:
 *
 *   - Choosing IS selecting. Clicking a card selects it and the selected card
 *     is highlighted. That replaces the global list's "Set default" button and
 *     the project dropdown's value — in `global` mode the selection is the
 *     default card, in `project` mode it is the card that project runs on.
 *   - Exactly three actions per card: test connection, edit, delete. Nothing
 *     else. Anything list-level (Add, one-click sign-in) belongs to the parent,
 *     above the list.
 *   - `project` mode adds one leading tile, "Global default", which is
 *     `card_id: null` — a reference to whatever the default is at run time,
 *     not a copy of its id.
 *
 * The three actions live here rather than in each parent so the two surfaces
 * cannot drift: a card behaves identically wherever it is shown.
 *
 * The AGENT_OS_API_KEY card is `read_only`. It cannot be edited, deleted, or
 * deselected while the variable is set, so it renders inert with its badge
 * rather than showing three dead controls.
 */
import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Check,
  CircleAlert,
  CircleCheck,
  CircleDashed,
  Loader2,
  SquarePen,
  Trash,
  X,
} from 'lucide-react';
import type {
  CardDeleteResponse,
  CardMutationResponse,
  CredentialCard,
  ProviderRegistry,
} from '../types';
import { api } from '../config';
import { cardHealth } from '../utils/cardHealth';
import type { CardHealth } from '../utils/cardHealth';
import { useT, translate } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import LLMProviderSettings from './LLMProviderSettings';

/**
 * Provider display names, fetched once per page load and shared by every
 * mount. Both surfaces render the same list, so neither may show a raw
 * registry key where the other shows a display name — and threading the
 * registry down as a prop from two unrelated parents is exactly how that
 * drift starts. The caller may still pass `providers` to skip the fetch.
 */
let registryCache: ProviderRegistry | null = null;
let registryInFlight: Promise<ProviderRegistry> | null = null;

function useProviderRegistry(supplied?: ProviderRegistry): ProviderRegistry {
  const [registry, setRegistry] = useState<ProviderRegistry>(
    supplied ?? registryCache ?? {},
  );
  useEffect(() => {
    if (supplied || registryCache) return;
    let cancelled = false;
    registryInFlight =
      registryInFlight ?? api<ProviderRegistry>('/api/v2/providers').catch(() => ({}));
    void registryInFlight.then((data) => {
      registryCache = data;
      if (!cancelled) setRegistry(data);
    });
    return () => {
      cancelled = true;
    };
  }, [supplied]);
  return supplied ?? registry;
}

/** Sentinel for the "follow the global default" tile. */
export const GLOBAL_DEFAULT_VALUE = '__global_default__';

/** A row's in-flight action, so only that row spins. */
type BusyKind = 'select' | 'test' | 'delete';

export interface CardListProps {
  cards: CredentialCard[];
  /** The EFFECTIVE default id ('env' while AGENT_OS_API_KEY is set). */
  defaultCardId: string | null;
  /**
   * `global`  — selection sets the default card (the daemon call is made here).
   * `project` — selection is the caller's `value`; `onChange` owns persistence.
   */
  mode: 'global' | 'project';
  /** project mode: the referenced card id; null = follow the global default. */
  value?: string | null;
  /** project mode: called with the new card id, or null for global default. */
  onChange?: (cardId: string | null) => void;
  /** Re-read after a mutation (delete, or a default change). */
  onRefresh: () => Promise<void> | void;
  /** Splice one card in place after a test/edit, without a round trip. */
  onCardUpdated: (card: CredentialCard) => void;
  providers?: ProviderRegistry;
  loading?: boolean;
  'data-testid'?: string;
}

export default function CardList({
  cards,
  defaultCardId,
  mode,
  value,
  onChange,
  onRefresh,
  onCardUpdated,
  providers,
  loading,
  'data-testid': testId = 'card-list',
}: CardListProps) {
  const t = useT();
  const { locale } = useLocale();
  const tr = useMemo(
    () => (key: Parameters<typeof t>[0], vars?: Parameters<typeof t>[1]) =>
      translate(locale, key, vars),
    [locale],
  );

  const registry = useProviderRegistry(providers);
  const [busy, setBusy] = useState<{ id: string; kind: BusyKind } | null>(null);
  const [rowError, setRowError] = useState<{ id: string; message: string } | null>(null);
  const [rowNote, setRowNote] = useState<{ id: string; message: string } | null>(null);
  const [editing, setEditing] = useState<CredentialCard | null>(null);

  function errorText(err: unknown): string {
    return err instanceof Error && err.message ? err.message : t('cards.action.failed');
  }

  function clearRow() {
    setRowError(null);
    setRowNote(null);
  }

  /** Selecting a card. In global mode that is a daemon call; in project mode
   *  the parent persists it with the rest of its settings form. */
  async function select(card: CredentialCard | null) {
    clearRow();
    if (mode === 'project') {
      onChange?.(card ? card.id : null);
      return;
    }
    if (!card || card.id === defaultCardId) return;
    setBusy({ id: card.id, kind: 'select' });
    try {
      const res = await api<{ default_card_id: string | null; applied: boolean }>(
        `/api/v2/settings/cards/${encodeURIComponent(card.id)}/default`,
        { method: 'PUT' },
      );
      // The env card outranks a stored default while AGENT_OS_API_KEY is set:
      // the choice IS recorded, it just is not what runs yet. Saying so beats
      // a click that appears to do nothing.
      if (!res.applied) setRowNote({ id: card.id, message: t('cards.default.envWins') });
      await onRefresh();
    } catch (err: unknown) {
      setRowError({ id: card.id, message: errorText(err) });
    } finally {
      setBusy(null);
    }
  }

  async function testCard(card: CredentialCard) {
    setBusy({ id: card.id, kind: 'test' });
    clearRow();
    try {
      const res = await api<CardMutationResponse>(
        `/api/v2/settings/cards/${encodeURIComponent(card.id)}/test`,
        { method: 'POST' },
      );
      onCardUpdated(res.card);
      // HTTP 200 even for a failed test — the verdict is in the body, and the
      // refreshed card already carries verified_at / last_error for the health
      // line. The note is only the immediate "I just clicked that".
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
    clearRow();
    try {
      const res = await api<CardDeleteResponse>(
        `/api/v2/settings/cards/${encodeURIComponent(card.id)}`,
        { method: 'DELETE' },
      );
      // Deleting a referenced card moves those projects onto the default — the
      // one behaviour worth naming out loud, so they are listed rather than
      // silently repointed.
      if (res.reassigned_projects?.length) {
        setRowNote({
          id: card.id,
          message: t('cards.delete.reassigned', {
            projects: res.reassigned_projects.map((p) => p.name).join(', '),
          }),
        });
      }
      // A project pinned to the card just deleted follows the global default
      // again; keep this surface's own selection honest about that.
      if (mode === 'project' && value === card.id) onChange?.(null);
      await onRefresh();
    } catch (err: unknown) {
      setRowError({ id: card.id, message: errorText(err) });
    } finally {
      setBusy(null);
    }
  }

  const selectedId = mode === 'global' ? defaultCardId : (value ?? null);
  const defaultCard = cards.find((c) => c.id === defaultCardId);
  // A pinned id with no matching card is one deleted out from under this
  // referrer. The daemon repoints on delete, but a stale render must not
  // silently look like "global default".
  const dangling = mode === 'project' && !!value && !cards.some((c) => c.id === value);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-secondary py-4">
        <Loader2 className="w-4 h-4 animate-spin" />
        {t('llm.loading')}
      </div>
    );
  }

  if (cards.length === 0) {
    return (
      <p className="text-sm text-secondary/70 italic py-2" data-testid={`${testId}-empty`}>
        {mode === 'global' ? t('cards.empty') : t('cards.picker.empty')}
      </p>
    );
  }

  return (
    <div data-testid={testId}>
      {dangling && (
        <p className="text-xs text-error mb-2" data-testid={`${testId}-dangling`}>
          {t('cards.picker.missing')}
        </p>
      )}

      <ul className="space-y-2" role="radiogroup" aria-label={t('cards.picker.aria')}>
        {/* project mode only: follow whatever the default is at run time. */}
        {mode === 'project' && (
          <li>
            <button
              type="button"
              role="radio"
              aria-checked={selectedId === null && !dangling}
              data-testid={`${testId}-global-default`}
              onClick={() => select(null)}
              className={`w-full text-left border rounded-lg px-3 py-2.5 transition-all duration-150 ${
                selectedId === null && !dangling
                  ? 'border-accent bg-accent/5 ring-1 ring-accent/30'
                  : 'border-border bg-sidebar hover:border-accent/40'
              }`}
            >
              <span className="flex items-center gap-2">
                <SelectionDot selected={selectedId === null && !dangling} />
                <span className="text-sm font-medium text-primary">
                  {defaultCard
                    ? t('cards.picker.globalDefaultNamed', { name: defaultCard.name })
                    : t('cards.picker.globalDefaultNone')}
                </span>
              </span>
              <span className="block text-xs text-secondary mt-1 pl-6">
                {t('cards.select.globalDefaultHint')}
              </span>
            </button>
          </li>
        )}

        {cards.map((card) => {
          const health = cardHealth(card, tr);
          const providerName = registry[card.provider]?.display_name || card.provider;
          const isSelected = card.id === selectedId && !(mode === 'project' && dangling);
          const rowBusy = busy?.id === card.id ? busy.kind : null;
          return (
            <li key={card.id} data-testid={`card-row-${card.id}`}>
              <div
                className={`border rounded-lg transition-all duration-150 ${
                  isSelected
                    ? 'border-accent bg-accent/5 ring-1 ring-accent/30'
                    : 'border-border bg-sidebar hover:border-accent/40'
                }`}
              >
                <div className="flex items-start gap-3 px-3 py-2.5">
                  {/* The tile body is the selector. */}
                  <button
                    type="button"
                    role="radio"
                    aria-checked={isSelected}
                    disabled={rowBusy === 'select'}
                    data-testid={`card-select-${card.id}`}
                    onClick={() => select(card)}
                    className="min-w-0 flex-1 text-left disabled:cursor-wait"
                  >
                    <span className="flex items-center gap-2 flex-wrap">
                      {rowBusy === 'select' ? (
                        <Loader2 className="w-4 h-4 shrink-0 animate-spin text-accent" />
                      ) : (
                        <SelectionDot selected={isSelected} />
                      )}
                      <span className="text-sm font-medium text-primary truncate min-w-0">
                        {card.name}
                      </span>
                      <HealthMark health={health} cardId={card.id} />
                      {isSelected && (
                        <span
                          data-testid={`card-selected-${card.id}`}
                          className="shrink-0 inline-flex items-center gap-1 text-[11px] rounded-full px-2 py-0.5 bg-accent/10 text-accent"
                        >
                          <Check className="w-3 h-3" />
                          {mode === 'global'
                            ? t('cards.badge.default')
                            : t('cards.badge.inUse')}
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
                    </span>

                    <span className="block text-xs text-secondary mt-1 truncate pl-6">
                      {/* A card's default name IS "<Provider> · <model>", so
                          repeating it here was pure noise. Show it only when
                          the card has been renamed and the name no longer
                          says what it runs on. */}
                      {namesItsSetup(card, providerName) ? null : (
                        <>
                          {providerName}
                          {' · '}
                          <span className="font-mono">
                            {card.model || t('cards.needsModel')}
                          </span>
                          {' · '}
                        </>
                      )}
                      {card.key_masked ? (
                        <span className="font-mono">{card.key_masked}</span>
                      ) : (
                        t('cards.noKey')
                      )}
                    </span>

                  </button>

                  {/* Exactly three actions. Never a fourth. */}
                  <div className="flex items-center gap-1 shrink-0">
                    <IconAction
                      testId={`card-test-${card.id}`}
                      label={t('cards.action.test')}
                      busy={rowBusy === 'test'}
                      onClick={() => testCard(card)}
                    >
                      <Activity className="w-4 h-4" strokeWidth={1.5} />
                    </IconAction>
                    <IconAction
                      testId={`card-edit-${card.id}`}
                      label={t('cards.action.edit')}
                      disabled={card.read_only}
                      onClick={() => {
                        clearRow();
                        setEditing(card);
                      }}
                    >
                      <SquarePen className="w-4 h-4" strokeWidth={1.5} />
                    </IconAction>
                    <IconAction
                      testId={`card-delete-${card.id}`}
                      label={t('cards.action.delete')}
                      disabled={card.read_only}
                      busy={rowBusy === 'delete'}
                      danger
                      onClick={() => deleteCard(card)}
                    >
                      <Trash className="w-4 h-4" strokeWidth={1.5} />
                    </IconAction>
                  </div>
                </div>

                {(rowNote?.id === card.id || rowError?.id === card.id) && (
                  <p
                    data-testid={
                      rowError?.id === card.id ? `card-error-${card.id}` : `card-note-${card.id}`
                    }
                    className={`text-xs px-3 pb-2.5 pl-9 ${
                      rowError?.id === card.id ? 'text-error' : 'text-secondary'
                    }`}
                  >
                    {rowError?.id === card.id ? rowError.message : rowNote?.message}
                  </p>
                )}

              </div>
            </li>
          );
        })}
      </ul>

      {editing && (
        <CardFormModal
          card={editing}
          onClose={() => setEditing(null)}
          onSaved={(res) => {
            onCardUpdated(res.card);
            void onRefresh();
          }}
        />
      )}
    </div>
  );
}

/**
 * Health as a symbol, not a sentence.
 *
 * The status line used to spell itself out on every row — "Verified 3m ago",
 * or an entire provider paragraph about credit limits — which made a list of
 * four cards read like a changelog. The state is one of three things, so it
 * is one of three marks, and the words appear on hover for the one card whose
 * words you actually want.
 */
function HealthMark({ health, cardId }: { health: CardHealth; cardId: string }) {
  const Icon =
    health.kind === 'verified'
      ? CircleCheck
      : health.kind === 'error'
        ? CircleAlert
        : CircleDashed;
  const tone =
    health.kind === 'verified'
      ? 'text-success'
      : health.kind === 'error'
        ? 'text-error'
        : 'text-secondary/60';
  return (
    <span className="relative shrink-0 inline-flex group/health">
      {/* role=img + aria-label: the text still reaches a screen reader, and
          the button this sits inside keeps a complete accessible name. */}
      <span role="img" aria-label={health.text} data-testid={`card-health-${cardId}`}>
        <Icon className={`w-4 h-4 ${tone}`} strokeWidth={1.5} />
      </span>
      {/* Absolutely positioned, so revealing it never reflows the row. */}
      <span
        aria-hidden="true"
        data-testid={`card-health-tip-${cardId}`}
        className="pointer-events-none absolute left-1/2 top-full z-30 mt-1.5 hidden w-64 -translate-x-1/2 rounded-md border border-border bg-card px-2.5 py-1.5 text-xs leading-relaxed text-primary shadow-lg group-hover/health:block"
      >
        {health.text}
      </span>
    </span>
  );
}

/** True when the card's name already spells out its provider and model, which
 *  is what the daemon names a card by default. */
function namesItsSetup(card: CredentialCard, providerName: string): boolean {
  if (!card.model) return false;
  const name = card.name.toLowerCase();
  return name.includes(card.model.toLowerCase()) && name.includes(providerName.toLowerCase());
}

/** The selected/unselected mark. A ring rather than a checkbox: the whole tile
 *  is the control, and a checkbox would invite a second click target. */
function SelectionDot({ selected }: { selected: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={`shrink-0 w-4 h-4 rounded-full border grid place-items-center ${
        selected ? 'border-accent bg-accent' : 'border-border bg-card'
      }`}
    >
      {selected && <Check className="w-2.5 h-2.5 text-white" strokeWidth={3} />}
    </span>
  );
}

function IconAction({
  testId,
  label,
  onClick,
  children,
  busy,
  disabled,
  danger,
}: {
  testId: string;
  label: string;
  onClick: () => void;
  children: React.ReactNode;
  busy?: boolean;
  disabled?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      data-testid={testId}
      title={label}
      aria-label={label}
      disabled={disabled || busy}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={`p-1.5 rounded transition-colors text-secondary disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center ${
        danger ? 'hover:text-error' : 'hover:text-accent'
      }`}
    >
      {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : children}
    </button>
  );
}

/**
 * The Add/Edit dialog. It wraps the existing provider form unchanged and
 * deliberately stays open after a save so the connection-test verdict the form
 * renders is readable; the user closes it when they have read it.
 */
export function CardFormModal({
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
