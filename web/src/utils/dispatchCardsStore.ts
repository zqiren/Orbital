// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 195: ChatView owns the conversation's dispatch cards (history plus the
 * live steps) and publishes them here by reference; the docked panel
 * (ChatTab) reads the one it shows. Same shape as sessionMessagesStore: two
 * siblings, no prop plumbing, and only the panel re-renders on a step.
 */
import { useSyncExternalStore } from 'react';
import type { DispatchCardItem } from './dispatchCards';

/** What the panel shows: a dispatch card, or one of a fanout's workers. */
export type WorkerPanelTarget =
  | { kind: 'card'; cardId: string; dispatchId?: string; handle: string }
  | { kind: 'fanout'; handle: string; label: string };

const store = new Map<string, DispatchCardItem[]>();
const listeners = new Set<() => void>();
const EMPTY: DispatchCardItem[] = [];

export function publishDispatchCards(key: string, cards: DispatchCardItem[]): void {
  if (store.get(key) === cards) return;
  store.set(key, cards);
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

export function useDispatchCards(key: string): DispatchCardItem[] {
  return useSyncExternalStore(subscribe, () => store.get(key) ?? EMPTY, () => EMPTY);
}

/** The card a target names: by dispatch id, then card id (a live card's id
 *  changes once history replaces it), then the handle's latest card. */
export function findTargetCard(
  cards: DispatchCardItem[],
  target: Extract<WorkerPanelTarget, { kind: 'card' }>,
): DispatchCardItem | undefined {
  if (target.dispatchId) {
    const byDispatch = cards.find((c) => c.dispatch_id === target.dispatchId);
    if (byDispatch) return byDispatch;
  }
  const byId = cards.find((c) => c.card_id === target.cardId);
  if (byId) return byId;
  if (target.dispatchId) return undefined;
  for (let i = cards.length - 1; i >= 0; i--) {
    if (cards[i].handle === target.handle) return cards[i];
  }
  return undefined;
}

/** Test helper. */
export function __resetDispatchCardsStore(): void {
  store.clear();
  listeners.forEach((l) => l());
}
