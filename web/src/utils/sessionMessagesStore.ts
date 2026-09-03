// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §11.5 — one transcript, two readers. ChatView owns the session's
 * message array (its `rawMessages`) and publishes it here by reference; the
 * workspace panel (ChatTab) subscribes and derives touched files and the
 * fallback screenshot from the same array instead of fetching the history a
 * second time. Module-level so the two siblings need no prop plumbing.
 */
import { useSyncExternalStore } from 'react';
import type { ChatMessage } from '../types';

const store = new Map<string, ChatMessage[]>();
const listeners = new Set<() => void>();

export function sessionMessagesKey(projectId: string, sessionId: string | null | undefined): string {
  return `${projectId}:${sessionId ?? ''}`;
}

export function publishSessionMessages(key: string, messages: ChatMessage[]): void {
  if (store.get(key) === messages) return;
  store.set(key, messages);
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => { listeners.delete(l); };
}

/** The published transcript for the key, or null when nothing has been published yet. */
export function useSessionMessages(key: string): ChatMessage[] | null {
  return useSyncExternalStore(subscribe, () => store.get(key) ?? null, () => null);
}

/** Test helper. */
export function __resetSessionMessagesStore(): void {
  store.clear();
  listeners.forEach((l) => l());
}
