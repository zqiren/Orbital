// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * useSeenReplies — which reply the user has checked, per session (spec 108,
 * moved into the daemon by spec 154 §5.1).
 *
 * The daemon keeps one checked store for every device (`checked.json`); each
 * session-list and Workbench row carries its `checked_reply_at`. Checking a
 * reply (viewing the session, or ✓ on a Workbench card) POSTs
 * `/api/v2/workbench/checked`, and the daemon broadcasts `workbench.changed`
 * so the other device updates.
 *
 * Between the POST and the refetch that carries the new time, this module
 * holds an optimistic mark per (project, session key) so the dot and the card
 * leave at once. The effective checked time is the later of the row's and the
 * mark. A failed POST drops the mark again.
 *
 * Retired (spec 154): the per-device map in
 * `localStorage['orbital:seenReplies:<pid>']`. It is no longer read or
 * written, and it is left in place, so a downgrade finds its old state.
 */

import { useCallback, useSyncExternalStore } from 'react';
import { api } from '../config';
import type { SessionListEntry } from '../types';
import { seenKey } from '../utils/sessionUnread';
import { laterOf } from '../utils/waitingOnYou';

/** The retired spec-108 key (kept for reference and the downgrade test). */
export function seenRepliesStorageKey(projectId: string): string {
  return `orbital:seenReplies:${projectId}`;
}

type Marks = ReadonlyMap<string, string>;
let marks: Marks = new Map();
const listeners = new Set<() => void>();

function markKey(projectId: string, key: string): string {
  return `${projectId}\u0000${key}`;
}

function emit(): void {
  for (const fn of listeners) fn();
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function snapshot(): Marks {
  return marks;
}

/** The optimistic mark for one session, if any. */
export function optimisticChecked(projectId: string, key: string): string | undefined {
  return marks.get(markKey(projectId, key));
}

/** The effective checked time of a row: the later of the daemon's and the mark. */
export function effectiveChecked(
  projectId: string,
  key: string,
  serverValue: string | null | undefined,
): string | null {
  return laterOf(serverValue ?? null, optimisticChecked(projectId, key));
}

/**
 * Record that the user checked `replyAt` in this session: an optimistic
 * mark now, the daemon write behind it. Resolves true when the daemon took
 * it; on failure the mark is dropped (the row badges again).
 */
export async function markChecked(projectId: string, key: string, replyAt: string): Promise<boolean> {
  const k = markKey(projectId, key);
  const prev = marks.get(k);
  if (prev === replyAt) return true;
  const next = new Map(marks);
  next.set(k, laterOf(prev, replyAt) ?? replyAt);
  marks = next;
  emit();
  try {
    await api('/api/v2/workbench/checked', {
      method: 'POST',
      body: JSON.stringify({ project_id: projectId, session_id: key, reply_at: replyAt }),
    });
    return true;
  } catch {
    const back = new Map(marks);
    if (prev === undefined) back.delete(k);
    else back.set(k, prev);
    marks = back;
    emit();
    return false;
  }
}

/** Re-render on every optimistic mark (the Workbench and the sidebars). */
export function useCheckedMarks(): Marks {
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}

/** Test seam. */
export function __resetCheckedMarks(): void {
  marks = new Map();
  emit();
}

export function useSeenReplies(
  projectId: string | null,
  sessions: readonly SessionListEntry[],
): {
  seenAt: (key: string) => string | undefined;
  markSeen: (key: string, replyAt: string) => void;
} {
  const current = useCheckedMarks();
  const seenAt = useCallback(
    (key: string) => {
      if (!projectId) return undefined;
      const row = sessions.find((s) => seenKey(s) === key);
      const mark = current.get(markKey(projectId, key));
      return laterOf(row?.checked_reply_at ?? null, mark) ?? undefined;
    },
    [projectId, sessions, current],
  );
  const markSeen = useCallback(
    (key: string, replyAt: string) => {
      if (!projectId) return;
      void markChecked(projectId, key, replyAt);
    },
    [projectId],
  );
  return { seenAt, markSeen };
}
