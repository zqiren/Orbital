// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * useSeenReplies — spec 108: which reply has this device shown the user, per
 * session, per project.
 *
 * `localStorage['orbital:seenReplies:<projectId>']` → `{ [sessionKey]: ISO }`
 * where the value is the server's `last_reply_at` the user has looked at
 * (`sessionKey` = `session_uuid ?? session_id`, see `seenKey`). Per device by
 * design: the phone (relay) and the desktop each keep their own map (spec 108
 * §8 Q2 deferred cross-device read state).
 *
 * Two rules:
 *   - First visit seeds, not floods. When the key is absent for a project,
 *     the first non-empty list writes every row's current reply as seen —
 *     existing history is "read"; only replies that land from now on count.
 *   - Prune on write to the keys present in the current list, so deleted
 *     sessions never accumulate.
 *
 * Storage failures are swallowed (the filter precedent in SessionSidebar):
 * the map still works in memory for this visit.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { SessionListEntry } from '../types';
import { seenKey } from '../utils/sessionUnread';

export type SeenReplies = Readonly<Record<string, string>>;

export function seenRepliesStorageKey(projectId: string): string {
  return `orbital:seenReplies:${projectId}`;
}

/** `null` when the key is absent (first visit); `{}` when present but
 * unparseable (treated as present so a corrupt value never re-seeds). */
function read(storageKey: string | null): SeenReplies | null {
  if (!storageKey) return null;
  let raw: string | null;
  try {
    raw = localStorage.getItem(storageKey);
  } catch {
    return null;
  }
  if (raw === null) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      const out: Record<string, string> = {};
      for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
        if (typeof v === 'string') out[k] = v;
      }
      return out;
    }
  } catch {
    /* corrupt — fall through */
  }
  return {};
}

function write(storageKey: string | null, map: SeenReplies): void {
  if (!storageKey) return;
  try {
    localStorage.setItem(storageKey, JSON.stringify(map));
  } catch {
    /* storage unavailable — the map still works for this visit */
  }
}

function pruneTo(map: SeenReplies, keep: ReadonlySet<string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(map)) {
    if (keep.has(k)) out[k] = v;
  }
  return out;
}

export function useSeenReplies(
  projectId: string | null,
  sessions: readonly SessionListEntry[],
): {
  seenAt: (key: string) => string | undefined;
  markSeen: (key: string, replyAt: string) => void;
} {
  const storageKey = projectId ? seenRepliesStorageKey(projectId) : null;

  // The map for the current project. Derived during render on a project
  // switch (React's "adjust state from props" pattern) so the first render
  // of the new project already reads its own map — never the previous one.
  const [state, setState] = useState<{ key: string | null; map: SeenReplies }>(() => ({
    key: storageKey,
    map: read(storageKey) ?? {},
  }));
  let current = state;
  if (state.key !== storageKey) {
    current = { key: storageKey, map: read(storageKey) ?? {} };
    setState(current);
  }
  // Writes go through a ref (the source of truth between renders — never a
  // closure variable read inside a setState updater) and are mirrored into
  // state so consumers re-render.
  const mapRef = useRef<SeenReplies>(current.map);
  mapRef.current = current.map;
  // Keys in the current list — what a write prunes to.
  const presentRef = useRef<ReadonlySet<string>>(new Set());
  presentRef.current = new Set(sessions.map(seenKey));

  const commit = useCallback(
    (key: string | null, next: SeenReplies) => {
      mapRef.current = next;
      write(key, next);
      setState({ key, map: next });
    },
    [],
  );

  // Seed on first visit: the key is absent and the list is non-empty.
  useEffect(() => {
    if (!storageKey || sessions.length === 0) return;
    if (read(storageKey) !== null) return; // present (even empty): never re-seed
    const seeded: Record<string, string> = {};
    for (const s of sessions) {
      if (s.last_reply_at) seeded[seenKey(s)] = s.last_reply_at;
    }
    commit(storageKey, seeded);
  }, [storageKey, sessions, commit]);

  const map = current.map;
  const seenAt = useCallback((key: string) => map[key], [map]);

  const markSeen = useCallback(
    (key: string, replyAt: string) => {
      if (mapRef.current[key] === replyAt) return;
      const next = pruneTo(mapRef.current, presentRef.current);
      next[key] = replyAt;
      commit(storageKey, next);
    },
    [storageKey, commit],
  );

  return { seenAt, markSeen };
}
