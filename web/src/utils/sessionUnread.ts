// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * sessionUnread — spec 108: is there a reply in this session the user has not
 * looked at?
 *
 * The server says WHEN the agent last replied and WHEN the user last spoke
 * (`last_reply_at` / `last_user_at`, both daemon-clock ISO strings on every
 * list entry). This device remembers WHICH reply it has shown the user
 * (`useSeenReplies`, keyed on the server's reply timestamp — the device clock
 * never enters the comparison). A row is unread when the reply is newer than
 * the user's last message, the row is resting (a lit status glyph outranks
 * the badge), and the reply is newer than the one last seen.
 */

import type { SessionListEntry } from '../types';
import { getStatusDisplay, rowDisplayStatus } from '../components/sessionStatus';

/** The per-device seen-map key for a row: the uuid, else the session id. */
export function seenKey(s: Pick<SessionListEntry, 'session_id' | 'session_uuid'>): string {
  return s.session_uuid ?? s.session_id;
}

export function hasUnseenReply(
  s: SessionListEntry,
  seenReplyAt: string | undefined,
): boolean {
  if (!s.last_reply_at) return false; // nothing answered yet, or an older daemon
  const reply = Date.parse(s.last_reply_at);
  if (Number.isNaN(reply)) return false;
  if (s.last_user_at) {
    const user = Date.parse(s.last_user_at);
    if (!Number.isNaN(user) && user > reply) return false; // the user spoke last
  }
  // A running / waiting / blocked / queued row (or a resting row whose worker
  // is working, spec 102) already has a lit glyph; the badge lights when the
  // row comes to rest with a reply.
  if (!getStatusDisplay(rowDisplayStatus(s)).resting) return false;
  if (seenReplyAt === undefined) return true;
  const seen = Date.parse(seenReplyAt);
  return Number.isNaN(seen) || reply > seen;
}
