// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * sessionUnread — spec 108: is there a reply in this session the user has not
 * looked at?
 *
 * The server says WHEN the agent last replied and WHEN the user last spoke
 * (`last_reply_at` / `last_user_at`, both daemon-clock ISO strings on every
 * list entry), and since spec 154 which reply the user has CHECKED
 * (`checked_reply_at`, the daemon's store, shared by every device). The
 * predicate itself is `isUnchecked` in `waitingOnYou` — one definition for
 * the session dot, the Workbench, the project count and the automation dot —
 * including its 14-day window: a reply nobody looked at for two weeks stops
 * badging.
 */

import type { SessionListEntry } from '../types';
import { isUnchecked } from './waitingOnYou';

/** The checked-store key for a row: the uuid, else the session id. */
export function seenKey(s: Pick<SessionListEntry, 'session_id' | 'session_uuid'>): string {
  return s.session_uuid ?? s.session_id;
}

export function hasUnseenReply(
  s: SessionListEntry,
  checkedReplyAt: string | null | undefined,
): boolean {
  return isUnchecked({ ...s, checked_reply_at: checkedReplyAt ?? null });
}
