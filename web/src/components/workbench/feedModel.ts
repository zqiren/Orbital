// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 154 §5.7 — from the feed store to the list the Workbench draws.
 *
 * ONE list by recency (2026-11-05, owner): pending sends first (newest
 * first), then every other card newest first by its time, whatever its
 * state; the dot (need / new / run) says the state. A task just sent from
 * the composer therefore never lands below the 等你 cards (spec 154's own
 * rejected-alternatives table: a just-sent task must not sit below the fold).
 *
 * The two groups are still computed, for the summary line and the project
 * counts: 等你 (waiting on you) and 进行中 (running), each newest first,
 * pending sends at the top of 进行中. A row is in exactly one group. A card
 * the user just answered (from its field or the reading pane: held) is in
 * 进行中 at once, before the daemon reports the turn, and sorts by the moment
 * it was answered; it comes back to 等你 as finished and unchecked when the
 * turn ends (2026-11-04).
 *
 * Pure: the store, the optimistic checked marks and "now" are inputs, so the
 * strip dot, the project counts and the page agree by construction.
 */

import type { WorkbenchFeedRow } from '../../types';
import { isRunning, isUnchecked, isWaiting, laterOf, needsAction } from '../../utils/waitingOnYou';
import { sessionKeyOf, type FeedState, type PendingTask } from '../../hooks/useWorkbenchFeed';

export type CardDot = 'need' | 'new' | 'run';

export interface FeedItem {
  key: string;
  /** The row as the card draws it (checked marks and closed asks applied). */
  row: WorkbenchFeedRow | null;
  pending: PendingTask | null;
  dot: CardDot;
  waiting: boolean;
  /** The time the card shows (ISO), and the one it sorts by. */
  time: string | null;
}

export interface FeedGroups {
  waiting: FeedItem[];
  running: FeedItem[];
  /** What the page draws: pending sends, then everything newest first. */
  items: FeedItem[];
}

type Marks = ReadonlyMap<string, string>;

function markFor(marks: Marks, row: WorkbenchFeedRow): string | undefined {
  const key = row.session_uuid ?? row.session_id;
  return key ? marks.get(`${row.project_id}\u0000${key}`) : undefined;
}

function cardTime(row: WorkbenchFeedRow, running: boolean): string | null {
  if (row.ask) return row.last_reply_at ?? row.ask.created ?? null;
  if (running) return row.last_user_at ?? row.last_activity_at ?? null;
  return row.last_reply_at ?? row.last_activity_at ?? null;
}

function newestFirst(a: string | null, b: string | null): number {
  const ta = a ? Date.parse(a) : NaN;
  const tb = b ? Date.parse(b) : NaN;
  return (Number.isNaN(tb) ? -Infinity : tb) - (Number.isNaN(ta) ? -Infinity : ta);
}

function itemKey(row: WorkbenchFeedRow): string {
  return row.session_id
    ? sessionKeyOf(row.project_id, row.session_id)
    : `${row.project_id}\u0000ask:${row.ask?.id ?? ''}`;
}

export function buildFeedGroups(feed: FeedState, marks: Marks, now: number = Date.now()): FeedGroups {
  const waiting: FeedItem[] = [];
  const running: FeedItem[] = [];
  const seen = new Set<string>();
  const rows = [...feed.rows];
  // A held card the latest feed momentarily lacks is drawn from its snapshot.
  for (const [key, h] of feed.held) {
    if (!rows.some((r) => itemKey(r) === key)) rows.push(h.row);
  }
  for (const raw of rows) {
    const key = itemKey(raw);
    if (seen.has(key)) continue;
    seen.add(key);
    const row: WorkbenchFeedRow = {
      ...raw,
      checked_reply_at: laterOf(raw.checked_reply_at, markFor(marks, raw)),
      ask: raw.ask && feed.closedAsks.has(`${raw.project_id}\u0000${raw.ask.id}`) ? null : raw.ask,
    };
    if (!row.session_id && !row.ask) continue; // a session-less ask, closed from its card
    const hold = feed.held.get(key);
    const held = !!hold;
    const live = isRunning(row) || held;
    const wait = !held && isWaiting(row, now);
    if (!wait && !live) continue;
    const dot: CardDot = wait && needsAction(row) ? 'need' : live ? 'run' : 'new';
    // A card just answered here is as recent as the answer, even before the
    // daemon reports the new turn's last_user_at.
    const time = hold ? laterOf(cardTime(row, live), new Date(hold.at).toISOString()) : cardTime(row, live);
    const item: FeedItem = { key, row, pending: null, dot, waiting: wait, time };
    (wait ? waiting : running).push(item);
  }
  // Pending sends the feed does not carry yet.
  const pendingItems: FeedItem[] = feed.pending
    .filter((p) => !(p.project_id && p.session_id && seen.has(sessionKeyOf(p.project_id, p.session_id))))
    .map((p) => ({
      key: `pending:${p.nonce}`,
      row: null,
      pending: p,
      dot: 'run' as const,
      waiting: false,
      time: new Date(p.created_at).toISOString(),
    }));

  // 等你 by recency alone (2026-11-05): the newest thing waiting sits right
  // under the composer. Overdue and amber still show on the card itself.
  waiting.sort((a, b) => newestFirst(a.time, b.time));
  running.sort((a, b) => newestFirst(a.time, b.time));
  pendingItems.sort((a, b) => newestFirst(a.time, b.time));
  const items = [...pendingItems, ...[...waiting, ...running].sort((a, b) => newestFirst(a.time, b.time))];
  return { waiting, running: [...pendingItems, ...running], items };
}

/**
 * Per project: how many sessions hold a reply the user has not checked, the
 * same sessions the session list dots (2026-11-05; it used to count every
 * 等你 card). An open ask or an approval alone is not unread: it waits on the
 * Workbench, and the project's own dot already says so.
 */
export function unreadCounts(groups: FeedGroups, now: number = Date.now()): Record<string, number> {
  const out: Record<string, number> = {};
  for (const item of groups.waiting) {
    const row = item.row;
    if (!row?.session_id || !isUnchecked(row, now)) continue;
    out[row.project_id] = (out[row.project_id] ?? 0) + 1;
  }
  return out;
}
