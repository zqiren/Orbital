// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 154 §5.7 — from the feed store to the two groups the Workbench draws.
 *
 * 等你 (waiting on you): overdue asks, then amber rows newest first, then the
 * rest newest first. 进行中 (running): pending sends first, then newest
 * `last_user_at` first. A row is in exactly one group; a card answered from
 * the Workbench (held) stays in 等你 until its turn ends.
 *
 * Pure: the store, the optimistic checked marks and "now" are inputs, so the
 * strip dot, the project counts and the page agree by construction.
 */

import type { WorkbenchFeedRow } from '../../types';
import { isRunning, isWaiting, laterOf, needsAction } from '../../utils/waitingOnYou';
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
    const live = isRunning(row);
    const held = feed.held.has(key);
    const wait = isWaiting(row, now) || held;
    if (!wait && !live) continue;
    const dot: CardDot = wait && needsAction(row) ? 'need' : live ? 'run' : 'new';
    const item: FeedItem = { key, row, pending: null, dot, waiting: wait, time: cardTime(row, live) };
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

  const overdue = (i: FeedItem) => (i.row?.ask?.overdue ? 0 : 1);
  const amber = (i: FeedItem) => (i.dot === 'need' ? 0 : 1);
  waiting.sort((a, b) => overdue(a) - overdue(b) || amber(a) - amber(b) || newestFirst(a.time, b.time));
  running.sort((a, b) => newestFirst(a.time, b.time));
  return { waiting, running: [...pendingItems, ...running] };
}

/** Per project: how many 等你 cards it has (spec 151's count). */
export function waitingCounts(groups: FeedGroups): Record<string, number> {
  const out: Record<string, number> = {};
  for (const item of groups.waiting) {
    const pid = item.row?.project_id;
    if (pid) out[pid] = (out[pid] ?? 0) + 1;
  }
  return out;
}
