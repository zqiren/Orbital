// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * 最近 / Recent (batch 1123) — the Workbench's last section: the most recent
 * sessions across every project (imported ones included) that are not in the
 * live list. Flat rows: agent, project, title, time. No state: we cannot tell
 * an unfinished task from a done one the user simply did not answer.
 *
 * A row is in exactly ONE place. The live list (feedModel's `items`) wins: a
 * session that becomes running or unread leaves 最近 at once. And a card that
 * leaves the live list on this device — an unread reply the user opened (the
 * checked mark lands before the feed refetches) — comes back to 最近 at once,
 * drawn from its feed row, rather than vanishing until the next fetch.
 *
 * Pure: the feed store, the live groups and the projects are inputs.
 */

import type { Project, WorkbenchFeedRow, WorkbenchRecentRow } from '../../types';
import { sessionKeyOf } from '../../hooks/useWorkbenchFeed';
import type { FeedGroups } from './feedModel';

/** How many rows 最近 shows. */
export const RECENT_LIMIT = 5;

export interface RecentItem {
  key: string;
  projectId: string;
  sessionId: string;
  projectName: string;
  title: string | null;
  agent: string | null;
  time: string;
}

const ts = (iso: string | null | undefined): number => {
  const n = iso ? Date.parse(iso) : NaN;
  return Number.isNaN(n) ? -Infinity : n;
};

function fromFeedRow(row: WorkbenchFeedRow, projectName: string): RecentItem | null {
  if (!row.session_id || row.trigger_type) return null;
  const time = row.last_activity_at ?? row.last_reply_at ?? row.last_user_at;
  if (!time) return null;
  return {
    key: sessionKeyOf(row.project_id, row.session_id),
    projectId: row.project_id,
    sessionId: row.session_id,
    projectName,
    title: row.name,
    agent: row.agent,
    time,
  };
}

export function buildRecent(
  feed: { rows: readonly WorkbenchFeedRow[]; recent: readonly WorkbenchRecentRow[] },
  groups: Pick<FeedGroups, 'items'>,
  projects: readonly Pick<Project, 'project_id' | 'name'>[],
  limit: number = RECENT_LIMIT,
): RecentItem[] {
  // Everything the live list draws, by session.
  const live = new Set<string>();
  for (const item of groups.items) {
    const pid = item.row?.project_id ?? item.pending?.project_id;
    const sid = item.row?.session_id ?? item.pending?.session_id;
    if (pid && sid) live.add(sessionKeyOf(pid, sid));
  }
  const known = new Map(projects.map((p) => [p.project_id, p.name]));
  const out = new Map<string, RecentItem>();
  for (const r of feed.recent) {
    const key = sessionKeyOf(r.project_id, r.session_id);
    if (live.has(key) || out.has(key)) continue;
    out.set(key, {
      key,
      projectId: r.project_id,
      sessionId: r.session_id,
      projectName: known.get(r.project_id) ?? r.project_name,
      title: r.title,
      agent: r.agent,
      time: r.updated_at,
    });
  }
  // A feed row the live list no longer draws (checked here, before the feed
  // caught up) belongs in 最近 now.
  for (const row of feed.rows) {
    if (!row.session_id) continue;
    const key = sessionKeyOf(row.project_id, row.session_id);
    if (live.has(key) || out.has(key)) continue;
    const name = known.get(row.project_id);
    if (name === undefined) continue;
    const item = fromFeedRow(row, name);
    if (item) out.set(key, item);
  }
  return [...out.values()].sort((a, b) => ts(b.time) - ts(a.time)).slice(0, limit);
}
