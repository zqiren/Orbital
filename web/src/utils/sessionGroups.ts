// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Session-list shaping for SessionSidebar: the All | Chats | Automations
 * filter, and collapsing an automation's runs into one group.
 *
 * Why: a daily automation mints a session every day, forever. Measured on
 * real projects, 40–63% of all sessions were automation runs. Listing them
 * flat buried the chats; parking them in a section BELOW the chats (the
 * first fix) meant scrolling past hundreds of rows to reach them. An
 * automation is one standing thing with a run history — so it gets one row,
 * and a filter makes either half of the list one click away.
 *
 * Pure and order-preserving: the caller sorts (pinned first, most recent
 * first); nothing here re-sorts, so a group sits where its latest run was.
 */

import type { SessionListEntry } from '../types';
import { classifySessionName } from '../lib/sessionLabel';

export type SessionFilter = 'all' | 'chats' | 'automations';
export type AutomationKind = 'schedule' | 'file_watch' | 'queue';

export interface AutomationGroup {
  /** `${kind}:${name}` — stable while the trigger keeps its name. */
  key: string;
  kind: AutomationKind;
  /** The trigger's name; null for the queue and for runs the user renamed
   * (the trigger name only survives in the auto-derived session name). */
  name: string | null;
  /** Most recent first — the caller's order. Always 2 or more. */
  runs: SessionListEntry[];
}

export type SessionListNode =
  | { type: 'session'; session: SessionListEntry }
  | { type: 'group'; group: AutomationGroup };

/** Which automation a session belongs to, or null for a conversation.
 * `trigger_type` and `origin` come from the backend and survive a rename; the
 * name-prefix classifier covers rows from a backend that predates them. */
function automationOf(s: SessionListEntry): { kind: AutomationKind; name: string | null } | null {
  const label = classifySessionName(s.name, s.origin);
  if (s.origin === 'queue' || label.kind === 'queue') return { kind: 'queue', name: null };
  if (label.kind === 'schedule' || label.kind === 'file_watch') {
    return { kind: label.kind, name: label.displayName };
  }
  if (s.trigger_type) return { kind: s.trigger_type, name: null };
  return null;
}

export function isAutomationSession(s: SessionListEntry): boolean {
  return automationOf(s) !== null;
}

export function buildSessionList(
  sorted: readonly SessionListEntry[],
  filter: SessionFilter,
): {
  pinned: SessionListEntry[];
  nodes: SessionListNode[];
  counts: Record<SessionFilter, number>;
} {
  const pinned: SessionListEntry[] = [];
  const nodes: SessionListNode[] = [];
  const groupAt = new Map<string, number>(); // group key → index in `nodes`
  const counts = { all: sorted.length, chats: 0, automations: 0 };

  for (const session of sorted) {
    const automation = automationOf(session);
    counts[automation ? 'automations' : 'chats'] += 1;
    if (filter === 'chats' && automation) continue;
    if (filter === 'automations' && !automation) continue;

    // A pin is the user saying "keep THIS one at the top": it stays an
    // individual row and is not also folded into its automation's group.
    if (session.pinned) {
      pinned.push(session);
      continue;
    }
    if (!automation) {
      nodes.push({ type: 'session', session });
      continue;
    }
    const key = `${automation.kind}:${automation.name ?? ''}`;
    const at = groupAt.get(key);
    if (at === undefined) {
      // First (= most recent) run: holds the group's place in the list.
      groupAt.set(key, nodes.length);
      nodes.push({ type: 'session', session });
      continue;
    }
    const node = nodes[at];
    if (node.type === 'session') {
      // Second run: the placeholder becomes a real group.
      nodes[at] = {
        type: 'group',
        group: { key, kind: automation.kind, name: automation.name, runs: [node.session, session] },
      };
    } else {
      node.group.runs.push(session);
    }
  }
  return { pinned, nodes, counts };
}
