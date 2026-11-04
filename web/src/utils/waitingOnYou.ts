// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * waitingOnYou — spec 154 §5.1: the ONE definition of "waiting on you" and
 * "running". The Workbench's two groups, the edge-strip glyph's dot, the
 * project-list count (spec 151) and the automation-group dot (spec 150) all
 * read these predicates; nothing else decides these states.
 *
 * Works on a session-list row or a Workbench feed row: both carry the
 * manager `status`, `worker_running`, the two spec-108 timestamps and the
 * daemon's `checked_reply_at`.
 */

import type { AgentRunStatus, WorkbenchFeedAsk, WorkbenchFeedStop } from '../types';
import { getStatusDisplay, rowDisplayStatus } from '../components/sessionStatus';

/** A reply nobody checked for this long stops counting everywhere. */
export const WINDOW_DAYS = 14;
const WINDOW_MS = WINDOW_DAYS * 24 * 60 * 60 * 1000;

/** Stops the user must fix (amber), beyond asks and approvals. */
const NEEDS_ACTION_CAUSES = new Set(['usage_limit', 'model_unavailable', 'network_auth']);

const RUNNING_STATUSES = new Set<AgentRunStatus>(['running', 'waiting', 'queued', 'new_session']);

export interface WaitingRow {
  status: AgentRunStatus;
  worker_running?: boolean;
  last_user_at?: string | null;
  last_reply_at?: string | null;
  checked_reply_at?: string | null;
  ask?: WorkbenchFeedAsk | null;
  stop?: WorkbenchFeedStop | null;
}

function ts(value: string | null | undefined): number {
  if (!value) return NaN;
  return Date.parse(value);
}

/** Spec 152 A's `isSessionLive` minus approval: something is working in it. */
export function isRunning(row: Pick<WaitingRow, 'status' | 'worker_running'>): boolean {
  return RUNNING_STATUSES.has(rowDisplayStatus(row));
}

/**
 * Spec 108's `hasUnseenReply`, with the daemon's checked time in place of
 * the device's seen map and the 14-day window: the agent spoke last, the row
 * is resting, the reply is newer than the one the user checked, and it landed
 * within the window.
 */
export function isUnchecked(row: WaitingRow, now: number = Date.now()): boolean {
  const reply = ts(row.last_reply_at);
  if (Number.isNaN(reply)) return false; // nothing answered yet, or an older daemon
  const user = ts(row.last_user_at);
  if (!Number.isNaN(user) && user > reply) return false; // the user spoke last
  if (!getStatusDisplay(rowDisplayStatus(row)).resting) return false;
  if (now - reply > WINDOW_MS) return false;
  const checked = ts(row.checked_reply_at);
  return Number.isNaN(checked) || reply > checked;
}

/** Waiting on you: an open ask, an approval, or a reply not yet checked. */
export function isWaiting(row: WaitingRow, now: number = Date.now()): boolean {
  return !!row.ask || row.status === 'pending_approval' || isUnchecked(row, now);
}

/** Amber: the agent is stopped until the user acts. */
export function needsAction(row: WaitingRow): boolean {
  return (
    !!row.ask ||
    row.status === 'pending_approval' ||
    (!!row.stop && NEEDS_ACTION_CAUSES.has(row.stop.cause))
  );
}

/** Later of two ISO times (either may be missing). */
export function laterOf(a: string | null | undefined, b: string | null | undefined): string | null {
  if (!a) return b ?? null;
  if (!b) return a;
  const ta = Date.parse(a);
  const tb = Date.parse(b);
  if (Number.isNaN(ta)) return b;
  if (Number.isNaN(tb)) return a;
  return tb > ta ? b : a;
}
