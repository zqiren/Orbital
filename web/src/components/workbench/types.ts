// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Frontend mirror of the Workbench surface's wire shapes (spec §5.3, §5.4,
 * §6, §8). These match `GET /api/v2/workbench` and the exit/dismiss/open/
 * migrate routes byte-for-byte — see `agent_os/api/routes/workbench.py`.
 */

/** One flagged `[user]` entry, lensed to a project. */
export interface WorkbenchEntry {
  project_id: string;
  id: string;
  text: string;
  due: string | null;
  evidence: string | null;
  from_session: string | null;
  confidence: 'stated' | 'unconfirmed' | null;
  created: string | null;
  touched: string | null;
  /** Server-computed (project tz) — age since `created`. */
  age_days: number | null;
  /** Server-computed (project tz) — `due` has passed. */
  overdue: boolean;
}

export type WorkbenchComputedType = 'overdue' | 'broken_automation' | 'paused_thread';

/** A daemon-computed card (zero LLM) — overdue dated fact, broken automation,
 *  or a paused thread. `key` is the dismissal/identity key: entry id for
 *  `overdue`, trigger id/name for `broken_automation`, session uuid for
 *  `paused_thread` (spec §6 — "Dismiss keyed by session id"). */
export interface WorkbenchComputedCard {
  type: WorkbenchComputedType;
  project_id: string;
  key: string;
  text: string;
  since: string | null;
}

/** Response of `GET /api/v2/workbench`. */
export interface WorkbenchResponse {
  entries: WorkbenchEntry[];
  computed: WorkbenchComputedCard[];
}

/** A merged, render-ready row — the "ONE list, no bands" surface (spec §6)
 *  interleaves flagged entries and computed cards into a single sort. */
export type WorkbenchListItem =
  | { kind: 'entry'; data: WorkbenchEntry }
  | { kind: 'computed'; data: WorkbenchComputedCard };
