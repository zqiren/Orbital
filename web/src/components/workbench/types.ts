// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Frontend mirror of the Workbench surface's wire shapes (spec §5.3, §5.4,
 * §8). These match `GET /api/v2/workbench` and the exit/open/migrate routes
 * byte-for-byte — see `agent_os/api/routes/workbench.py`.
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
  /** Server-computed (project tz) — whole days `due` is past; null unless
   *  `overdue` is true. Never recompute this client-side (spec §7.3) — the
   *  browser's local clock/tz can disagree with the project tz. */
  days_late: number | null;
  /** The `## ` heading in PROJECT_STATE.md this entry was found under, or
   *  null when the entry has no section (e.g. not sourced from the file). */
  section: string | null;
}

/** Per-project "what's in flight" summary — global (unlensed) view only. */
export interface WorkbenchDigest {
  project_id: string;
  /** Raw markdown lines, or null when there's nothing to show. */
  in_progress: string | null;
  /** Raw markdown lines, or null when there's nothing to show. */
  next_steps: string | null;
}

/** Response of `GET /api/v2/workbench`. */
export interface WorkbenchResponse {
  entries: WorkbenchEntry[];
  /** Only projects where at least one side is non-null; server-filtered for
   *  privacy already (no need to re-check workbench_exclude_global here). */
  digests?: WorkbenchDigest[];
}
