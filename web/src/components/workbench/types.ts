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
}

/** Response of `GET /api/v2/workbench`. */
export interface WorkbenchResponse {
  entries: WorkbenchEntry[];
}
