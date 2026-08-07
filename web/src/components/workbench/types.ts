// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Frontend mirror of the Workbench surface's wire shapes (spec §5.3, §5.4,
 * §8). These match `GET /api/v2/workbench` and the exit/migrate routes
 * byte-for-byte — see `agent_os/api/routes/workbench.py`. Card tap is a
 * client-side prefill (no doorway POST) — see WorkbenchPage.tsx.
 */

/** One flagged `[user]` entry, lensed to a project. */
export interface WorkbenchEntry {
  project_id: string;
  /** The daemon-assigned id. The backend heals id-less flagged entries on read
   *  (bug #45), but a legacy/stale payload can still put `null` on the wire —
   *  so the frontend must treat this as nullable and never key on it alone. */
  id: string | null;
  text: string;
  due: string | null;
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

/** Response of `GET /api/v2/workbench`. */
export interface WorkbenchResponse {
  entries: WorkbenchEntry[];
}
