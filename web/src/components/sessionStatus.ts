// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * sessionStatus.ts — maps AgentRunStatus → display glyph + CSS color token.
 *
 * These glyphs come from the V1 mockup. The colours are the readable status
 * tokens in web/src/index.css (spec 145; the bright fills read at ~2:1 on the
 * light sidebar), referenced as CSS variables so they follow the theme:
 *   success-text → #15803D / dark #4ADE80   (running)
 *   accent-text  → #2563EB / dark #539AF8   (waiting, queued, starting)
 *   warning-text → #B45309 / dark #FBBF24   (pending_approval / Blocked, error)
 *   idle         → #A1A1AA                  (idle; resting rows show no glyph)
 */

import type { AgentRunStatus, SessionListEntry } from '../types';
import type { StringKey } from '../i18n/strings';
import { displayRunStatus } from '../utils/projectStatus';

export interface StatusDisplay {
  glyph: string;
  /** Inline CSS color value (a `var(--color-*)` theme token). */
  color: string;
  /** Human-readable label (used for aria and text). English — this map is
   *  module-level and cannot call the useT() hook. */
  label: string;
  /**
   * Catalog key for `label`, where one exists. A caller that renders the
   * label (none does today — the glyph is aria-hidden and the label is
   * unused) passes this through t() and falls back to `label`, which stays
   * byte-identical English. Only states added after the i18n catalog carry
   * one; the six original labels predate it.
   */
  labelKey?: StringKey;
  /**
   * True for the resting state (idle, and the unknown-status fallback).
   * List rows render no glyph for resting sessions — a repeated ⏸ on every
   * idle row conveys nothing and reads as noise; only differentiating states
   * (running / waiting / blocked / starting / error) light up.
   */
  resting?: boolean;
}

/**
 * Display map covering every AgentRunStatus value. The four primary states
 * (running / waiting / pending_approval / idle) match the V1 mockup exactly.
 * The remaining states (new_session / error) get sensible glyph + label
 * entries so rows that render for them announce correctly (rather than
 * falling back to a misleading "Idle"):
 *   - new_session → accent ◐, "Starting" (active-looking; a session that just began).
 *   - error       → warning ⚠, "Error" (the persistent error indicator is also
 *                   rendered separately via SessionStatusGlyph).
 *   - queued      → accent ◔, "Queued" (spec 081; a first message waiting for
 *                   the project slot — nothing has run yet, so it is NOT
 *                   resting: the row must light up).
 */
const STATUS_DISPLAY_MAP: Record<AgentRunStatus, StatusDisplay> = {
  running: { glyph: '◐', color: 'var(--color-success-text)', label: 'Running' },
  waiting: { glyph: '⟳', color: 'var(--color-accent-text)', label: 'Waiting' },
  // Waiting's blue: both are "not my turn yet". A distinct glyph (a quarter
  // circle, less full than waiting's ⟳ spin) keeps the two readable apart.
  queued: { glyph: '◔', color: 'var(--color-accent-text)', label: 'Queued', labelKey: 'session.status.queued' },
  // pending_approval is rendered as "Blocked" in ALL UI copy per spec.
  pending_approval: { glyph: '⚠', color: 'var(--color-warning-text)', label: 'Blocked' },
  idle: { glyph: '⏸', color: '#A1A1AA', label: 'Idle', resting: true },
  new_session: { glyph: '◐', color: 'var(--color-accent-text)', label: 'Starting' },
  error: { glyph: '⚠', color: 'var(--color-warning-text)', label: 'Error' },
};

const FALLBACK_STATUS: StatusDisplay = { glyph: '⏸', color: '#A1A1AA', label: 'Idle', resting: true };

export function getStatusDisplay(status: AgentRunStatus): StatusDisplay {
  return STATUS_DISPLAY_MAP[status] ?? FALLBACK_STATUS;
}

/**
 * The status a session-list row displays (spec 102): the manager's `status`,
 * or `running` while a worker has an open turn in that session and the
 * manager rests. `worker_running` is additive and may be absent (older
 * daemon) — absent reads as false, so every pre-102 entry renders as before.
 */
export function rowDisplayStatus(
  entry: Pick<SessionListEntry, 'status' | 'worker_running'>,
): AgentRunStatus {
  return displayRunStatus(entry.status, entry.worker_running ?? false);
}
