// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * sessionStatus.ts — maps AgentRunStatus → display glyph + CSS color token.
 *
 * These glyphs/colors come from the V1 mockup. The color tokens are defined in
 * web/src/index.css under @theme:
 *   success → #22C55E   (running)
 *   accent  → #6366F1   (waiting)
 *   warning → #F59E0B   (pending_approval / Blocked)
 *   idle    → #A1A1AA   (idle)
 *
 * `stopped` sessions are treated as ARCHIVED in the UI (grouped separately in
 * the sidebar's archived section).
 */

import type { AgentRunStatus } from '../types';

export interface StatusDisplay {
  glyph: string;
  /** Inline CSS color value (matches the @theme token). */
  color: string;
  /** Human-readable label (used for aria and text). */
  label: string;
}

/**
 * Display map covering every AgentRunStatus value. The four primary states
 * (running / waiting / pending_approval / idle) match the V1 mockup exactly.
 * The remaining states (stopped / new_session / error) get sensible glyph +
 * label entries so rows that render for them announce correctly (rather than
 * falling back to a misleading "Idle"):
 *   - stopped     → muted ◌, "Archived" (shown in the archived section).
 *   - new_session → accent ◐, "Starting" (active-looking; a session that just began).
 *   - error       → warning ⚠, "Error" (the persistent error indicator is also
 *                   rendered separately via SessionStatusGlyph).
 */
const STATUS_DISPLAY_MAP: Record<AgentRunStatus, StatusDisplay> = {
  running: { glyph: '◐', color: '#22C55E', label: 'Running' },
  waiting: { glyph: '⟳', color: '#6366F1', label: 'Waiting' },
  // pending_approval is rendered as "Blocked" in ALL UI copy per spec.
  pending_approval: { glyph: '⚠', color: '#F59E0B', label: 'Blocked' },
  idle: { glyph: '⏸', color: '#A1A1AA', label: 'Idle' },
  stopped: { glyph: '◌', color: '#A1A1AA', label: 'Archived' },
  new_session: { glyph: '◐', color: '#6366F1', label: 'Starting' },
  error: { glyph: '⚠', color: '#F59E0B', label: 'Error' },
};

const FALLBACK_STATUS: StatusDisplay = { glyph: '⏸', color: '#A1A1AA', label: 'Idle' };

export function getStatusDisplay(status: AgentRunStatus): StatusDisplay {
  return STATUS_DISPLAY_MAP[status] ?? FALLBACK_STATUS;
}

/**
 * Returns true for statuses that are considered "archived".
 * Only `stopped` is archived; everything else is treated as active.
 */
export function isArchivedStatus(status: AgentRunStatus): boolean {
  return status === 'stopped';
}

/**
 * Returns true for statuses that are considered "active" (i.e. NOT archived).
 * Defined as the complement of isArchivedStatus so that no status value —
 * including `error`, `new_session`, or any future enum addition — can fall
 * through both predicates and vanish from the sidebar (DO-NOT #12: ALL
 * sessions must appear). An erroring session stays visible in the active
 * list and surfaces its error indicator via SessionStatusGlyph.
 */
export function isActiveStatus(status: AgentRunStatus): boolean {
  return !isArchivedStatus(status);
}
