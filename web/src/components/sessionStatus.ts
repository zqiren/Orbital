// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * sessionStatus.ts — maps AgentRunStatus → display glyph + CSS color token.
 *
 * These glyphs/colors come from the V1 mockup. The color tokens are defined in
 * web/src/index.css under @theme:
 *   success → #22C55E   (running)
 *   accent  → #539AF8   (waiting)
 *   warning → #F59E0B   (pending_approval / Blocked)
 *   idle    → #A1A1AA   (idle)
 */

import type { AgentRunStatus } from '../types';

export interface StatusDisplay {
  glyph: string;
  /** Inline CSS color value (matches the @theme token). */
  color: string;
  /** Human-readable label (used for aria and text). */
  label: string;
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
 */
const STATUS_DISPLAY_MAP: Record<AgentRunStatus, StatusDisplay> = {
  running: { glyph: '◐', color: '#22C55E', label: 'Running' },
  waiting: { glyph: '⟳', color: '#539AF8', label: 'Waiting' },
  // pending_approval is rendered as "Blocked" in ALL UI copy per spec.
  pending_approval: { glyph: '⚠', color: '#F59E0B', label: 'Blocked' },
  idle: { glyph: '⏸', color: '#A1A1AA', label: 'Idle', resting: true },
  new_session: { glyph: '◐', color: '#539AF8', label: 'Starting' },
  error: { glyph: '⚠', color: '#F59E0B', label: 'Error' },
};

const FALLBACK_STATUS: StatusDisplay = { glyph: '⏸', color: '#A1A1AA', label: 'Idle', resting: true };

export function getStatusDisplay(status: AgentRunStatus): StatusDisplay {
  return STATUS_DISPLAY_MAP[status] ?? FALLBACK_STATUS;
}
