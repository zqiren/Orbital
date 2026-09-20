// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * First-journey tour — the decisions, kept pure so they can be tested without
 * a DOM: who gets the tour, what triggers it, and where a coachmark goes.
 */

import type { Project } from '../types';

type ProjectLike = Pick<Project, 'project_id' | 'is_scratch'>;

/** Per-device, like `orbital.locale` — no backend. */
export const TOUR_STORAGE_KEY = 'orbital.tour.firstJourney';

/** `pending`: a new user who has not seen it yet. `done`: seen, skipped, or
 * opted out. */
export type TourState = 'pending' | 'done';

/**
 * Decide, once, whether this device's user is owed the tour.
 *
 * The first time a LOADED project list is seen with no stored decision: only
 * Quick Tasks → `pending`; already has a project → `done`. That second case is
 * the upgrade path — someone with hundreds of sessions must not be ambushed by
 * a tour because a new flag shipped. They can still replay it from Settings.
 *
 * Returns null while undecidable: an empty list means "not loaded / fetch
 * failed" (a loaded list always contains Quick Tasks), never "new user".
 */
export function decideTourState(
  stored: string | null,
  projects: readonly ProjectLike[],
): TourState | null {
  if (stored === 'pending' || stored === 'done') return stored;
  if (projects.length === 0) return null;
  return projects.some((p) => !p.is_scratch) ? 'done' : 'pending';
}

/** The trigger: entering a project the user created, while still pending.
 * Not at wizard completion — no project exists then, so most stops would have
 * nothing to point at. */
export function shouldStartTour(
  state: TourState | null,
  project: ProjectLike | undefined,
): boolean {
  return state === 'pending' && !!project && !project.is_scratch;
}

export function readTourState(): string | null {
  try {
    return localStorage.getItem(TOUR_STORAGE_KEY);
  } catch {
    return null; // storage unavailable (private/locked-down webview)
  }
}

export function writeTourState(state: TourState): void {
  try {
    localStorage.setItem(TOUR_STORAGE_KEY, state);
  } catch {
    /* storage unavailable — the tour simply is not remembered */
  }
}

// ── Coachmark placement ─────────────────────────────────────────────────────

export interface Box { left: number; top: number; width: number; height: number }
export interface Size { width: number; height: number }
export type CardSide = 'bottom' | 'top' | 'left' | 'right' | 'over';

const GAP = 14; // anchor ↔ card
const MARGIN = 12; // card ↔ viewport edge

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(v, max));

/**
 * Where the coachmark card goes: below the anchor if it fits, else above, else
 * beside it (tall anchors such as the docked panel), else over it. Always
 * fully inside the viewport.
 */
export function placeCard(
  anchor: Box,
  card: Size,
  viewport: Size,
): { left: number; top: number; side: CardSide } {
  const maxLeft = viewport.width - card.width - MARGIN;
  const maxTop = viewport.height - card.height - MARGIN;
  const centredLeft = clamp(anchor.left + anchor.width / 2 - card.width / 2, MARGIN, maxLeft);
  const centredTop = clamp(anchor.top + anchor.height / 2 - card.height / 2, MARGIN, maxTop);

  const below = anchor.top + anchor.height + GAP;
  if (below + card.height + MARGIN <= viewport.height) {
    return { left: centredLeft, top: below, side: 'bottom' };
  }
  const above = anchor.top - GAP - card.height;
  if (above >= MARGIN) return { left: centredLeft, top: above, side: 'top' };

  const leftOf = anchor.left - GAP - card.width;
  if (leftOf >= MARGIN) return { left: leftOf, top: centredTop, side: 'left' };
  const rightOf = anchor.left + anchor.width + GAP;
  if (rightOf + card.width + MARGIN <= viewport.width) {
    return { left: rightOf, top: centredTop, side: 'right' };
  }
  return { left: centredLeft, top: centredTop, side: 'over' };
}
