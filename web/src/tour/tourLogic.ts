// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * First-journey tour — the decisions, kept pure so they can be tested without
 * a DOM: the order of the stops and where a coachmark goes. The tour opens
 * only from Settings → About (batch 1123: never by itself).
 */

/**
 * The order of the stops that are on screen. With the left-off cards showing
 * (a project that has history), they come first and the agents stop second —
 * "open one and continue with any agent" is the point of the tour then.
 * Otherwise the order is as listed.
 */
export function orderTourStops<T extends { id: string }>(present: readonly T[]): T[] {
  const leftOff = present.find((s) => s.id === 'leftOff');
  if (!leftOff) return [...present];
  const agents = present.find((s) => s.id === 'agents');
  return [
    leftOff,
    ...(agents ? [agents] : []),
    ...present.filter((s) => s !== leftOff && s !== agents),
  ];
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
