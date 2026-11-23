// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The Welcome page's folding and its Start gate (batch 1123), kept pure so
 * the rules can be tested without a DOM.
 *
 * Three bands: Agents, Projects (only when the history scan found something),
 * the Orbital Agent. Everything starts open. Folded, a band is one line that
 * says its value; the Projects band's line is the import line itself, so it
 * has no folded form of its own — "查看全部 / Review" turns it into the
 * working area instead.
 *
 *  - Review takes the room: Agents and the Orbital Agent fold while it is
 *    open, and closing it (完成, its header, Esc) puts back what was there.
 *  - Starting to work in the Orbital band (focusing the key field, picking a
 *    tile) folds Agents. A passing key never folds the Orbital band.
 *  - A band folded by hand stays folded until it is opened by hand. Opening
 *    a band by hand ends the focus that pushed it aside.
 *  - With nothing ready (no signed-in agent, the Orbital Agent not
 *    connected) the bands that hold the way in never fold by themselves.
 */

export type FoldBand = 'agents' | 'orb';
export type Focus = null | 'orb' | 'review';

export interface FoldState {
  /** Folded by hand. */
  col: Record<FoldBand, boolean>;
  focus: Focus;
  /** What review replaced, restored when it closes. */
  prevFocus: Focus;
}

export interface FoldInputs {
  /** Agents installed AND signed in. */
  ready: number;
  /** The Orbital Agent passed its test or signed in. */
  connected: boolean;
  /** The Projects band is on screen. */
  hasProjects: boolean;
}

export const INITIAL_FOLD: FoldState = { col: { agents: false, orb: false }, focus: null, prevFocus: null };

/** Nothing can work yet: these bands hold the way in. */
export function isLocked(band: FoldBand, inp: FoldInputs): boolean {
  return inp.ready === 0 && !inp.connected && (band === 'agents' || band === 'orb');
}

export function isReviewing(s: FoldState, inp: FoldInputs): boolean {
  return s.focus === 'review' && inp.hasProjects;
}

export function isFolded(band: FoldBand, s: FoldState, inp: FoldInputs): boolean {
  if (s.col[band]) return true;
  if (isLocked(band, inp)) return false;
  if (band === 'agents') return isReviewing(s, inp) || s.focus === 'orb';
  return isReviewing(s, inp);
}

/** A band header was clicked. */
export function toggleBand(s: FoldState, band: FoldBand, inp: FoldInputs): FoldState {
  if (!isFolded(band, s, inp)) {
    return {
      ...s,
      col: { ...s.col, [band]: true },
      focus: band === 'orb' && s.focus === 'orb' ? null : s.focus,
    };
  }
  // Opened by hand: it stays open, and the focus that pushed it aside ends.
  let focus = s.focus;
  let prevFocus = s.prevFocus;
  if (focus === 'review') { focus = null; prevFocus = null; }
  if (focus === 'orb' && band !== 'orb') focus = null;
  return { col: { ...s.col, [band]: false }, focus, prevFocus };
}

/** The user started working in the Orbital band (key field, a tile). */
export function engageOrb(s: FoldState): FoldState {
  return s.focus === null ? { ...s, focus: 'orb' } : s;
}

/** 查看全部 / 完成 / the Projects header. */
export function toggleReview(s: FoldState, inp: FoldInputs): FoldState {
  if (isReviewing(s, inp)) return closeReview(s);
  return { ...s, prevFocus: s.focus, focus: 'review' };
}

/** Esc, 完成: put back what was there before. */
export function closeReview(s: FoldState): FoldState {
  if (s.focus !== 'review') return s;
  return { ...s, focus: s.prevFocus === 'orb' ? 'orb' : null, prevFocus: null };
}

/**
 * The Start gate: at least one agent installed AND signed in, or the Orbital
 * Agent connected (a passing key or a sign-in). A signed-out agent does not
 * count: it could not answer the first message.
 */
export function canStart(inp: Pick<FoldInputs, 'ready' | 'connected'>): boolean {
  return inp.ready > 0 || inp.connected;
}
