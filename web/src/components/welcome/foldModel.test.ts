// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import {
  canStart, closeReview, engageOrb, INITIAL_FOLD, isFolded, isLocked, isReviewing, toggleBand, toggleReview,
  type FoldInputs,
} from './foldModel';

const READY: FoldInputs = { ready: 2, connected: false, hasProjects: true };
const NOTHING: FoldInputs = { ready: 0, connected: false, hasProjects: true };

describe('Welcome folding', () => {
  it('starts with every band open', () => {
    expect(isFolded('agents', INITIAL_FOLD, READY)).toBe(false);
    expect(isFolded('orb', INITIAL_FOLD, READY)).toBe(false);
    expect(isReviewing(INITIAL_FOLD, READY)).toBe(false);
  });

  it('review folds the other bands and closing it restores them', () => {
    const open = toggleReview(INITIAL_FOLD, READY);
    expect(isReviewing(open, READY)).toBe(true);
    expect(isFolded('agents', open, READY)).toBe(true);
    expect(isFolded('orb', open, READY)).toBe(true);
    const closed = closeReview(open);
    expect(isFolded('agents', closed, READY)).toBe(false);
    expect(isFolded('orb', closed, READY)).toBe(false);
    // 完成 is the same toggle
    expect(toggleReview(open, READY)).toEqual(closed);
  });

  it('review restores the Orbital focus it replaced', () => {
    const working = engageOrb(INITIAL_FOLD);
    expect(isFolded('agents', working, READY)).toBe(true);
    const back = closeReview(toggleReview(working, READY));
    expect(back.focus).toBe('orb');
    expect(isFolded('agents', back, READY)).toBe(true);
    expect(isFolded('orb', back, READY)).toBe(false);
  });

  it('no Projects band: review never takes the room', () => {
    const inp = { ...READY, hasProjects: false };
    const s = toggleReview(INITIAL_FOLD, inp);
    expect(isReviewing(s, inp)).toBe(false);
    expect(isFolded('orb', s, inp)).toBe(false);
  });

  it('working in the Orbital band folds Agents, never the Orbital band', () => {
    const s = engageOrb(INITIAL_FOLD);
    expect(isFolded('agents', s, READY)).toBe(true);
    expect(isFolded('orb', s, READY)).toBe(false);
    // a passing key changes nothing about the Orbital band
    expect(isFolded('orb', s, { ...READY, connected: true })).toBe(false);
    // engaging again is a no-op
    expect(engageOrb(s)).toBe(s);
  });

  it('a band folded by hand stays folded through focus changes', () => {
    const s = toggleBand(INITIAL_FOLD, 'orb', READY);
    expect(isFolded('orb', s, READY)).toBe(true);
    const reviewed = closeReview(toggleReview(s, READY));
    expect(isFolded('orb', reviewed, READY)).toBe(true);
    expect(isFolded('orb', engageOrb(reviewed), READY)).toBe(true);
  });

  it('opening a band by hand ends the focus that pushed it aside', () => {
    const working = engageOrb(INITIAL_FOLD);
    const opened = toggleBand(working, 'agents', READY);
    expect(isFolded('agents', opened, READY)).toBe(false);
    expect(opened.focus).toBe(null);
    const reviewing = toggleReview(INITIAL_FOLD, READY);
    const orbOpen = toggleBand(reviewing, 'orb', READY);
    expect(isReviewing(orbOpen, READY)).toBe(false);
    expect(isFolded('orb', orbOpen, READY)).toBe(false);
  });

  it('folding the Orbital band by hand ends its own focus', () => {
    const working = engageOrb(INITIAL_FOLD);
    const s = toggleBand(working, 'orb', READY);
    expect(s.focus).toBe(null);
    expect(isFolded('agents', s, READY)).toBe(false);
  });

  it('with nothing ready, the bands that hold the way in never fold by themselves', () => {
    expect(isLocked('agents', NOTHING)).toBe(true);
    expect(isLocked('orb', NOTHING)).toBe(true);
    const working = engageOrb(INITIAL_FOLD);
    expect(isFolded('agents', working, NOTHING)).toBe(false);
    const reviewing = toggleReview(INITIAL_FOLD, NOTHING);
    expect(isFolded('agents', reviewing, NOTHING)).toBe(false);
    expect(isFolded('orb', reviewing, NOTHING)).toBe(false);
    // ...but the user can still fold one by hand
    expect(isFolded('agents', toggleBand(INITIAL_FOLD, 'agents', NOTHING), NOTHING)).toBe(true);
    // once the Orbital Agent connects, the rules apply again
    expect(isFolded('agents', working, { ...NOTHING, connected: true })).toBe(true);
  });
});

describe('Start gate', () => {
  it('needs a ready agent or a connected Orbital Agent', () => {
    expect(canStart({ ready: 0, connected: false })).toBe(false);
    expect(canStart({ ready: 1, connected: false })).toBe(true);
    expect(canStart({ ready: 0, connected: true })).toBe(true);
  });
});
