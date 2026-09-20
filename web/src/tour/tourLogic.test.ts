// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { decideTourState, placeCard, shouldStartTour } from './tourLogic';

const scratch = { project_id: 's', is_scratch: true };
const own = { project_id: 'a', is_scratch: false };

describe('decideTourState — who gets the tour', () => {
  it('a brand-new install (only Quick Tasks) is marked pending', () => {
    expect(decideTourState(null, [scratch])).toBe('pending');
  });

  it('an install that ALREADY has a project is opted out — never ambush an existing user', () => {
    expect(decideTourState(null, [scratch, own])).toBe('done');
  });

  it('decides nothing on an empty list (not loaded / fetch failed)', () => {
    // Deciding here would opt a new user out (or in) on a network blip.
    expect(decideTourState(null, [])).toBeNull();
  });

  it('never revisits a stored decision', () => {
    expect(decideTourState('pending', [scratch, own])).toBe('pending');
    expect(decideTourState('done', [scratch])).toBe('done');
  });

  it('treats a garbage stored value as undecided', () => {
    expect(decideTourState('banana', [scratch])).toBe('pending');
  });
});

describe('shouldStartTour — the trigger', () => {
  it('fires on entering a project the user created, while pending', () => {
    expect(shouldStartTour('pending', own)).toBe(true);
  });
  it('does not fire in Quick Tasks, with no project, or once done', () => {
    expect(shouldStartTour('pending', scratch)).toBe(false);
    expect(shouldStartTour('pending', undefined)).toBe(false);
    expect(shouldStartTour('done', own)).toBe(false);
    expect(shouldStartTour(null, own)).toBe(false);
  });
});

describe('placeCard', () => {
  const viewport = { width: 1280, height: 800 };
  const card = { width: 320, height: 180 };

  it('sits below the anchor when there is room, centred on it and clamped to the viewport', () => {
    const p = placeCard({ left: 600, top: 100, width: 100, height: 40 }, card, viewport);
    expect(p.side).toBe('bottom');
    expect(p.top).toBe(100 + 40 + 14);
    expect(p.left).toBe(600 + 50 - 160);

    const edge = placeCard({ left: 1240, top: 100, width: 30, height: 30 }, card, viewport);
    expect(edge.left).toBe(1280 - 320 - 12);
  });

  it('flips above for an anchor near the bottom (the composer)', () => {
    const p = placeCard({ left: 300, top: 720, width: 700, height: 60 }, card, viewport);
    expect(p.side).toBe('top');
    expect(p.top).toBe(720 - 14 - 180);
  });

  it('goes beside a tall anchor that leaves no room above or below (the docked panel)', () => {
    const p = placeCard({ left: 920, top: 60, width: 360, height: 740 }, card, viewport);
    expect(p.side).toBe('left');
    expect(p.left).toBe(920 - 14 - 320);
  });

  it('falls back to overlapping the anchor rather than leaving the screen', () => {
    const p = placeCard({ left: 0, top: 0, width: 1280, height: 800 }, card, viewport);
    expect(p.side).toBe('over');
    expect(p.left).toBeGreaterThanOrEqual(12);
    expect(p.top).toBeGreaterThanOrEqual(12);
  });
});
