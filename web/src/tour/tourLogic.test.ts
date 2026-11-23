// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { orderTourStops, placeCard } from './tourLogic';

describe('orderTourStops', () => {
  const ids = (xs: { id: string }[]) => xs.map((x) => x.id);
  const stops = (...names: string[]) => names.map((id) => ({ id }));

  it('keeps the listed order when there are no left-off cards (no history → today\'s tour)', () => {
    expect(ids(orderTourStops(stops('project', 'agents', 'panel', 'files')))).toEqual(['project', 'agents', 'panel', 'files']);
  });

  it('puts the left-off cards first and the agents stop second', () => {
    expect(ids(orderTourStops(stops('leftOff', 'project', 'agents', 'panel', 'quickTasks')))).toEqual(
      ['leftOff', 'agents', 'project', 'panel', 'quickTasks'],
    );
  });

  it('copes with no agents stop (no sub-agents installed)', () => {
    expect(ids(orderTourStops(stops('project', 'leftOff', 'files')))).toEqual(['leftOff', 'project', 'files']);
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

  it('flips above for an anchor near the bottom of the window', () => {
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
