// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { fitRect, frameRect } from './browserFrameRect';

describe('frameRect — where the live page is drawn', () => {
  it('owner at its own size: exactly 1:1, no band', () => {
    expect(frameRect(690, 820, 690, 820, true)).toEqual({ x: 0, y: 0, w: 690, h: 820 });
  });
  it('owner mid-drag: covers the box from the top-left, never a band', () => {
    for (const [bw, bh] of [[600, 820], [800, 820], [690, 700], [1000, 1000], [300, 400]]) {
      const r = frameRect(690, 820, bw, bh, true);
      expect(r.x).toBe(0);
      expect(r.y).toBe(0);
      expect(r.w).toBeGreaterThanOrEqual(bw - 1e-9);
      expect(r.h).toBeGreaterThanOrEqual(bh - 1e-9);
      expect(r.w / r.h).toBeCloseTo(690 / 820, 9); // never stretched
    }
  });
  it('not the owner: letterboxed, centred (fitRect)', () => {
    expect(frameRect(1280, 720, 690, 820, false)).toEqual(fitRect(1280, 720, 690, 820));
    const r = fitRect(1280, 720, 690, 820);
    expect(r.w).toBeCloseTo(690, 9);
    expect(r.y).toBeCloseTo((820 - 388.125) / 2, 9);
  });
});
