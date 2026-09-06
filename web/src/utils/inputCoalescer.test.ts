// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { createInputCoalescer, type PointerMessage } from './inputCoalescer';

/** A scheduler the test ticks by hand — one `tick()` = one animation frame. */
function fakeScheduler() {
  let queued: (() => void) | null = null;
  return {
    schedule: (cb: () => void) => {
      queued = cb;
      return 1;
    },
    cancel: () => {
      queued = null;
    },
    tick() {
      const cb = queued;
      queued = null;
      cb?.();
    },
  };
}

function setup() {
  const sent: PointerMessage[] = [];
  const sched = fakeScheduler();
  const c = createInputCoalescer((m) => sent.push(m), sched);
  return { sent, sched, c };
}

describe('inputCoalescer', () => {
  it('sends the first event of a quiet period immediately', () => {
    const { sent, c } = setup();
    c.move(10, 20, 0);
    expect(sent).toEqual([{ type: 'mouse', action: 'move', x: 10, y: 20, modifiers: 0 }]);
  });

  it('holds further moves until the next frame and keeps only the newest', () => {
    const { sent, sched, c } = setup();
    c.move(1, 1, 0);
    c.move(2, 2, 0);
    c.move(3, 3, 8);
    expect(sent).toHaveLength(1);
    sched.tick();
    expect(sent).toHaveLength(2);
    expect(sent[1]).toEqual({ type: 'mouse', action: 'move', x: 3, y: 3, modifiers: 8 });
  });

  it('adds up wheel deltas that arrive within one frame', () => {
    const { sent, sched, c } = setup();
    c.wheel(5, 5, 0, 8, 0);
    for (let i = 0; i < 10; i++) c.wheel(6, 6, 1, 8, 0);
    expect(sent).toHaveLength(1);
    sched.tick();
    expect(sent).toHaveLength(2);
    expect(sent[1]).toEqual({ type: 'mouse', action: 'wheel', x: 6, y: 6, deltaX: 10, deltaY: 80, modifiers: 0 });
  });

  it('sends a pending wheel before a pending move', () => {
    const { sent, sched, c } = setup();
    c.move(0, 0, 0);
    c.move(9, 9, 0);
    c.wheel(9, 9, 0, 30, 0);
    sched.tick();
    expect(sent.slice(1).map((m) => m.action)).toEqual(['wheel', 'move']);
  });

  it('a frame with nothing pending sends nothing and re-arms on the next event', () => {
    const { sent, sched, c } = setup();
    c.move(0, 0, 0);
    sched.tick();
    expect(sent).toHaveLength(1);
    c.move(1, 1, 0);
    expect(sent).toHaveLength(2); // quiet again → immediate
  });

  it('flush sends what is pending at once', () => {
    const { sent, c } = setup();
    c.wheel(0, 0, 0, 10, 0);
    c.wheel(0, 0, 0, 10, 0);
    c.flush();
    expect(sent).toHaveLength(2);
    expect(sent[1].deltaY).toBe(10);
  });

  it('cancel drops what is pending', () => {
    const { sent, sched, c } = setup();
    c.move(0, 0, 0);
    c.move(5, 5, 0);
    c.cancel();
    sched.tick();
    expect(sent).toHaveLength(1);
  });
});
