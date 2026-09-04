// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Merges the live browser view's pointer traffic down to one message of
 * each kind per animation frame (spec 078 §5.6, 2026-09-04 "scrolling feels
 * very laggy").
 *
 * A trackpad emits ~120 wheel events a second and a mouse ~120 moves; each
 * one used to become its own WebSocket message and its own ~20 ms round
 * trip in Chrome, so a two-second swipe queued five seconds of scrolling.
 * Here the first event of a quiet period goes out immediately (lowest
 * latency for a single click-and-look), and everything that follows within
 * the same frame is merged: moves keep the newest position, wheels add their
 * deltas. The route does the same merge on its side, so nothing can pile up
 * between the two.
 */

export type PointerMessage = {
  type: 'mouse';
  action: 'move' | 'wheel';
  x: number;
  y: number;
  modifiers: number;
  deltaX?: number;
  deltaY?: number;
}

export interface InputCoalescer {
  move(x: number, y: number, modifiers: number): void;
  wheel(x: number, y: number, deltaX: number, deltaY: number, modifiers: number): void;
  /** Send whatever is pending now (e.g. before a click, so it lands on the scrolled page). */
  flush(): void;
  /** Drop anything pending and stop the scheduled flush. */
  cancel(): void;
}

type Scheduler = (cb: () => void) => number;
type Canceller = (id: number) => void;

const FRAME_MS = 16;

function defaultScheduler(): { schedule: Scheduler; cancel: Canceller } {
  if (typeof requestAnimationFrame === 'function' && typeof cancelAnimationFrame === 'function') {
    return { schedule: (cb) => requestAnimationFrame(cb), cancel: (id) => cancelAnimationFrame(id) };
  }
  return {
    schedule: (cb) => setTimeout(cb, FRAME_MS) as unknown as number,
    cancel: (id) => clearTimeout(id as unknown as ReturnType<typeof setTimeout>),
  };
}

export function createInputCoalescer(
  send: (msg: PointerMessage) => void,
  scheduler: { schedule: Scheduler; cancel: Canceller } = defaultScheduler(),
): InputCoalescer {
  let pendingMove: PointerMessage | null = null;
  let pendingWheel: PointerMessage | null = null;
  let scheduled: number | null = null;

  function flushPending() {
    scheduled = null;
    // Wheel first: a following move should be interpreted on the scrolled page.
    const wheel = pendingWheel;
    const move = pendingMove;
    pendingWheel = null;
    pendingMove = null;
    if (wheel) send(wheel);
    if (move) send(move);
  }

  function arm() {
    if (scheduled === null) scheduled = scheduler.schedule(flushPending);
  }

  return {
    move(x, y, modifiers) {
      const msg: PointerMessage = { type: 'mouse', action: 'move', x, y, modifiers };
      if (scheduled === null) {
        // Leading edge: nothing in flight this frame — go now, then hold the
        // rest of the frame's traffic.
        send(msg);
        arm();
        return;
      }
      pendingMove = msg;
    },
    wheel(x, y, deltaX, deltaY, modifiers) {
      if (scheduled === null) {
        send({ type: 'mouse', action: 'wheel', x, y, deltaX, deltaY, modifiers });
        arm();
        return;
      }
      if (pendingWheel) {
        pendingWheel.x = x;
        pendingWheel.y = y;
        pendingWheel.modifiers = modifiers;
        pendingWheel.deltaX = (pendingWheel.deltaX ?? 0) + deltaX;
        pendingWheel.deltaY = (pendingWheel.deltaY ?? 0) + deltaY;
      } else {
        pendingWheel = { type: 'mouse', action: 'wheel', x, y, deltaX, deltaY, modifiers };
      }
    },
    flush() {
      if (scheduled !== null) scheduler.cancel(scheduled);
      flushPending();
    },
    cancel() {
      if (scheduled !== null) scheduler.cancel(scheduled);
      scheduled = null;
      pendingMove = null;
      pendingWheel = null;
    },
  };
}
