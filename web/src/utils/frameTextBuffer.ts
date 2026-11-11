// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Collects streamed text and hands it on at most once per animation frame
 * (spec 171 A7). The manager's reply arrives as many small deltas; each one
 * used to be its own React commit and its own markdown parse of the whole
 * growing text. Here every delta that lands inside one frame is joined and
 * committed together, in arrival order, so the committed text is always the
 * plain concatenation of the deltas.
 *
 * `key` names the stream (the session it belongs to). A push under another
 * key drops what the old key had pending, so text can never cross streams.
 * The buffer state lives here, outside any React updater.
 */
export interface FrameTextBuffer<K, M> {
  /** Add text for `key`; `meta` (e.g. the source) is the latest push's. */
  push(key: K, text: string, meta: M): void;
  /** Commit what is pending now and cancel the scheduled frame. */
  flush(): void;
  /** Drop what is pending and cancel the scheduled frame. */
  discard(): void;
}

export function createFrameTextBuffer<K, M>(
  commit: (key: K, text: string, meta: M) => void,
  schedule: (fn: () => void) => number,
  cancel: (id: number) => void,
): FrameTextBuffer<K, M> {
  let has = false;
  let key: K | undefined;
  let text = '';
  let meta: M | undefined;
  let frameId: number | null = null;
  let scheduled = false;
  // Bumped whenever a scheduled frame is superseded, so a stale callback
  // (e.g. a scheduler that cannot cancel) does nothing.
  let generation = 0;

  function reset() {
    has = false;
    key = undefined;
    text = '';
    meta = undefined;
  }

  function unschedule() {
    if (frameId !== null) cancel(frameId);
    frameId = null;
    scheduled = false;
    generation += 1;
  }

  function flush() {
    unschedule();
    if (!has) return;
    const k = key as K;
    const t = text;
    const m = meta as M;
    reset();
    commit(k, t, m);
  }

  function discard() {
    unschedule();
    reset();
  }

  function push(k: K, t: string, m: M) {
    if (has && key !== k) reset();
    has = true;
    key = k;
    text += t;
    meta = m;
    if (scheduled) return;
    scheduled = true;
    const mine = ++generation;
    const id = schedule(() => {
      if (mine !== generation) return;
      frameId = null;
      scheduled = false;
      flush();
    });
    // A scheduler that ran the callback synchronously has already flushed.
    if (scheduled && mine === generation) frameId = id;
  }

  return { push, flush, discard };
}
