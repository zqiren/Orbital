// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { createFrameTextBuffer } from './frameTextBuffer';

/** A frame clock the test advances by hand. */
function manualFrames() {
  let next = 1;
  const queued = new Map<number, () => void>();
  return {
    schedule: (fn: () => void) => {
      const id = next++;
      queued.set(id, fn);
      return id;
    },
    cancel: (id: number) => {
      queued.delete(id);
    },
    run() {
      const fns = [...queued.values()];
      queued.clear();
      fns.forEach((fn) => fn());
    },
    get pending() {
      return queued.size;
    },
  };
}

type Commit = { key: string; text: string; meta: string };

function setup() {
  const frames = manualFrames();
  const commits: Commit[] = [];
  const buf = createFrameTextBuffer<string, string>(
    (key, text, meta) => commits.push({ key, text, meta }),
    frames.schedule,
    frames.cancel,
  );
  return { frames, commits, buf };
}

describe('createFrameTextBuffer', () => {
  it('N pushes within one frame commit once, with the full concatenation', () => {
    const { frames, commits, buf } = setup();
    const parts = ['Hel', 'lo', ', ', 'wor', 'ld'];
    for (const p of parts) buf.push('s1', p, 'management');
    expect(commits).toEqual([]);
    expect(frames.pending).toBe(1);
    frames.run();
    expect(commits).toEqual([{ key: 's1', text: 'Hello, world', meta: 'management' }]);
  });

  it('pushes across frames commit once per frame, in order, losing nothing', () => {
    const { frames, commits, buf } = setup();
    buf.push('s1', 'a', 'm');
    buf.push('s1', 'b', 'm');
    frames.run();
    buf.push('s1', 'c', 'm');
    frames.run();
    frames.run(); // an idle frame commits nothing
    buf.push('s1', 'd', 'm');
    buf.push('s1', 'e', 'm');
    frames.run();
    expect(commits.map((c) => c.text)).toEqual(['ab', 'c', 'de']);
    expect(commits.map((c) => c.text).join('')).toBe('abcde');
  });

  it('flush() commits what is pending now and cancels the scheduled frame', () => {
    const { frames, commits, buf } = setup();
    buf.push('s1', 'x', 'm');
    buf.push('s1', 'y', 'm');
    buf.flush();
    expect(commits).toEqual([{ key: 's1', text: 'xy', meta: 'm' }]);
    expect(frames.pending).toBe(0);
    frames.run();
    expect(commits).toHaveLength(1);
    buf.flush(); // nothing pending: no commit
    expect(commits).toHaveLength(1);
  });

  it('discard() drops pending text; nothing commits later', () => {
    const { frames, commits, buf } = setup();
    buf.push('s1', 'gone', 'm');
    buf.discard();
    frames.run();
    buf.flush();
    expect(commits).toEqual([]);
    // Still usable afterwards.
    buf.push('s1', 'next', 'm');
    frames.run();
    expect(commits).toEqual([{ key: 's1', text: 'next', meta: 'm' }]);
  });

  it('a push under a different key drops the other key\'s pending text', () => {
    const { frames, commits, buf } = setup();
    buf.push('s1', 'old-session', 'm');
    buf.push('s2', 'new', 'm');
    frames.run();
    expect(commits).toEqual([{ key: 's2', text: 'new', meta: 'm' }]);
  });

  it('meta is the latest push\'s', () => {
    const { frames, commits, buf } = setup();
    buf.push('s1', 'a', 'first');
    buf.push('s1', 'b', 'second');
    frames.run();
    expect(commits).toEqual([{ key: 's1', text: 'ab', meta: 'second' }]);
  });

  it('an empty-text push still commits (it is still a delta)', () => {
    const { frames, commits, buf } = setup();
    buf.push('s1', '', 'm');
    frames.run();
    expect(commits).toEqual([{ key: 's1', text: '', meta: 'm' }]);
  });

  it('a scheduler that runs the callback synchronously keeps working', () => {
    const commits: string[] = [];
    const buf = createFrameTextBuffer<string, null>(
      (_k, text) => commits.push(text),
      (fn) => { fn(); return 7; },
      () => {},
    );
    buf.push('s1', 'a', null);
    buf.push('s1', 'b', null);
    buf.push('s1', 'c', null);
    expect(commits).toEqual(['a', 'b', 'c']);
  });
});
