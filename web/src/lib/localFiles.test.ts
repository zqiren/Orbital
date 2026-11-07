// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  canPickLocalFiles, nativeDropAvailable, splitDrop, waitForNativeDrop,
} from './localFiles';

type Win = { pywebview?: unknown; __orbitalNativeDrop?: boolean };

afterEach(() => {
  delete (window as unknown as Win).pywebview;
  delete (window as unknown as Win).__orbitalNativeDrop;
  vi.useRealTimers();
});

const e = (name: string) => ({ path: `/x/${name}`, name, size: 1, mime: 'text/plain' });

describe('localFiles', () => {
  it('only a shell with pick_files counts as the desktop app', () => {
    expect(canPickLocalFiles()).toBe(false);
    (window as unknown as Win).pywebview = { api: { pick_folder: () => null } };
    expect(canPickLocalFiles()).toBe(false);
    (window as unknown as Win).pywebview = { api: { pick_folder: () => null, pick_files: async () => [] } };
    expect(canPickLocalFiles()).toBe(true);
    expect(nativeDropAvailable()).toBe(false);
    (window as unknown as Win).__orbitalNativeDrop = true;
    expect(nativeDropAvailable()).toBe(true);
  });

  it('splitDrop matches by name, each path once', () => {
    const files = ['a.txt', 'a.txt', 'b.txt'].map((n) => new File(['x'], n));
    const out = splitDrop(files, [e('a.txt'), e('b.txt')]);
    expect(out.local.map((x) => x.name)).toEqual(['a.txt', 'b.txt']);
    expect(out.upload.map((f) => f.name)).toEqual(['a.txt']);
  });

  it('waitForNativeDrop resolves with the shell event, or empty on timeout', async () => {
    const p = waitForNativeDrop(1000);
    window.dispatchEvent(new CustomEvent('orbital:native-drop', { detail: [e('a.txt'), { bad: 1 }] }));
    await expect(p).resolves.toEqual([e('a.txt')]);
    vi.useFakeTimers();
    const q = waitForNativeDrop(50);
    vi.advanceTimersByTime(60);
    await expect(q).resolves.toEqual([]);
  });
});
