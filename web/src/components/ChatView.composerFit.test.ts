// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * fitComposerHeight: the shrink re-measure collapses the textarea to `auto`
 * for one forced layout. During that layout the composer's footprint must not
 * shrink, or the chat above gets taller, the browser clamps its scrollTop and
 * a view pinned to the bottom is left displaced after the restore.
 */
import { describe, expect, it } from 'vitest';
import { fitComposerHeight } from './ChatView';

function instrument(contentHeight: number, height = '') {
  const ta = document.createElement('textarea');
  const log: string[] = [];
  const state = { contentHeight, height, margin: '' };
  Object.defineProperty(ta, 'scrollHeight', {
    configurable: true,
    get: () => {
      log.push(`measure(height=${state.height}, margin=${state.margin})`);
      return state.contentHeight;
    },
  });
  Object.defineProperty(ta.style, 'height', {
    configurable: true,
    get: () => state.height,
    set: (v: string) => { log.push(`height=${v}`); state.height = v; },
  });
  Object.defineProperty(ta.style, 'marginBottom', {
    configurable: true,
    get: () => state.margin,
    set: (v: string) => { log.push(`margin=${v}`); state.margin = v; },
  });
  return { ta, log, state };
}

describe('fitComposerHeight', () => {
  it('a shrink holds the old height as a bottom margin while it measures, then drops it', () => {
    const { ta, log, state } = instrument(24, '113px');
    fitComposerHeight(ta, true);
    expect(log).toEqual([
      'height=auto',
      'margin=113px',
      'measure(height=auto, margin=113px)',
      'margin=',
      'height=24px',
    ]);
    expect(state.margin).toBe('');
  });

  it('growth (no reset) touches neither the margin nor an unchanged height', () => {
    const { ta, log } = instrument(44, '44px');
    fitComposerHeight(ta, false);
    expect(log).toEqual(['measure(height=44px, margin=)']);
  });

  it('a first fit with no height set yet has nothing to hold', () => {
    const { ta, log } = instrument(24, '');
    fitComposerHeight(ta, true);
    expect(log).toEqual(['height=auto', 'measure(height=auto, margin=)', 'height=24px']);
  });

  it('caps at 160px', () => {
    const { ta, state } = instrument(500, '44px');
    fitComposerHeight(ta, false);
    expect(state.height).toBe('160px');
  });
});
