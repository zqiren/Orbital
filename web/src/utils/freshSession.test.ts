// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { markFreshSession, consumeFreshSession } from './freshSession';

describe('freshSession (spec 107 FE-1)', () => {
  it('a mark is consumed exactly once', () => {
    markFreshSession('p1', 'sess_a');
    expect(consumeFreshSession('p1', 'sess_a')).toBe(true);
    expect(consumeFreshSession('p1', 'sess_a')).toBe(false);
  });

  it('is keyed by project AND session id', () => {
    markFreshSession('p1', 'sess_b');
    expect(consumeFreshSession('p2', 'sess_b')).toBe(false);
    expect(consumeFreshSession('p1', 'sess_other')).toBe(false);
    expect(consumeFreshSession('p1', 'sess_b')).toBe(true);
  });

  it('an unmarked id is never fresh', () => {
    expect(consumeFreshSession('p1', 'never_marked')).toBe(false);
  });
});
