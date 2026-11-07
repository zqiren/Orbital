// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { skipsFirstRunSetup } from './config';

describe('skipsFirstRunSetup', () => {
  it('runs setup on the desktop', () => {
    expect(skipsFirstRunSetup('localhost', false, false)).toBe(false);
    expect(skipsFirstRunSetup('127.0.0.1', false, false)).toBe(false);
  });

  it('leaves setup to the desktop when a phone reaches its daemon', () => {
    expect(skipsFirstRunSetup('192.168.1.20', false, false)).toBe(true);
  });

  it('skips setup through the relay', () => {
    expect(skipsFirstRunSetup('relay.example.com', true, false)).toBe(true);
    expect(skipsFirstRunSetup('relay.example.com', true, true)).toBe(true);
  });

  it('runs setup from any client of a headless daemon', () => {
    expect(skipsFirstRunSetup('server.example.ts.net', false, true)).toBe(false);
  });
});
