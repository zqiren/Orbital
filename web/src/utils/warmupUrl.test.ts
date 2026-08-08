// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { warmupUrlForLocale } from './warmupUrl';

describe('warmupUrlForLocale', () => {
  it('sends zh users to Bing (accounts.google.com is unreachable from mainland China)', () => {
    expect(warmupUrlForLocale('zh')).toBe('https://www.bing.com');
  });

  it('keeps the Google accounts default for en', () => {
    expect(warmupUrlForLocale('en')).toBe('https://accounts.google.com');
  });
});
