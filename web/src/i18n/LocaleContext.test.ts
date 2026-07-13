// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * readInitialLocale — Spec 008 §8 Q3 (B1): navigator.language fallback.
 * localStorage wins when set; otherwise a zh* device language resolves to
 * 'zh', anything else (including a missing navigator.language) to 'en'.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { readInitialLocale } from './LocaleContext';
import { LOCALE_STORAGE_KEY } from './locales';

describe('readInitialLocale', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('prefers localStorage over navigator.language', () => {
    localStorage.setItem(LOCALE_STORAGE_KEY, 'en');
    vi.stubGlobal('navigator', { language: 'zh-CN' });
    expect(readInitialLocale()).toBe('en');
  });

  it('falls back to zh for a zh-CN navigator.language when localStorage is empty', () => {
    vi.stubGlobal('navigator', { language: 'zh-CN' });
    expect(readInitialLocale()).toBe('zh');
  });

  it('falls back to en for an en-US navigator.language', () => {
    vi.stubGlobal('navigator', { language: 'en-US' });
    expect(readInitialLocale()).toBe('en');
  });

  it('falls back to en when navigator.language is undefined', () => {
    vi.stubGlobal('navigator', {});
    expect(readInitialLocale()).toBe('en');
  });
});
