// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { resolve, interpolate, translate } from './useT';
import type { StringKey } from './strings';

describe('resolve', () => {
  it('returns zh for zh locale when present', () => {
    expect(resolve({ en: 'Save', zh: '保存' }, 'zh', 'k')).toBe('保存');
  });
  it('returns en for en locale', () => {
    expect(resolve({ en: 'Save', zh: '保存' }, 'en', 'k')).toBe('Save');
  });
  it('falls back to en when zh is empty', () => {
    expect(resolve({ en: 'Save', zh: '' }, 'zh', 'k')).toBe('Save');
  });
  it('falls back to en when zh is missing', () => {
    expect(resolve({ en: 'Save' }, 'zh', 'k')).toBe('Save');
  });
  it('returns the key itself when the entry is undefined', () => {
    expect(resolve(undefined, 'en', 'a.b.c')).toBe('a.b.c');
  });
});

describe('interpolate', () => {
  it('replaces a {name} placeholder', () => {
    expect(interpolate('Hi {name}', { name: 'Acme' })).toBe('Hi Acme');
  });
  it('replaces numeric vars and leaves unknown placeholders intact', () => {
    expect(interpolate('{n} left {x}', { n: 3 })).toBe('3 left {x}');
  });
  it('returns text unchanged when no vars given', () => {
    expect(interpolate('Hi {name}')).toBe('Hi {name}');
  });
});

describe('translate (uses real catalog)', () => {
  it('translates a known key into zh', () => {
    expect(translate('zh', 'app.loading')).toBe('加载中…');
  });
  it('returns the key for an unknown key', () => {
    expect(translate('en', 'totally.unknown.key' as StringKey)).toBe('totally.unknown.key');
  });
});
