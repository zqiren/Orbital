// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import type { StringEntry } from '../strings';
import { buildTextIndex, candidateTexts, findMatches, matchText, normalizeText, searchCatalog } from './matchCatalog';

const FAKE: Record<string, StringEntry> = {
  'a.save': { en: 'Save', zh: '保存' },
  'a.dup': { en: 'Save', zh: '存储' },
  'a.count': { en: '{n} sessions', zh: '{n} 个会话' },
  'a.enOnly': { en: 'English only' },
  'a.regex': { en: 'Cost ($) for {name}', zh: '{name} 的费用 ($)' },
};

describe('buildTextIndex / matchText', () => {
  it('matches exact text in the active locale', () => {
    expect(matchText(buildTextIndex('zh', FAKE), '保存')).toEqual([{ key: 'a.save', text: '保存', exact: true }]);
    expect(matchText(buildTextIndex('en', FAKE), 'Save').map((m) => m.key)).toEqual(['a.save', 'a.dup']);
  });

  it('indexes the English fallback when zh is missing', () => {
    expect(matchText(buildTextIndex('zh', FAKE), 'English only')[0]?.key).toBe('a.enOnly');
  });

  it('matches placeholder strings by pattern, flagged as inexact', () => {
    expect(matchText(buildTextIndex('zh', FAKE), '5 个会话')).toEqual([{ key: 'a.count', text: '5 个会话', exact: false }]);
    expect(matchText(buildTextIndex('en', FAKE), 'Cost ($) for demo')[0]?.key).toBe('a.regex');
  });

  it('normalizes nbsp and whitespace runs', () => {
    expect(normalizeText('  Save  \n')).toBe('Save');
    expect(matchText(buildTextIndex('en', FAKE), '  Save ')[0]?.key).toBe('a.save');
  });

  it('ignores empty and very long text', () => {
    const idx = buildTextIndex('en', FAKE);
    expect(matchText(idx, '   ')).toEqual([]);
    expect(matchText(idx, 'x'.repeat(400))).toEqual([]);
  });
});

describe('candidateTexts / findMatches', () => {
  it('walks from an icon inside a button up to the button label', () => {
    document.body.innerHTML = '<button><svg></svg><span>Save</span></button>';
    const svg = document.querySelector('svg')!;
    const res = findMatches(svg, buildTextIndex('en', FAKE));
    expect(res.matches[0]?.key).toBe('a.save');
    expect(res.el?.tagName).toBe('BUTTON');
  });

  it('reads aria-label, placeholder and title before text', () => {
    document.body.innerHTML = '<input placeholder="English only"><button aria-label="Save" title="ignored">x</button>';
    const input = document.querySelector('input')!;
    expect(candidateTexts(input)[0]).toMatchObject({ text: 'English only', via: 'placeholder' });
    const btn = document.querySelector('button')!;
    expect(findMatches(btn, buildTextIndex('en', FAKE)).matches[0]?.key).toBe('a.save');
  });

  it('reports every text it tried when nothing matches', () => {
    document.body.innerHTML = '<div><p>Not in catalog</p></div>';
    const p = document.querySelector('p')!;
    const res = findMatches(p, buildTextIndex('en', FAKE));
    expect(res.matches).toEqual([]);
    expect(res.el).toBeNull();
    expect(res.tried).toContain('Not in catalog');
  });
});

describe('searchCatalog', () => {
  it('searches keys and both languages, case-insensitively', () => {
    expect(searchCatalog('SAVE', 30, FAKE).map((h) => h.key)).toEqual(['a.save', 'a.dup']);
    expect(searchCatalog('会话', 30, FAKE).map((h) => h.key)).toEqual(['a.count']);
    expect(searchCatalog('a.enonly', 30, FAKE).map((h) => h.key)).toEqual(['a.enOnly']);
    expect(searchCatalog('', 30, FAKE)).toEqual([]);
  });
});
