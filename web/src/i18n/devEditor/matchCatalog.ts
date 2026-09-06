// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Reverse lookup for the dev translation editor: from the text a DOM element
// shows to the catalog key(s) that produced it. Pure functions so the matching
// rules are unit-testable without the overlay.

import { STRINGS, type StringEntry } from '../strings';
import { resolve } from '../useT';
import type { Locale } from '../locales';

export interface CatalogMatch {
  key: string;
  /** The normalized text that matched. */
  text: string;
  /** false when matched through a {placeholder} pattern. */
  exact: boolean;
}

export interface TextIndex {
  exact: Map<string, string[]>;
  patterns: { key: string; re: RegExp }[];
}

const PLACEHOLDER = /\{\w+\}/g;
const MAX_TEXT = 300;

export function normalizeText(s: string): string {
  return s.replace(/ /g, ' ').replace(/\s+/g, ' ').trim();
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Index every catalog string as rendered in `locale` (with the zh -> en fallback). */
export function buildTextIndex(
  locale: Locale,
  catalog: Record<string, StringEntry> = STRINGS as Record<string, StringEntry>,
): TextIndex {
  const exact = new Map<string, string[]>();
  const patterns: TextIndex['patterns'] = [];
  for (const [key, entry] of Object.entries(catalog)) {
    const text = normalizeText(resolve(entry, locale, key));
    if (!text) continue;
    if (PLACEHOLDER.test(text)) {
      PLACEHOLDER.lastIndex = 0;
      const src = text.split(PLACEHOLDER).map(escapeRe).join('(.+?)');
      patterns.push({ key, re: new RegExp(`^${src}$`) });
    } else {
      const list = exact.get(text) ?? [];
      list.push(key);
      exact.set(text, list);
    }
    PLACEHOLDER.lastIndex = 0;
  }
  return { exact, patterns };
}

export function matchText(index: TextIndex, raw: string): CatalogMatch[] {
  const text = normalizeText(raw);
  if (!text || text.length > MAX_TEXT) return [];
  const out: CatalogMatch[] = (index.exact.get(text) ?? []).map((key) => ({ key, text, exact: true }));
  if (out.length === 0) {
    for (const p of index.patterns) if (p.re.test(text)) out.push({ key: p.key, text, exact: false });
  }
  return out;
}

export interface Candidate {
  text: string;
  el: Element;
  /** Where the text came from: an attribute name, 'text' (own text nodes) or 'textContent'. */
  via: string;
}

const TEXT_ATTRS = ['aria-label', 'title', 'placeholder', 'alt'];

/**
 * Candidate strings for an element, nearest first: its own attributes and
 * direct text nodes, then its full text, then the same for each ancestor.
 * Walking up is what makes a click on an icon inside a button find the
 * button's label.
 */
export function candidateTexts(start: Element, maxDepth = 6): Candidate[] {
  const out: Candidate[] = [];
  const seen = new Set<string>();
  const push = (raw: string | null, el: Element, via: string) => {
    const text = normalizeText(raw ?? '');
    if (text && !seen.has(text)) { seen.add(text); out.push({ text, el, via }); }
  };
  let el: Element | null = start;
  for (let depth = 0; el && depth < maxDepth; depth += 1, el = el.parentElement) {
    for (const attr of TEXT_ATTRS) push(el.getAttribute(attr), el, attr);
    if (el instanceof HTMLInputElement && (el.type === 'button' || el.type === 'submit')) push(el.value, el, 'value');
    const own = Array.from(el.childNodes)
      .filter((n) => n.nodeType === Node.TEXT_NODE)
      .map((n) => n.textContent ?? '')
      .join(' ');
    push(own, el, 'text');
    push(el.textContent, el, 'textContent');
    if (el === document.body) break;
  }
  return out;
}

export interface FindResult {
  matches: CatalogMatch[];
  /** The element whose text matched (null when nothing matched). */
  el: Element | null;
  /** Every text that was tried, for the "no match" explanation. */
  tried: string[];
}

export function findMatches(start: Element, index: TextIndex): FindResult {
  const candidates = candidateTexts(start);
  const tried = candidates.map((c) => c.text);
  for (const c of candidates) {
    const matches = matchText(index, c.text);
    if (matches.length) return { matches, el: c.el, tried };
  }
  return { matches: [], el: null, tried };
}

export interface SearchHit { key: string; en: string; zh?: string }

/** Substring search over keys and both languages, for the manual fallback. */
export function searchCatalog(
  query: string,
  limit = 30,
  catalog: Record<string, StringEntry> = STRINGS as Record<string, StringEntry>,
): SearchHit[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const out: SearchHit[] = [];
  for (const [key, e] of Object.entries(catalog)) {
    if (key.toLowerCase().includes(q) || e.en.toLowerCase().includes(q) || (e.zh ?? '').toLowerCase().includes(q)) {
      out.push({ key, en: e.en, zh: e.zh });
      if (out.length >= limit) break;
    }
  }
  return out;
}
