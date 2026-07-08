// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from 'vitest';
import { fetchPathWithFallback } from './openPathWithFallback';
import type { FileContent } from '../types';

function fileAt(path: string): FileContent {
  return { path, content: 'x', type: 'text', size: 1 } as FileContent;
}

/** getContent stub that only knows the given paths. */
function contentStore(...paths: string[]) {
  return async (p: string) => (paths.includes(p) ? fileAt(p) : null);
}

describe('fetchPathWithFallback', () => {
  test('opens an exact path without calling resolve', async () => {
    let resolveCalled = false;
    const result = await fetchPathWithFallback(
      contentStore('content/drafts/003.md'),
      async () => {
        resolveCalled = true;
        return [];
      },
      'content/drafts/003.md',
    );
    expect(result).toEqual({
      status: 'ok',
      content: fileAt('content/drafts/003.md'),
      path: 'content/drafts/003.md',
    });
    expect(resolveCalled).toBe(false);
  });

  test('falls back to a unique resolve match on 404', async () => {
    const result = await fetchPathWithFallback(
      contentStore('orbital/DECISIONS.md'),
      async () => ['orbital/DECISIONS.md'],
      'DECISIONS.md',
    );
    expect(result).toEqual({
      status: 'ok',
      content: fileAt('orbital/DECISIONS.md'),
      path: 'orbital/DECISIONS.md',
    });
  });

  test('reports ambiguous when resolve returns multiple matches', async () => {
    const result = await fetchPathWithFallback(
      contentStore(),
      async () => ['a/todo.md', 'b/todo.md'],
      'todo.md',
    );
    expect(result).toEqual({ status: 'ambiguous', matches: ['a/todo.md', 'b/todo.md'] });
  });

  test('reports not_found when resolve has no matches', async () => {
    const result = await fetchPathWithFallback(contentStore(), async () => [], 'nope.md');
    expect(result).toEqual({ status: 'not_found' });
  });

  test('reports not_found when resolve itself fails', async () => {
    const result = await fetchPathWithFallback(contentStore(), async () => null, 'nope.md');
    expect(result).toEqual({ status: 'not_found' });
  });

  test('reports not_found when the resolved path then fails to fetch', async () => {
    // Race: file deleted between resolve and fetch.
    const result = await fetchPathWithFallback(
      contentStore(),
      async () => ['gone/nope.md'],
      'nope.md',
    );
    expect(result).toEqual({ status: 'not_found' });
  });
});
