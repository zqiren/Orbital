// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 166: the catalog cache is keyed by project AND the composer's pin, so
 * a worker's own commands never show under another pin (or under Orbital).
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

const calls: string[] = [];
vi.mock('../config', () => ({
  api: vi.fn(async (path: string) => {
    calls.push(path);
    const worker = new URL(path, 'http://x').searchParams.get('worker');
    return [{ key: worker ?? 'plain', name: worker ?? 'plain', description: '', source: 'x', path: '', dir_name: '' }];
  }),
}));

import {
  __clearSkillCatalogCacheForTests,
  cachedSkillCatalog,
  loadSkillCatalog,
  matchSkills,
  type SkillCatalogEntry,
} from './skill-catalog';

beforeEach(() => {
  calls.length = 0;
  __clearSkillCatalogCacheForTests();
});

describe('skill catalog cache keyed by pin (spec 166)', () => {
  it('no worker: the unchanged path and its own cache slot', async () => {
    await loadSkillCatalog('p1');
    expect(calls).toEqual(['/api/v2/projects/p1/skill-catalog']);
    expect(cachedSkillCatalog('p1')?.[0].name).toBe('plain');
    expect(cachedSkillCatalog('p1', 'codex')).toBeNull();
  });

  it('a worker pin queries ?worker= and caches separately', async () => {
    await loadSkillCatalog('p1', 'claude-code');
    await loadSkillCatalog('p1', 'codex');
    expect(calls).toEqual([
      '/api/v2/projects/p1/skill-catalog?worker=claude-code',
      '/api/v2/projects/p1/skill-catalog?worker=codex',
    ]);
    expect(cachedSkillCatalog('p1', 'claude-code')?.[0].name).toBe('claude-code');
    expect(cachedSkillCatalog('p1', 'codex')?.[0].name).toBe('codex');
    expect(cachedSkillCatalog('p1')).toBeNull();
    // Fresh: a second load is answered from the cache.
    await loadSkillCatalog('p1', 'codex');
    expect(calls).toHaveLength(2);
  });

  it('null worker is Orbital (the unpinned key)', async () => {
    await loadSkillCatalog('p1', null);
    expect(calls).toEqual(['/api/v2/projects/p1/skill-catalog']);
    expect(cachedSkillCatalog('p1', null)?.[0].name).toBe('plain');
  });
});

describe('matching a worker command (spec 166)', () => {
  const imagegen: SkillCatalogEntry = {
    key: 'imagegen', name: 'imagegen', description: 'Make images', source: 'worker',
    path: '', dir_name: '', invoke: '$imagegen ', hint: '',
  };
  const review: SkillCatalogEntry = {
    key: 'review', name: 'review', description: 'Review (project)', source: 'worker',
    path: '', dir_name: '', invoke: '/review ', hint: '<pr>',
  };

  it('matches by name; the invoke token plays no part', () => {
    expect(matchSkills([imagegen, review], 'ima')).toEqual([imagegen]);
    expect(matchSkills([imagegen, review], '$')).toEqual([]);
    expect(matchSkills([imagegen, review], '')).toEqual([imagegen, review]);
  });
});
