// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

const apiFn = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiFn(...args),
}));

import { useProjects, applyManualOrder } from './useProjects';
import type { Project } from '../types';

function makeProject(id: string, extra: Partial<Project> = {}): Project {
  return {
    project_id: id,
    name: id,
    workspace: `/tmp/${id}`,
    model: '',
    api_key: '',
    base_url: null,
    autonomy: 'hands_off',
    instructions: '',
    ...extra,
  };
}

const ids = (projects: Project[]) => projects.map((p) => p.project_id);

// ─── Pure reorder logic ──────────────────────────────────────────────────────
// jsdom cannot simulate a real pointer drag, so the ordering itself is proven
// here as ids-in / ids-out (spec 056 §5 "keep the reorder computation pure").

describe('applyManualOrder', () => {
  const a = makeProject('a');
  const b = makeProject('b');
  const c = makeProject('c');

  it('reorders to match the given id list', () => {
    expect(ids(applyManualOrder([a, b, c], ['c', 'a', 'b']))).toEqual(['c', 'a', 'b']);
  });

  it('is a no-op when the ids already match the current order', () => {
    expect(ids(applyManualOrder([a, b, c], ['a', 'b', 'c']))).toEqual(['a', 'b', 'c']);
  });

  it('does not mutate the input array', () => {
    const input = [a, b, c];
    applyManualOrder(input, ['c', 'b', 'a']);
    expect(ids(input)).toEqual(['a', 'b', 'c']);
  });

  it('keeps unlisted projects at the bottom in their original order', () => {
    const d = makeProject('d');
    // Only b and a are placed; c and d were never dragged.
    expect(ids(applyManualOrder([a, b, c, d], ['b', 'a']))).toEqual(['b', 'a', 'c', 'd']);
  });

  it('keeps the scratch project pinned above everything', () => {
    const scratch = makeProject('scratch', { is_scratch: true });
    // The id list deliberately tries to bury scratch at the bottom.
    const out = applyManualOrder([a, scratch, b], ['a', 'b', 'scratch']);
    expect(ids(out)).toEqual(['scratch', 'a', 'b']);
  });

  it('ignores ids that are not in the list', () => {
    expect(ids(applyManualOrder([a, b], ['b', 'ghost', 'a']))).toEqual(['b', 'a']);
  });
});

// ─── reorderProjects: optimistic apply + revert ──────────────────────────────

describe('useProjects.reorderProjects', () => {
  beforeEach(() => {
    apiFn.mockReset();
  });

  async function mountWith(projects: Project[]) {
    apiFn.mockResolvedValueOnce(projects);
    const hook = renderHook(() => useProjects());
    await act(async () => {
      await hook.result.current.listProjects();
    });
    return hook;
  }

  it('applies the new order optimistically, before the POST resolves', async () => {
    const seed = [makeProject('a'), makeProject('b'), makeProject('c')];
    const { result } = await mountWith(seed);

    // A POST that never settles during this assertion window.
    let resolvePost: (value: Project[]) => void = () => {};
    apiFn.mockReturnValueOnce(new Promise<Project[]>((r) => { resolvePost = r; }));

    let pending!: Promise<unknown>;
    act(() => {
      pending = result.current.reorderProjects(['c', 'a', 'b']);
    });

    // Painted from the local list — no round-trip needed.
    expect(ids(result.current.projects)).toEqual(['c', 'a', 'b']);

    await act(async () => {
      resolvePost([makeProject('c'), makeProject('a'), makeProject('b')]);
      await pending;
    });
  });

  it('POSTs the ordered ids to /projects/reorder', async () => {
    const { result } = await mountWith([makeProject('a'), makeProject('b')]);
    apiFn.mockResolvedValueOnce([makeProject('b'), makeProject('a')]);

    await act(async () => {
      await result.current.reorderProjects(['b', 'a']);
    });

    expect(apiFn).toHaveBeenLastCalledWith('/api/v2/projects/reorder', {
      method: 'POST',
      body: JSON.stringify({ ordered_ids: ['b', 'a'] }),
    });
  });

  it('adopts the server response as canonical', async () => {
    const { result } = await mountWith([makeProject('a'), makeProject('b')]);
    // The server's answer wins even when it differs from the optimistic guess
    // (e.g. another device reordered concurrently).
    apiFn.mockResolvedValueOnce([makeProject('a'), makeProject('b'), makeProject('z')]);

    await act(async () => {
      await result.current.reorderProjects(['b', 'a']);
    });

    expect(ids(result.current.projects)).toEqual(['a', 'b', 'z']);
  });

  it('reverts to the pre-drag order when the POST fails', async () => {
    const { result } = await mountWith([makeProject('a'), makeProject('b'), makeProject('c')]);
    apiFn.mockRejectedValueOnce(new Error('boom'));

    await act(async () => {
      await expect(result.current.reorderProjects(['c', 'b', 'a'])).rejects.toThrow('boom');
    });

    await waitFor(() => {
      expect(ids(result.current.projects)).toEqual(['a', 'b', 'c']);
    });
  });

  it('keeps a mid-flight refetch when the POST fails, and only undoes the order', async () => {
    // App refetches projects from several event paths, so a listProjects()
    // can resolve while the reorder POST is still open. The rollback must
    // undo the ordering without throwing that fresher data away.
    const { result } = await mountWith([makeProject('a'), makeProject('b')]);

    let rejectPost: (e: Error) => void = () => {};
    apiFn.mockReturnValueOnce(new Promise<Project[]>((_, rej) => { rejectPost = rej; }));

    let pending!: Promise<unknown>;
    act(() => {
      pending = result.current.reorderProjects(['b', 'a']);
    });
    expect(ids(result.current.projects)).toEqual(['b', 'a']);

    // A refetch lands mid-flight: 'a' was renamed elsewhere and 'c' is new.
    apiFn.mockResolvedValueOnce([
      makeProject('a', { name: 'Renamed' }),
      makeProject('b'),
      makeProject('c'),
    ]);
    await act(async () => {
      await result.current.listProjects();
    });

    await act(async () => {
      rejectPost(new Error('boom'));
      await expect(pending).rejects.toThrow('boom');
    });

    await waitFor(() => {
      // Pre-drag order restored, and 'c' stays at the bottom as an unplaced
      // project rather than vanishing with the stale snapshot.
      expect(ids(result.current.projects)).toEqual(['a', 'b', 'c']);
    });
    // The rename survived the rollback.
    expect(result.current.projects[0].name).toBe('Renamed');
  });

  it('does not resurrect a project deleted while the POST was in flight', async () => {
    const { result } = await mountWith([makeProject('a'), makeProject('b'), makeProject('c')]);

    let rejectPost: (e: Error) => void = () => {};
    apiFn.mockReturnValueOnce(new Promise<Project[]>((_, rej) => { rejectPost = rej; }));

    let pending!: Promise<unknown>;
    act(() => {
      pending = result.current.reorderProjects(['c', 'b', 'a']);
    });

    apiFn.mockResolvedValueOnce(undefined); // DELETE
    await act(async () => {
      await result.current.deleteProject('b');
    });

    await act(async () => {
      rejectPost(new Error('boom'));
      await expect(pending).rejects.toThrow('boom');
    });

    await waitFor(() => {
      expect(ids(result.current.projects)).toEqual(['a', 'c']);
    });
  });

  it('reverts cleanly twice in a row (the rollback snapshot stays current)', async () => {
    const { result } = await mountWith([makeProject('a'), makeProject('b')]);

    apiFn.mockRejectedValueOnce(new Error('first'));
    await act(async () => {
      await expect(result.current.reorderProjects(['b', 'a'])).rejects.toThrow('first');
    });
    await waitFor(() => expect(ids(result.current.projects)).toEqual(['a', 'b']));

    apiFn.mockRejectedValueOnce(new Error('second'));
    await act(async () => {
      await expect(result.current.reorderProjects(['b', 'a'])).rejects.toThrow('second');
    });
    await waitFor(() => expect(ids(result.current.projects)).toEqual(['a', 'b']));
  });
});
