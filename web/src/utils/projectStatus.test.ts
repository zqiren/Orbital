// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect, vi } from 'vitest';
import type { AgentRunStatus } from '../types';
import {
  createSubAgentsRunningSync,
  getProjectDotColor,
} from './projectStatus';

describe('getProjectDotColor — precedence (spec 095)', () => {
  const dot = (
    status: AgentRunStatus | undefined,
    { approvals = 0, workers = false }: { approvals?: number; workers?: boolean } = {},
  ) =>
    getProjectDotColor(
      'p',
      status ? { p: status } : {},
      approvals ? { p: approvals } : {},
      { p: workers },
    );

  it('shows green while a worker runs with the manager idle', () => {
    expect(dot('idle', { workers: true })).toBe('bg-success');
    expect(dot(undefined, { workers: true })).toBe('bg-success');
  });

  it('stays grey with the manager idle and no worker running', () => {
    expect(dot('idle')).toBe('bg-idle');
  });

  it('keeps a manager error red while a worker runs', () => {
    expect(dot('error', { workers: true })).toBe('bg-error');
  });

  it('keeps pending approvals amber over a running worker', () => {
    expect(dot('idle', { approvals: 1, workers: true })).toBe('bg-warning');
  });

  it('is unchanged for manager-running states', () => {
    expect(dot('running')).toBe('bg-success');
    expect(dot('waiting')).toBe('bg-success');
  });

  it('treats a missing map as no worker running (older callers)', () => {
    expect(getProjectDotColor('p', { p: 'idle' }, {})).toBe('bg-idle');
  });
});

/** A fetch whose responses the test resolves by hand, in any order. */
function manualFetch() {
  const pending: Array<{ projectId: string; resolve: (v: boolean | undefined) => void; reject: (e: unknown) => void }> = [];
  const fetchFlag = vi.fn(
    (projectId: string) =>
      new Promise<boolean | undefined>((resolve, reject) => {
        pending.push({ projectId, resolve, reject });
      }),
  );
  return { fetchFlag, pending };
}

async function settle() {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

describe('createSubAgentsRunningSync', () => {
  it('keeps one request in flight per project and queues exactly one more', async () => {
    const { fetchFlag, pending } = manualFetch();
    const apply = vi.fn();
    const sync = createSubAgentsRunningSync(fetchFlag, apply);

    sync.refresh('p');
    sync.refresh('p');
    sync.refresh('p');
    expect(fetchFlag).toHaveBeenCalledTimes(1);

    pending[0].resolve(true);
    await settle();
    // The burst collapsed into one follow-up that starts after the event.
    expect(fetchFlag).toHaveBeenCalledTimes(2);
    pending[1].resolve(false);
    await settle();

    expect(fetchFlag).toHaveBeenCalledTimes(2);
    expect(apply.mock.calls).toEqual([['p', true], ['p', false]]);
  });

  it('runs projects independently', () => {
    const { fetchFlag } = manualFetch();
    const sync = createSubAgentsRunningSync(fetchFlag, vi.fn());
    sync.refresh('a');
    sync.refresh('b');
    expect(fetchFlag.mock.calls.map((c) => c[0])).toEqual(['a', 'b']);
  });

  it('drops a response that lands after a newer one was applied', async () => {
    const { fetchFlag, pending } = manualFetch();
    const apply = vi.fn();
    const sync = createSubAgentsRunningSync(fetchFlag, apply);

    // Hydration issued first (worker still running at that moment)...
    const hydrate = sync.track('p');
    // ...then a terminal event's refetch, which comes back first.
    sync.refresh('p');
    pending[0].resolve(false);
    await settle();
    hydrate(true);

    expect(apply.mock.calls).toEqual([['p', false]]);
  });

  it('reads a missing field (older daemon) as false', async () => {
    const { fetchFlag, pending } = manualFetch();
    const apply = vi.fn();
    const sync = createSubAgentsRunningSync(fetchFlag, apply);
    sync.refresh('p');
    pending[0].resolve(undefined);
    await settle();
    expect(apply).toHaveBeenCalledWith('p', false);
  });

  it('still runs the queued refetch after a failed request', async () => {
    const { fetchFlag, pending } = manualFetch();
    const apply = vi.fn();
    const sync = createSubAgentsRunningSync(fetchFlag, apply);
    sync.refresh('p');
    sync.refresh('p');
    pending[0].reject(new Error('tunnel dropped'));
    await settle();
    expect(fetchFlag).toHaveBeenCalledTimes(2);
    pending[1].resolve(true);
    await settle();
    expect(apply.mock.calls).toEqual([['p', true]]);
  });
});
