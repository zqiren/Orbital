// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: vi.fn() }));
vi.mock('../config', () => ({ api: apiMock }));

import {
  __resetAgentAvailability,
  applyQuotaUpdate,
  fetchAgentAvailability,
  setAgentAvailability,
  settleInstalledAgents,
  useInstalledAgents,
} from './useAgentAvailability';

afterEach(() => {
  apiMock.mockReset();
  act(() => __resetAgentAvailability());
});

const codex = { slug: 'codex', name: 'Codex', installed: true };
const claude = { slug: 'claude-code', name: 'Claude Code', installed: true };

describe('useInstalledAgents', () => {
  it('is null until the first answer, then leaves out built-in and absent agents', () => {
    const { result } = renderHook(() => useInstalledAgents());
    expect(result.current).toBeNull();
    act(() => setAgentAvailability([
      { slug: 'built-in', name: 'Orbital', installed: true },
      claude,
      { ...codex, installed: false },
    ]));
    expect(result.current).toEqual([{ slug: 'claude-code', name: 'Claude Code' }]);
  });

  it('picks up an agent that appears on a later fetch, without a reload', async () => {
    // The report: Codex located in Settings never reached the pin menu.
    const { result } = renderHook(() => useInstalledAgents());
    apiMock.mockResolvedValueOnce([claude, { ...codex, installed: false }]);
    await act(async () => { await fetchAgentAvailability(); });
    expect(result.current?.map((a) => a.slug)).toEqual(['claude-code']);

    apiMock.mockResolvedValueOnce([claude, codex]);
    await act(async () => { await fetchAgentAvailability(); });
    expect(result.current?.map((a) => a.slug)).toEqual(['claude-code', 'codex']);
  });

  it('keeps the same array when the set is unchanged, so quota pushes cost no re-render', () => {
    const { result } = renderHook(() => useInstalledAgents());
    act(() => setAgentAvailability([codex]));
    const first = result.current;
    act(() => setAgentAvailability([codex]));
    act(() => applyQuotaUpdate('codex', null));
    expect(result.current).toBe(first);
  });

  it('settles a failed first load on empty, and a failed refetch keeps the list', () => {
    const { result } = renderHook(() => useInstalledAgents());
    act(() => settleInstalledAgents());
    expect(result.current).toEqual([]);
    act(() => setAgentAvailability([codex]));
    act(() => settleInstalledAgents());
    expect(result.current).toEqual([{ slug: 'codex', name: 'Codex' }]);
  });
});
