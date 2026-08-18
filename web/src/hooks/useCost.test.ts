// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

const onMock = vi.fn();
const offMock = vi.fn();
// Mutable so the reconnect tests can flip it between rerenders; the factory
// reads it on every render.
let wsConnectionState = 'connected';

vi.mock('./useWebSocket', () => ({
  useWebSocket: () => ({
    on: onMock,
    off: offMock,
    connectionState: wsConnectionState,
    subscribe: vi.fn(),
  }),
}));

let apiFn = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiFn(...args),
  ApiError: class ApiError extends Error {},
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000/ws',
}));

import { useCost } from './useCost';
import { bumpPricingVersion } from '../budget/pricingVersion';
import type { CostResponse } from '../budget/types';

/**
 * The payload the daemon ACTUALLY sends. `SpendBroadcaster.submit`
 * (agent_os/budget/spend_broadcast.py) builds exactly these five fields;
 * `WSManager.broadcast(project_id, payload)` uses the project id for ROUTING
 * and forwards the payload verbatim, so there is NO `project_id` on the wire.
 *
 * Every fixture here used to hand-build the event WITH `project_id`, which is
 * why the suite stayed green while the corner never refreshed in production:
 * the tests proved the handler works on a message that does not exist. Keep
 * this the default; only the two tests that specifically exercise the
 * presence-guarded filter add the field.
 */
function daemonSpendEvent(spend = 3) {
  return { type: 'budget.spend_updated', window: 'daily', spend, limit: 10, currency: 'USD' };
}

function makeCost(amount: number): CostResponse {
  return {
    window: 'daily',
    by_currency: { USD: amount },
    converted_total: { currency: 'USD', amount, estimated: true },
    breakdown: [],
    subagents: [],
  };
}

describe('useCost', () => {
  beforeEach(() => {
    onMock.mockClear();
    offMock.mockClear();
    apiFn = vi.fn();
    wsConnectionState = 'connected';
  });

  it('fetches GET /cost with the window on mount', async () => {
    apiFn.mockResolvedValueOnce(makeCost(1.5));
    const { result } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.cost?.converted_total.amount).toBe(1.5);
    expect(apiFn).toHaveBeenCalledWith('/api/v2/projects/p1/cost?window=daily');
  });

  it('reports loading=true from the FIRST render (no fabricated-$0.00 frame, #40)', () => {
    apiFn.mockReturnValueOnce(new Promise(() => {})); // never resolves
    const { result } = renderHook(() => useCost('p1', 'daily'));
    expect(result.current.loading).toBe(true);
    expect(result.current.cost).toBeNull();
  });

  it('reports loading=false from the first render when projectId is null', () => {
    const { result } = renderHook(() => useCost(null, 'daily'));
    expect(result.current.loading).toBe(false);
  });

  it('passes the configured window through to the query', async () => {
    apiFn.mockResolvedValueOnce(makeCost(0));
    renderHook(() => useCost('p1', 'total'));
    await waitFor(() =>
      expect(apiFn).toHaveBeenCalledWith('/api/v2/projects/p1/cost?window=total'),
    );
  });

  it('does not fetch when projectId is null', async () => {
    renderHook(() => useCost(null, 'daily'));
    // Give the effect a tick.
    await act(async () => { await Promise.resolve(); });
    expect(apiFn).not.toHaveBeenCalled();
  });

  it('clears the previous cost synchronously when the project changes (#40 stale bleed)', async () => {
    apiFn.mockResolvedValueOnce(makeCost(9));
    const { result, rerender } = renderHook(
      ({ pid }: { pid: string }) => useCost(pid, 'daily'),
      { initialProps: { pid: 'p1' } },
    );
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(9));

    // Second project's fetch never resolves — the old number must NOT survive.
    apiFn.mockReturnValueOnce(new Promise(() => {}));
    rerender({ pid: 'p2' });
    expect(result.current.cost).toBeNull();
    expect(result.current.loading).toBe(true);
  });

  it('keeps the last known cost visible during a same-query re-fetch', async () => {
    apiFn.mockResolvedValueOnce(makeCost(1));
    const { result } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(1));

    apiFn.mockReturnValueOnce(new Promise(() => {}));
    const handler = onMock.mock.calls.find((c) => c[0] === 'budget.spend_updated')![1] as (e: unknown) => void;
    await act(async () => {
      handler(daemonSpendEvent());
    });
    // Refetch in flight: previous number still shown, not blanked.
    expect(result.current.cost?.converted_total.amount).toBe(1);
    expect(result.current.loading).toBe(true);
  });

  it('subscribes to budget.spend_updated on mount and unsubscribes on unmount', async () => {
    apiFn.mockResolvedValue(makeCost(0));
    const { unmount } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => {
      expect(onMock.mock.calls.map((c) => c[0])).toContain('budget.spend_updated');
    });
    unmount();
    expect(offMock.mock.calls.map((c) => c[0])).toContain('budget.spend_updated');
  });

  it('re-fetches on the daemon\'s REAL spend payload (no project_id field)', async () => {
    // The regression this pins: the handler used to compare `event.project_id`
    // (undefined on the wire) to the project id, so the refetch never ran and
    // the corner sat frozen until an unrelated remount/reconnect.
    apiFn.mockResolvedValueOnce(makeCost(1)).mockResolvedValueOnce(makeCost(3));
    const { result } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(1));

    const handler = onMock.mock.calls.find((c) => c[0] === 'budget.spend_updated')![1] as (e: unknown) => void;
    await act(async () => {
      handler(daemonSpendEvent());
    });
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(3));
    expect(apiFn).toHaveBeenCalledTimes(2);
  });

  it('re-fetches when the payload DOES carry a matching project_id', async () => {
    // The daemon may start stamping project_id on the broadcast; the guarded
    // filter must accept that shape too, not just the bare one.
    apiFn.mockResolvedValueOnce(makeCost(1)).mockResolvedValueOnce(makeCost(3));
    const { result } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(1));

    const handler = onMock.mock.calls.find((c) => c[0] === 'budget.spend_updated')![1] as (e: unknown) => void;
    await act(async () => {
      handler({ ...daemonSpendEvent(), project_id: 'p1' });
    });
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(3));
    expect(apiFn).toHaveBeenCalledTimes(2);
  });

  it('ignores budget.spend_updated for a different project', async () => {
    apiFn.mockResolvedValueOnce(makeCost(1));
    const { result } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(1));

    const handler = onMock.mock.calls.find((c) => c[0] === 'budget.spend_updated')![1] as (e: unknown) => void;
    // The field IS present here, so the guarded filter must still reject it.
    await act(async () => {
      handler({ ...daemonSpendEvent(99), project_id: 'OTHER', limit: null });
    });
    // No second fetch — still the initial value.
    expect(apiFn).toHaveBeenCalledTimes(1);
    expect(result.current.cost?.converted_total.amount).toBe(1);
  });

  it('surfaces an error and leaves cost null when the fetch fails', async () => {
    apiFn.mockRejectedValueOnce(new Error('500'));
    const { result } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe('500');
    expect(result.current.cost).toBeNull();
  });

  it('re-fetches when the pricing version is bumped (pricing-edit recompute, #40)', async () => {
    apiFn.mockResolvedValueOnce(makeCost(1)).mockResolvedValueOnce(makeCost(5));
    const { result } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(1));
    expect(apiFn).toHaveBeenCalledTimes(1);

    // As PricingEditorPage.onSaved does — NO prop threading: every mounted
    // useCost (corner, queue banner, settings meter) re-reads under new rates.
    await act(async () => {
      bumpPricingVersion();
    });
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(5));
    expect(apiFn).toHaveBeenCalledTimes(2);
  });

  it('re-fetches when the WS reconnects (missed spend events, #40)', async () => {
    apiFn.mockResolvedValueOnce(makeCost(1)).mockResolvedValueOnce(makeCost(4));
    const { result, rerender } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(1));
    expect(apiFn).toHaveBeenCalledTimes(1);

    wsConnectionState = 'disconnected';
    rerender();
    await act(async () => { await Promise.resolve(); });
    expect(apiFn).toHaveBeenCalledTimes(1); // going down does not fetch

    wsConnectionState = 'connected';
    rerender();
    await waitFor(() => expect(result.current.cost?.converted_total.amount).toBe(4));
    expect(apiFn).toHaveBeenCalledTimes(2);
  });

  it('does NOT re-fetch on rerenders with an unchanged query', async () => {
    apiFn.mockResolvedValue(makeCost(1));
    const { rerender } = renderHook(() => useCost('p1', 'daily'));
    await waitFor(() => expect(apiFn).toHaveBeenCalledTimes(1));
    rerender();
    await act(async () => { await Promise.resolve(); });
    expect(apiFn).toHaveBeenCalledTimes(1);
  });
});
