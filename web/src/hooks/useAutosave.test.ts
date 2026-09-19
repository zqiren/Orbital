// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAutosave } from './useAutosave';

type Patch = { name?: string; card_id?: string | null; rules?: string };

function deferred() {
  let resolve!: () => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('useAutosave', () => {
  it('saveNow sends the patch at once and reports saving → saved', async () => {
    const save = vi.fn(() => Promise.resolve());
    const { result } = renderHook(() => useAutosave<Patch>(save));
    expect(result.current.status).toBe('idle');

    act(() => result.current.saveNow({ card_id: 'card_b' }));
    expect(save).toHaveBeenCalledWith({ card_id: 'card_b' });
    expect(result.current.status).toBe('saving');
    await act(async () => {});
    expect(result.current.status).toBe('saved');
  });

  it('saveSoon waits for typing to stop and sends only the latest value', async () => {
    const save = vi.fn(() => Promise.resolve());
    const { result } = renderHook(() => useAutosave<Patch>(save, 800));

    act(() => result.current.saveSoon({ rules: 'a' }));
    act(() => vi.advanceTimersByTime(500));
    act(() => result.current.saveSoon({ rules: 'ab' }));
    act(() => vi.advanceTimersByTime(500));
    expect(save).not.toHaveBeenCalled();
    await act(async () => vi.advanceTimersByTime(300));
    expect(save).toHaveBeenCalledTimes(1);
    expect(save).toHaveBeenCalledWith({ rules: 'ab' });
  });

  it('flush sends pending edits without waiting (leaving a field)', () => {
    const save = vi.fn(() => Promise.resolve());
    const { result } = renderHook(() => useAutosave<Patch>(save));
    act(() => result.current.saveSoon({ name: 'Ada' }));
    act(() => {
      void result.current.flush();
    });
    expect(save).toHaveBeenCalledWith({ name: 'Ada' });
    act(() => vi.advanceTimersByTime(2000));
    expect(save).toHaveBeenCalledTimes(1);
  });

  it('a click merges with pending typing into one request', () => {
    const save = vi.fn(() => Promise.resolve());
    const { result } = renderHook(() => useAutosave<Patch>(save));
    act(() => result.current.saveSoon({ name: 'Ada' }));
    act(() => result.current.saveNow({ card_id: 'card_b' }));
    expect(save).toHaveBeenCalledTimes(1);
    expect(save).toHaveBeenCalledWith({ name: 'Ada', card_id: 'card_b' });
  });

  it('never has two requests in flight: a later save waits for the earlier one', async () => {
    const first = deferred();
    const save = vi.fn().mockReturnValueOnce(first.promise).mockResolvedValue(undefined);
    const { result } = renderHook(() => useAutosave<Patch>(save));

    act(() => result.current.saveNow({ card_id: 'card_a' }));
    act(() => result.current.saveNow({ card_id: 'card_b' }));
    expect(save).toHaveBeenCalledTimes(1);

    await act(async () => first.resolve());
    expect(save).toHaveBeenCalledTimes(2);
    expect(save).toHaveBeenLastCalledWith({ card_id: 'card_b' });
    await act(async () => {});
    expect(result.current.status).toBe('saved');
  });

  it('a failure says so, and Retry re-sends it under any newer edit', async () => {
    const save = vi
      .fn()
      .mockRejectedValueOnce(new Error('Project not found'))
      .mockResolvedValue(undefined);
    const { result } = renderHook(() => useAutosave<Patch>(save));

    act(() => result.current.saveNow({ name: 'Ada', rules: 'old' }));
    await act(async () => {});
    expect(result.current.status).toBe('error');
    expect(result.current.error).toBe('Project not found');

    act(() => result.current.saveSoon({ rules: 'new' }));
    await act(async () => result.current.retry());
    expect(save).toHaveBeenLastCalledWith({ name: 'Ada', rules: 'new' });
    expect(result.current.status).toBe('saved');
    expect(result.current.error).toBe('');
  });

  it('a later successful save does not hide an earlier failure', async () => {
    const save = vi
      .fn()
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValue(undefined);
    const { result } = renderHook(() => useAutosave<Patch>(save));
    act(() => result.current.saveNow({ name: 'Ada' }));
    await act(async () => {});
    act(() => result.current.saveNow({ card_id: 'card_b' }));
    await act(async () => {});
    expect(result.current.status).toBe('error');
  });

  it('unmounting sends what is still pending, through the save it was given', () => {
    const save = vi.fn(() => Promise.resolve());
    const { result, unmount } = renderHook(() => useAutosave<Patch>(save));
    act(() => result.current.saveSoon({ rules: 'typed then left' }));
    unmount();
    expect(save).toHaveBeenCalledWith({ rules: 'typed then left' });
  });

  it('an empty flush sends nothing', () => {
    const save = vi.fn(() => Promise.resolve());
    const { result } = renderHook(() => useAutosave<Patch>(save));
    act(() => {
      void result.current.flush();
    });
    expect(save).not.toHaveBeenCalled();
    expect(result.current.status).toBe('idle');
  });
});
