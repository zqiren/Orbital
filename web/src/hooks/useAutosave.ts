// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ApiError } from '../config';

export type AutosaveStatus = 'idle' | 'saving' | 'saved' | 'error';

/**
 * Save-as-you-go for a settings form: every edit becomes a partial update.
 *
 * - `saveNow(patch)` — clicks, toggles, selects: sent at once.
 * - `saveSoon(patch)` — typing: sent once the user pauses for `delayMs`, or
 *   earlier on `flush()` (leaving the field) or unmount (leaving the page).
 * - Pending edits merge into one request, and requests run strictly one at a
 *   time, so an older value can never land after a newer one.
 * - A failed patch is kept until `retry()` sends it again under any newer edit
 *   of the same field; the status stays 'error' until then.
 *
 * `save` must be bound to the record being edited: the hook calls the latest
 * one it was rendered with, including from the unmount flush.
 */
export function useAutosave<T extends object>(
  save: (patch: Partial<T>) => Promise<unknown> | void,
  delayMs = 800,
) {
  const saveRef = useRef(save);
  useLayoutEffect(() => {
    saveRef.current = save;
  });

  const pending = useRef<Partial<T>>({});
  const failed = useRef<Partial<T> | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const chain = useRef<Promise<void>>(Promise.resolve());
  const inFlight = useRef(0);
  const mounted = useRef(true);
  const [status, setStatus] = useState<AutosaveStatus>('idle');
  const [error, setError] = useState('');

  const flush = useCallback((): Promise<void> => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
    const patch = pending.current;
    if (Object.keys(patch).length === 0) return chain.current;
    pending.current = {};
    const idle = inFlight.current === 0;
    inFlight.current += 1;
    if (mounted.current) setStatus('saving');
    const run = async () => {
      try {
        await saveRef.current(patch);
      } catch (err: unknown) {
        failed.current = { ...patch, ...(failed.current ?? {}) };
        if (mounted.current) {
          setError(
            err instanceof ApiError ? err.detail : err instanceof Error ? err.message : String(err),
          );
        }
      } finally {
        inFlight.current -= 1;
        if (mounted.current && inFlight.current === 0) {
          setStatus(failed.current ? 'error' : 'saved');
        }
      }
    };
    // Nothing in flight: send now (the request leaves before this returns,
    // which also matters for the unmount flush). Otherwise queue behind it.
    chain.current = idle ? run() : chain.current.then(run);
    return chain.current;
  }, []);

  const saveNow = useCallback(
    (patch: Partial<T>) => {
      pending.current = { ...pending.current, ...patch };
      void flush();
    },
    [flush],
  );

  const saveSoon = useCallback(
    (patch: Partial<T>) => {
      pending.current = { ...pending.current, ...patch };
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => void flush(), delayMs);
    },
    [flush, delayMs],
  );

  const retry = useCallback((): Promise<void> => {
    if (failed.current) {
      pending.current = { ...failed.current, ...pending.current };
      failed.current = null;
      setError('');
    }
    return flush();
  }, [flush]);

  useEffect(() => {
    mounted.current = true;
    // Closing the window mid-pause must not drop what was typed.
    const onPageHide = () => void flush();
    window.addEventListener('pagehide', onPageHide);
    return () => {
      window.removeEventListener('pagehide', onPageHide);
      mounted.current = false;
      void flush();
    };
  }, [flush]);

  return { status, error, saveNow, saveSoon, flush, retry };
}
