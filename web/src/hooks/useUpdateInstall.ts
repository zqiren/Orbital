// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * The in-app upgrade's state (agent_os/updater.py) for the UpdatePill.
 *
 * Seeded by the pill's mount fetch of /api/v2/update-status (its `install`
 * field), then by the POST that starts a run, then polled from
 * GET /api/v2/update/install once a second while a run is in flight. The
 * daemon also broadcasts `update.state` over the WS; the pill does not rely
 * on it (adding it to the typed event union is not needed for one window,
 * and the poll survives a WS reconnect mid-download).
 */

const API_BASE = import.meta.env.VITE_API_BASE || '';
export const UPDATE_POLL_MS = 1000;

export type InstallPhase =
  | 'idle' | 'downloading' | 'verifying' | 'ready' | 'installing' | 'failed';

export interface InstallState {
  state: InstallPhase;
  version: string | null;
  progress: number;
  error: string | null;
  can_install: boolean;
}

const ACTIVE: ReadonlySet<InstallPhase> = new Set(['downloading', 'verifying', 'installing']);

export function isActive(s: InstallState | null): boolean {
  return !!s && ACTIVE.has(s.state);
}

export function useUpdateInstall(initial: InstallState | null) {
  const [install, setInstall] = useState<InstallState | null>(initial);
  // The daemon refused the click (409: this app cannot update itself).
  const [refused, setRefused] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (initial) setInstall(initial);
  }, [initial]);

  useEffect(() => {
    if (!isActive(install)) return;
    timer.current = setTimeout(() => {
      fetch(`${API_BASE}/api/v2/update/install`)
        .then((r) => (r.ok ? r.json() : null))
        .then((data: InstallState | null) => {
          // While installing, the daemon going away is the expected outcome:
          // keep showing "Restarting…" (a new copy of this state re-arms the poll).
          setInstall((prev) => (data ? data : prev ? { ...prev } : prev));
        })
        .catch(() => setInstall((prev) => (prev ? { ...prev } : prev)));
    }, UPDATE_POLL_MS);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [install]);

  const start = useCallback(async (region: 'global' | 'china') => {
    setRefused(false);
    try {
      const r = await fetch(`${API_BASE}/api/v2/update/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ region }),
      });
      if (!r.ok) {
        setRefused(true);
        return false;
      }
      setInstall((await r.json()) as InstallState);
      return true;
    } catch {
      setRefused(true);
      return false;
    }
  }, []);

  /** Forget a finished run (the pill's dismiss). */
  const reset = useCallback(() => {
    setInstall(null);
    setRefused(false);
  }, []);

  return { install, refused, start, reset };
}
