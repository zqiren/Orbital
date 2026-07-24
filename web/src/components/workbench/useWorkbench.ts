// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Data hook for the Workbench surface (spec §5.3, §5.4, §8).
 *
 * Fetches `GET /api/v2/workbench[?project_id=]` and exposes the flagged
 * entries in server order — the backend sorts overdue-first-then-oldest, so
 * this hook never re-sorts client-side.
 *
 * Exits are optimistic (removed from local state immediately) and revert on
 * a non-2xx response; a 409 on exit (concurrent PROJECT_STATE.md write)
 * additionally triggers a refetch and a brief conflict flag so the page can
 * show a notice. `openEntry`/`migrate` spawn a seeded session through the
 * doorway routes and return the new session id for navigation.
 *
 * This hook dispatches a global `orbital:workbench-changed` event so the
 * sidebar's nav badge (fetched independently, see Sidebar.tsx's
 * `useWorkbenchCount`) can refetch without polling, whenever entry count is
 * known (or provably about) to have changed: `exitEntry` 2xx, `exitEntry` 409
 * (the refetch proves the file changed server-side, even though this
 * client's optimistic write lost), and `migrate` 2xx (a session spawn, not
 * the count change itself — the agent edits PROJECT_STATE.md minutes later,
 * so this only marks that a migration started). It is never dispatched on
 * the optimistic update itself or on a non-409 error path.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from '../../config';
import type { WorkbenchDigest, WorkbenchEntry, WorkbenchResponse } from './types';

export interface UseWorkbenchArgs {
  /** Project lens: fetch only this project's entries (omit for the global,
   *  privacy-filtered surface). */
  projectId?: string;
}

export interface UseWorkbenchResult {
  entries: WorkbenchEntry[];
  /** Per-project "in flight" summaries, passed through as-is from the
   *  response (default []). Callers decide whether to render them — the
   *  page only shows this UI on the global (unlensed) surface. */
  digests: WorkbenchDigest[];
  loading: boolean;
  error: string | null;
  /** True briefly after a 409 exit conflict (auto-clears). */
  conflict: boolean;
  refetch: () => Promise<void>;
  exitEntry: (
    projectId: string,
    memId: string,
    kind: 'fulfilled' | 'irrelevant',
    reason?: string,
  ) => Promise<void>;
  /** Returns the new session id (for navigation to the project chat). */
  openEntry: (projectId: string, memId: string) => Promise<string>;
  /** Returns the new session id (for navigation to the project chat). */
  migrate: (projectId: string) => Promise<string>;
}

function workbenchUrl(projectId?: string): string {
  return projectId
    ? `/api/v2/workbench?project_id=${encodeURIComponent(projectId)}`
    : '/api/v2/workbench';
}

export function useWorkbench({ projectId }: UseWorkbenchArgs): UseWorkbenchResult {
  const [entries, setEntries] = useState<WorkbenchEntry[]>([]);
  const [digests, setDigests] = useState<WorkbenchDigest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);

  // Monotonic epoch: only the newest fetch may write state (project switches
  // mid-flight must not have a slow stale response clobber the fresh one).
  const epochRef = useRef(0);

  const fetchAll = useCallback(async () => {
    const epoch = ++epochRef.current;
    setLoading(true);
    setError(null);
    try {
      const res = await api<WorkbenchResponse>(workbenchUrl(projectId));
      if (epochRef.current === epoch) {
        setEntries(res?.entries ?? []);
        setDigests(res?.digests ?? []);
      }
    } catch (e) {
      if (epochRef.current === epoch) {
        setError(e instanceof Error ? e.message : 'error');
        setEntries([]);
        setDigests([]);
      }
    } finally {
      if (epochRef.current === epoch) setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const exitEntry = useCallback(
    async (pid: string, memId: string, kind: 'fulfilled' | 'irrelevant', reason = '') => {
      const prevEntries = entries;
      setEntries((cur) => cur.filter((e) => !(e.project_id === pid && e.id === memId)));
      try {
        await api(
          `/api/v2/workbench/${encodeURIComponent(pid)}/entries/${encodeURIComponent(memId)}/exit`,
          { method: 'POST', body: JSON.stringify({ kind, reason }) },
        );
        window.dispatchEvent(new CustomEvent('orbital:workbench-changed'));
      } catch (e) {
        setEntries(prevEntries); // revert the optimistic removal
        if (e instanceof ApiError && e.status === 409) {
          setConflict(true);
          await fetchAll();
          window.dispatchEvent(new CustomEvent('orbital:workbench-changed'));
          setTimeout(() => setConflict(false), 4000);
        }
      }
    },
    [entries, fetchAll],
  );

  const openEntry = useCallback(async (pid: string, memId: string): Promise<string> => {
    const res = await api<{ session_id: string }>(
      `/api/v2/workbench/${encodeURIComponent(pid)}/entries/${encodeURIComponent(memId)}/open`,
      { method: 'POST' },
    );
    return res.session_id;
  }, []);

  const migrate = useCallback(async (pid: string): Promise<string> => {
    const res = await api<{ session_id: string }>(
      `/api/v2/workbench/${encodeURIComponent(pid)}/migrate`,
      { method: 'POST' },
    );
    window.dispatchEvent(new CustomEvent('orbital:workbench-changed'));
    return res.session_id;
  }, []);

  return {
    entries,
    digests,
    loading,
    error,
    conflict,
    refetch: fetchAll,
    exitEntry,
    openEntry,
    migrate,
  };
}
