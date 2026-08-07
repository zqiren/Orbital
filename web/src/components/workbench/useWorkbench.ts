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
 * show a notice. `migrate` spawns a seeded session through the migrate route
 * and returns the new session id for navigation. Card tap (the entry
 * doorway) is NOT in this hook — it's a client-side composer prefill with no
 * network call, handled entirely in WorkbenchPage (spec 2026-07-24).
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
import type { WorkbenchEntry, WorkbenchResponse } from './types';

export interface UseWorkbenchArgs {
  /** Project lens: fetch only this project's entries (omit for the global,
   *  privacy-filtered surface). */
  projectId?: string;
  /** Fired when an exit (Done/Delete) fails in a way the user should see: a
   *  non-409 server error, or an entry with no persisted id (which can never be
   *  exited server-side). Surfacing lives with the page, not the hook, so the
   *  hook's returned field set stays stable (bug #45 — this used to revert
   *  silently). */
  onExitError?: () => void;
}

export interface UseWorkbenchResult {
  entries: WorkbenchEntry[];
  loading: boolean;
  error: string | null;
  /** True briefly after a 409 exit conflict (auto-clears). */
  conflict: boolean;
  refetch: () => Promise<void>;
  exitEntry: (
    projectId: string,
    memId: string | null,
    kind: 'fulfilled' | 'irrelevant',
    reason?: string,
  ) => Promise<void>;
  /** Returns the new session id (for navigation to the project chat). */
  migrate: (projectId: string) => Promise<string>;
}

function workbenchUrl(projectId?: string): string {
  return projectId
    ? `/api/v2/workbench?project_id=${encodeURIComponent(projectId)}`
    : '/api/v2/workbench';
}

export function useWorkbench({ projectId, onExitError }: UseWorkbenchArgs): UseWorkbenchResult {
  const [entries, setEntries] = useState<WorkbenchEntry[]>([]);
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
      }
    } catch (e) {
      if (epochRef.current === epoch) {
        setError(e instanceof Error ? e.message : 'error');
        setEntries([]);
      }
    } finally {
      if (epochRef.current === epoch) setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const exitEntry = useCallback(
    async (pid: string, memId: string | null, kind: 'fulfilled' | 'irrelevant', reason = '') => {
      // Defence in depth (bug #45): an entry with no persisted id — null, or
      // the literal "null" a template-coerced id produces — can never be exited
      // server-side (its POST hits /entries/null/exit → 404). Surface it rather
      // than firing a doomed request or silently doing nothing.
      if (memId == null || memId === 'null') {
        onExitError?.();
        return;
      }
      const id = memId;
      const prevEntries = entries;
      setEntries((cur) => cur.filter((e) => !(e.project_id === pid && e.id === id)));
      try {
        await api(
          `/api/v2/workbench/${encodeURIComponent(pid)}/entries/${encodeURIComponent(id)}/exit`,
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
        } else {
          // Non-409 failure (404/500/network). This used to revert SILENTLY —
          // a failed delete looked like a no-op (bug #45). Surface it.
          onExitError?.();
        }
      }
    },
    [entries, fetchAll, onExitError],
  );

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
    loading,
    error,
    conflict,
    refetch: fetchAll,
    exitEntry,
    migrate,
  };
}
