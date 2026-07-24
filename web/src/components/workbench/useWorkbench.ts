// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Data hook for the Workbench surface (spec §5.3, §5.4, §6, §8).
 *
 * Fetches `GET /api/v2/workbench[?project_id=]` and exposes a single merged,
 * sorted list (`items`) — spec §6's "ONE list, no bands": flagged entries and
 * daemon-computed cards interleave by the same rank the backend uses for
 * entries alone (overdue first, then oldest first), since the backend only
 * sorts `entries` and leaves the merge with `computed` to the surface.
 *
 * Exits/dismissals are optimistic (removed from local state immediately) and
 * revert on a non-2xx response; a 409 on exit (concurrent PROJECT_STATE.md
 * write) additionally triggers a refetch and a brief conflict flag so the
 * page can show a notice. `openEntry`/`migrate` spawn a seeded session
 * through the doorway routes and return the new session id for navigation.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from '../../config';
import type {
  WorkbenchComputedCard,
  WorkbenchEntry,
  WorkbenchListItem,
  WorkbenchResponse,
} from './types';

export interface UseWorkbenchArgs {
  /** Project lens: fetch only this project's entries/computed cards (omit
   *  for the global, privacy-filtered surface). */
  projectId?: string;
}

export interface UseWorkbenchResult {
  entries: WorkbenchEntry[];
  computed: WorkbenchComputedCard[];
  /** Merged + sorted for rendering — see `mergeAndSort`. */
  items: WorkbenchListItem[];
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
  dismissComputed: (projectId: string, type: string, key: string) => Promise<void>;
  /** Returns the new session id (for navigation to the project chat). */
  openEntry: (projectId: string, memId: string) => Promise<string>;
  /** Seeded doorway for overdue / broken_automation computed cards. */
  openComputed: (projectId: string, type: string, key: string) => Promise<string>;
  /** Disable a broken automation (PATCH trigger enabled=false) + refetch. */
  disableTrigger: (projectId: string, triggerId: string) => Promise<void>;
  /** Returns the new session id (for navigation to the project chat). */
  migrate: (projectId: string) => Promise<string>;
}

function workbenchUrl(projectId?: string): string {
  return projectId
    ? `/api/v2/workbench?project_id=${encodeURIComponent(projectId)}`
    : '/api/v2/workbench';
}

/** Sort rank mirroring the backend's `(not overdue, created)` tuple sort
 *  (`agent_os/api/routes/workbench.py::get_workbench`): overdue first, then
 *  ascending by date — oldest float up. An `overdue`-type computed card
 *  asserts the same "overdue" fact as an overdue entry, so it ranks with
 *  them; other computed cards rank with the not-yet-overdue entries, keyed
 *  by `since`. */
function sortRank(item: WorkbenchListItem): [number, string] {
  if (item.kind === 'entry') {
    return [item.data.overdue ? 0 : 1, item.data.created || '9999-99-99'];
  }
  const isOverdue = item.data.type === 'overdue';
  return [isOverdue ? 0 : 1, item.data.since || '9999-99-99'];
}

export function mergeAndSort(
  entries: WorkbenchEntry[],
  computed: WorkbenchComputedCard[],
): WorkbenchListItem[] {
  const items: WorkbenchListItem[] = [
    ...entries.map((e) => ({ kind: 'entry' as const, data: e })),
    ...computed.map((c) => ({ kind: 'computed' as const, data: c })),
  ];
  return items
    .map((item) => ({ item, rank: sortRank(item) }))
    .sort((a, b) => {
      if (a.rank[0] !== b.rank[0]) return a.rank[0] - b.rank[0];
      if (a.rank[1] < b.rank[1]) return -1;
      if (a.rank[1] > b.rank[1]) return 1;
      return 0;
    })
    .map(({ item }) => item);
}

export function useWorkbench({ projectId }: UseWorkbenchArgs): UseWorkbenchResult {
  const [entries, setEntries] = useState<WorkbenchEntry[]>([]);
  const [computed, setComputed] = useState<WorkbenchComputedCard[]>([]);
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
        setComputed(res?.computed ?? []);
      }
    } catch (e) {
      if (epochRef.current === epoch) {
        setError(e instanceof Error ? e.message : 'error');
        setEntries([]);
        setComputed([]);
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
      } catch (e) {
        setEntries(prevEntries); // revert the optimistic removal
        if (e instanceof ApiError && e.status === 409) {
          setConflict(true);
          await fetchAll();
          setTimeout(() => setConflict(false), 4000);
        }
      }
    },
    [entries, fetchAll],
  );

  const dismissComputed = useCallback(
    async (pid: string, type: string, key: string) => {
      const prevComputed = computed;
      setComputed((cur) => cur.filter((c) => !(c.project_id === pid && c.type === type && c.key === key)));
      try {
        await api(
          `/api/v2/workbench/${encodeURIComponent(pid)}/computed/${encodeURIComponent(type)}/${encodeURIComponent(key)}/dismiss`,
          { method: 'POST' },
        );
      } catch {
        setComputed(prevComputed); // revert
      }
    },
    [computed],
  );

  const openEntry = useCallback(async (pid: string, memId: string): Promise<string> => {
    const res = await api<{ session_id: string }>(
      `/api/v2/workbench/${encodeURIComponent(pid)}/entries/${encodeURIComponent(memId)}/open`,
      { method: 'POST' },
    );
    return res.session_id;
  }, []);

  const openComputed = useCallback(
    async (pid: string, type: string, key: string): Promise<string> => {
      const res = await api<{ session_id: string }>(
        `/api/v2/workbench/${encodeURIComponent(pid)}/computed/${encodeURIComponent(type)}/${encodeURIComponent(key)}/open`,
        { method: 'POST' },
      );
      return res.session_id;
    },
    [],
  );

  const disableTrigger = useCallback(
    async (pid: string, triggerId: string): Promise<void> => {
      await api(
        `/api/v2/projects/${encodeURIComponent(pid)}/triggers/${encodeURIComponent(triggerId)}`,
        { method: 'PATCH', body: JSON.stringify({ enabled: false }) },
      );
      await fetchAll(); // the detector skips disabled triggers — card clears
    },
    [fetchAll],
  );

  const migrate = useCallback(async (pid: string): Promise<string> => {
    const res = await api<{ session_id: string }>(
      `/api/v2/workbench/${encodeURIComponent(pid)}/migrate`,
      { method: 'POST' },
    );
    return res.session_id;
  }, []);

  return {
    entries,
    computed,
    items: mergeAndSort(entries, computed),
    loading,
    error,
    conflict,
    refetch: fetchAll,
    exitEntry,
    dismissComputed,
    openEntry,
    openComputed,
    disableTrigger,
    migrate,
  };
}
