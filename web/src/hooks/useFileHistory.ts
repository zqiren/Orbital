// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 065 slice two — the client side of `/files/history`.
 *
 * Three small module stores, shared by every surface that shows history (the
 * file header's 历史 glyph, the history column, each turn's changed-files
 * line, the undo sheet), so one project costs one status request however many
 * of them are on screen:
 *
 * - **status** per project. `has_changes` is the D10 gate: until the daemon
 *   has recorded one change, nothing history-related renders at all.
 * - **epoch** per project, bumped by `bumpFileHistory` after anything that can
 *   add a version (a turn ending, a restore, a name, the Files panel save).
 *   Lists and turns re-read when it moves. The daemon emits no event when a
 *   version lands (slice one), so this is the refresh: on open and after
 *   each turn, never a poll.
 * - **focus**: "open this file's history on this version", set by the undo
 *   sheet's 查看 and the changed-files chips, taken by the FilePreview that
 *   shows the file.
 */
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { api } from '../config';
import { isMemoryFile } from '../components/history/historyModel';
import type {
  AgentRunStatus,
  FileHistoryList,
  HistoryRestoreItem,
  HistoryRestoreResult,
  HistoryStatus,
  HistoryView,
  TurnFiles,
} from '../types';

const base = (projectId: string) =>
  `/api/v2/projects/${encodeURIComponent(projectId)}/files/history`;

function qs(params: Record<string, string | number | null | undefined>): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined || v === '') continue;
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  }
  return parts.length ? `?${parts.join('&')}` : '';
}

/** Raw calls. Each rejects on a non-2xx answer (ApiError from `api`). */
export const historyApi = {
  status: (projectId: string) => api<HistoryStatus>(`${base(projectId)}/status`),
  list: (projectId: string, path: string) =>
    api<FileHistoryList>(`${base(projectId)}${qs({ path })}`),
  view: (projectId: string, path: string, version: number, fromVersion?: number | null) =>
    api<HistoryView>(
      `${base(projectId)}/view${qs({ path, version, from_version: fromVersion ?? undefined })}`,
    ),
  turn: (projectId: string, sessionId: string, fromTs: string, toTs?: string | null) =>
    api<TurnFiles>(
      `${base(projectId)}/turn${qs({ session_id: sessionId, from_ts: fromTs, to_ts: toTs ?? undefined })}`,
    ),
  restore: (projectId: string, items: HistoryRestoreItem[]) =>
    api<{ results: HistoryRestoreResult[] }>(`${base(projectId)}/restore`, {
      method: 'POST',
      body: JSON.stringify({ items }),
    }),
  name: (projectId: string, path: string, sha: string, name: string) =>
    api<{ path: string; sha: string; name: string | null }>(`${base(projectId)}/name`, {
      method: 'POST',
      body: JSON.stringify({ path, sha, name }),
    }),
};

// ── tiny pub/sub ──────────────────────────────────────────────────────────

type Listener = () => void;
const listeners = new Map<string, Set<Listener>>();

function subscribeKey(key: string, fn: Listener): () => void {
  let set = listeners.get(key);
  if (!set) {
    set = new Set();
    listeners.set(key, set);
  }
  set.add(fn);
  return () => {
    set?.delete(fn);
  };
}

function notify(key: string) {
  for (const fn of listeners.get(key) ?? []) fn();
}

// ── status + epoch ────────────────────────────────────────────────────────

const statusByProject = new Map<string, HistoryStatus | null>();
const statusInFlight = new Map<string, Promise<void>>();
const epochByProject = new Map<string, number>();

function fetchStatus(projectId: string): Promise<void> {
  const running = statusInFlight.get(projectId);
  if (running) return running;
  const p = historyApi
    .status(projectId)
    .then(
      (s) => {
        // An older daemon (no history routes) or a stub answers something
        // that is not a status: treat it as "no history", never as an error.
        const ok = s && typeof s === 'object' && !Array.isArray(s) && 'status' in s;
        statusByProject.set(projectId, ok ? s : null);
        if (ok && s.has_changes) rememberHadChanges(projectId);
      },
      () => {
        if (!statusByProject.has(projectId)) statusByProject.set(projectId, null);
      },
    )
    .finally(() => {
      statusInFlight.delete(projectId);
      notify(`status:${projectId}`);
    });
  statusInFlight.set(projectId, p);
  return p;
}

/**
 * Something may have added a version: re-read the status, and let every
 * mounted list / turn line of the project re-fetch. `delayMs` re-runs it once
 * more after the watcher's quiet window (spec 065 §5.1: 2 s), so a shell or
 * worker write that settles just after the turn ends still shows.
 */
export function bumpFileHistory(projectId: string, opts: { again?: number } = {}) {
  epochByProject.set(projectId, (epochByProject.get(projectId) ?? 0) + 1);
  void fetchStatus(projectId);
  notify(`epoch:${projectId}`);
  if (opts.again && opts.again > 0) {
    setTimeout(() => bumpFileHistory(projectId), opts.again);
  }
}

/** The watcher settles writes over a 2 s quiet window; re-read once after it. */
export const SETTLE_MS = 3500;

const IN_RUN: ReadonlySet<AgentRunStatus> = new Set<AgentRunStatus>(['running', 'waiting', 'pending_approval']);
const runningByProject = new Set<string>();

/**
 * Every project's agent.status, from App. A run that ends re-reads the
 * project's history when this device already holds an answer for it: the
 * ChatTab bump only sees runs in the open project, so a run started and
 * watched from the Workbench left a cached "no changes" (a project's first
 * change never showed its changed-files line until a reload). A project
 * nothing has asked about stays unfetched; its first surface reads it fresh.
 */
export function noteAgentStatus(projectId: string, status: AgentRunStatus) {
  if (IN_RUN.has(status)) {
    runningByProject.add(projectId);
    return;
  }
  if (!runningByProject.delete(projectId)) return;
  if (statusByProject.has(projectId)) bumpFileHistory(projectId, { again: SETTLE_MS });
}

/** The project's history health; null until known or when the daemon has none. */
export function useHistoryStatus(projectId: string | undefined): HistoryStatus | null {
  const key = projectId ?? '';
  const subscribe = useCallback((fn: Listener) => subscribeKey(`status:${key}`, fn), [key]);
  const value = useSyncExternalStore(
    subscribe,
    () => (key ? statusByProject.get(key) ?? null : null),
    () => null,
  );
  useEffect(() => {
    if (key && !statusByProject.has(key)) void fetchStatus(key);
  }, [key]);
  return value;
}

// A project whose store became unreadable reports has_changes false (the
// daemon cannot read it). Spec 065 §5.8: a project that HAD history and goes
// unavailable keeps the glyph and says 历史不可用, so this device remembers
// which projects it has seen with changes. Per device, additive, no backend.
const HAD_KEY = (projectId: string) => `orbital.history.had.${projectId}`;

function rememberHadChanges(projectId: string) {
  try {
    localStorage.setItem(HAD_KEY(projectId), '1');
  } catch {
    /* storage unavailable: the glyph just follows has_changes */
  }
}

function hadChanges(projectId: string): boolean {
  try {
    return localStorage.getItem(HAD_KEY(projectId)) === '1';
  } catch {
    return false;
  }
}

/**
 * Whether the project's history surfaces render at all (D10): once it has a
 * recorded change, or when it is unavailable after having had one.
 */
export function historyVisible(projectId: string | undefined, status: HistoryStatus | null): boolean {
  if (!projectId || !status) return false;
  if (status.has_changes) return true;
  return status.status === 'unavailable' && hadChanges(projectId);
}

/** Re-read signal: moves on every `bumpFileHistory(projectId)`. */
export function useHistoryEpoch(projectId: string | undefined): number {
  const key = projectId ?? '';
  const subscribe = useCallback((fn: Listener) => subscribeKey(`epoch:${key}`, fn), [key]);
  return useSyncExternalStore(
    subscribe,
    () => epochByProject.get(key) ?? 0,
    () => 0,
  );
}

// ── one turn's changed files ──────────────────────────────────────────────

interface TurnEntry {
  epoch: number;
  data: TurnFiles | null;
  loading: boolean;
}
const turnCache = new Map<string, TurnEntry>();

function turnKey(projectId: string, sessionId: string, fromTs: string, toTs: string | null) {
  return `${projectId}|${sessionId}|${fromTs}|${toTs ?? ''}`;
}

/**
 * The files one turn of `sessionId` changed, fetched once per turn and again
 * whenever the project's epoch moves. `enabled` false (no history yet, the
 * turn still running, off screen) fetches nothing and returns what is cached.
 */
export function useTurnFiles(
  projectId: string | undefined,
  sessionId: string | undefined,
  fromTs: string | undefined,
  toTs: string | null,
  enabled: boolean,
): TurnFiles | null {
  const epoch = useHistoryEpoch(projectId);
  const key =
    projectId && sessionId && fromTs ? turnKey(projectId, sessionId, fromTs, toTs) : '';
  const subscribe = useCallback((fn: Listener) => subscribeKey(`turn:${key}`, fn), [key]);
  const entry = useSyncExternalStore(
    subscribe,
    () => (key ? turnCache.get(key) : undefined),
    () => undefined,
  );
  useEffect(() => {
    if (!enabled || !key || !projectId || !sessionId || !fromTs) return;
    const held = turnCache.get(key);
    // Up to date, or a read is in flight: when that read lands the entry
    // changes, this effect runs again, and an epoch that moved meanwhile
    // (a bump during the read) triggers one more.
    if (held && (held.epoch === epoch || held.loading)) return;
    turnCache.set(key, { epoch, data: held?.data ?? null, loading: true });
    void historyApi.turn(projectId, sessionId, fromTs, toTs).then(
      (data) => {
        const ok = data && typeof data === 'object' && Array.isArray((data as TurnFiles).files);
        turnCache.set(key, { epoch, data: ok ? data : null, loading: false });
        notify(`turn:${key}`);
      },
      () => {
        turnCache.set(key, { epoch, data: held?.data ?? null, loading: false });
        notify(`turn:${key}`);
      },
    );
  }, [enabled, key, epoch, projectId, sessionId, fromTs, toTs, entry]);
  return entry?.data ?? null;
}

// ── one file's versions ───────────────────────────────────────────────────

/**
 * A file's versions while `active` (the column is open), re-read on every
 * epoch move. Holds the last answer through a re-read so the timeline never
 * blanks; `reload` forces one (after a restore or a name).
 */
export function useFileVersions(
  projectId: string | undefined,
  path: string | null,
  active: boolean,
): { list: FileHistoryList | null; loading: boolean; failed: boolean; reload: () => void } {
  const epoch = useHistoryEpoch(projectId);
  const [tick, setTick] = useState(0);
  const key = projectId && path ? `${projectId}|${path}` : '';
  // What this render wants; an answer is "settled" once it carries this id.
  const want = `${key}|${epoch}|${tick}`;
  const [settled, setSettled] = useState<{
    key: string;
    want: string;
    list: FileHistoryList | null;
    failed: boolean;
  } | null>(null);
  useEffect(() => {
    if (!active || !projectId || !path) return;
    let cancelled = false;
    historyApi.list(projectId, path).then(
      (list) => {
        if (!cancelled) setSettled({ key, want, list, failed: false });
      },
      () => {
        if (!cancelled) {
          setSettled((prev) => ({ key, want, list: prev?.key === key ? prev.list : null, failed: true }));
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [active, projectId, path, key, want]);
  const reload = useCallback(() => setTick((n) => n + 1), []);
  const mine = settled !== null && settled.key === key;
  return {
    // The last answer for this file stays on screen through a re-read.
    list: mine ? settled.list : null,
    loading: active && (!mine || settled.want !== want),
    failed: mine && settled.failed,
    reload,
  };
}

// ── when an element has been on screen ───────────────────────────────────

/** Once the element has been on screen (no observer, as in tests: at once). */
export function useSeenOnce<T extends Element>(): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === 'undefined');
  useEffect(() => {
    if (seen) return;
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        setSeen(true);
        obs.disconnect();
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [seen]);
  return [ref, seen];
}

/**
 * Spec 065 §5.9 — a capsule's turn: its changed files once the project has
 * history, the turn has ended and the capsule has been seen; null otherwise
 * (the capsule then renders exactly as before).
 */
export function useCapsuleTurn(
  projectId: string | undefined,
  sessionId: string | undefined,
  fromTs: string | undefined,
  toTs: string | null,
  ended: boolean,
  seen: boolean,
): TurnFiles | null {
  const status = useHistoryStatus(fromTs ? projectId : undefined);
  const enabled = !!fromTs && !!status?.has_changes && ended && seen;
  const data = useTurnFiles(projectId, sessionId, fromTs, toTs, enabled);
  // Memory files are Orbital's bookkeeping, not the agent's work on the
  // user's files: a turn that only touched them keeps the capsule as it was.
  if (!enabled || !data || !data.files.some((f) => !isMemoryFile(f.path))) return null;
  return data;
}

// ── focus: open a file's history on one version ───────────────────────────

export interface HistoryFocus {
  path: string;
  /** The version to select (a turn's last write); null = just open. */
  v: number | null;
  nonce: number;
}
const focusByProject = new Map<string, HistoryFocus>();
let focusNonce = 0;

export function normalizeHistoryPath(path: string): string {
  return path.replace(/^\.\//, '').replace(/\/+$/, '');
}

/** Ask the FilePreview that next shows `path` to open its history on `v`. */
export function requestHistoryFocus(projectId: string, path: string, v: number | null) {
  focusNonce += 1;
  focusByProject.set(projectId, { path: normalizeHistoryPath(path), v, nonce: focusNonce });
  notify(`focus:${projectId}`);
}

/** The pending focus for this project's `path`, if any. */
export function useHistoryFocus(projectId: string | undefined, path: string | null): HistoryFocus | null {
  const key = projectId ?? '';
  const subscribe = useCallback((fn: Listener) => subscribeKey(`focus:${key}`, fn), [key]);
  const focus = useSyncExternalStore(
    subscribe,
    () => focusByProject.get(key) ?? null,
    () => null,
  );
  if (!focus || !path) return null;
  const want = normalizeHistoryPath(path);
  return focus.path === want || want.endsWith(`/${focus.path}`) || focus.path.endsWith(`/${want}`)
    ? focus
    : null;
}

/** Mark a focus handled so a remount does not reopen it. */
export function clearHistoryFocus(projectId: string, nonce: number) {
  const held = focusByProject.get(projectId);
  if (held && held.nonce === nonce) {
    focusByProject.delete(projectId);
    notify(`focus:${projectId}`);
  }
}

/** Test seam: forget every cached answer. */
export function __resetFileHistoryStores() {
  statusByProject.clear();
  statusInFlight.clear();
  epochByProject.clear();
  runningByProject.clear();
  turnCache.clear();
  focusByProject.clear();
  listeners.clear();
}
