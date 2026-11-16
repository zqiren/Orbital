// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Task sources — the client's one copy of the daemon's sources, shared by the
 * Workbench's 任务来源 button (its waiting count) and the panel.
 *
 * Module-level like the Workbench feed (`useWorkbenchFeed`): `GET /sources`
 * at connect and reconnect, then the three pushed events (`sources.changed`,
 * `sources.removed`, `sources.entry`) applied by a pure reducer. A source's
 * detail (its settings and history) is loaded when the panel opens it and
 * kept current by the same events; its sessions' states are not pushed, so an
 * open detail is re-read on `workbench.changed`.
 */

import { useEffect, useSyncExternalStore } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import type { WebSocketEvent } from '../../types';
import { fetchCatalog, fetchSource, fetchSources } from './sourcesApi';
import type { Catalog, HistoryEntry, SourceDetail, SourceSummary } from './types';

export interface SourcesState {
  loaded: boolean;
  list: SourceSummary[];
  /** Arrivals queued across sources that are on (the entry button's count). */
  waiting: number;
  catalog: Catalog | null;
  details: Readonly<Record<string, SourceDetail>>;
}

export const EMPTY_SOURCES: SourcesState = { loaded: false, list: [], waiting: 0, catalog: null, details: {} };

/** The 任务来源 button keeps its amber dot while any source is failed (paused ones too). */
export function anyFailed(list: readonly SourceSummary[]): boolean {
  return list.some((s) => s.status === 'failed');
}

export function totalWaiting(list: readonly SourceSummary[]): number {
  return list.reduce((n, s) => n + (s.on ? s.waiting : 0), 0);
}

const entryTime = (e: HistoryEntry) => Date.parse(e.kind === 'gap' ? e.start : e.arrived_at) || 0;

/** Newest first: by arrival, a gap by its start. */
export function sortHistory(h: HistoryEntry[]): HistoryEntry[] {
  return [...h].sort((a, b) => entryTime(b) - entryTime(a));
}

function summaryFields(s: SourceSummary): SourceSummary {
  const {
    id, type, family, name, display_name, label, on, is_new, registered_by, status, failure,
    last_check_at, waiting, held, catch_up, created_at,
  } = s;
  return {
    id, type, family, name, display_name, label, on, is_new, registered_by, status, failure,
    last_check_at, waiting, held, catch_up, created_at,
  };
}

function upsertSummary(state: SourcesState, s: SourceSummary): SourcesState {
  const row = summaryFields(s);
  const i = state.list.findIndex((x) => x.id === s.id);
  const list = i < 0 ? [row, ...state.list] : state.list.map((x, k) => (k === i ? row : x));
  const d = state.details[s.id];
  const details = d ? { ...state.details, [s.id]: { ...d, ...row } } : state.details;
  return { ...state, list, details, waiting: totalWaiting(list) };
}

/** A detail the daemon answered (a GET, a PATCH, an action): it and its row. */
export function applyDetail(state: SourcesState, d: SourceDetail): SourcesState {
  const next = upsertSummary(state, d);
  return { ...next, details: { ...next.details, [d.id]: { ...d, history: sortHistory(d.history) } } };
}

export type SourcesEvent =
  | { type: 'sources.changed'; source: SourceSummary }
  | { type: 'sources.removed'; source_id: string }
  | { type: 'sources.entry'; source_id: string; entry: HistoryEntry };

export function applySourcesEvent(state: SourcesState, e: SourcesEvent): SourcesState {
  switch (e.type) {
    case 'sources.changed':
      return e.source ? upsertSummary(state, e.source) : state;
    case 'sources.removed': {
      const list = state.list.filter((x) => x.id !== e.source_id);
      const details = { ...state.details };
      delete details[e.source_id];
      return { ...state, list, details, waiting: totalWaiting(list) };
    }
    case 'sources.entry': {
      const d = state.details[e.source_id];
      if (!d || !e.entry) return state;
      const rest = d.history.filter((h) => h.id !== e.entry.id);
      return { ...state, details: { ...state.details, [d.id]: { ...d, history: sortHistory([e.entry, ...rest]) } } };
    }
    default:
      return state;
  }
}

// ── the module store ──────────────────────────────────────────────────────

let state: SourcesState = EMPTY_SOURCES;
const listeners = new Set<() => void>();
function set(next: SourcesState) {
  if (next === state) return;
  state = next;
  for (const fn of listeners) fn();
}
const subscribe = (fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn); }; };
const getSnapshot = () => state;

export function getSourcesState(): SourcesState { return state; }
export function useSources(): SourcesState { return useSyncExternalStore(subscribe, getSnapshot, getSnapshot); }

export async function refreshSources(): Promise<void> {
  try {
    const res = await fetchSources();
    const list = (Array.isArray(res?.sources) ? res.sources : []).map(summaryFields);
    // Details stay; their rows refresh from the list.
    const details: Record<string, SourceDetail> = {};
    for (const s of list) if (state.details[s.id]) details[s.id] = { ...state.details[s.id], ...s };
    set({ ...state, loaded: true, list, details, waiting: totalWaiting(list) });
  } catch {
    set({ ...state, loaded: true });
  }
}

let catalogLoad: Promise<void> | null = null;
export function loadCatalog(force = false): Promise<void> {
  if (catalogLoad && !force) return catalogLoad;
  catalogLoad = fetchCatalog()
    .then((catalog) => { if (Array.isArray(catalog?.types)) set({ ...state, catalog }); else catalogLoad = null; })
    .catch(() => { catalogLoad = null; });
  return catalogLoad;
}

export async function loadDetail(id: string): Promise<SourceDetail | null> {
  try {
    const d = await fetchSource(id);
    if (!d || d.id !== id || !Array.isArray(d.history)) return null;
    set(applyDetail(state, d));
    return d;
  } catch {
    return null;
  }
}

/** Older history (`?before=`), appended to the loaded detail. */
export async function loadOlder(id: string): Promise<void> {
  const d = state.details[id];
  const last = d?.history[d.history.length - 1];
  if (!d || !last) return;
  const more = await fetchSource(id, last.id);
  const cur = state.details[id];
  if (!cur) return;
  const seen = new Set(cur.history.map((h) => h.id));
  const history = sortHistory([...cur.history, ...more.history.filter((h) => !seen.has(h.id))]);
  set({ ...state, details: { ...state.details, [id]: { ...cur, history, history_more: more.history_more } } });
}

export function putDetail(d: SourceDetail): void { set(applyDetail(state, d)); }
export function dropSource(id: string): void { set(applySourcesEvent(state, { type: 'sources.removed', source_id: id })); }
/** Local edit of a loaded detail (the instruction as typed), before the daemon answers. */
export function patchLocal(id: string, patch: Partial<SourceDetail>): void {
  const d = state.details[id];
  if (d) set({ ...state, details: { ...state.details, [id]: { ...d, ...patch } } });
}

/** Test seam. */
export function __resetSources(): void { state = EMPTY_SOURCES; catalogLoad = null; for (const fn of listeners) fn(); }

/** Which detail the panel shows (an open detail is re-read on workbench.changed). */
let openDetail: string | null = null;
export function setOpenDetail(id: string | null): void { openDetail = id; }

/** WS wiring: mounted while the Workbench is on screen. */
export function useSourcesSync(enabled = true): void {
  const ws = useWebSocket();
  const connected = ws.connectionState === 'connected';
  useEffect(() => {
    if (enabled && connected) void refreshSources();
  }, [enabled, connected]);
  useEffect(() => {
    if (!enabled) return;
    const onEvent = (e: WebSocketEvent) => set(applySourcesEvent(state, e as SourcesEvent));
    let pending: ReturnType<typeof setTimeout> | null = null;
    const onWorkbench = () => {
      const id = openDetail;
      if (!id || pending) return;
      pending = setTimeout(() => { pending = null; void loadDetail(id); }, 400);
    };
    ws.on('sources.changed', onEvent);
    ws.on('sources.removed', onEvent);
    ws.on('sources.entry', onEvent);
    ws.on('workbench.changed', onWorkbench);
    ws.on('agent.status', onWorkbench);
    return () => {
      ws.off('sources.changed', onEvent);
      ws.off('sources.removed', onEvent);
      ws.off('sources.entry', onEvent);
      ws.off('workbench.changed', onWorkbench);
      ws.off('agent.status', onWorkbench);
      if (pending) clearTimeout(pending);
    };
  }, [ws, enabled]);
}
