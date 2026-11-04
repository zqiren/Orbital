// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * useWorkbenchFeed — spec 154 §5.2: the App-level store behind the
 * Workbench, the strip glyph's dot and the project-list counts.
 *
 * One `GET /api/v2/workbench/feed` serves every consumer: the daemon selects
 * the few rows that can be on the Workbench (running, approvals, open asks,
 * replies not yet checked inside the window). The store refetches at connect
 * and reconnect and on the events that move a row (one request in flight,
 * one trailing request after it — the 151 coalescing scheme).
 *
 * It also holds what only this client knows:
 *   - pending sends: a task typed on the Workbench, shown at the top of
 *     进行中 from the moment it is sent until the feed carries its session;
 *   - live activity per session (`agent.status_summary`, else
 *     `agent.activity`), shown on running cards and never persisted;
 *   - held cards: a card answered from the Workbench keeps its place in 等你
 *     until the agent's turn ends (§5.8);
 *   - asks closed from a card, hidden until the next feed agrees;
 *   - text to put back in the composer when a send could not start.
 *
 * Module-level on purpose: leaving the Workbench mid-send loses nothing.
 */

import { useEffect, useSyncExternalStore } from 'react';
import { api } from '../config';
import type { WebSocketEvent, WorkbenchFeedRow } from '../types';
import { useWebSocket } from './useWebSocket';
import { isRunning } from '../utils/waitingOnYou';

export interface PendingTask {
  nonce: string;
  text: string;
  /** Known once routed (or picked). */
  project_id: string | null;
  /** Known once minted. */
  session_id: string | null;
  /** The picked agent ('orbital' for Orbital), or null for 自动. */
  agent: string | null;
  created_at: number;
}

interface Held {
  sawRunning: boolean;
  at: number;
  /** The last row seen, drawn while a refetch momentarily lacks it. */
  row: WorkbenchFeedRow;
}

export interface FeedState {
  rows: WorkbenchFeedRow[];
  loaded: boolean;
  error: boolean;
  pending: PendingTask[];
  /** Live text per session: the manager's status line, else the last tool step. */
  activity: Readonly<Record<string, { summary?: string; tool?: string }>>;
  held: ReadonlyMap<string, Held>;
  closedAsks: ReadonlySet<string>;
  composerRestore: string | null;
}

/** A held card that never saw its turn start is let go after this long. */
const HOLD_SAFETY_MS = 2 * 60 * 1000;

const EMPTY: FeedState = {
  rows: [],
  loaded: false,
  error: false,
  pending: [],
  activity: {},
  held: new Map(),
  closedAsks: new Set(),
  composerRestore: null,
};

let state: FeedState = EMPTY;
const listeners = new Set<() => void>();

function set(patch: Partial<FeedState>): void {
  state = { ...state, ...patch };
  for (const fn of listeners) fn();
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function getSnapshot(): FeedState {
  return state;
}

export function sessionKeyOf(projectId: string, sessionId: string | null | undefined): string {
  return `${projectId}\u0000${sessionId ?? ''}`;
}

export function getFeedState(): FeedState {
  return state;
}

// ── Fetch, coalesced ──────────────────────────────────────────────────────

let inFlight = false;
let queued = false;

function applyRows(rows: WorkbenchFeedRow[]): void {
  const byKey = new Map(rows.map((r) => [sessionKeyOf(r.project_id, r.session_id), r]));
  // A pending send leaves once the feed carries its session.
  const pending = state.pending.filter(
    (p) => !(p.project_id && p.session_id && byKey.has(sessionKeyOf(p.project_id, p.session_id))),
  );
  // Held cards: release once the turn that started after the reply ended.
  const held = new Map(state.held);
  const now = Date.now();
  for (const [key, h] of held) {
    const row = byKey.get(key);
    if (row && isRunning(row)) {
      held.set(key, { ...h, sawRunning: true, row });
    } else if (h.sawRunning || now - h.at > HOLD_SAFETY_MS) {
      held.delete(key);
    } else if (row) {
      held.set(key, { ...h, row });
    }
  }
  // An ask closed from a card stays hidden until the feed agrees.
  const openAsks = new Set(
    rows.filter((r) => r.ask).map((r) => `${r.project_id}\u0000${r.ask!.id}`),
  );
  const closedAsks = new Set([...state.closedAsks].filter((k) => openAsks.has(k)));
  // Live activity only for sessions still running.
  const activity: FeedState['activity'] = Object.fromEntries(
    Object.entries(state.activity).filter(([key]) => {
      const row = byKey.get(key);
      return !!row && isRunning(row);
    }),
  );
  set({ rows, loaded: true, error: false, pending, held, closedAsks, activity });
}

/** Refetch the feed: one request in flight, one trailing request after it. */
export function refreshFeed(): Promise<void> {
  if (inFlight) {
    queued = true;
    return Promise.resolve();
  }
  inFlight = true;
  return api<{ rows?: WorkbenchFeedRow[] }>('/api/v2/workbench/feed')
    .then((data) => applyRows(Array.isArray(data?.rows) ? data.rows : []))
    .catch(() => set({ error: true, loaded: true }))
    .finally(() => {
      inFlight = false;
      if (queued) {
        queued = false;
        void refreshFeed();
      }
    });
}

// ── Client-side state ─────────────────────────────────────────────────────

export function addPending(task: PendingTask): void {
  set({ pending: [task, ...state.pending.filter((p) => p.nonce !== task.nonce)] });
}

export function updatePending(nonce: string, patch: Partial<PendingTask>): void {
  set({ pending: state.pending.map((p) => (p.nonce === nonce ? { ...p, ...patch } : p)) });
}

export function removePending(nonce: string): void {
  set({ pending: state.pending.filter((p) => p.nonce !== nonce) });
}

/** Keep this card in 等你 until the turn its reply starts has ended. */
export function holdCard(row: WorkbenchFeedRow): void {
  const held = new Map(state.held);
  held.set(sessionKeyOf(row.project_id, row.session_id), { sawRunning: false, at: Date.now(), row });
  set({ held });
}

export function hideClosedAsk(projectId: string, askId: string): void {
  set({ closedAsks: new Set([...state.closedAsks, `${projectId}\u0000${askId}`]) });
}

export function restoreComposer(text: string | null): void {
  set({ composerRestore: text });
}

export function setActivity(
  projectId: string, sessionId: string, kind: 'summary' | 'tool', text: string,
): void {
  const key = sessionKeyOf(projectId, sessionId);
  const cur = state.activity[key] ?? {};
  if (cur[kind] === text) return;
  set({ activity: { ...state.activity, [key]: { ...cur, [kind]: text } } });
}

/** The live line of a running session, if any. */
export function activityText(feed: FeedState, projectId: string, sessionId: string | null): string | null {
  if (!sessionId) return null;
  const a = feed.activity[sessionKeyOf(projectId, sessionId)];
  return a?.summary ?? a?.tool ?? null;
}

/** Test seam. */
export function __resetWorkbenchFeed(): void {
  inFlight = false;
  queued = false;
  state = EMPTY;
  for (const fn of listeners) fn();
}

export function useWorkbenchFeed(): FeedState {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

// ── WS wiring (mounted once, in App) ─────────────────────────────────────

/** Events that can move a row on or off the Workbench. Not
 * `fanout.task_update` or `chat.sub_agent_message`: both fire per chunk. */
export const FEED_EVENTS: ReadonlyArray<WebSocketEvent['type']> = [
  'workbench.changed',
  'agent.status',
  'sub_agent.dispatched',
  'sub_agent.started',
  'sub_agent.completed',
  'sub_agent.continued',
  'sub_agent.error',
  'sub_agent.failed',
  'sub_agent.stopped',
  'sub_agent.turn_interrupted',
  'chat.pending_enqueued',
  'chat.pending_dispatched',
  'chat.pending_cancelled',
  'fanout.started',
  'fanout.completed',
  'approval.request',
  'approval.resolved',
];

export function useWorkbenchFeedSync(enabled: boolean): void {
  const ws = useWebSocket();
  const connected = ws.connectionState === 'connected';
  useEffect(() => {
    if (enabled && connected) void refreshFeed();
  }, [enabled, connected]);
  useEffect(() => {
    if (!enabled) return;
    const onEvent = () => void refreshFeed();
    const onSummary = (event: WebSocketEvent) => {
      const e = event as { project_id?: string; session_id?: string; summary?: string };
      if (e.project_id && e.session_id && e.summary) {
        setActivity(e.project_id, e.session_id, 'summary', e.summary);
      }
    };
    const onActivity = (event: WebSocketEvent) => {
      const e = event as { project_id?: string; session_id?: string; description?: string };
      if (e.project_id && e.session_id && e.description) {
        setActivity(e.project_id, e.session_id, 'tool', e.description);
      }
    };
    for (const type of FEED_EVENTS) ws.on(type, onEvent);
    ws.on('agent.status_summary', onSummary);
    ws.on('agent.activity', onActivity);
    return () => {
      for (const type of FEED_EVENTS) ws.off(type, onEvent);
      ws.off('agent.status_summary', onSummary);
      ws.off('agent.activity', onActivity);
    };
  }, [ws, enabled]);
}
