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
 *   - held cards: a card answered from the Workbench (its reply field or the
 *     reading pane) is drawn in 进行中 until the agent's turn ends, and then
 *     comes back to 等你 as finished and unchecked (2026-11-04);
 *   - asks closed from a card, hidden until the next feed agrees;
 *   - text to put back in the composer when a send could not start.
 *
 * Module-level on purpose: leaving the Workbench mid-send loses nothing.
 */

import { useEffect, useRef, useSyncExternalStore } from 'react';
import { api } from '../config';
import type { ActivityEvent, WebSocketEvent, WorkbenchFeedRow } from '../types';
import { describeLiveActivity, describeWorkerTool, type ActivityTranslate } from '../utils/chatTransform';
import { translate } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
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
  /** Spec 180: 自动 is still picking (the card shows a shimmering 自动). */
  routing?: boolean;
  /** Spec 180: the route answered with a project it chose (the card's ✦). */
  routed_by?: 'model' | 'fallback' | null;
}

/**
 * Spec 180: a session being moved from a card. `moving` while the daemon
 * stops and moves it (the card's dot rests, its project does not open);
 * `moved` once it answered, until the feed carries the session in `to`.
 */
export interface MoveOverride {
  from: string;
  to: string;
  phase: 'moving' | 'moved';
  /** The daemon restarted the work in `to`. */
  woke: boolean;
}

interface Held {
  sawRunning: boolean;
  at: number;
  /** Spec 180: held because a move restarted the work (正在开始… until the
   *  run is seen). */
  starting?: boolean;
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
  /** Spec 180: sessions whose ✦ the user cleared here (by session key),
   *  hidden until the feed agrees. */
  unmarked: ReadonlySet<string>;
  /** Spec 180: card moves in flight or just done, by session id. */
  moves: ReadonlyMap<string, MoveOverride>;
}

/** A held card that never saw its turn start is let go after this long. */
export const HOLD_SAFETY_MS = 2 * 60 * 1000;

/** Spec 169 B: each hold's own expiry, so a hold whose turn never starts
 *  lets go at HOLD_SAFETY_MS even when no feed fetch comes (or it fails).
 *  Keyed like `held`; cleared whenever the hold goes by any route. */
const holdTimers = new Map<string, ReturnType<typeof setTimeout>>();

function clearHoldTimer(key: string): void {
  const id = holdTimers.get(key);
  if (id === undefined) return;
  clearTimeout(id);
  holdTimers.delete(key);
}

function expireHold(key: string, at: number): void {
  holdTimers.delete(key);
  const h = state.held.get(key);
  // A newer hold has its own timer; a hold whose run was seen is the feed's.
  if (!h || h.at !== at || h.sawRunning) return;
  const left = at + HOLD_SAFETY_MS - Date.now();
  if (left > 0) {
    holdTimers.set(key, setTimeout(() => expireHold(key, at), left));
    return;
  }
  const held = new Map(state.held);
  held.delete(key);
  set({ held });
}

const EMPTY: FeedState = {
  rows: [],
  loaded: false,
  error: false,
  pending: [],
  activity: {},
  held: new Map(),
  closedAsks: new Set(),
  composerRestore: null,
  unmarked: new Set(),
  moves: new Map(),
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
      // The run is seen: its end releases the hold, not the clock.
      clearHoldTimer(key);
    } else if (h.sawRunning || now - h.at >= HOLD_SAFETY_MS) {
      held.delete(key);
      clearHoldTimer(key);
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
  // A cleared ✦ stays hidden until the feed drops `routed_by` too.
  const marked = new Set(
    rows.filter((r) => r.routed_by).map((r) => sessionKeyOf(r.project_id, r.session_id)),
  );
  const unmarked = new Set([...state.unmarked].filter((k) => marked.has(k)));
  // A finished move is the feed's once the session is no longer listed in
  // the project it left.
  const moves = new Map(state.moves);
  for (const [sid, m] of moves) {
    if (m.phase === 'moved' && !byKey.has(sessionKeyOf(m.from, sid))) moves.delete(sid);
  }
  set({ rows, loaded: true, error: false, pending, held, closedAsks, activity, unmarked, moves });
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

/** Draw this card in 进行中 until the turn its reply starts has ended. */
export function holdCard(row: WorkbenchFeedRow): void {
  const key = sessionKeyOf(row.project_id, row.session_id);
  const at = Date.now();
  const held = new Map(state.held);
  held.set(key, { sawRunning: false, at, row });
  clearHoldTimer(key);
  holdTimers.set(key, setTimeout(() => expireHold(key, at), HOLD_SAFETY_MS));
  set({ held });
}

/** Spec 160 round 4: the reply started no turn (the daemon recorded a 对 on
 *  the import scan's confirm card): let the held card go at once. */
export function releaseCard(projectId: string, sessionId: string | null | undefined): void {
  const key = sessionKeyOf(projectId, sessionId ?? null);
  clearHoldTimer(key);
  const held = new Map(state.held);
  if (!held.delete(key)) return;
  set({ held });
}

/**
 * Spec 169 B: the step shared by both routes that answer a card (its reply
 * field, the reading pane's composer), after the card was held. `result`
 * is the inject response, or `null` when the send failed. A reply that
 * started no turn (the daemon recorded it, `status: "recorded"`) or never
 * got through lets the hold go at once; then refetch.
 */
export function settleReply(
  projectId: string, sessionId: string | null | undefined, result: unknown,
): void {
  const recorded = !!result && typeof result === 'object'
    && (result as { status?: unknown }).status === 'recorded';
  if (result === null || recorded) releaseCard(projectId, sessionId);
  void refreshFeed();
}

/**
 * Spec 180 §5.1: the user opened a card Orbital routed — its ✦ goes at once
 * (optimistic, like a checked reply) and the session PATCH clears
 * `routed_by` for every device. A session with no file yet answers 404:
 * the mark stays hidden here, nothing else to do.
 */
export function clearRoutedMark(projectId: string, sessionId: string): void {
  const key = sessionKeyOf(projectId, sessionId);
  if (!state.unmarked.has(key)) set({ unmarked: new Set([...state.unmarked, key]) });
  api(`/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    body: JSON.stringify({ routed_by: null }),
  }).catch(() => { /* hidden here; the next feed decides */ });
}

function renameKey<V>(map: ReadonlyMap<string, V>, from: string, to: string): Map<string, V> {
  const next = new Map(map);
  const v = next.get(from);
  if (v !== undefined) {
    next.delete(from);
    next.set(to, v);
  }
  return next;
}

/**
 * Spec 180 §5.5: move a session to another project from its card. The card
 * stays in place (the move changes neither of its times); while the daemon
 * stops and moves it the dot rests, then the name changes. A held card
 * follows the session to its new project; the stopped run's live line is
 * dropped. Resolves with the daemon's answer; rejects (and puts
 * everything back) when the move failed.
 */
export async function moveSession(
  from: string, sessionId: string, to: string,
): Promise<{ project_id: string; session_id: string; woke: boolean }> {
  const moves = new Map(state.moves);
  moves.set(sessionId, { from, to, phase: 'moving', woke: false });
  set({ moves });
  try {
    const res = await api<{ project_id?: string; session_id?: string; woke?: boolean }>(
      `/api/v2/agents/${encodeURIComponent(from)}/sessions/${encodeURIComponent(sessionId)}/move`,
      { method: 'POST', body: JSON.stringify({ to_project_id: to }) },
    );
    const dest = res?.project_id || to;
    const sid = res?.session_id || sessionId;
    const oldKey = sessionKeyOf(from, sessionId);
    const newKey = sessionKeyOf(dest, sid);
    const timer = holdTimers.get(oldKey);
    if (timer !== undefined) {
      holdTimers.delete(oldKey);
      holdTimers.set(newKey, timer);
    }
    // The run it had was stopped: its live line goes with it (the run in
    // the new project reports its own).
    const activity = { ...state.activity };
    delete activity[oldKey];
    const done = new Map(state.moves);
    done.set(sid, { from, to: dest, phase: 'moved', woke: !!res?.woke });
    const held = renameKey(state.held, oldKey, newKey);
    // The daemon restarted the work in the new project: the card is drawn
    // running from now, as after a reply, until that run ends.
    const snapshot = state.rows.find((r) => r.session_id === sessionId && r.project_id === from)
      ?? state.held.get(oldKey)?.row;
    if (res?.woke && snapshot) {
      const at = Date.now();
      held.set(newKey, { sawRunning: false, at, starting: true, row: { ...snapshot, project_id: dest, routed_by: null } });
      clearHoldTimer(newKey);
      holdTimers.set(newKey, setTimeout(() => expireHold(newKey, at), HOLD_SAFETY_MS));
    }
    set({ moves: done, held, activity });
    void refreshFeed();
    return { project_id: dest, session_id: sid, woke: !!res?.woke };
  } catch (e) {
    const back = new Map(state.moves);
    back.delete(sessionId);
    set({ moves: back });
    void refreshFeed();
    throw e;
  }
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

/**
 * A running card's line for one `agent.activity` event, worded as the chat
 * words the same event. A tool result is not a new step (the chat only marks
 * the call done), so it leaves the line as it was: null.
 */
export function activityLine(e: Partial<ActivityEvent>, tr: ActivityTranslate): string | null {
  if (e.category === 'tool_result' || e.worker_event === 'tool_result') return null;
  if (e.worker_event === 'tool_call') {
    return describeWorkerTool(e.tool_name || 'tool', e.arguments, undefined, tr);
  }
  if (!e.description) return null;
  return e.tool_name ? describeLiveActivity(e.tool_name, e.arguments, undefined, tr, e.description) : e.description;
}

/** The live line of a running session, if any. */
export function activityText(feed: FeedState, projectId: string, sessionId: string | null): string | null {
  if (!sessionId) return null;
  const a = feed.activity[sessionKeyOf(projectId, sessionId)];
  return a?.summary ?? a?.tool ?? null;
}

/** Test seam. */
export function __resetWorkbenchFeed(): void {
  for (const id of holdTimers.values()) clearTimeout(id);
  holdTimers.clear();
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
  'chat.pending_dispatched',
  'chat.pending_cancelled',
  'fanout.started',
  'fanout.completed',
  'approval.request',
  'approval.resolved',
];

export function useWorkbenchFeedSync(enabled: boolean): void {
  const ws = useWebSocket();
  const { locale } = useLocale();
  const localeRef = useRef(locale);
  useEffect(() => { localeRef.current = locale; }, [locale]);
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
      const e = event as Partial<ActivityEvent>;
      if (!e.project_id || !e.session_id) return;
      const line = activityLine(e, (k, v) => translate(localeRef.current, k, v));
      if (line) setActivity(e.project_id, e.session_id, 'tool', line);
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
