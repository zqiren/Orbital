// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 195: an agent that Orbital (or the pinned worker) dispatched shows in
 * the conversation as ONE card; its steps live inside the card and open in
 * the docked panel. Pure helpers shared by the history transform and the live
 * handlers in ChatView, so the two build the same item.
 *
 * Who started a dispatch decides card vs inline (marker `_meta.initiator`,
 * live events' `initiator`):
 *   management_agent → card, the worker's final reply a bubble under it;
 *   worker:<x>       → card, the reply folded inside it (x writes its own answer);
 *   user_pinned / queue_item → inline, exactly as before.
 * A row written before the initiator existed: no suppress_wake → card,
 * suppress_wake → inline (spec F1).
 */
import type { DisplayItem } from './chatTransform';
import { skillAwarePreview } from '../lib/skill-parsing';

export type DispatchCardItem = Extract<DisplayItem, { type: 'dispatch_card' }>;
export type CardStatus = DispatchCardItem['status'];

const WORKER_PREFIX = 'worker:';

/** A card's status dot: green pulsing = working, amber = waits on you, red =
 *  failed, hollow = stopped, none = done. */
export type CardDot = 'running' | 'needs-you' | 'error' | 'stopped' | 'none';

export function cardDot(status: CardStatus, needsYou: boolean): CardDot {
  if (status === 'running') return needsYou ? 'needs-you' : 'running';
  if (status === 'error') return 'error';
  if (status === 'stopped') return 'stopped';
  return 'none';
}

export function isCardedInitiator(initiator: string | undefined, suppressWake = false): boolean {
  if (typeof initiator === 'string' && initiator) {
    return initiator === 'management_agent' || initiator.startsWith(WORKER_PREFIX);
  }
  return !suppressWake;
}

/** A worker dispatched it: the reply goes back to that worker, not into the chat. */
export function isFoldedInitiator(initiator: string | undefined): boolean {
  return typeof initiator === 'string' && initiator.startsWith(WORKER_PREFIX);
}

const SENT_RE = /^\[Sub-agent\]\s+Message sent to\s+([\w.-]+):\s*([\s\S]*)$/;
const TRANSCRIPT_TAIL_RE = /\.\s*Transcript:[\s\S]*$/;

/** The handle and the brief's preview from a dispatch marker's text. */
export function parseDispatchMarker(content: string): { handle: string; preview: string } | null {
  const m = content.match(SENT_RE);
  if (!m) return null;
  let preview = m[2].replace(TRANSCRIPT_TAIL_RE, '').trim();
  if (preview.length >= 2 && preview.startsWith('"') && preview.endsWith('"')) {
    preview = preview.slice(1, -1);
  }
  return { handle: m[1], preview };
}

/** The card's title: the first non-empty line of the brief. */
export function cardLabel(brief: string): string {
  const shown = skillAwarePreview(brief);
  for (const line of shown.split('\n')) {
    const t = line.trim();
    if (t) return t;
  }
  return '';
}

/** The brief Orbital last sent `handle`, from its agent_message tool row
 *  (the live card's title before the marker's preview is known). */
export function latestBriefFor(items: DisplayItem[], handle: string): string | undefined {
  for (let i = items.length - 1; i >= 0; i--) {
    const it = items[i];
    if (it.type === 'user_message') return undefined;
    if (it.type !== 'agent_run') continue;
    for (let k = it.items.length - 1; k >= 0; k--) {
      const c = it.items[k];
      if (c.type === 'tool_call_row' && c.dispatch && c.dispatch.agent === handle) {
        return c.dispatch.message;
      }
    }
  }
  return undefined;
}

/** Index of the card a live event belongs to: by dispatch id when the event
 *  carries one, else the handle's latest running card (a handle has at most
 *  one live dispatch per session, so this is exact). -1 when none. */
export function findCardIndex(
  items: DisplayItem[],
  key: { handle: string; dispatchId?: string; runningOnly?: boolean },
): number {
  for (let i = items.length - 1; i >= 0; i--) {
    const it = items[i];
    if (it.type !== 'dispatch_card' || it.handle !== key.handle) continue;
    if (key.dispatchId && it.dispatch_id && it.dispatch_id !== key.dispatchId) continue;
    if (key.runningOnly !== false && it.status !== 'running') {
      // An exact id match is the card even after it ended (a late event).
      if (key.dispatchId && it.dispatch_id === key.dispatchId) return i;
      continue;
    }
    return i;
  }
  return -1;
}

export interface LiveCardSpec {
  handle: string;
  dispatchId?: string;
  initiator?: string;
  /** The brief (or its preview); the title is its first line. */
  brief?: string;
  timestamp: string;
}

function newCard(spec: LiveCardSpec, items: DisplayItem[]): DispatchCardItem {
  const ms = Date.parse(spec.timestamp);
  const startedAt = Number.isFinite(ms) ? ms : Date.now();
  const brief = spec.brief ?? latestBriefFor(items, spec.handle) ?? '';
  return {
    type: 'dispatch_card',
    card_id: spec.dispatchId ?? `live:${spec.handle}:${startedAt}`,
    ...(spec.dispatchId ? { dispatch_id: spec.dispatchId } : {}),
    handle: spec.handle,
    label: cardLabel(brief),
    ...(spec.initiator ? { initiator: spec.initiator } : {}),
    ...(isFoldedInitiator(spec.initiator) ? { folded: true } : {}),
    status: 'running',
    started_at: startedAt,
    ended_at: null,
    steps: [],
    timestamp: spec.timestamp,
  };
}

/** The card for this dispatch, created at the end of the list when none
 *  exists yet. A running card of the same handle without an id adopts the
 *  id (it was opened by an event that arrived first); one with ANOTHER id
 *  missed its end and is closed. */
export function ensureLiveCard(
  items: DisplayItem[],
  spec: LiveCardSpec,
): { items: DisplayItem[]; idx: number } {
  const idx = findCardIndex(items, { handle: spec.handle, dispatchId: spec.dispatchId });
  if (idx >= 0) {
    const card = items[idx] as DispatchCardItem;
    const needsId = !!spec.dispatchId && !card.dispatch_id;
    const betterLabel = !!spec.brief && !card.label;
    const needsInitiator = !!spec.initiator && !card.initiator;
    if (!needsId && !betterLabel && !needsInitiator) return { items, idx };
    const next = [...items];
    next[idx] = {
      ...card,
      ...(needsId ? { dispatch_id: spec.dispatchId } : {}),
      ...(betterLabel ? { label: cardLabel(spec.brief!) } : {}),
      ...(needsInitiator
        ? { initiator: spec.initiator, ...(isFoldedInitiator(spec.initiator) ? { folded: true } : {}) }
        : {}),
    };
    return { items: next, idx };
  }
  let base = items;
  const stale = findCardIndex(items, { handle: spec.handle });
  if (stale >= 0) {
    const now = Date.parse(spec.timestamp);
    base = [...items];
    const old = base[stale] as DispatchCardItem;
    base[stale] = { ...old, status: 'completed', ended_at: Number.isFinite(now) ? now : Date.now() };
  }
  return absorbDispatchRow([...base, newCard(spec, base)], base.length);
}

/**
 * Orbital's `agent_message` send that a card stands for leaves Orbital's
 * capsule: the card already says it was sent, and to whom. A capsule left
 * with nothing else goes too. A worker's own dispatch (folded) has no such
 * row in Orbital's capsules. Returns the list and the card's new index.
 */
export function absorbDispatchRow(items: DisplayItem[], cardIdx: number): { items: DisplayItem[]; idx: number } {
  const card = items[cardIdx] as DispatchCardItem;
  if (card.folded) return { items, idx: cardIdx };
  for (let k = cardIdx - 1; k >= 0; k--) {
    const it = items[k];
    if (it.type === 'user_message') break;
    if (it.type !== 'agent_run' || it.source || it.memory_record || it.memory_tidy) continue;
    const r = it.items.findIndex((c) => c.type === 'tool_call_row' && c.dispatch?.agent === card.handle);
    if (r < 0) continue;
    const row = it.items[r] as Extract<DisplayItem, { type: 'tool_call_row' }>;
    const rest = it.items.filter((_, j) => j !== r);
    const next = [...items];
    if (rest.some((c) => c.type !== 'agent_message')) {
      const counts = { ...it.tool_call_count_by_name };
      counts[row.tool_name] = (counts[row.tool_name] ?? 1) - 1;
      if (counts[row.tool_name] <= 0) delete counts[row.tool_name];
      next[k] = { ...it, items: rest, tool_call_count_by_name: counts };
      return { items: next, idx: cardIdx };
    }
    next.splice(k, 1);
    return { items: next, idx: cardIdx - 1 };
  }
  return { items, idx: cardIdx };
}

/** Apply `fn` to the steps of the card at `idx`. */
export function updateCardSteps(
  items: DisplayItem[],
  idx: number,
  fn: (steps: DisplayItem[]) => DisplayItem[],
): DisplayItem[] {
  const card = items[idx] as DispatchCardItem;
  const steps = fn(card.steps);
  if (steps === card.steps) return items;
  const next = [...items];
  next[idx] = { ...card, steps };
  return next;
}

/** The worker's last visible message in a card's steps: its reply. */
export function lastReplyOf(card: DispatchCardItem): string | undefined {
  for (let k = card.steps.length - 1; k >= 0; k--) {
    const s = card.steps[k];
    if ((s.type === 'agent_message' || s.type === 'sub_agent_message') && s.source === card.handle) {
      if ('isHeaderOnly' in s && s.isHeaderOnly) continue;
      const text = s.content.trim();
      if (text) return s.content;
    }
  }
  return undefined;
}

function hasReplyAfter(items: DisplayItem[], idx: number, handle: string): boolean {
  const next = items[idx + 1];
  return !!next && next.type === 'agent_message' && next.source === handle && !next.isHeaderOnly;
}

/** The reply bubble right under the card (an Orbital dispatch only). */
export function placeReplyUnderCard(
  items: DisplayItem[],
  idx: number,
  reply: string,
  timestamp: string,
): DisplayItem[] {
  const card = items[idx] as DispatchCardItem;
  if (card.folded || !reply.trim() || hasReplyAfter(items, idx, card.handle)) return items;
  const next = [...items];
  next.splice(idx + 1, 0, { type: 'agent_message', content: reply, source: card.handle, timestamp });
  return next;
}

/** Close the card: its status, its end time, its open step capsule, and (an
 *  Orbital dispatch) its reply under it. `closeSteps` finalizes the running
 *  capsule inside the steps (ChatView's finalizeLiveCapsule). */
export function endCard(
  items: DisplayItem[],
  idx: number,
  status: Exclude<CardStatus, 'running'>,
  timestamp: string,
  closeSteps: (steps: DisplayItem[], status: Exclude<CardStatus, 'running'>) => DisplayItem[],
  replyFallback?: string,
): DisplayItem[] {
  const card = items[idx] as DispatchCardItem;
  const ms = Date.parse(timestamp);
  const ended: DispatchCardItem = {
    ...card,
    status,
    ended_at: card.ended_at ?? (Number.isFinite(ms) ? ms : Date.now()),
    steps: closeSteps(card.steps, status),
  };
  let next = [...items];
  next[idx] = ended;
  if (status === 'completed') {
    const reply = lastReplyOf(ended) ?? replyFallback;
    if (reply) next = placeReplyUnderCard(next, idx, reply, timestamp);
  }
  return next;
}

/** A card opened for a dispatch that turned out to be the user's own pinned
 *  turn: its steps go back inline where the card stood. */
export function unwrapCard(items: DisplayItem[], idx: number): DisplayItem[] {
  const card = items[idx] as DispatchCardItem;
  return [...items.slice(0, idx), ...card.steps, ...items.slice(idx + 1)];
}

/** Split off the run of still-RUNNING cards at the end of the list that a
 *  worker itself dispatched (initiator `worker:<owner>`). The owner's events
 *  of the turn that dispatched them go BEFORE them, which is where a reload
 *  places them (that turn hangs off its own marker, which precedes its
 *  child's). Once a child ended, its result reaches the owner as a new turn
 *  (spec 161), whose marker follows the child's: those events go after. */
export function splitTrailingChildCards(
  items: DisplayItem[],
  owner: string,
): { head: DisplayItem[]; tail: DisplayItem[] } {
  let k = items.length;
  while (k > 0) {
    const it = items[k - 1];
    if (it.type !== 'dispatch_card' || it.initiator !== `${WORKER_PREFIX}${owner}` || it.status !== 'running') break;
    k--;
  }
  if (k === items.length) return { head: items, tail: [] };
  return { head: items.slice(0, k), tail: items.slice(k) };
}

/** True when any card is in the conversation (spec 195 item 7). */
export function hasDispatchCards(items: DisplayItem[]): boolean {
  return items.some((it) => it.type === 'dispatch_card');
}

/** How a live worker event says which dispatch it belongs to (spec 195). */
export interface WorkerEventRoute {
  /** On the event (contract §2): absent on a worker's own turn (spec 129). */
  initiator?: string;
  dispatchId?: string;
  /** The handle's last `sub_agent.dispatched` initiator, for a daemon whose
   *  events carry none; cleared when the handle's turn ends. */
  fallbackInitiator?: string;
}

/**
 * Spec 195: apply one live worker event (`fn` over a list of items) where it
 * belongs. A dispatch Orbital or a worker started goes into its card's steps
 * (opened here when its first event beats the dispatched event), so two
 * agents streaming at once never interleave in the conversation; the user's
 * own pinned chat and a worker's own turn stay inline, as before. A worker's
 * inline events go before the cards it dispatched itself, where a reload puts
 * them.
 */
export function applyWorkerEvent(
  prev: DisplayItem[],
  owner: string,
  route: WorkerEventRoute,
  fn: (list: DisplayItem[]) => DisplayItem[],
): DisplayItem[] {
  let carded: boolean;
  if (route.initiator) carded = isCardedInitiator(route.initiator);
  else if (findCardIndex(prev, { handle: owner, dispatchId: route.dispatchId }) >= 0) carded = true;
  else if (route.fallbackInitiator) carded = isCardedInitiator(route.fallbackInitiator);
  else carded = false;
  if (carded) {
    const { items, idx } = ensureLiveCard(prev, {
      handle: owner,
      dispatchId: route.dispatchId,
      initiator: route.initiator ?? route.fallbackInitiator,
      timestamp: new Date().toISOString(),
    });
    let next = updateCardSteps(items, idx, fn);
    const card = next[idx] as DispatchCardItem;
    // A reply that lands after the card already ended still goes under it.
    if (card.status === 'completed') {
      const reply = lastReplyOf(card);
      if (reply) next = placeReplyUnderCard(next, idx, reply, new Date().toISOString());
    }
    return next;
  }
  const { head, tail } = splitTrailingChildCards(prev, owner);
  const out = fn(head);
  return tail.length > 0 ? [...out, ...tail] : out;
}
