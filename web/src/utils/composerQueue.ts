// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The composer's queue block (2026-11-24): messages that wait for their turn
 * live as rows inside the composer, not as bubbles in the chat, until they are
 * really sent. Two kinds share one row:
 *
 *   own     waits behind this chat's own agent (the daemon's session queue,
 *           keyed by nonce; drained when the agent takes its next step)
 *   worker  waits behind an outside agent's current task (its prompt queue,
 *           keyed by dispatch id); its chat row is written at dispatch
 *
 * Pure helpers: build the rows the block shows, rebuild an entry from GET
 * /pending after a reload, and merge an edited message back into a draft.
 */

import { parseAttachmentsBlock } from '../lib/attachment-parsing';
import { splitUserWire } from './editMessage';

/** One attachment of a queued message, enough to restore it as a chip. */
export interface QueuedChip {
  filename: string;
  mime: string;
  size: number;
  /** An uploaded file: its workspace path. */
  uploadedPath?: string;
  /** Desktop app: a file attached by its absolute local path. */
  localPath?: string;
  /** The picked File, when this client still has it (image thumbnail). */
  file?: File;
}

/** One attachment on the wire / in GET /pending (exact size). */
export interface WireAttachment {
  path: string;
  mime: string;
  size: number;
  local?: boolean;
}

export interface QueueRow {
  /** `n:<nonce>` for an own row, `d:<dispatch id>` for a worker row. */
  key: string;
  kind: 'own' | 'worker';
  /** The outside agent a worker row waits for. */
  handle?: string;
  /** One line of text (the typed words). */
  text: string;
  chips: QueuedChip[];
  /** False only for an old entry whose attachments cannot be restored. */
  editable: boolean;
  queuedAt: string;
}

export interface OwnQueued {
  sessionId: string;
  text: string;
  chips: QueuedChip[];
  editable: boolean;
  queuedAt: string;
}

export interface WorkerQueued {
  dispatchId: string;
  handle: string;
  sessionId: string;
  text: string;
  chips: QueuedChip[];
  editable: boolean;
  queuedAt: string;
}

/** The rows for the viewed session, oldest first, both kinds together. */
export function queueRows(
  own: ReadonlyMap<string, OwnQueued>,
  worker: readonly WorkerQueued[],
  sessionId: string | undefined,
): QueueRow[] {
  if (sessionId === undefined) return [];
  const rows: QueueRow[] = [];
  for (const [nonce, e] of own) {
    if (e.sessionId !== sessionId) continue;
    rows.push({ key: `n:${nonce}`, kind: 'own', text: e.text, chips: e.chips, editable: e.editable, queuedAt: e.queuedAt });
  }
  for (const w of worker) {
    if (w.sessionId !== sessionId) continue;
    rows.push({
      key: `d:${w.dispatchId}`, kind: 'worker', handle: w.handle,
      text: w.text, chips: w.chips, editable: w.editable, queuedAt: w.queuedAt,
    });
  }
  // Stable: equal times keep insertion order (own first, as they were added).
  return rows
    .map((r, i) => ({ r, i }))
    .sort((a, b) => (a.r.queuedAt < b.r.queuedAt ? -1 : a.r.queuedAt > b.r.queuedAt ? 1 : a.i - b.i))
    .map(({ r }) => r);
}

/** Composer uploads are stored timestamp-prefixed (attachment-upload.ts);
 *  the chip shows the name the user picked. */
const UPLOAD_TS_RE = /^\d{4}-\d{2}-\d{2}T\d{6}-/;

function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

/** Chips from the exact attachment list the daemon kept. */
export function chipsFromWire(list: readonly WireAttachment[] | undefined | null): QueuedChip[] {
  if (!list) return [];
  return list.map((a) => (a.local
    ? { filename: baseName(a.path), mime: a.mime, size: a.size, localPath: a.path }
    : { filename: baseName(a.path).replace(UPLOAD_TS_RE, ''), mime: a.mime, size: a.size, uploadedPath: a.path }));
}

/** What GET /pending gives back for one queued message. */
export interface PendingWire {
  content: string;
  text?: string;
  attachments?: WireAttachment[] | null;
}

/**
 * Rebuild a row's text and chips from GET /pending (after a reload). The
 * daemon keeps the typed words and the exact attachments; an entry without
 * them (sent by an older client) shows the words from the wire content, and
 * is editable only if it carries no attachments (a chip without its exact
 * size could not be sent again).
 */
export function fromPendingWire(p: PendingWire): { text: string; chips: QueuedChip[]; editable: boolean } {
  const chips = chipsFromWire(p.attachments ?? undefined);
  const text = typeof p.text === 'string' ? p.text : splitUserWire(p.content).body;
  const unrecoverable = chips.length === 0 && parseAttachmentsBlock(p.content).attachments.length > 0;
  return { text: splitUserWire(text).body, chips, editable: !unrecoverable };
}

/** An edited message goes back into the composer; a draft already there
 *  keeps its place and the message follows on a new line. */
export function mergeIntoDraft(draft: string, text: string): string {
  if (!draft.trim()) return text;
  if (!text) return draft;
  return `${draft.replace(/\s+$/, '')}\n${text}`;
}
