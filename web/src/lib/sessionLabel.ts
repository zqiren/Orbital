// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * sessionLabel.ts — classify a session's auto-derived name into a display
 * kind + cleaned label, purely from frontend pattern matching.
 *
 * Backends stamp machine prefixes into the first user message, and the
 * session name is auto-derived from that message's first ~50 chars
 * (word-boundary truncated, `…`-suffixed). So machine-originated sessions
 * carry recognizable name prefixes:
 *
 *   queue     "[QUEUE ITEM | id=… | attempt=N]" + header contract
 *             (agent_os/queue/dispatcher.py — the item's own text never
 *             survives the 50-char cap, so nothing is recoverable)
 *   schedule  "[Triggered by schedule 'NAME' (…)]…"
 *   file_watch"[Triggered by file_watch 'NAME']…"
 *             (agent_os/daemon_v2/trigger_manager.py — NAME is recoverable)
 *   attachment"<attached_files>\n- path (mime, size)…"
 *             (agent_os/api/routes/_attachment_formatter.py — the file
 *             basename is recoverable)
 *
 * All patterns must tolerate mid-token truncation: the stored name is capped
 * at 50 chars, so closing quotes/brackets may be missing and a `…` may sit
 * anywhere. Queue detection prefers the API's `origin === 'queue'` field
 * (persisted in session meta) and uses the name prefix only as fallback;
 * trigger/attachment kinds exist only as name patterns. A user rename
 * replaces the machine name, so pattern-derived kinds (and their chips)
 * intentionally disappear on rename — the queue chip survives because its
 * signal is the origin field, and the renamed text then wins as the label.
 */

export type SessionKind = 'plain' | 'queue' | 'schedule' | 'file_watch' | 'attachment';

export interface SessionLabelInfo {
  kind: SessionKind;
  /**
   * Human-readable label, or null when nothing human-authored survives the
   * machine prefix (the row then renders the kind chip alone, or a generic
   * fallback for attachment/plain).
   */
  displayName: string | null;
  /** Tooltip detail, e.g. "schedule 'Daily check'". */
  detail?: string;
}

const TRIGGER_RE = /^\[Triggered by (schedule|file_watch) '([^'\n]*)/;
const QUEUE_RE = /^\[QUEUE ITEM/;
const ATTACHED_RE = /^<attached_files>/;
/** Upload names are timestamp-prefixed by the composer (attachment-upload.ts). */
const UPLOAD_TS_RE = /^\d{4}-\d{2}-\d{2}T\d{6}-/;

/** Strip the derive-time ellipsis and surrounding whitespace. */
function tidy(fragment: string): string {
  return fragment.replace(/…+$/, '').trim();
}

/**
 * Pull the file basename out of a (possibly truncated) <attached_files>
 * block head: "<attached_files>\n- some/path.pdf (application/pdf, 1.2 MB)…".
 * Returns null when truncation ate the path entirely.
 */
function attachmentBasename(name: string): string | null {
  const dash = name.indexOf('- ');
  if (dash < 0) return null;
  let path = name.slice(dash + 2);
  // The path ends at " (" (mime paren) or the line end, whichever comes first.
  for (const stop of [' (', '\n']) {
    const idx = path.indexOf(stop);
    if (idx >= 0) path = path.slice(0, idx);
  }
  const base = tidy(path.slice(path.lastIndexOf('/') + 1)).replace(UPLOAD_TS_RE, '');
  return base || null;
}

export function classifySessionName(
  name: string | null | undefined,
  origin?: string,
): SessionLabelInfo {
  const n = (name ?? '').trim();

  const isQueue = origin === 'queue' || QUEUE_RE.test(n);

  if (ATTACHED_RE.test(n)) {
    return {
      kind: isQueue ? 'queue' : 'attachment',
      displayName: attachmentBasename(n),
    };
  }

  const trigger = TRIGGER_RE.exec(n);
  if (!isQueue && trigger) {
    const kind = trigger[1] as 'schedule' | 'file_watch';
    const triggerName = tidy(trigger[2]);
    return {
      kind,
      displayName: triggerName || null,
      detail: triggerName ? `${kind} '${triggerName}'` : undefined,
    };
  }

  if (isQueue) {
    // The queue header + contract consume the whole derived name; only a
    // user rename (which drops the "[QUEUE ITEM" prefix) is worth showing.
    return { kind: 'queue', displayName: QUEUE_RE.test(n) ? null : n || null };
  }

  return { kind: 'plain', displayName: n || null };
}
