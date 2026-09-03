// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §6 — pure selectors over the session's ChatMessage[] the chat
 * already holds. CONTRACT FILE: signatures are final.
 *
 * Tool identity note (verified against a real session JSONL,
 * `orbital-data/p5-cred/orbital/sessions/p5-cred_7c20adc6.jsonl`, and against
 * `chatTransform.ts`'s TOOL_NAME_TO_CATEGORY at :262): the persisted tool
 * names are the bare `read` / `write` / `edit` / `browser`. The file tools put
 * the target under `path` (older rows may use `file_path`).
 *
 * The browser tool's screenshot lands on the **tool RESULT** row's `_meta`
 * (`browser.py` returns `meta={"url", "title", "screenshot_path", …}`;
 * `session.py` writes it verbatim as `_meta`; the `/chat` reader returns the
 * parsed JSONL dict untouched, so it survives to the frontend).
 * `_meta.screenshot_path` is an ABSOLUTE path — the files content route joins
 * it onto the workspace, and `os.path.join` keeps an absolute second argument,
 * so it still resolves as long as the file is inside the workspace.
 */
import type { ActivityEvent, ChatMessage } from '../types';

export type TouchedOp = 'read' | 'edited' | 'written';
export interface TouchedFile { path: string; op: TouchedOp; lastAt?: string }
export type PanelView = 'files' | 'browser';

/** Persisted tool name → the touched-file operation it represents. */
const FILE_TOOL_OPS: Record<string, TouchedOp> = {
  read: 'read',
  write: 'written',
  edit: 'edited',
};

/** The persisted name of the browser tool (chatTransform.ts:273). */
const BROWSER_TOOL = 'browser';

/** written > edited > read — the strongest op a path saw wins its badge. */
const OP_RANK: Record<TouchedOp, number> = { read: 0, edited: 1, written: 2 };

function parseArgs(raw: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(raw);
    return parsed !== null && typeof parsed === 'object'
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

function pathFromArgs(args: Record<string, unknown>): string | null {
  const raw = args.path ?? args.file_path;
  return typeof raw === 'string' && raw.length > 0 ? raw : null;
}

/** Files this session touched, newest first, one row per path (strongest op wins: written > edited > read). */
export function touchedFiles(messages: ChatMessage[]): TouchedFile[] {
  // seq = position of the LAST touch, so "newest first" is a descending sort
  // on it. One entry per path; op only ever escalates.
  const byPath = new Map<string, { file: TouchedFile; seq: number }>();
  let seq = 0;

  for (const message of messages) {
    if (!message.tool_calls) continue;
    for (const call of message.tool_calls) {
      const op = FILE_TOOL_OPS[call.function?.name ?? ''];
      if (!op) continue;
      const path = pathFromArgs(parseArgs(call.function.arguments ?? '{}'));
      if (!path) continue;

      seq += 1;
      const existing = byPath.get(path);
      if (existing === undefined) {
        byPath.set(path, { file: { path, op, lastAt: message.timestamp }, seq });
        continue;
      }
      existing.seq = seq;
      existing.file.lastAt = message.timestamp;
      if (OP_RANK[op] > OP_RANK[existing.file.op]) existing.file.op = op;
    }
  }

  return [...byPath.values()]
    .sort((a, b) => b.seq - a.seq)
    .map((entry) => entry.file);
}

/** Latest browser screenshot for the session (from tool-result `_meta`), or null. */
export function latestScreenshot(
  messages: ChatMessage[],
): { path: string; url?: string; title?: string } | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const meta = messages[i]._meta;
    if (!meta) continue;
    const path = meta.screenshot_path;
    if (typeof path !== 'string' || path.length === 0) continue;
    const url = typeof meta.url === 'string' ? meta.url : undefined;
    const title = typeof meta.title === 'string' ? meta.title : undefined;
    return { path, url, title };
  }
  return null;
}

/** Which panel view a live activity event belongs to; null when it should not move the panel. */
export function viewForEvent(event: ActivityEvent): PanelView | null {
  switch (event.category) {
    case 'file_read':
    case 'file_write':
    case 'file_edit':
      return 'files';
    case 'browser_automation':
      return 'browser';
    default:
      break;
  }
  // Older daemons categorize unknown tools as `tool_use`; fall back to the
  // tool name so a browser/file call still moves the panel.
  if (event.tool_name === BROWSER_TOOL) return 'browser';
  if (event.tool_name in FILE_TOOL_OPS) return 'files';
  return null;
}

/**
 * The workspace path a file activity event refers to, when the daemon shipped
 * parsed `arguments` alongside the event. Null for browser/command events and
 * for older daemons that only send the English `description`.
 */
export function pathForEvent(event: ActivityEvent): string | null {
  if (viewForEvent(event) !== 'files' || !event.arguments) return null;
  return pathFromArgs(event.arguments);
}

/**
 * A live activity event as the synthetic assistant row `touchedFiles` reads,
 * so the panel's touched list stays current mid-run without refetching the
 * session. Returns null when the event carries no usable tool call.
 */
export function eventAsMessage(event: ActivityEvent): ChatMessage | null {
  if (!event.arguments || !(event.tool_name in FILE_TOOL_OPS)) return null;
  return {
    role: 'assistant',
    content: null,
    source: event.source,
    timestamp: event.timestamp,
    tool_calls: [
      {
        id: event.id,
        type: 'function',
        function: { name: event.tool_name, arguments: JSON.stringify(event.arguments) },
      },
    ],
  };
}
