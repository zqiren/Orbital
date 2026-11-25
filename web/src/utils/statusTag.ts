// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The manager's `[STATUS: short text]` tag (prompt_builder.py asks for it; the
 * daemon's activity_translator.py extracts it for the one-line activity
 * summary). The chat never shows the raw tag: these helpers strip it for
 * display only. Stored message content is never rewritten, so every content
 * comparison in the app (dedup, truncation repair) keeps seeing the raw text.
 *
 * Code is left alone: a tag inside a fenced block or an inline code span is
 * content, not a status.
 */
import type { DisplayItem } from './chatTransform';

// Same shape the daemon extracts (`\[STATUS:\s*(.+?)\]`, one line), case
// insensitive like the left-off reader's strip.
const TAG_RE = /\[STATUS:[ \t]*([^\]\n]*)\]/gi;
// Fenced blocks (an unclosed fence runs to the end, as while streaming) and
// inline code spans. Captured so `split` keeps them at odd indexes.
const CODE_RE = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`)/g;
const OPENER = '[status:';

function proseParts(text: string): string[] {
  return text.split(CODE_RE);
}

/** A trailing fragment that may still become a tag while it streams in:
 *  `[`, `[STA`, `[STATUS: half the text`. */
function trailingPartialTag(prose: string): number {
  const at = prose.lastIndexOf('[');
  if (at < 0) return -1;
  const tail = prose.slice(at);
  if (tail.includes(']') || tail.includes('\n')) return -1;
  const low = tail.toLowerCase();
  if (low.length <= OPENER.length ? OPENER.startsWith(low) : low.startsWith(OPENER)) return at;
  return -1;
}

/**
 * `text` without its status tags. Returns `text` unchanged (same string) when
 * it holds none. With `streaming`, a tag still arriving at the end is hidden
 * too, so no `[STATUS:` flashes before its `]` lands.
 */
export function stripStatusTags(text: string, opts: { streaming?: boolean } = {}): string {
  if (!text || !/\[/.test(text)) return text;
  const parts = proseParts(text);
  let changed = false;
  for (let k = 0; k < parts.length; k += 2) {
    let prose = parts[k];
    if (!prose) continue;
    const next = prose.replace(TAG_RE, () => {
      changed = true;
      return '\u0000';
    });
    prose = next;
    if (opts.streaming && k === parts.length - 1) {
      const cut = trailingPartialTag(prose);
      if (cut >= 0) {
        prose = prose.slice(0, cut) + '\u0000';
        changed = true;
      }
    }
    parts[k] = prose;
  }
  if (!changed) return text;
  return parts
    .join('')
    // A tag alone on its line takes the line with it; one inside a line
    // takes the spaces before it.
    .replace(/^[ \t]*\u0000[ \t]*(?:\r?\n|$)/gm, '')
    .replace(/[ \t]*\u0000/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/** The reply's status text (the first tag outside code, as the daemon reads
 *  it), or null. */
export function statusOf(text: string | null | undefined): string | null {
  if (!text || !/\[/.test(text)) return null;
  const parts = proseParts(text);
  for (let k = 0; k < parts.length; k += 2) {
    TAG_RE.lastIndex = 0;
    const m = TAG_RE.exec(parts[k]);
    TAG_RE.lastIndex = 0;
    if (m && m[1].trim()) return m[1].trim();
  }
  return null;
}

/** What an agent bubble shows and copies: the text without its tags. A reply
 *  that is nothing but a tag (and has nowhere else to put it) shows the
 *  status as plain text, never the brackets. */
export function displayAgentText(content: string): string {
  const stripped = stripStatusTags(content);
  if (stripped.trim() || stripped === content) return stripped;
  return statusOf(content) ?? '';
}

/** True when the reply's only text is its status tag(s). */
export function isStatusOnly(content: string | null | undefined): boolean {
  if (!content || !statusOf(content)) return false;
  return !stripStatusTags(content).trim();
}

// ---------------------------------------------------------------- the fold

type AgentRun = Extract<DisplayItem, { type: 'agent_run' }>;
type CapsuleChild = AgentRun['items'][number];

export interface StatusStep {
  /** The reply's status; undefined for a step whose reply carried none. */
  status?: string;
  children: CapsuleChild[];
  tool_call_count_by_name: Record<string, number>;
  started_at: number;
  ended_at: number | null;
  /** Internal: the marker came after a text bubble (see statusSteps). */
  afterText?: boolean;
}

function ms(ts: string): number {
  const n = Date.parse(ts);
  return Number.isFinite(n) ? n : 0;
}

/**
 * A capsule's steps, one per assistant reply, when at least one reply in it
 * carried a status (the transform and the live path put the status on the
 * empty `agent_message` marker that opens the reply's batch). Null for a
 * capsule with no status at all: it keeps today's capsule.
 *
 * A step runs from its marker to the next one; the last runs to the
 * capsule's end (null while it is still running).
 */
export function statusSteps(capsule: AgentRun, opts: { running?: boolean } = {}): StatusStep[] | null {
  if (capsule.memory_record || capsule.memory_tidy) return null;
  if (!capsule.items.some((c) => c.type === 'agent_message' && !!c.status)) return null;
  const steps: StatusStep[] = [];
  let cur: StatusStep | null = null;
  for (const child of capsule.items) {
    if (child.type === 'agent_message') {
      cur = {
        ...(child.status ? { status: child.status } : {}),
        ...(child.statusAfterText ? { afterText: true } : {}),
        children: [],
        tool_call_count_by_name: {},
        started_at: ms(child.timestamp) || capsule.started_at,
        ended_at: null,
      };
      steps.push(cur);
      continue;
    }
    if (!cur) {
      cur = { children: [], tool_call_count_by_name: {}, started_at: capsule.started_at, ended_at: null };
      steps.push(cur);
    }
    cur.children.push(child);
    if (child.type === 'tool_call_row') {
      cur.tool_call_count_by_name[child.tool_name] = (cur.tool_call_count_by_name[child.tool_name] ?? 0) + 1;
    }
  }
  // A finished capsule's last marker that no tool or thought followed (live:
  // a reply with text and a status but no tools) is not a step. While the
  // run goes on it is the step in progress, and its status is the live one.
  // A reply with text keeps its bubble, so its status waits for a tool
  // before it shows as a step.
  const real = steps.filter((s, k) =>
    s.children.length > 0 || k < steps.length - 1 || (opts.running && !s.afterText));
  for (const s of real) delete s.afterText;
  if (real.length === 0) return null;
  for (let k = 0; k < real.length; k++) {
    real[k].ended_at = k < real.length - 1
      ? real[k + 1].started_at
      : opts.running ? null : capsule.ended_at;
  }
  return real;
}

/** The capsule renders as the status fold (cheap check, no step split). */
export function isStatusFold(capsule: AgentRun): boolean {
  if (capsule.memory_record || capsule.memory_tidy) return false;
  return capsule.items.some((c) => c.type === 'agent_message' && !!c.status);
}

/** The fold's line: the latest status any step carried. */
export function latestStatus(steps: StatusStep[]): string | undefined {
  for (let k = steps.length - 1; k >= 0; k--) {
    if (steps[k].status) return steps[k].status;
  }
  return undefined;
}
