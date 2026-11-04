// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 065 slice two — pure helpers behind the history column, the tinted
 * version view, the changed-files line and the undo sheet. No React here, so
 * every rule the design file states is unit-tested on its own.
 */
import type {
  HistoryEditor,
  HistoryVersion,
  TurnFile,
} from '../../types';
import type { StringKey } from '../../i18n/strings';
import { workerToolCategory } from '../../utils/chatTransform';

export type Tr = (key: StringKey, vars?: Record<string, string | number>) => string;

// ── timeline rows ─────────────────────────────────────────────────────────

/**
 * One row of the timeline. Writes of one agent turn fold into one row (the
 * design's "3 次修改"); storage keeps one version per write, so `members`
 * holds them newest first and the chevron expands one sub-row per member.
 */
export interface TimelineRow {
  key: string;
  /** The newest member: what the row shows and what selecting it restores to. */
  head: HistoryVersion;
  members: HistoryVersion[];
  /** The version just before the group (the view's `from`), if any. */
  before: HistoryVersion | null;
  current: boolean;
  dayKey: string;
}

function dayKeyOf(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return '';
  return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
}

/** Versions arrive newest first; consecutive members of a group fold. */
export function buildRows(versions: HistoryVersion[]): TimelineRow[] {
  const rows: TimelineRow[] = [];
  for (let i = 0; i < versions.length; i++) {
    const v = versions[i];
    const last = rows[rows.length - 1];
    if (last && last.head.group === v.group && v.kind !== 'baseline' && v.turn) {
      last.members.push(v);
      continue;
    }
    rows.push({
      key: `g:${v.v}`,
      head: v,
      members: [v],
      before: null,
      current: v.current,
      dayKey: dayKeyOf(v.ts),
    });
  }
  // `before` = the version right after the group's oldest member in the list.
  let idx = 0;
  for (const row of rows) {
    idx += row.members.length;
    row.before = versions[idx] ?? null;
  }
  return rows;
}

/** Selection key of a sub-row (one write inside a folded turn). */
export const memberKey = (v: HistoryVersion) => `v:${v.v}`;

export interface Selection {
  /** The version the view shows and a restore writes back. */
  target: HistoryVersion;
  /** `from_version` for the view (the version before a folded group). */
  fromV: number | null;
  isCurrent: boolean;
}

export function selectionFor(rows: TimelineRow[], key: string | null): Selection | null {
  if (rows.length === 0) return null;
  for (const row of rows) {
    if (key === null ? row.current : row.key === key) {
      return {
        target: row.head,
        fromV: row.members.length > 1 ? row.before?.v ?? null : null,
        isCurrent: row.current,
      };
    }
    for (const m of row.members) {
      if (memberKey(m) === key) return { target: m, fromV: null, isCurrent: m.current };
    }
  }
  return null;
}

/** The row holding version `v` (a turn's last write, from the undo sheet). */
export function rowKeyForVersion(rows: TimelineRow[], v: number): string | null {
  for (const row of rows) if (row.members.some((m) => m.v === v)) return row.key;
  return null;
}

// ── labels ────────────────────────────────────────────────────────────────

export function formatClock(ts: string | number): string {
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  if (Number.isNaN(d.getTime())) return '';
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/** 今天 / 昨天 / 11月2日 (Today / Yesterday / Nov 2). */
export function dayLabel(ts: string | number, locale: string, tr: Tr, now: Date = new Date()): string {
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  if (Number.isNaN(d.getTime())) return '';
  const startOf = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((startOf(now) - startOf(d)) / 86_400_000);
  if (diff === 0) return tr('history.today');
  if (diff === 1) return tr('history.yesterday');
  const opts: Intl.DateTimeFormatOptions =
    d.getFullYear() === now.getFullYear()
      ? { month: locale === 'zh' ? 'long' : 'short', day: 'numeric' }
      : { year: 'numeric', month: locale === 'zh' ? 'long' : 'short', day: 'numeric' };
  return d.toLocaleDateString(locale === 'zh' ? 'zh-CN' : 'en-US', opts);
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function linesLabel(n: number, tr: Tr): string {
  return n === 1 ? tr('history.lines.one', { n }) : tr('history.lines.other', { n });
}

/** The third line of a version row: what changed, in plain words. */
export function changeLabel(row: TimelineRow, tr: Tr, older: HistoryVersion | null): string {
  const head = row.head;
  if (head.kind === 'baseline') return tr('history.baseline');
  if (head.kind === 'restore' && head.attrib.from_ts) {
    return tr('history.restoredFrom', { t: formatClock(head.attrib.from_ts) });
  }
  if (head.kind === 'delete') return tr('history.deleted');
  if (row.members[row.members.length - 1].kind === 'create' && row.members.length === 1) {
    return tr('history.created');
  }
  const counted = row.members.filter((m) => m.lines !== null);
  if (counted.length === 0) {
    // Office files: restore-only, so the row says how the size moved.
    const from = older?.size ?? null;
    return from !== null && head.size !== null
      ? tr('history.size', { from: formatBytes(from), to: formatBytes(head.size) })
      : formatBytes(head.size);
  }
  return linesLabel(
    counted.reduce((sum, m) => sum + (m.lines ?? 0), 0),
    tr,
  );
}

/** Line two's subject: the agents (logos) and the session name, or a phrase. */
export interface Author {
  agents: string[];
  you: boolean;
  label: string;
  /** Ambiguous: "{a} 或 {b}（同时在运行）" for the tooltip. */
  either: string | null;
}

export function agentName(slug: string): string {
  const known: Record<string, string> = {
    orbital: 'Orbital',
    'claude-code': 'Claude Code',
    claude: 'Claude Code',
    codex: 'Codex',
    pi: 'Pi',
    codebuddy: 'CodeBuddy',
    cursor: 'Cursor',
    dsh: 'DeepSeek',
    gemini: 'Gemini',
    grok: 'Grok',
    kimi: 'Kimi',
    minimax: 'MiniMax',
  };
  return known[slug] ?? slug;
}

export function authorOf(v: HistoryVersion, tr: Tr): Author {
  const a = v.attrib;
  const sessions = a.sessions ?? [];
  switch (a.kind) {
    case 'user':
    case 'restore':
      return { agents: [], you: true, label: tr('history.you'), either: null };
    case 'outside':
      return { agents: [], you: false, label: tr('history.outside'), either: null };
    case 'closed':
      return { agents: [], you: false, label: tr('history.closed'), either: null };
    case 'baseline':
      return { agents: [], you: false, label: '', either: null };
    case 'orbital':
      return { agents: ['orbital'], you: false, label: 'Orbital', either: null };
    default: {
      if (sessions.length === 0) {
        return { agents: [], you: false, label: tr('history.outside'), either: null };
      }
      const agents = [...new Set(sessions.flatMap((s) => (s.agents.length ? s.agents : ['orbital'])))];
      const names = sessions.map((s) => s.name || '').filter(Boolean);
      if (v.ambiguous || sessions.length > 1) {
        const [x, y] = sessions.map((s) => s.name || agentName(s.agents[0] ?? 'orbital'));
        return {
          agents,
          you: false,
          label: names.join(' / '),
          either: tr('history.either', { a: x ?? '', b: y ?? '' }),
        };
      }
      return { agents, you: false, label: names[0] ?? agentName(agents[0]), either: null };
    }
  }
}

// ── editor palette (design: one calm colour per agent) ────────────────────

/** CSS custom properties per slug; the view's container sets the dark values. */
const KNOWN_PALETTE: Record<string, string> = {
  orbital: 'var(--hist-c-orbital)',
  'claude-code': 'var(--hist-c-claude-code)',
  claude: 'var(--hist-c-claude-code)',
  codex: 'var(--hist-c-codex)',
  pi: 'var(--hist-c-pi)',
};
/** Other agents get a stable slot keyed by slug. */
const EXTRA_PALETTE = ['#D946A8', '#0EA5E9', '#CA8A04', '#4F46E5', '#DC2626', '#059669'];
export const NEUTRAL_TINT = 'var(--hist-c-you)';

export function agentColor(slug: string): string {
  const hit = KNOWN_PALETTE[slug];
  if (hit) return hit;
  let h = 0;
  for (let i = 0; i < slug.length; i++) h = (h * 31 + slug.charCodeAt(i)) >>> 0;
  return EXTRA_PALETTE[h % EXTRA_PALETTE.length];
}

/** The tint of a version: its single agent, else grey (§5.8). */
export function editorColor(editor: HistoryEditor | null | undefined): string {
  if (!editor || editor.neutral || editor.ambiguous || editor.agents.length === 0) return NEUTRAL_TINT;
  // A manager and its worker in one session: the worker did the writing.
  const writer = editor.agents.find((a) => a !== 'orbital') ?? editor.agents[0];
  return agentColor(writer);
}

/** Agents that tint something in this view: the legend's entries. */
export function legendAgents(editors: HistoryEditor[]): string[] {
  const out: string[] = [];
  for (const e of editors) {
    if (e.neutral || e.ambiguous || e.agents.length === 0) continue;
    const writer = e.agents.find((a) => a !== 'orbital') ?? e.agents[0];
    if (!out.includes(writer)) out.push(writer);
  }
  return out;
}

/**
 * Tailwind classes that define the palette and the tint strengths on a
 * container: 18 % / 15 % in light, 26 % / 22 % in dark (design file).
 */
export const PALETTE_VARS =
  '[--hist-c-orbital:#539AF8] [--hist-c-claude-code:#E07B39] [--hist-c-codex:#12A386] [--hist-c-pi:#8B5CF6] [--hist-c-you:#8A8A94] [--hist-ins:18%] [--hist-del:15%] ' +
  'dark:[--hist-c-claude-code:#F0915A] dark:[--hist-c-codex:#2CC4A4] dark:[--hist-c-pi:#A78BFA] dark:[--hist-ins:26%] dark:[--hist-del:22%]';

/** Inline style of a changed line: insertions tinted, deletions tinted + struck. */
export function tintStyle(op: 'add' | 'del', color: string): Record<string, string> {
  const base: Record<string, string> = {
    borderRadius: '3px',
    WebkitBoxDecorationBreak: 'clone',
    boxDecorationBreak: 'clone',
    padding: '1px 1px',
  };
  if (op === 'add') {
    return { ...base, backgroundColor: `color-mix(in srgb, ${color} var(--hist-ins), transparent)` };
  }
  return {
    ...base,
    backgroundColor: `color-mix(in srgb, ${color} var(--hist-del), transparent)`,
    textDecorationLine: 'line-through',
    textDecorationThickness: '1.5px',
    textDecorationColor: `color-mix(in srgb, ${color} 75%, currentColor)`,
    color: 'var(--color-secondary)',
  };
}

/** The same style as a CSS declaration string (for hast `style`). */
export function cssText(style: Record<string, string>): string {
  return Object.entries(style)
    .map(([k, v]) => `${k.replace(/^Webkit/, '-webkit-').replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}: ${v}`)
    .join('; ');
}

// ── undo sheet ────────────────────────────────────────────────────────────

/**
 * §5.9 / §7: checked only when nobody touched the file since the turn. A file
 * the turn created is never pre-checked (restoring it deletes it), nor is a
 * version written while two sessions ran (ambiguous).
 */
export function defaultChecked(file: TurnFile): boolean {
  if (isMemoryFile(file.path)) return false;
  if (file.kind === 'create') return false;
  if (file.ambiguous) return false;
  return file.clean;
}

/**
 * The four Layer-1 memory files (D11). The memory record writes them as
 * Orbital's own bookkeeping, so they never count as "what the agent did to my
 * files": not in the changed-files line, and in the undo sheet listed last,
 * unchecked, under 记忆, so undoing a turn never silently rewinds memory.
 */
const MEMORY_FILES = new Set([
  'orbital/PROJECT_STATE.md',
  'orbital/DECISIONS.md',
  'orbital/LESSONS.md',
  'orbital/INDEX.md',
]);

export function isMemoryFile(path: string): boolean {
  return MEMORY_FILES.has(path.replace(/^\.\//, ''));
}

export function splitTurnFiles(files: TurnFile[]): { work: TurnFile[]; memory: TurnFile[] } {
  return {
    work: files.filter((f) => !isMemoryFile(f.path)),
    memory: files.filter((f) => isMemoryFile(f.path)),
  };
}

/** Work files whose current content is what the turn started from: undone already. */
export function undoneCount(files: TurnFile[]): number {
  return files.filter((f) => !isMemoryFile(f.path) && !f.clean && f.current_sha === f.before_sha).length;
}

// ── chat capsule ──────────────────────────────────────────────────────────

/** The breakdown minus its file-writing counts (edit / write / file change). */
export function stripFileEditCounts(counts: Record<string, number>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [name, n] of Object.entries(counts)) {
    const cat = workerToolCategory(name);
    if (cat === 'file_write' || cat === 'file_edit') continue;
    if (/^(multiedit|multi_edit|apply_patch|file_change|notebookedit|str_replace_editor|create_file)$/i.test(name)) continue;
    out[name] = n;
  }
  return out;
}

/** A turn: from one user row to the next (§5.5). */
export interface TurnSpan {
  fromTs: string;
  toTs: string | null;
}

interface SpanItem {
  type: string;
  timestamp?: string;
  capsule_id?: string;
  memory_record?: boolean;
  memory_tidy?: unknown;
}

/**
 * For each turn in a transcript, its last capsule (the one the changed-files
 * line goes on) mapped to the turn's span. Memory record / tidy capsules are
 * the daemon's bookkeeping, not the agent's work, and never carry the line.
 */
export function lastCapsuleOfTurns(items: SpanItem[]): Map<string, TurnSpan> {
  const out = new Map<string, TurnSpan>();
  let fromTs: string | null = null;
  let lastCapsule: string | null = null;
  const close = (toTs: string | null) => {
    if (fromTs && lastCapsule) out.set(lastCapsule, { fromTs, toTs });
  };
  for (const item of items) {
    if (item.type === 'user_message' && item.timestamp) {
      close(item.timestamp);
      fromTs = item.timestamp;
      lastCapsule = null;
      continue;
    }
    if (item.type === 'session_separator') {
      close(item.timestamp ?? null);
      fromTs = null;
      lastCapsule = null;
      continue;
    }
    if (item.type === 'agent_run' && item.capsule_id && !item.memory_record && !item.memory_tidy) {
      lastCapsule = item.capsule_id;
    }
  }
  close(null);
  return out;
}

/**
 * Split a translated string on a `{x}` slot the caller fills with a React
 * node (a logo): returns the text before and after it.
 */
export function splitOnSlot(text: string, marker: string): [string, string] {
  const i = text.indexOf(marker);
  if (i < 0) return [text, ''];
  return [text.slice(0, i), text.slice(i + marker.length)];
}
