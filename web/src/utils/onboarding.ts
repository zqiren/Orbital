// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 138 — value-first onboarding from the user's agent history. The wire
 * shapes of the onboarding routes (history scan, importable projects, batch
 * import, left-off cards) and the pure decisions the wizard, the import
 * dialog and the left-off cards share, kept here so they test without a DOM.
 */

import type { useT } from '../i18n/useT';

// ── Wire shapes ─────────────────────────────────────────────────────────────

/** GET /api/v2/onboarding/history — also starts the scan on first call. */
export interface HistoryStatus {
  status: 'scanning' | 'ready';
  sessions_total: number;
  /** A key per agent whose history reader loaded (0 allowed). */
  by_agent: Record<string, number>;
  /** Spec 153: private notes per agent (absent on an older daemon). */
  notes_by_agent?: Record<string, number>;
  /** Spec 153: the same notes by the folder they were kept about; `folder`
   *  null = the agent's dir holds notes but no transcript names its folder. */
  notes_by_folder?: NotesFolderRow[];
}

export interface NotesFolderRow {
  agent?: string;
  folder: string | null;
  /** The agent's own (lossy) dir name, shown when the folder is unknown. */
  encoded: string;
  n: number;
}

/** GET /api/v2/projects/{pid}/imported-notes (spec 153). */
export interface ImportedNotes {
  agent: string;
  count: number;
  /** The notes are in the project (the fold completed). */
  folded: boolean;
  /** Workspace-relative path of the imported index. */
  index_path: string;
}

/** One folder from GET /api/v2/onboarding/importable-projects. */
export interface ImportCandidate {
  source: 'claude-code' | 'codex' | 'obsidian' | string;
  name: string;
  path: string;
  session_count: number;
  last_activity: string | null;
  /** The folder is already a project's workspace (spec 084). */
  already_imported?: boolean;
  kind?: string;
  /** Qualifies only by one or two conversations: offered collapsed, unchecked. */
  minor?: boolean;
  is_git?: boolean;
  by_agent?: Record<string, number>;
  /** Spec 153: private notes per agent kept about this folder. */
  notes?: Record<string, number>;
}

export interface GroupReason {
  kind: 'same_skill' | 'same_files' | 'same_request' | 'same_folder' | string;
  value: string;
  count: number;
}

/** Conversations that ran outside any project but belong together. */
export interface ImportGroup {
  id: string;
  session_count: number;
  last_activity: string | null;
  by_agent: Record<string, number>;
  reasons: GroupReason[];
  suggested_folder: string | null;
  members: Array<{ agent: string; native_id: string }>;
}

export interface ImportableResponse {
  /** Absent on an older daemon, which answers in one shot. */
  status?: 'scanning' | 'ready';
  candidates: ImportCandidate[];
  /** `[]` until the scan is ready; absent on an older daemon. */
  groups?: ImportGroup[];
}

export interface ImportRequest {
  folders: string[];
  groups: Array<{ id: string; workspace: string }>;
}

export interface ImportResult {
  created: Array<{ project_id: string; name: string; workspace: string; from: 'folder' | 'group' }>;
  skipped: Array<{ path: string; reason: string }>;
}

export const STOP_CAUSES = ['usage_limit', 'model_unavailable', 'network_auth', 'interrupted', 'no_reply'] as const;
export type StopCause = (typeof STOP_CAUSES)[number];

export interface LeftOffGit {
  branch: string | null;
  session_branch?: string | null;
  uncommitted: number;
  commits_since: number;
}

export interface LeftOffLastTime {
  session_id: string;
  /** The slug the session ran on; null = the Orbital agent. */
  agent: string | null;
  user_text: string;
  at: string;
  git: LeftOffGit | null;
}

export interface LeftOffUnfinished {
  session_id: string;
  agent: string | null;
  user_text: string;
  at: string;
  cause: StopCause;
  /** Only while still in the future. */
  resets_at: string | null;
}

/** GET /api/v2/projects/{pid}/left-off */
export interface LeftOffResponse {
  last_time: LeftOffLastTime | null;
  unfinished: LeftOffUnfinished[];
}

// ── Agent names ─────────────────────────────────────────────────────────────

/** Product names of the agents with a history reader. Not translated. */
const KNOWN_AGENT_NAMES: Record<string, string> = {
  'claude-code': 'Claude Code',
  codex: 'Codex',
  pi: 'Pi',
  'kimi-code': 'Kimi Code',
  'minimax-code': 'MiniMax Code',
  obsidian: 'Obsidian',
};

/** `null` is the Orbital agent itself. */
export function agentDisplayName(
  slug: string | null,
  installed: ReadonlyArray<{ slug: string; name: string }> = [],
): string {
  if (!slug) return 'Orbital';
  return installed.find((a) => a.slug === slug)?.name ?? KNOWN_AGENT_NAMES[slug] ?? slug;
}

// ── The project checklist ───────────────────────────────────────────────────

/** Tick overrides keyed by folderKey/groupKey; absent = the default tick. */
export type TickOverrides = Record<string, boolean>;

export const folderKey = (path: string) => `f:${path}`;
export const groupKey = (id: string) => `g:${id}`;

/** Everything is offered checked except the minor folders; an imported
 *  folder is not offered at all. */
export function isFolderChecked(c: ImportCandidate, ticks: TickOverrides): boolean {
  if (c.already_imported) return false;
  return ticks[folderKey(c.path)] ?? !c.minor;
}

export function isGroupChecked(g: ImportGroup, ticks: TickOverrides): boolean {
  return ticks[groupKey(g.id)] ?? true;
}

/** The folder a group will be imported into: the user's choice, else the
 *  suggestion. null = it cannot be imported yet. */
export function groupFolder(g: ImportGroup, chosen: Record<string, string>): string | null {
  return chosen[g.id] ?? g.suggested_folder ?? null;
}

export function buildImportRequest(
  data: ImportableResponse,
  ticks: TickOverrides,
  chosen: Record<string, string>,
): ImportRequest {
  const folders = data.candidates.filter((c) => isFolderChecked(c, ticks)).map((c) => c.path);
  const groups: ImportRequest['groups'] = [];
  for (const g of data.groups ?? []) {
    const workspace = groupFolder(g, chosen);
    if (isGroupChecked(g, ticks) && workspace) groups.push({ id: g.id, workspace });
  }
  return { folders, groups };
}

/** Items that were offered checked and the user unchecked. */
export function uncheckedCount(data: ImportableResponse, ticks: TickOverrides): number {
  const folders = data.candidates.filter(
    (c) => !c.already_imported && !c.minor && !isFolderChecked(c, ticks),
  ).length;
  const groups = (data.groups ?? []).filter((g) => !isGroupChecked(g, ticks)).length;
  return folders + groups;
}

/** Whether the Projects step has anything to offer. */
export function hasImportables(data: ImportableResponse): boolean {
  return data.candidates.some((c) => !c.already_imported) || (data.groups?.length ?? 0) > 0;
}

const ts = (iso: string | null | undefined) => {
  const n = iso ? Date.parse(iso) : NaN;
  return Number.isNaN(n) ? -Infinity : n;
};

/** The created project whose conversations were most recently active. */
export function landingProjectId(
  result: ImportResult,
  request: ImportRequest,
  data: ImportableResponse,
): string | null {
  if (result.created.length === 0) return null;
  const activity = (c: ImportResult['created'][number]): number => {
    if (c.from === 'group') {
      const sent = request.groups.find((g) => g.workspace === c.workspace);
      return ts((data.groups ?? []).find((g) => g.id === sent?.id)?.last_activity);
    }
    return ts(data.candidates.find((x) => x.path === c.workspace)?.last_activity);
  };
  let best = result.created[0];
  for (const c of result.created) if (activity(c) > activity(best)) best = c;
  return best.project_id;
}

/** "Found in {n} conversations with your agents. …" — n from the history
 *  scan, else the candidates' own counts while it is still running. */
export function importIntro(
  t: ReturnType<typeof useT>,
  total: number | undefined,
  data: { candidates: Array<{ session_count: number }> } | null,
): string {
  // While the history scan is still counting, its total lags the rows that
  // are already on screen; never say "0 conversations" above a list of them.
  const fromRows = (data?.candidates ?? []).reduce((sum, c) => sum + (c.session_count || 0), 0);
  const n = Math.max(total ?? 0, fromRows);
  return t(n === 1 ? 'importProjects.intro.one' : 'importProjects.intro.other', { n });
}

// ── Left-off cards ──────────────────────────────────────────────────────────

const isStr = (v: unknown): v is string => typeof v === 'string';

/** Validate the left-off body; anything malformed reads as "nothing to
 *  show", so the empty chat falls back to its example prompts. */
export function parseLeftOff(raw: unknown): LeftOffResponse | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const body = raw as { last_time?: unknown; unfinished?: unknown };
  let last: LeftOffLastTime | null = null;
  const lt = body.last_time as Partial<LeftOffLastTime> | null | undefined;
  if (lt && isStr(lt.session_id) && isStr(lt.user_text) && isStr(lt.at)) {
    last = {
      session_id: lt.session_id,
      agent: isStr(lt.agent) ? lt.agent : null,
      user_text: lt.user_text,
      at: lt.at,
      git: lt.git && typeof lt.git === 'object' ? lt.git : null,
    };
  }
  const rows = Array.isArray(body.unfinished) ? body.unfinished : [];
  const unfinished: LeftOffUnfinished[] = [];
  for (const r of rows as Array<Partial<LeftOffUnfinished>>) {
    if (!r || !isStr(r.session_id) || !isStr(r.user_text) || !isStr(r.at)) continue;
    if (!(STOP_CAUSES as readonly string[]).includes(r.cause as string)) continue;
    unfinished.push({
      session_id: r.session_id,
      agent: isStr(r.agent) ? r.agent : null,
      user_text: r.user_text,
      at: r.at,
      cause: r.cause as StopCause,
      resets_at: isStr(r.resets_at) ? r.resets_at : null,
    });
    if (unfinished.length === 3) break;
  }
  return { last_time: last, unfinished };
}

export function hasLeftOff(r: LeftOffResponse | null): boolean {
  return !!r && (r.last_time !== null || r.unfinished.length > 0);
}

/** The "N conversations" count the Agents step shows for one agent: null
 *  (no count) when the agent has no history reader, and also when it has
 *  none on this computer — "0 conversations" reads like something broke. */
export function conversationCount(
  byAgent: Record<string, number> | undefined,
  slug: string,
): number | null {
  const n = byAgent?.[slug];
  return typeof n === 'number' && n > 0 ? n : null;
}

/** Spec 153: the "N memories only it can use" count of one agent's wizard
 *  row; null hides the segment (none, or an older daemon). */
export function notesCount(history: HistoryStatus | null | undefined, slug: string): number | null {
  const n = history?.notes_by_agent?.[slug];
  return typeof n === 'number' && n > 0 ? n : null;
}

export interface NotesBreakdownRow {
  /** Folder basename, or the agent's dir name when the folder is unknown. */
  label: string;
  notFound: boolean;
  n: number;
}

/** The click-to-expand breakdown under an agent's row: one row per folder,
 *  most notes first (the daemon's order; re-sorted so a reorder there cannot
 *  break it). Rows of other agents are skipped; a row without `agent` (an
 *  older shape) is Claude Code's, the only reader. */
export function notesBreakdown(history: HistoryStatus | null | undefined, slug: string): NotesBreakdownRow[] {
  const rows = (history?.notes_by_folder ?? []).filter(
    (r) => (r.agent ?? 'claude-code') === slug && typeof r.n === 'number' && r.n > 0,
  );
  return rows
    .map((r) => {
      const folder = typeof r.folder === 'string' && r.folder ? r.folder : null;
      const base = folder ? folder.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || folder : r.encoded;
      return { label: base, notFound: folder === null, n: r.n };
    })
    .sort((a, b) => b.n - a.n || Number(a.notFound) - Number(b.notFound) || a.label.localeCompare(b.label));
}

export function parseImportedNotes(raw: unknown): ImportedNotes | null {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  if (typeof r.count !== 'number' || typeof r.index_path !== 'string' || !r.index_path) return null;
  return {
    agent: typeof r.agent === 'string' ? r.agent : 'claude-code',
    count: r.count,
    folded: r.folded === true,
    index_path: r.index_path,
  };
}

/** Spec 153 §4.4: the private-notes card shows in an imported project with
 *  notes to share and at least two workers to share them with — with fewer,
 *  "every agent here can use them" would be false. */
export function showNotesCard(
  notes: ImportedNotes | null,
  project: { imported?: boolean; available_workers?: string[] } | null | undefined,
): boolean {
  if (!notes || notes.count <= 0 || !project?.imported) return false;
  return (project.available_workers?.length ?? 0) >= 2;
}
