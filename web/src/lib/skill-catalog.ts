// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The composer's `/` skill picker data (spec 132): the per-project catalog
 * from `GET /api/v2/projects/{pid}/skill-catalog`, cached per project, and
 * the name matcher.
 */

import { api } from '../config';

export interface SkillCatalogEntry {
  key: string;
  name: string;
  description: string;
  /** Where it was found (project, workspace, agents, claude, codex). Not shown. */
  source: string;
  /** Absolute SKILL.md path; the inject route accepts only catalog paths. */
  path: string;
  dir_name: string;
  /**
   * Spec 137: a built-in entry the client adds to the list (no SKILL.md on
   * disk). Its text rides the message as an inline `<use_skill>` block, so
   * the inject route never sees a catalog path it cannot resolve. Built-ins
   * look like any other skill in the menu (no badge, per spec 132).
   */
  builtin?: {
    /** The skill text the agent follows. Agent-facing, so not translated. */
    body: string;
    /** What the composer holds after the pick (translated). */
    prefill: string;
    /** Always to the manager: only it has the trigger tools. */
    managerOnly: boolean;
  };
  /**
   * Spec 166: a command the pinned worker itself reported. A pick puts
   * exactly this in the composer (`/review ` or Codex's `$name `), with no
   * chip and no payload: the worker reads its own invocation.
   */
  invoke?: string;
  /** The worker's argument hint (`<pr>`), shown muted after the name. */
  hint?: string;
}

/** The path a built-in carries in its block; never a file. */
export const BUILTIN_SKILL_PATH_PREFIX = 'orbital:builtin/';

export interface BuiltinSkillStrings {
  description: string;
  prefill: string;
}

/**
 * The `/automate` instructions. The time zone is the browser's, so "9am"
 * means the user's 9am rather than the tool's UTC default.
 */
export function automateSkillBody(timeZone: string): string {
  return [
    'Create an automation for this project from what the user wrote below, using the `create_trigger` tool.',
    '- A time or a rhythm ("every weekday at 9am") is `type="schedule"`: write the cron expression, a short `human` description, and `timezone` = `' +
      timeZone +
      '` unless the user names another.',
    '- "When files appear or change in <folder>" is `type="file_watch"`: `watch_path` relative to the workspace, plus `patterns` if the user named file types.',
    '- `task` is the instruction the automation will run each time, written so it stands on its own.',
    '- Give it a short `name`.',
    'Ask one question only if the schedule or the folder is genuinely unclear. After creating it, reply in one or two sentences: what it does and when it runs. Do not run the task now.',
  ].join('\n');
}

/** The client-side entries merged into every project's `/` menu. */
export function builtinSkills(strings: BuiltinSkillStrings, timeZone: string): SkillCatalogEntry[] {
  return [
    {
      key: 'builtin:automate',
      name: 'automate',
      description: strings.description,
      source: 'builtin',
      path: `${BUILTIN_SKILL_PATH_PREFIX}automate`,
      dir_name: 'automate',
      builtin: { body: automateSkillBody(timeZone), prefill: strings.prefill, managerOnly: true },
    },
  ];
}

/** Built-ins first; an installed skill with the same name replaces nothing. */
export function withBuiltinSkills(
  entries: SkillCatalogEntry[] | null,
  builtins: SkillCatalogEntry[],
): SkillCatalogEntry[] | null {
  if (entries === null) return null;
  return [...builtins, ...entries];
}

/**
 * How a picked skill travels (spec 132 + 137). A catalog skill is a payload
 * the backend re-resolves plus the pointer block it prepends; a built-in is
 * an inline block inside the message text and no payload.
 */
export function skillSendParts(
  skill: SkillCatalogEntry | null | undefined,
  buildPointerBlock: (s: { name: string; path: string }) => string,
  buildInlineBlock: (name: string, path: string, body: string) => string,
): { payload?: { key: string; path: string }; pointerBlock: string; contentPrefix: string } {
  if (!skill) return { pointerBlock: '', contentPrefix: '' };
  if (skill.builtin) {
    return { pointerBlock: '', contentPrefix: buildInlineBlock(skill.name, skill.path, skill.builtin.body) };
  }
  return {
    payload: { key: skill.key, path: skill.path },
    pointerBlock: buildPointerBlock({ name: skill.name, path: skill.path }),
    contentPrefix: '',
  };
}

/** A cached catalog older than this is refreshed in the background on open. */
export const SKILL_CATALOG_REFRESH_MS = 30_000;

interface CacheEntry {
  entries: SkillCatalogEntry[] | null;
  fetchedAt: number;
  inflight: Promise<SkillCatalogEntry[]> | null;
}

const cache = new Map<string, CacheEntry>();

export function __clearSkillCatalogCacheForTests(): void {
  cache.clear();
}

/** Spec 166: one slot per project and pin (`null` = Orbital, no worker). */
function cacheKey(projectId: string, worker: string | null | undefined): string {
  return worker ? `${projectId}\u0000${worker}` : projectId;
}

/** The cached list for `projectId` under this pin, or null before the first fetch lands. */
export function cachedSkillCatalog(
  projectId: string,
  worker?: string | null,
): SkillCatalogEntry[] | null {
  return cache.get(cacheKey(projectId, worker))?.entries ?? null;
}

/**
 * Fetch the catalog unless a fresh copy is cached. Resolves with the list
 * (the cached one when no fetch was needed). A failed fetch keeps whatever
 * was cached and resolves with it, or with `[]`. With a worker pin the
 * daemon folds that worker's own commands in (spec 166).
 */
export function loadSkillCatalog(
  projectId: string,
  worker?: string | null,
  now = Date.now(),
): Promise<SkillCatalogEntry[]> {
  const key = cacheKey(projectId, worker);
  const hit = cache.get(key);
  if (hit?.inflight) return hit.inflight;
  if (hit?.entries && now - hit.fetchedAt < SKILL_CATALOG_REFRESH_MS) {
    return Promise.resolve(hit.entries);
  }
  const slot: CacheEntry = hit ?? { entries: null, fetchedAt: 0, inflight: null };
  const query = worker ? `?worker=${encodeURIComponent(worker)}` : '';
  slot.inflight = api<SkillCatalogEntry[]>(
    `/api/v2/projects/${encodeURIComponent(projectId)}/skill-catalog${query}`,
  )
    .then((data) => {
      slot.entries = Array.isArray(data) ? data : [];
      slot.fetchedAt = Date.now();
      return slot.entries;
    })
    .catch(() => slot.entries ?? [])
    .finally(() => {
      slot.inflight = null;
    });
  cache.set(key, slot);
  return slot.inflight;
}

const SEGMENT_SPLIT = /[-_\s]+/;

function rank(entry: SkillCatalogEntry, q: string): number | null {
  const names = [entry.name.toLowerCase(), entry.dir_name.toLowerCase()];
  if (names.some((n) => n.startsWith(q))) return 0;
  if (names.some((n) => n.split(SEGMENT_SPLIT).some((seg) => seg.startsWith(q)))) return 1;
  return null;
}

/**
 * Case-insensitive prefix match on the skill name (or its directory name),
 * then prefix match on each `-`/`_`/space-separated segment: "msg" finds
 * `wecomcli-msg`, "wec" every `wecomcli-*`. Full-name matches sort first,
 * then segment matches, then alphabetical. Descriptions are never matched.
 */
export function matchSkills(entries: SkillCatalogEntry[], query: string): SkillCatalogEntry[] {
  const q = query.toLowerCase();
  const ranked: Array<{ e: SkillCatalogEntry; r: number }> = [];
  for (const e of entries) {
    const r = q === '' ? 0 : rank(e, q);
    if (r !== null) ranked.push({ e, r });
  }
  ranked.sort((a, b) => {
    if (a.r !== b.r) return a.r - b.r;
    const an = a.e.name.toLowerCase();
    const bn = b.e.name.toLowerCase();
    return an < bn ? -1 : an > bn ? 1 : 0;
  });
  return ranked.map((x) => x.e);
}

/**
 * The `/` query when the composer is in picker position, else null: `/` is
 * the first character and the rest has no whitespace or second `/` (so
 * `/Users/…`, `/tmp/x` and `/ 这个` are never picker input).
 */
export function slashQuery(text: string): string | null {
  if (!text.startsWith('/')) return null;
  const rest = text.slice(1);
  return /^[^\s/]*$/.test(rest) ? rest : null;
}
