// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The Welcome page's data rules, pure: an agent's state, how many are ready,
 * and the Projects band's one line (its count and its live checkbox).
 */

import { isSignedIn, type SubAgentListEntry } from '../../hooks/useSubAgentList';
import {
  buildImportRequest, folderKey, groupFolder, groupKey, rowKeys, withLateRowsUnchecked, withTicksChange,
  type ImportChoices, type ImportableResponse, type ImportRequest,
} from '../../utils/onboarding';

/** ready = could answer now (the daemon's `ready`: installed, signed in,
 *  nothing blocking it); out = installed, signed out, has a login; key =
 *  installed, takes a key instead (its key lives in Settings); off = signed
 *  in but something else blocks it (its vendor refused a turn, its version
 *  floor, a failed handshake, a dependency): the tile names it
 *  (`not_ready`) and offers the fix when there is one. */
export type AgentState = 'ready' | 'out' | 'key' | 'off';

/** The daemon's `ready` (and not a login the CLI said expired); an older
 *  daemon without `ready`: installed and signed in. */
export function isReady(e: SubAgentListEntry): boolean {
  return e.installed && (e.ready ?? isSignedIn(e)) && e.auth_expired !== true;
}

export function agentStateOf(e: SubAgentListEntry): AgentState {
  if (isReady(e)) return 'ready';
  if (!isSignedIn(e)) return e.supports_login === false ? 'key' : 'out';
  return 'off';
}

/** Agents that could answer the first message right now. */
export function readyAgentCount(entries: readonly SubAgentListEntry[] | null): number {
  return (entries ?? []).filter(isReady).length;
}

/** What Start would import: the list as the checklist shows it. */
export function importRequestOf(data: ImportableResponse | null, choices: ImportChoices): ImportRequest {
  if (!data) return { folders: [], groups: [] };
  const ticks = withLateRowsUnchecked(data, choices.ticks, choices.offered);
  return buildImportRequest(data, ticks, choices.chosen);
}

export function importCount(data: ImportableResponse | null, choices: ImportChoices): number {
  const r = importRequestOf(data, choices);
  return r.folders.length + r.groups.length;
}

/** Everything the list would import by default (folders that are not minor,
 *  groups that have a folder). The line's "all". */
export function defaultImportCount(data: ImportableResponse | null, choices: ImportChoices): number {
  if (!data) return 0;
  const folders = data.candidates.filter((c) => !c.already_imported && !c.minor).length;
  const groups = (data.groups ?? []).filter((g) => !!groupFolder(g, choices.chosen)).length;
  return folders + groups;
}

export type LineTick = 'all' | 'some' | 'none';

export function importLineTick(data: ImportableResponse | null, choices: ImportChoices): LineTick {
  const n = importCount(data, choices);
  if (n === 0) return 'none';
  return n >= defaultImportCount(data, choices) ? 'all' : 'some';
}

/**
 * The line's checkbox: anything ticked → nothing; nothing ticked → the
 * default ticks (folders that are not minor, and the groups). Pins the list
 * exactly as a click in the checklist does, so rows that arrive later stay
 * unchecked once the user has chosen.
 */
export function toggleImportLine(prev: ImportChoices, data: ImportableResponse | null): ImportChoices {
  if (!data) return prev;
  if (importCount(data, prev) > 0) return withTicksChange(prev, data, rowKeys(data), false);
  const on = [
    ...data.candidates.filter((c) => !c.already_imported && !c.minor).map((c) => folderKey(c.path)),
    ...(data.groups ?? []).map((g) => groupKey(g.id)),
  ];
  const off = data.candidates.filter((c) => !c.already_imported && c.minor).map((c) => folderKey(c.path));
  return withTicksChange(withTicksChange(prev, data, off, false), data, on, true);
}

/** The agents whose history the found projects come from, most first. */
export function importAgents(data: ImportableResponse | null): string[] {
  const n: Record<string, number> = {};
  for (const c of data?.candidates ?? []) {
    if (c.already_imported) continue;
    // A vault or a bare repo found on disk is no agent's history.
    const by = c.by_agent ?? (c.source === 'obsidian' || c.source === 'git' ? {} : { [c.source]: c.session_count });
    for (const [slug, k] of Object.entries(by)) if (k > 0) n[slug] = (n[slug] ?? 0) + k;
  }
  for (const g of data?.groups ?? []) {
    for (const [slug, k] of Object.entries(g.by_agent ?? {})) if (k > 0) n[slug] = (n[slug] ?? 0) + k;
  }
  return Object.entries(n).sort((a, b) => b[1] - a[1]).map(([slug]) => slug);
}
