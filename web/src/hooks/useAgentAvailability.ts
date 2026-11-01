// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Spec 099 — readiness + subscription quota per agent, from
// GET /api/v2/agents/available, kept live by `agent.quota_updated`.
//
// A module-level store rather than props: the pin control is mounted three
// levels under App (ChatView, QueueComposer, AutomationForm) and the data is
// account-wide, not per project. App owns the fetch-at-mount and the WS
// subscription; the pin menu re-fetches on open (one backend-cached REST
// hit, which also starts the codex read-through refresh).

import { useSyncExternalStore } from 'react';
import { api } from '../config';
import type { AgentAvailability, QuotaSnapshot } from '../types';

export interface InstalledAgent { slug: string; name: string }

let bySlug: Record<string, AgentAvailability> = {};
// The pin menu's rows. Null until the first answer; a new array only when the
// set changes, so quota pushes don't re-render the app root.
let installedAgents: InstalledAgent[] | null = null;
const listeners = new Set<() => void>();
let inflight: Promise<AgentAvailability[]> | null = null;

function emit() {
  listeners.forEach((l) => l());
}

function observedMs(q: QuotaSnapshot | undefined): number {
  const t = q ? Date.parse(q.observed_at) : NaN;
  return Number.isNaN(t) ? -Infinity : t;
}

/** Replace the list. A quota that a WS push delivered after this response
 *  was produced is newer than the response's, so it is kept. */
export function setAgentAvailability(list: AgentAvailability[]): void {
  const next: Record<string, AgentAvailability> = {};
  for (const entry of list) {
    const prev = bySlug[entry.slug]?.quota;
    const keep = prev && observedMs(prev) > observedMs(entry.quota);
    next[entry.slug] = keep ? { ...entry, quota: prev } : entry;
  }
  bySlug = next;
  const installed = list
    .filter((a) => a.slug !== 'built-in' && a.installed)
    .map((a) => ({ slug: a.slug, name: a.name }));
  if (JSON.stringify(installed) !== JSON.stringify(installedAgents)) {
    installedAgents = installed;
  }
  emit();
}

/** Apply `agent.quota_updated` (null = the daemon cleared it). */
export function applyQuotaUpdate(agent: string, snapshot: QuotaSnapshot | null): void {
  const prev = bySlug[agent];
  if (!prev && !snapshot) return;
  const base = prev ?? { slug: agent, name: agent };
  const { quota: _old, ...rest } = base;
  bySlug = { ...bySlug, [agent]: snapshot ? { ...rest, quota: snapshot } : rest };
  emit();
}

/** Fetch the list (single-flight) and update the store. Rejects on failure
 *  so a caller can tell a first-load failure from an empty list. */
export function fetchAgentAvailability(): Promise<AgentAvailability[]> {
  if (inflight) return inflight;
  inflight = api<AgentAvailability[]>('/api/v2/agents/available')
    .then((list) => {
      const safe = Array.isArray(list) ? list : [];
      setAgentAvailability(safe);
      return safe;
    })
    .finally(() => { inflight = null; });
  return inflight;
}

/** Fire-and-forget refresh (pin menu open). Failures keep the last state. */
export function refreshAgentAvailability(): void {
  fetchAgentAvailability().catch(() => { /* keep the last good state */ });
}

function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

const getSnapshot = () => bySlug;

export function useAgentAvailability(): Record<string, AgentAvailability> {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/** The installed agents, refreshed by every fetch: the pin menu opening, the
 *  settings page changing one, the daemon's sweep finding a change. Null until
 *  the first answer. */
export function useInstalledAgents(): InstalledAgent[] | null {
  const get = () => installedAgents;
  return useSyncExternalStore(subscribe, get, get);
}

/** A first load that failed settles on empty; a failed refetch keeps the list. */
export function settleInstalledAgents(): void {
  if (installedAgents !== null) return;
  installedAgents = [];
  emit();
}

/** Test seam: reset module state between tests. */
export function __resetAgentAvailability(): void {
  bySlug = {};
  installedAgents = null;
  inflight = null;
  emit();
}
