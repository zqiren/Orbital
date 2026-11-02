// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 149 — the pin menu's model + thinking per worker, and Orbital's card.
 *
 * Three existing reads, layered:
 *  - `GET /settings/sub-agents`: each agent's choices (`param_schema`, with
 *    the live model lists and their labels) and its Settings values (`config`);
 *  - `GET /agents/{pid}/sessions/{sid}/sub-agent-params`: this conversation's
 *    overrides (`PATCH` the same path to set/clear one);
 *  - `GET /settings`: the credential cards Orbital's own row offers.
 * Everything loads when the menu opens (`refresh`), never on mount: the line
 * only shows inside the open menu.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../config';
import type { CredentialCard } from '../types';

interface ParamSchemaLite {
  allowed: string[] | null;
  default?: string | null;
  labels?: Record<string, string> | null;
}

export interface SubAgentModelEntry {
  slug: string;
  config?: Record<string, string> | null;
  param_schema?: Record<string, ParamSchemaLite> | null;
}

export interface ModelChoice { value: string; label: string }

/** What the menu needs about one worker. */
export interface WorkerModelInfo {
  slug: string;
  /** Listed models, or null when the model is free text. */
  models: ModelChoice[] | null;
  /** Thinking levels the agent supports, or null (no thinking control). */
  levels: string[] | null;
  /** The Settings values the conversation falls back to (null = CLI default). */
  settingsModel: string | null;
  settingsEffort: string | null;
  /** This conversation's overrides (null = follow Settings). */
  model: string | null;
  effort: string | null;
}

export type ParamsPatch = { model?: string | null; effort?: string | null };
type Overrides = Record<string, { model?: string; effort?: string }>;

export function labelFor(info: WorkerModelInfo, value: string): string {
  return info.models?.find((m) => m.value === value)?.label ?? value;
}

/** Fold one Settings entry and the conversation's override into the menu's
 *  view of that worker; null for an agent with neither model nor thinking. */
export function workerModelInfo(
  entry: SubAgentModelEntry,
  override: { model?: string; effort?: string } | undefined,
): WorkerModelInfo | null {
  const schema = entry.param_schema ?? {};
  if (!schema.model && !schema.effort) return null;
  const labels = schema.model?.labels ?? {};
  const allowed = schema.model?.allowed ?? null;
  return {
    slug: entry.slug,
    models: allowed ? allowed.map((v) => ({ value: v, label: labels[v] ?? v })) : null,
    levels: schema.effort?.allowed ?? null,
    settingsModel: entry.config?.model || schema.model?.default || null,
    settingsEffort: entry.config?.effort || schema.effort?.default || null,
    model: override?.model || null,
    effort: override?.effort || null,
  };
}

/** The row's second line: "<model> · <thinking>", the effective values. */
export function modelLineText(info: WorkerModelInfo, defaultLabel: string): string {
  const model = info.model ?? info.settingsModel;
  const effort = info.effort ?? info.settingsEffort;
  const head = model ? labelFor(info, model) : defaultLabel;
  return effort ? `${head} · ${effort}` : head;
}

// Choices and Settings values change rarely: one copy for every chat view, so
// a second open shows the lines at once while a refresh runs behind it.
let entriesCache: SubAgentModelEntry[] | null = null;
const NO_OVERRIDES: Overrides = {};
const NO_CARDS: CredentialCard[] = [];
let cardsCache: { cards: CredentialCard[]; defaultCardId: string | null } | null = null;

/** Test seam. */
export function __resetConversationModels(): void {
  entriesCache = null;
  cardsCache = null;
}

export interface ConversationModels {
  workers: Record<string, WorkerModelInfo>;
  cards: CredentialCard[];
  defaultCardId: string | null;
  /** Re-read all three sources (call when the menu opens). */
  refresh: () => void;
  /** Set / clear (null) a worker's model or thinking for this conversation. */
  setWorkerParams: (slug: string, patch: ParamsPatch) => Promise<boolean>;
}

export function useConversationModels(
  projectId: string,
  sessionId: string | undefined,
): ConversationModels {
  const [entries, setEntries] = useState<SubAgentModelEntry[] | null>(entriesCache);
  const [cards, setCards] = useState(cardsCache);
  const [overrides, setOverrides] = useState<{ sid: string | undefined; params: Overrides }>(
    { sid: sessionId, params: {} },
  );
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  const loadOverrides = useCallback(() => {
    if (!sessionId) return;
    api<{ params?: Overrides }>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/sub-agent-params`,
    )
      .then((r) => {
        if (alive.current) setOverrides({ sid: sessionId, params: r.params ?? {} });
      })
      .catch(() => {});
  }, [projectId, sessionId]);

  // A different conversation has its own overrides.
  useEffect(() => {
    setOverrides({ sid: sessionId, params: {} });
    if (entriesCache !== null) loadOverrides();
  }, [sessionId, loadOverrides]);

  const refresh = useCallback(() => {
    api<SubAgentModelEntry[]>('/api/v2/settings/sub-agents')
      .then((list) => {
        entriesCache = Array.isArray(list) ? list : [];
        if (alive.current) setEntries(entriesCache);
      })
      .catch(() => {});
    api<{ credential_cards?: CredentialCard[]; default_card_id?: string | null }>('/api/v2/settings')
      .then((s) => {
        cardsCache = { cards: s.credential_cards ?? [], defaultCardId: s.default_card_id ?? null };
        if (alive.current) setCards(cardsCache);
      })
      .catch(() => {});
    loadOverrides();
  }, [loadOverrides]);

  const setWorkerParams = useCallback(async (slug: string, patch: ParamsPatch) => {
    if (!sessionId) return false;
    const before = overrides.params;
    // Applied at once; the daemon's answer (or a failure) settles it.
    const optimistic = { ...(before[slug] ?? {}) };
    for (const k of ['model', 'effort'] as const) {
      if (!(k in patch)) continue;
      const v = patch[k];
      if (v) optimistic[k] = v; else delete optimistic[k];
    }
    setOverrides({ sid: sessionId, params: { ...before, [slug]: optimistic } });
    try {
      const r = await api<{ params: { model?: string; effort?: string } }>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/sub-agent-params`,
        { method: 'PATCH', body: JSON.stringify({ slug, ...patch }) },
      );
      if (alive.current) {
        setOverrides((cur) => (cur.sid === sessionId
          ? { sid: sessionId, params: { ...cur.params, [slug]: r.params ?? {} } }
          : cur));
      }
      return true;
    } catch {
      if (alive.current) setOverrides({ sid: sessionId, params: before });
      return false;
    }
  }, [projectId, sessionId, overrides.params]);

  const params = overrides.sid === sessionId ? overrides.params : NO_OVERRIDES;
  const workers = useMemo(() => {
    const out: Record<string, WorkerModelInfo> = {};
    for (const entry of entries ?? []) {
      const info = workerModelInfo(entry, params[entry.slug]);
      if (info) out[entry.slug] = info;
    }
    return out;
  }, [entries, params]);
  return {
    workers,
    cards: cards?.cards ?? NO_CARDS,
    defaultCardId: cards?.defaultCardId ?? null,
    refresh,
    setWorkerParams,
  };
}
