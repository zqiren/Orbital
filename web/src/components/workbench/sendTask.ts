// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 154 §5.6 — sending a task from the Workbench composer.
 *
 * Composes the two primitives the card doorway already uses, never a new
 * daemon endpoint that re-implements the inject route:
 *   1. a pending card at the top of 进行中 (正在开始…);
 *   2. project chip on 自动 → `POST /workbench/route` (one model call;
 *      Quick Tasks on any failure), else the picked project;
 *   3. `POST /agents/{pid}/new-session` → `{session_id, pinned_target}`;
 *   4. the chat's own send: target = the picked agent (null for Orbital),
 *      else the minted `pinned_target`. The targeted first send stamps the
 *      session's pin; the Workbench never PATCHes it, so a per-task pick never
 *      moves the project's sticky pin.
 *
 * Failure rule (one rule, no inference from error codes): after a failure,
 * if the session exists in that project the text was persisted — the card
 * stays and shows the failure like any failed run. Otherwise the pending card
 * goes, the text returns to the composer and the open-error banner shows.
 *
 * The draft is kept in localStorage (`{text, nonce, project_id, session_id}`,
 * the last two filled as soon as known) until the inject returns. After a
 * reload it is restored only when no session with that id exists in that
 * project; otherwise it is dropped, because the text is already there.
 */

import { api } from '../../config';
import type { SessionListEntry } from '../../types';
import {
  addPending, refreshFeed, removePending, restoreComposer, updatePending,
} from '../../hooks/useWorkbenchFeed';

export const DRAFT_KEY = 'orbital:workbench.draft';

interface Draft {
  text: string;
  nonce: string;
  project_id: string | null;
  session_id: string | null;
}

function saveDraft(d: Draft): void {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(d));
  } catch {
    /* storage unavailable — the in-memory pending card still holds the text */
  }
}

function clearDraft(): void {
  try {
    localStorage.removeItem(DRAFT_KEY);
  } catch {
    /* ignore */
  }
}

function readDraft(): Draft | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw) as Partial<Draft>;
    if (typeof d?.text !== 'string' || !d.text.trim()) return null;
    return {
      text: d.text,
      nonce: typeof d.nonce === 'string' ? d.nonce : '',
      project_id: typeof d.project_id === 'string' ? d.project_id : null,
      session_id: typeof d.session_id === 'string' ? d.session_id : null,
    };
  } catch {
    return null;
  }
}

function newNonce(): string {
  const c = globalThis.crypto as Crypto | undefined;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  return `wb-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** Does this session exist (has a first message) in that project? */
async function sessionExists(projectId: string, sessionId: string): Promise<boolean> {
  try {
    const resp = await api<{ sessions?: SessionListEntry[] }>(
      `/api/v2/projects/${encodeURIComponent(projectId)}/sessions`,
    );
    return (resp?.sessions ?? []).some(
      (s) => s.session_id === sessionId || s.session_uuid === sessionId,
    );
  } catch {
    return false;
  }
}

export interface SendDeps {
  newSession: (projectId: string) => Promise<{ session_id?: string; pinned_target?: string | null } | undefined>;
  injectMessage: (
    projectId: string, content: string, target?: string, nonce?: string,
    attachments?: undefined, sessionId?: string, pinned?: boolean,
  ) => Promise<unknown>;
}

export interface SendChoice {
  text: string;
  /** null = 自动 (route it). */
  projectId: string | null;
  /** null = 自动 (the project's pin); 'orbital' = the manager. */
  agent: string | null;
}

export type SendOutcome =
  | { ok: true; projectId: string; sessionId: string }
  | { ok: false; restored: boolean };

export async function sendWorkbenchTask(choice: SendChoice, deps: SendDeps): Promise<SendOutcome> {
  const text = choice.text.trim();
  const nonce = newNonce();
  const draft: Draft = { text, nonce, project_id: choice.projectId, session_id: null };
  saveDraft(draft);
  addPending({
    nonce, text, project_id: choice.projectId, session_id: null,
    agent: choice.agent, created_at: Date.now(),
  });

  let projectId = choice.projectId;
  let sessionId: string | null = null;
  try {
    if (!projectId) {
      const routed = await api<{ project_id?: string | null }>('/api/v2/workbench/route', {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      projectId = routed?.project_id ?? null;
      if (!projectId) throw new Error('no project to route to');
      draft.project_id = projectId;
      saveDraft(draft);
      updatePending(nonce, { project_id: projectId });
    }
    const minted = await deps.newSession(projectId);
    if (!minted?.session_id) throw new Error('mint failed');
    sessionId = minted.session_id;
    draft.session_id = sessionId;
    saveDraft(draft);
    updatePending(nonce, { session_id: sessionId });
    const target = choice.agent === 'orbital'
      ? undefined
      : (choice.agent ?? minted.pinned_target ?? undefined);
    await deps.injectMessage(projectId, text, target, nonce, undefined, sessionId, !!target);
    clearDraft();
    void refreshFeed();
    return { ok: true, projectId, sessionId };
  } catch {
    clearDraft();
    void refreshFeed();
    if (projectId && sessionId && (await sessionExists(projectId, sessionId))) {
      // The text was persisted: the session's own row shows the failure.
      return { ok: true, projectId, sessionId };
    }
    removePending(nonce);
    restoreComposer(text);
    return { ok: false, restored: true };
  }
}

/** After a reload: put an unsent draft back in the composer, unless its
 * session exists (then the text is already in that session). */
export async function recoverDraft(): Promise<void> {
  const d = readDraft();
  if (!d) return;
  if (d.project_id && d.session_id && (await sessionExists(d.project_id, d.session_id))) {
    clearDraft();
    return;
  }
  clearDraft();
  restoreComposer(d.text);
}
