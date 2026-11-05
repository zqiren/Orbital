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
 *      Quick Tasks on any failure), else the picked project. The route sees
 *      the attached filenames as an `<attached_files>` list after the text;
 *   3. staged attachments upload into that project (`deps.uploadAll`). One
 *      path whatever the chip: uploads are per project, and on 自动 the
 *      project is only known after step 2;
 *   4. `POST /agents/{pid}/new-session` → `{session_id, pinned_target}`;
 *   5. the chat's own send: target = the picked agent (null for Orbital),
 *      else the minted `pinned_target`. The targeted first send stamps the
 *      session's pin; the Workbench never PATCHes it, so a per-task pick never
 *      moves the project's sticky pin. The attachments ride as the inject's
 *      `attachments` payload and the daemon writes the `<attached_files>`
 *      prefix, exactly as the chat composer sends them.
 *
 * Failure rule (one rule, no inference from error codes): after a failure,
 * if the session exists in that project the text was persisted — the card
 * stays and shows the failure like any failed run. Otherwise the pending card
 * goes, the text returns to the composer and the open-error banner shows; the
 * composer keeps its chips (it clears them only after a send that went out),
 * so a failed upload is retried by sending again.
 *
 * The draft is kept in localStorage (`{text, nonce, project_id, session_id}`,
 * the last two filled as soon as known) until the inject returns. After a
 * reload it is restored only when no session with that id exists in that
 * project; otherwise it is dropped, because the text is already there.
 * The draft carries text only: Files cannot go into localStorage, so after a
 * reload the attachments of an unsent task are gone and only the text comes
 * back.
 */

import { api } from '../../config';
import type { SessionListEntry } from '../../types';
import type { UploadedAttachment } from '../../hooks/useAttachments';
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
    attachments?: UploadedAttachment[], sessionId?: string, pinned?: boolean,
  ) => Promise<unknown>;
  /** Upload the staged chips with these ids into the project (throws when
   *  one fails). Required when the choice carries attachments. */
  uploadAll?: (projectId: string, ids: readonly string[]) => Promise<UploadedAttachment[]>;
}

/** A file staged in the composer: its chip id and name. */
export interface StagedFile {
  id: string;
  filename: string;
}

/** The task text as routing sees it: the attached filenames listed after it. */
export function routeText(text: string, files: readonly StagedFile[]): string {
  if (files.length === 0) return text;
  const block = `<attached_files>\n${files.map((f) => `- ${f.filename}`).join('\n')}\n</attached_files>`;
  return text ? `${text}\n\n${block}` : block;
}

export interface SendChoice {
  text: string;
  /** null = 自动 (route it). */
  projectId: string | null;
  /** null = 自动 (the project's pin); 'orbital' = the manager. */
  agent: string | null;
  /** Staged attachments (uploaded once the project is known). */
  files?: StagedFile[];
}

export type SendOutcome =
  | { ok: true; projectId: string; sessionId: string }
  | { ok: false; restored: boolean };

export async function sendWorkbenchTask(choice: SendChoice, deps: SendDeps): Promise<SendOutcome> {
  const text = choice.text.trim();
  const files = choice.files ?? [];
  const nonce = newNonce();
  const draft: Draft = { text, nonce, project_id: choice.projectId, session_id: null };
  saveDraft(draft);
  addPending({
    // An attachment-only task is named by its files on the pending card.
    nonce, text: text || files.map((f) => f.filename).join(', '),
    project_id: choice.projectId, session_id: null,
    agent: choice.agent, created_at: Date.now(),
  });

  let projectId = choice.projectId;
  let sessionId: string | null = null;
  try {
    if (!projectId) {
      const routed = await api<{ project_id?: string | null }>('/api/v2/workbench/route', {
        method: 'POST',
        body: JSON.stringify({ text: routeText(text, files) }),
      });
      projectId = routed?.project_id ?? null;
      if (!projectId) throw new Error('no project to route to');
      draft.project_id = projectId;
      saveDraft(draft);
      updatePending(nonce, { project_id: projectId });
    }
    let attachments: UploadedAttachment[] | undefined;
    if (files.length > 0) {
      if (!deps.uploadAll) throw new Error('no uploader');
      attachments = await deps.uploadAll(projectId, files.map((f) => f.id));
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
    await deps.injectMessage(
      projectId, text, target, nonce,
      attachments && attachments.length > 0 ? attachments : undefined,
      sessionId, !!target,
    );
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
