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
 *      project is only known after step 2. A desktop-app chip attached by
 *      its local path uploads nothing: `uploadAll` passes it straight through;
 *   4. `POST /agents/{pid}/new-session` → `{session_id, pinned_target}`;
 *   5. the chat's own send: target = the picked agent (null for Orbital),
 *      else the minted `pinned_target`. The targeted first send stamps the
 *      session's pin; the Workbench never PATCHes it, so a per-task pick never
 *      moves the project's sticky pin. The attachments ride as the inject's
 *      `attachments` payload and the daemon writes the `<attached_files>`
 *      prefix, exactly as the chat composer sends them.
 *
 * Spec 180 §5.2 — while 自动 picks, the pending card shows a shimmering 自动
 * where the name will land, and the user may decide first. One decision per
 * send, the first one wins:
 *   - the route answer (`by: model | fallback` is stamped on the session as
 *     `routed_by`, so its card carries ✦);
 *   - a project picked on the card (`choosePending`): the send goes there at
 *     once, unmarked; a route answer arriving later is ignored;
 *   - ✕ (`unsendPending`): nothing is minted or uploaded, the card leaves and
 *     the text returns to the composer, with no error line.
 * A route answer that arrives while the card's picker is open waits until it
 * closes (`holdPending`): a pick there wins, a close without one applies it.
 * The cutoff is the route answer: after it the pending card's project only
 * shows (the card that carries the session is the one that moves it).
 *
 * Batch 1118 — a pick from the `/` menu (a worker command such as
 * `/compact`, a skill chip, `/automate`) skips step 2: it goes to the picked
 * project, or on 自动 to `fallbackProjectId` (Quick Tasks, whose list the menu
 * showed). The routing model never reads a command as a task.
 *
 * Routing that finds no project (2026-11-07: the call failed after its one
 * retry, `by: "failed"`) is not a send: nothing is minted or uploaded, the
 * text comes back and the outcome says `unrouted`, so the composer shows its
 * one error line and opens the project picker.
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
import { codedErrorOf } from '../../utils/providerError';
import type { SessionListEntry } from '../../types';
import type { UploadedAttachment } from '../../hooks/useAttachments';
import {
  addPending, refreshFeed, removePending, restoreComposer, updatePending,
} from '../../hooks/useWorkbenchFeed';
import { skillSendParts, type SkillCatalogEntry } from '../../lib/skill-catalog';
import { buildInlineSkillBlock, buildSkillBlock } from '../../lib/skill-parsing';

/**
 * Batch 1118: does the text start with a command (`/compact`, `/review 12`)?
 * The picker's rule (`slashQuery`, then whitespace or the end), the same one
 * the daemon uses to send a command alone: `/Users/me/a.txt look` is text.
 */
export function startsWithCommand(text: string): boolean {
  return /^\/[^\s/]+(\s|$)/.test(text);
}

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
    skill?: { key: string; path: string }, answering?: undefined,
    /** Spec 180: the route chose the project; stamped when the send creates the session. */
    routedBy?: RoutedBy,
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
  /** The skill or built-in picked in the `/` menu (a chip). */
  skill?: SkillCatalogEntry | null;
  /** Where a pick from the `/` menu goes when the project is 自动: the
   *  project whose list the menu showed (Quick Tasks). */
  fallbackProjectId?: string | null;
}

/** The agent a keyless send could not go to, and why (the notice's input). */
export interface AgentBlocked {
  agent: string;
  agentName?: string;
  /** Absent = `agent_signed_out`. */
  code?: 'agent_not_ready';
  reason?: string;
  detail?: string;
}

export type SendOutcome =
  | { ok: true; projectId: string; sessionId: string }
  /** `unrouted`: 自动 found no project; the user picks one. `unsent`: ✕ while
   *  picking — the text is back, nothing went out, nothing to report. */
  | {
    ok: false; restored: boolean; unrouted?: boolean; unsent?: boolean;
    /** Batch 1123: no key and the project's agent is signed out — offer its
     *  sign-in. 2026-11-25: or signed in but not ready (`code`
     *  `agent_not_ready`, with the daemon's reason and the vendor's words). */
    signedOut?: AgentBlocked;
  };

/** `agent`: no key, a signed-in Claude Code or Codex chose (batch 1123). */
export type RoutedBy = 'model' | 'agent' | 'fallback';

type Decision =
  | { kind: 'pick'; projectId: string }
  | { kind: 'unsend' }
  | { kind: 'route'; projectId: string | null; by: unknown }
  | { kind: 'routeError' };

interface Decider {
  decide: (d: Decision) => void;
  hold: (open: boolean) => void;
}

/** The sends still picking, by nonce. */
const deciders = new Map<string, Decider>();

/** A route answer that waited for the card's picker applies this long after
 *  the picker closed without a pick (the prototype's beat). */
export const HOLD_RELEASE_MS = 400;

/** A project picked on a picking card: the send goes there, unmarked. */
export function choosePending(nonce: string, projectId: string): void {
  deciders.get(nonce)?.decide({ kind: 'pick', projectId });
}

/** ✕ on a picking card: the send stops before anything is minted. */
export function unsendPending(nonce: string): void {
  deciders.get(nonce)?.decide({ kind: 'unsend' });
}

/** The card's picker opened (true) or closed (false) while picking. */
export function holdPending(nonce: string, open: boolean): void {
  deciders.get(nonce)?.hold(open);
}

/** The first of: the route answer (held while the card's picker is open),
 *  a pick on the card, ✕. */
function decide(nonce: string, route: Promise<Decision>): Promise<Decision> {
  return new Promise<Decision>((resolve) => {
    let done = false;
    let held = false;
    let due: Decision | null = null;
    const finish = (d: Decision) => {
      if (done) return;
      done = true;
      deciders.delete(nonce);
      resolve(d);
    };
    deciders.set(nonce, {
      decide: finish,
      hold: (open) => {
        held = open;
        if (!open && due) {
          const waiting = due;
          setTimeout(() => { if (!held) finish(waiting); }, HOLD_RELEASE_MS);
        }
      },
    });
    void route.then((d) => {
      if (held) due = d;
      else finish(d);
    });
  });
}

export async function sendWorkbenchTask(choice: SendChoice, deps: SendDeps): Promise<SendOutcome> {
  const text = choice.text.trim();
  const files = choice.files ?? [];
  const skill = choice.skill ?? null;
  const nonce = newNonce();
  // Batch 1118: a pick from the `/` menu (a command typed or prefilled, a
  // skill chip, `/automate`) is never routing input: the routing model would
  // read `/compact` as a task. It goes to the project whose list was shown —
  // the picked one, or on 自动 the fallback (Quick Tasks, the route's own
  // fallback); with none it is unrouted like a route that found nothing.
  const direct = startsWithCommand(text) || skill !== null;
  let projectId = choice.projectId ?? (direct ? choice.fallbackProjectId ?? null : null);
  const draft: Draft = { text, nonce, project_id: projectId, session_id: null };
  saveDraft(draft);
  addPending({
    // An attachment-only task is named by its files on the pending card.
    nonce, text: text || (skill ? skill.name : files.map((f) => f.filename).join(', ')),
    project_id: projectId, session_id: null,
    agent: choice.agent, created_at: Date.now(),
    routing: !projectId && !direct,
  });

  let sessionId: string | null = null;
  let unrouted = false;
  let routedBy: RoutedBy | undefined;
  try {
    if (!projectId && direct) {
      unrouted = true;
      throw new Error('no project for a command');
    }
    if (!projectId) {
      const route = api<{ project_id?: string | null; by?: unknown }>('/api/v2/workbench/route', {
        method: 'POST',
        body: JSON.stringify({ text: routeText(text, files) }),
      }).then(
        (r): Decision => ({ kind: 'route', projectId: r?.project_id ?? null, by: r?.by }),
        (): Decision => ({ kind: 'routeError' }),
      );
      const d = await decide(nonce, route);
      if (d.kind === 'unsend') {
        clearDraft();
        removePending(nonce);
        restoreComposer(text);
        return { ok: false, restored: true, unsent: true };
      }
      if (d.kind === 'routeError') throw new Error('route failed');
      if (d.kind === 'pick') {
        projectId = d.projectId;
      } else {
        projectId = d.projectId;
        if (!projectId) {
          unrouted = true;
          throw new Error('no project to route to');
        }
        if (d.by === 'model' || d.by === 'agent' || d.by === 'fallback') routedBy = d.by;
      }
      draft.project_id = projectId;
      saveDraft(draft);
      updatePending(nonce, { project_id: projectId, routing: false, routed_by: routedBy ?? null });
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
    // Spec 137: `/automate` goes to the manager whatever the pick — only it
    // has the trigger tools (the composer offers it only when there is one).
    const target = choice.agent === 'orbital' || skill?.builtin?.managerOnly
      ? undefined
      : (choice.agent ?? minted.pinned_target ?? undefined);
    // The chat composer's skill parts: a catalog skill is a payload the
    // daemon re-resolves, a built-in an inline block in the text.
    const parts = skillSendParts(skill, buildSkillBlock, buildInlineSkillBlock);
    await deps.injectMessage(
      projectId, parts.contentPrefix + text, target, nonce,
      attachments && attachments.length > 0 ? attachments : undefined,
      sessionId, !!target, parts.payload, undefined, routedBy,
    );
    clearDraft();
    void refreshFeed();
    return { ok: true, projectId, sessionId };
  } catch (err) {
    const coded = codedErrorOf(err);
    const blocked = coded?.code === 'agent_signed_out' || coded?.code === 'agent_not_ready';
    const signedOut: AgentBlocked | undefined = blocked && coded?.agent
      ? {
        agent: coded.agent,
        ...(coded.agent_name ? { agentName: coded.agent_name } : {}),
        ...(coded.code === 'agent_not_ready' ? { code: 'agent_not_ready' as const } : {}),
        ...(coded.reason ? { reason: coded.reason } : {}),
        ...(coded.detail ? { detail: coded.detail } : {}),
      }
      : undefined;
    clearDraft();
    void refreshFeed();
    if (projectId && sessionId && (await sessionExists(projectId, sessionId))) {
      // The text was persisted: the session's own row shows the failure.
      return { ok: true, projectId, sessionId };
    }
    removePending(nonce);
    restoreComposer(text);
    if (unrouted) return { ok: false, restored: true, unrouted: true };
    return signedOut ? { ok: false, restored: true, signedOut } : { ok: false, restored: true };
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
