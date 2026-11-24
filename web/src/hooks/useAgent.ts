// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback } from 'react';
import { api } from '../config';
import type { SubAgentTranscriptResult } from '../types';

interface ActionResult {
  status: string;
}

export interface NewSessionResult extends ActionResult {
  /** The id of the session. For a fresh-create (no session_id supplied) this
   *  is a brand-new id like `sess_xxxxxxxx`; the session materializes on the
   *  server only when the first message is injected into it. */
  session_id?: string;
  /** Opaque uuid for the minted session (debug / forward-compat). */
  session_uuid?: string;
}

export interface InjectResult extends ActionResult {
  /** True when inject_message auto-denied a pending approval because the
   *  user sent a new message while the agent was paused. */
  approval_dismissed?: boolean;
  /** tool_call_id of the approval that was dismissed (present when
   *  approval_dismissed is true). */
  dismissed_tool_call_id?: string;
  /** The request's nonce, echoed back for queued same-session messages. */
  nonce?: string;
  /** Spec 180: present only when the session had moved and the daemon
   *  delivered the message into this other project. */
  project_id?: string;
  /** Spec 195 (contract §3): a pinned send's dispatch. With `status:
   *  "queued_behind"` the message waits in `handle`'s queue at `position`. */
  dispatch_id?: string;
  handle?: string;
  position?: number;
  /** The composer queue block (2026-11-24): with `queued_behind`, the chat
   *  row is held back until the message is dispatched. */
  held?: boolean;
}

/**
 * GET /agents/{id}/pending — mobile/relay reconnect recovery (spec 006 §3f,
 * §11d). v3: entries carry the FULL `content` (not a preview) so recall can
 * rebuild the composer text. Every entry is a message queued in a running
 * session's own queue (`kind: 'same'`); spec 155 removed the cross-session
 * kind.
 */
export interface PendingEntry {
  session_id: string;
  nonce: string | null;
  content: string;
  position: number;
  /** `same`: behind this chat's own agent. `worker` (2026-11-24): a direct
   *  send held back behind an outside agent's current task. */
  kind: 'same' | 'worker';
  /** The typed words and the exact attachments (the queue block's row and
   *  its edit); absent on an entry the daemon has no record of them for. */
  text?: string;
  attachments?: Array<{ path: string; mime: string; size: number; local?: boolean }> | null;
  /** When it queued (the block's row order). */
  queued_at?: string;
  /** kind `worker` only. */
  dispatch_id?: string;
  handle?: string;
}

export interface PendingListResult {
  /** Legacy field: a running session of the project, or null. Unused. */
  holder: string | null;
  pending: PendingEntry[];
}

/**
 * POST /agents/{id}/pending/cancel response (spec 006 v3 §12 R2). `removed`
 * is the SERVER-AUTHORITATIVE signal for recall: true when a still-queued
 * entry was actually removed (safe to load its text into the composer), false
 * when the message had already dispatched / drained (do NOT load — it is
 * running; loading would double-send).
 */
export interface CancelPendingResult extends ActionResult {
  removed: boolean;
}

/** POST /agents/{id}/pending/send-now: `sent` is false when the message had
 *  already gone out or been removed (nothing is sent twice). */
export interface SendPendingNowResult extends ActionResult {
  sent: boolean;
}

export function useAgent() {
  const startAgent = useCallback(
    async (projectId: string, initialMessage?: string) => {
      return api<ActionResult>('/api/v2/agents/start', {
        method: 'POST',
        body: JSON.stringify({
          project_id: projectId,
          ...(initialMessage !== undefined && { initial_message: initialMessage }),
        }),
      });
    },
    [],
  );

  // The UI Stop button uses cancelMessage (/cancel). There is no user-facing
  // /stop endpoint: runtime-resource teardown is automatic via the daemon's
  // idle-eviction sweep. See TASK-idle-eviction-and-remove-stop.md.
  const cancelMessage = useCallback(async (projectId: string, sessionId?: string) => {
    return api<ActionResult>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/cancel`,
      {
        method: 'POST',
        body: JSON.stringify({
          ...(sessionId !== undefined && { session_id: sessionId }),
        }),
      },
    );
  }, []);

  const newSession = useCallback(async (projectId: string, sessionId?: string) => {
    return api<NewSessionResult>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/new-session`,
      {
        method: 'POST',
        body: JSON.stringify({
          ...(sessionId !== undefined && { session_id: sessionId }),
        }),
      },
    );
  }, []);

  // `pinned` (spec 074): true when `target` is the composer's sticky "Talking
  // to" pin — since spec 091 the only way the composer sets `target`. The
  // backend maps every `target` send to initiator="user_pinned"
  // (wake-suppressed dispatch; the management agent takes zero turns) and
  // accepts but ignores this field; the composer still sends it so an older
  // daemon keeps pinned semantics. Omitted from the body when false.
  const injectMessage = useCallback(
    async (
      projectId: string,
      content: string,
      target?: string,
      nonce?: string,
      // `local` (desktop app): `path` is the user's file by absolute path.
      attachments?: Array<{ path: string; mime: string; size: number; local?: boolean }>,
      sessionId?: string,
      pinned?: boolean,
      // Spec 132: the skill picked in the `/` menu; the backend re-resolves
      // `path` against the project's skill catalog and adds the block.
      skill?: { key: string; path: string },
      // Spec 146: this message answers the ask in the composer's tray.
      answering?: { id: string; words: string },
      // Spec 180: the Workbench route chose this project ('model' |
      // 'fallback'); the daemon stamps it only when this send creates the
      // session, and an older daemon ignores it.
      routedBy?: 'model' | 'agent' | 'fallback',
    ) => {
      return api<InjectResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/inject`,
        {
          method: 'POST',
          body: JSON.stringify({
            content,
            ...(target !== undefined && { target }),
            ...(nonce !== undefined && { nonce }),
            ...(attachments && attachments.length > 0 && { attachments }),
            ...(sessionId !== undefined && { session_id: sessionId }),
            ...(pinned && { pinned: true }),
            ...(skill && { skill }),
            ...(answering && { ask_id: answering.id, ask_answer: answering.words }),
            ...(routedBy && { routed_by: routedBy }),
          }),
        },
      );
    },
    [],
  );

  // Pending-input queue (spec 006). Cancel/dequeue a queued pending input for
  // session B (cross- OR same-session). Omit `nonce` to cancel all of B's
  // pending entries. v3: returns `{status, removed}` — `removed` tells the
  // caller whether a still-queued entry was actually removed (server-
  // authoritative recall, §12 R2).
  const cancelPendingInput = useCallback(
    async (projectId: string, sessionId: string, nonce?: string, dispatchId?: string) => {
      return api<CancelPendingResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/pending/cancel`,
        {
          method: 'POST',
          body: JSON.stringify({
            session_id: sessionId,
            ...(nonce !== undefined && { nonce }),
            ...(dispatchId !== undefined && { dispatch_id: dispatchId }),
          }),
        },
      );
    },
    [],
  );

  // The composer queue block's "send now" (2026-11-24): stop what is running
  // and send this queued message next. `nonce` names one queued behind this
  // chat's own agent, `dispatchId` one waiting behind an outside agent.
  const sendPendingNow = useCallback(
    async (projectId: string, sessionId: string, which: { nonce?: string; dispatchId?: string }) => {
      return api<SendPendingNowResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/pending/send-now`,
        {
          method: 'POST',
          body: JSON.stringify({
            session_id: sessionId,
            ...(which.nonce !== undefined && { nonce: which.nonce }),
            ...(which.dispatchId !== undefined && { dispatch_id: which.dispatchId }),
          }),
        },
      );
    },
    [],
  );

  // Pending-input queue (spec 006). Recovery fetch for reconnect / switch-back:
  // returns the slot holder plus all queued pending inputs for the project.
  const getPending = useCallback(async (projectId: string) => {
    return api<PendingListResult>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/pending`,
    );
  }, []);

  // Sub-agent fanout drill-in (spec 009 §0.5, Task 4's contract): read-only
  // transcript for one worker/cli handle. `session_id` scopes the lookup to
  // the parent chat session that dispatched it (same query param convention
  // as `/sub-agents/status`).
  const getSubAgentTranscript = useCallback(
    async (projectId: string, handle: string, sessionId?: string) => {
      const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      return api<SubAgentTranscriptResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/sub-agents/${encodeURIComponent(handle)}/transcript${qs}`,
      );
    },
    [],
  );

  const approveToolCall = useCallback(
    async (
      projectId: string,
      toolCallId: string,
      replyText?: string,
      approveAll?: boolean,
      sessionId?: string,
      // Spec 117 D5: the unsandboxed_run card's answer ("once" | "always" |
      // "disable_safe_mode") rides the existing approve route's
      // response_payload field, as MFA codes already do.
      responsePayload?: string,
    ) => {
      return api<ActionResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/approve`,
        {
          method: 'POST',
          body: JSON.stringify({
            tool_call_id: toolCallId,
            ...(replyText !== undefined && { reply_text: replyText }),
            ...(approveAll && { approve_all: true }),
            ...(sessionId !== undefined && { session_id: sessionId }),
            ...(responsePayload !== undefined && { response_payload: responsePayload }),
          }),
        },
      );
    },
    [],
  );

  const denyToolCall = useCallback(
    async (projectId: string, toolCallId: string, reason: string, sessionId?: string, stopTurn?: boolean) => {
      return api<ActionResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/deny`,
        {
          method: 'POST',
          body: JSON.stringify({
            tool_call_id: toolCallId,
            reason,
            ...(sessionId !== undefined && { session_id: sessionId }),
            ...(stopTurn && { stop_turn: true }),
          }),
        },
      );
    },
    [],
  );

  return { startAgent, cancelMessage, newSession, injectMessage, cancelPendingInput, sendPendingNow, getPending, getSubAgentTranscript, approveToolCall, denyToolCall };
}
