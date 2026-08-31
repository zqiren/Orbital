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
  /**
   * Slot-enforcement Phase 1 (Track J): present when the backend returned
   * 202 with `status: "slot_held"` because the requested session_id does
   * not currently hold the project's active-loop slot. The frontend should
   * render SlotHeldNotice and offer cancel-and-resend.
   *
   * Pending-input queue (spec 006): also present on the new happy-path 202
   * `status: "queued_pending_slot"` — the holder (A) of the slot, shown in
   * the "Waiting for {holder}…" affordance.
   */
  holding_session_id?: string;
  /** Human-readable description that accompanies `status: "slot_held"`. */
  message?: string;
  /**
   * Pending-input queue (spec 006) 202 `queued_pending_slot` fields: the
   * message was accepted and queued (NOT rejected) and will auto-dispatch
   * when the slot frees. The optimistic bubble is KEPT and a waiting
   * affordance is shown (see ChatView `pendingInputs`).
   */
  queued_session_id?: string;
  nonce?: string;
  position?: number;
}

/**
 * GET /agents/{id}/pending — mobile/relay reconnect recovery (spec 006 §3f,
 * §11d). v3: entries carry the FULL `content` (not a preview) so recall can
 * rebuild the composer text, and a `kind` discriminating cross-session
 * (`_pending_inject`, slot held by another session) from same-session
 * (`session._queue`, this session's turn mid-flight) queued messages.
 */
export interface PendingEntry {
  session_id: string;
  nonce: string | null;
  content: string;
  position: number;
  kind: 'cross' | 'same';
}

export interface PendingListResult {
  /** The session currently holding the slot, or null if free. */
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

  const coldStartScan = useCallback(async (projectId: string) => {
    return api<{ status: string; session_id?: string }>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/cold-start-scan`,
      { method: 'POST' },
    );
  }, []);

  // `pinned` (spec 074): true when `target` came from the composer's sticky
  // "Talking to" dropdown (the session pin) rather than a leading @mention —
  // the backend maps it to initiator="user_pinned" (wake-suppressed dispatch;
  // the management agent takes zero turns). Omitted from the body when false.
  const injectMessage = useCallback(
    async (
      projectId: string,
      content: string,
      target?: string,
      nonce?: string,
      attachments?: Array<{ path: string; mime: string; size: number }>,
      sessionId?: string,
      pinned?: boolean,
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
    async (projectId: string, sessionId: string, nonce?: string) => {
      return api<CancelPendingResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/pending/cancel`,
        {
          method: 'POST',
          body: JSON.stringify({
            session_id: sessionId,
            ...(nonce !== undefined && { nonce }),
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

  return { startAgent, cancelMessage, newSession, coldStartScan, injectMessage, cancelPendingInput, getPending, getSubAgentTranscript, approveToolCall, denyToolCall };
}
