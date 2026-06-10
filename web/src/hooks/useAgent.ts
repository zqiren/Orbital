// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback } from 'react';
import { api } from '../config';

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
   */
  holding_session_id?: string;
  /** Human-readable description that accompanies `status: "slot_held"`. */
  message?: string;
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

  const injectMessage = useCallback(
    async (
      projectId: string,
      content: string,
      target?: string,
      nonce?: string,
      attachments?: Array<{ path: string; mime: string; size: number }>,
      sessionId?: string,
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
          }),
        },
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

  return { startAgent, cancelMessage, newSession, coldStartScan, injectMessage, approveToolCall, denyToolCall };
}
