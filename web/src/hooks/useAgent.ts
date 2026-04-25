// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback } from 'react';
import { api } from '../config';

interface ActionResult {
  status: string;
}

export interface InjectResult extends ActionResult {
  /** True when inject_message auto-denied a pending approval because the
   *  user sent a new message while the agent was paused. */
  approval_dismissed?: boolean;
  /** tool_call_id of the approval that was dismissed (present when
   *  approval_dismissed is true). */
  dismissed_tool_call_id?: string;
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

  // internal: full teardown, not used by UI
  const stopAgent = useCallback(async (projectId: string) => {
    return api<ActionResult>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/stop`,
      { method: 'POST' },
    );
  }, []);

  const cancelMessage = useCallback(async (projectId: string) => {
    return api<ActionResult>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/cancel`,
      { method: 'POST' },
    );
  }, []);

  const newSession = useCallback(async (projectId: string) => {
    return api<ActionResult>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/new-session`,
      { method: 'POST' },
    );
  }, []);

  const injectMessage = useCallback(
    async (projectId: string, content: string, target?: string, nonce?: string) => {
      return api<InjectResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/inject`,
        {
          method: 'POST',
          body: JSON.stringify({
            content,
            ...(target !== undefined && { target }),
            ...(nonce !== undefined && { nonce }),
          }),
        },
      );
    },
    [],
  );

  const approveToolCall = useCallback(
    async (projectId: string, toolCallId: string, replyText?: string, approveAll?: boolean) => {
      return api<ActionResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/approve`,
        {
          method: 'POST',
          body: JSON.stringify({
            tool_call_id: toolCallId,
            ...(replyText !== undefined && { reply_text: replyText }),
            ...(approveAll && { approve_all: true }),
          }),
        },
      );
    },
    [],
  );

  const denyToolCall = useCallback(
    async (projectId: string, toolCallId: string, reason: string) => {
      return api<ActionResult>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/deny`,
        {
          method: 'POST',
          body: JSON.stringify({
            tool_call_id: toolCallId,
            reason,
          }),
        },
      );
    },
    [],
  );

  return { startAgent, stopAgent, cancelMessage, newSession, injectMessage, approveToolCall, denyToolCall };
}
