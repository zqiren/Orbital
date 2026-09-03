// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useState } from 'react';
import { api } from '../config';
import type {
  ChatMessage,
  StreamDeltaEvent,
  ActivityEvent,
  StatusSummaryEvent,
  ApprovalRequestEvent,
  ApprovalResolvedEvent,
  WebSocketEvent,
} from '../types';

export interface UseChatHistoryOptions {
  /**
   * When provided, history is filtered to messages belonging to this session
   * (appends `?session_id=<sessionId>` to the fetch URL). When omitted, the
   * full merged history across all sessions is returned — unchanged behaviour.
   */
  sessionId?: string | null;
}

export function useChatHistory(options: UseChatHistoryOptions = {}) {
  const { sessionId } = options;
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  /**
   * Spec 078 §9.8 — the workspace panel needs to see EVERY activity event,
   * `tool_result` included, to follow what the agent is doing. Those are
   * deliberately still dropped from `messages` (a "Tool result received" row
   * has nothing to render), so instead of un-dropping them they are exposed
   * here as the last event seen.
   *
   * Caveat for consumers, verified against
   * `agent_os/daemon_v2/activity_translator.py:193`: the live `tool_result`
   * event carries only `{category, description: "Tool result received",
   * tool_name: <tool_call_id>}` — NO `_meta`, so no `screenshot_path`. The
   * browser screenshot is only readable from the persisted history row's
   * `_meta` (or, once it lands, the live CDP stream).
   */
  const [lastEvent, setLastEvent] = useState<ActivityEvent | null>(null);

  const loadHistory = useCallback(async (projectId: string) => {
    setLoading(true);
    try {
      let url = `/api/v2/agents/${encodeURIComponent(projectId)}/chat`;
      if (sessionId) {
        url += `?session_id=${encodeURIComponent(sessionId)}`;
      }
      const data = await api<ChatMessage[]>(url);
      setMessages(data);
      return data;
    } catch {
      setMessages([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  const mergeRealtimeEvent = useCallback((event: WebSocketEvent) => {
    switch (event.type) {
      case 'chat.stream_delta':
        handleStreamDelta(event);
        break;
      case 'agent.activity':
        handleActivity(event);
        break;
      case 'agent.status_summary':
        handleStatusSummary(event);
        break;
      case 'approval.request':
        handleApprovalRequest(event);
        break;
      case 'approval.resolved':
        handleApprovalResolved(event);
        break;
    }
  }, []);

  function handleStreamDelta(event: StreamDeltaEvent) {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (
        last &&
        last.role === 'assistant' &&
        last.source === event.source &&
        !last.tool_calls
      ) {
        const updated = { ...last, content: (last.content ?? '') + event.text };
        return [...prev.slice(0, -1), updated];
      }
      return [
        ...prev,
        {
          role: 'assistant' as const,
          content: event.text,
          source: event.source,
          timestamp: new Date().toISOString(),
        },
      ];
    });
  }

  function handleActivity(event: ActivityEvent) {
    // Published to subscribers BEFORE the tool_result drop below, so the panel
    // sees the full stream (spec 078 §9.8).
    setLastEvent(event);
    setMessages((prev) => {
      if (event.category === 'tool_result') return prev;
      const last = prev[prev.length - 1];
      if (
        last &&
        last.role === 'assistant' &&
        last.source === event.source &&
        last._status
      ) {
        const updated = { ...last, _status: event.description };
        return [...prev.slice(0, -1), updated];
      }
      return [
        ...prev,
        {
          role: 'assistant' as const,
          content: null,
          source: event.source,
          timestamp: event.timestamp,
          _status: event.description,
        },
      ];
    });
  }

  function handleStatusSummary(event: StatusSummaryEvent) {
    setMessages((prev) => [
      ...prev,
      {
        role: 'system' as const,
        content: event.summary,
        source: 'management',
        timestamp: event.timestamp,
      },
    ]);
  }

  function handleApprovalRequest(event: ApprovalRequestEvent) {
    setMessages((prev) => [
      ...prev,
      {
        role: 'system' as const,
        content: event.what,
        source: 'management',
        timestamp: new Date().toISOString(),
        _meta: {
          approval_request: true,
          tool_call_id: event.tool_call_id,
          tool_name: event.tool_name,
          tool_args: event.tool_args,
        },
      },
    ]);
  }

  function handleApprovalResolved(event: ApprovalResolvedEvent) {
    setMessages((prev) => [
      ...prev,
      {
        role: 'system' as const,
        content: `Tool call ${event.resolution}: ${event.tool_call_id}`,
        source: 'management',
        timestamp: new Date().toISOString(),
        _meta: {
          approval_resolved: true,
          tool_call_id: event.tool_call_id,
          resolution: event.resolution,
        },
      },
    ]);
  }

  const clearMessages = useCallback(() => {
    setMessages([]);
    setLastEvent(null);
  }, []);

  return { messages, loading, lastEvent, loadHistory, mergeRealtimeEvent, clearMessages };
}
