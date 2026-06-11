// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { ChatMessage, ToolCall, ActivityCategory } from '../types';
import { translate } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

/**
 * Locale-aware translator for activity descriptions. Defaults to English so
 * callers (and unit tests) that omit it get the original English strings,
 * keeping output byte-identical when no locale is threaded in.
 */
export type ActivityTranslate = (
  key: StringKey,
  vars?: Record<string, string | number>,
) => string;
const EN_ACTIVITY: ActivityTranslate = (key, vars) => translate('en', key, vars);

export interface Activity {
  id: string;
  category: ActivityCategory;
  description: string;
  toolName: string;
  timestamp: string;
}

export type DisplayItem =
  | { type: 'user_message'; content: string; timestamp: string; target?: string; isHistorical?: boolean }
  | {
      type: 'agent_message';
      content: string;
      source: string;
      timestamp: string;
      isHistorical?: boolean;
      /**
       * When true, the renderer shows only the avatar + "agent · HH:MM" header
       * row with no body. Used to give a content-null (tool-only) assistant
       * turn a visible agent anchor above its capsule so the capsule does not
       * visually attach to the preceding user message. See FE-A3.
       */
      isHeaderOnly?: boolean;
    }
  | { type: 'sub_agent_message'; content: string; source: string; timestamp: string; isHistorical?: boolean }
  | { type: 'reasoning_block'; content: string; timestamp: string; turn_id: string; isHistorical?: boolean }
  | {
      type: 'tool_call_row';
      tool_name: string;
      target_description: string;
      tool_call_id: string;
      category: ActivityCategory;
      timestamp: string;
      result_content: string | null;
      result_status: 'pending' | 'received' | 'error';
      isHistorical?: boolean;
    }
  | {
      type: 'agent_run';
      capsule_id: string;
      status: 'running' | 'completed' | 'error' | 'stopped';
      items: DisplayItem[];
      tool_call_count_by_name: Record<string, number>;
      has_thinking: boolean;
      started_at: number;
      ended_at: number | null;
      isHistorical?: boolean;
      /**
       * When true, the renderer starts this capsule expanded so the reasoning
       * is visible without the user having to click the chevron. Set by the
       * transform when a content-null turn with reasoning opens the capsule.
       * See FE-A3 / FE-2.
       */
      defaultExpanded?: boolean;
    }
  | { type: 'session_separator'; timestamp: string }
  | {
      /**
       * A compact, one-line marker rendered in the chat flow for the daemon's
       * [Sub-agent] lifecycle system messages — start, message-sent, completion
       * (with summary), failure. Surfaces the persisted record that a
       * sub-agent ran; without this, the messages are silently dropped. See
       * FE-A2.
       */
      type: 'sub_agent_activity';
      action: 'started' | 'sent' | 'completed' | 'failed';
      handle: string;
      timestamp: string;
      /** Present for action='completed'. Trimmed; may be empty. */
      summary?: string;
      /** Present for action='sent'. The first chunk of the sent message. */
      preview?: string;
      /** Present for action='failed'. */
      error?: string;
      isHistorical?: boolean;
    }
  | {
      type: 'approval_card';
      what: string;
      tool_name: string;
      tool_call_id: string;
      tool_args: Record<string, unknown>;
      recent_activity: ChatMessage[];
      reasoning?: string;
      resolved?: 'approved' | 'denied';
    }
  | {
      type: 'agent_notify';
      title: string;
      body: string;
      urgency: 'high' | 'normal' | 'low';
      timestamp: string;
    }
  | {
      type: 'refresh_status';
      status: 'in_progress' | 'done' | 'failed' | 'skipped';
      trigger: 'turn_count' | 'agent_decided' | 'token_pressure';
      timestamp: string;
    }
  | {
      // Budget enforcement trip (P3-G). Built from a session row written by the
      // loop: {role:'system', source:'budget', event:'budget_blocked',
      // payload:{action,window,spend,limit,currency}}. Codes/numbers only — the
      // client localizes the message. (budget_threshold is WS-only; it never
      // lands in a session row, so it isn't represented here.)
      type: 'budget_event';
      event: 'budget_blocked';
      action: 'pause' | 'stop';
      window: 'daily' | 'weekly' | 'monthly' | 'total';
      spend: number;
      limit: number | null;
      currency: string;
      timestamp: string;
    };

const WINDOWS_PATH_RE = /[A-Za-z]:\\(?:Users|Windows|Program)[^\s"';&|>]*/gi;
const UNIX_PATH_RE = /\/(?:home|Users|etc|var|root)\/[^\s"';&|>]*/g;
const ENV_VAR_RE = /(?:\$HOME|\$USERPROFILE|%USERPROFILE%|%APPDATA%|%LOCALAPPDATA%)/gi;

export function containsExternalPaths(command: string, workspace: string): boolean {
  const normalizedWs = workspace.replace(/\\/g, '/').toLowerCase();
  const allPaths: string[] = [];

  for (const re of [WINDOWS_PATH_RE, UNIX_PATH_RE]) {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(command)) !== null) {
      allPaths.push(m[0]);
    }
  }

  ENV_VAR_RE.lastIndex = 0;
  if (ENV_VAR_RE.test(command)) {
    return true;
  }

  for (const p of allPaths) {
    const normalized = p.replace(/\\/g, '/').toLowerCase();
    if (!normalized.startsWith(normalizedWs)) {
      return true;
    }
  }

  return false;
}

const TOOL_NAME_TO_CATEGORY: Record<string, ActivityCategory> = {
  read: 'file_read',
  write: 'file_write',
  edit: 'file_edit',
  glob: 'file_search',
  grep: 'content_search',
  shell: 'command_exec',
  web_search: 'web_search',
  web_fetch: 'web_fetch',
  request_access: 'request_access',
  agent_message: 'agent_message',
  browser: 'browser_automation',
};

function toolCallToActivity(
  tc: ToolCall,
  timestamp: string,
  message?: ChatMessage,
  workspace?: string,
  tr: ActivityTranslate = EN_ACTIVITY,
): Activity {
  // Check for persisted description first (from JSONL _activity_descriptions).
  // These are backend-authored English strings; surfaced verbatim.
  const persisted = message?._activity_descriptions?.[tc.id];
  if (persisted) {
    const name = tc.function.name;
    const category = TOOL_NAME_TO_CATEGORY[name] ?? 'tool_use';
    return { id: tc.id, category, description: persisted, toolName: name, timestamp };
  }

  const name = tc.function.name;
  const category = TOOL_NAME_TO_CATEGORY[name] ?? 'tool_use';
  let args: Record<string, unknown> = {};
  try {
    args = JSON.parse(tc.function.arguments);
  } catch {
    // ignore
  }

  let description: string;
  switch (category) {
    case 'file_read':
      description = tr('activity.read', { path: String(args.path ?? args.file_path ?? name) });
      break;
    case 'file_write':
      description = tr('activity.created', { path: String(args.path ?? args.file_path ?? name) });
      break;
    case 'file_edit':
      description = tr('activity.edited', { path: String(args.path ?? args.file_path ?? name) });
      break;
    case 'file_search':
      description = tr('activity.searchingFiles', { pattern: String(args.pattern ?? '?') });
      break;
    case 'content_search':
      description = args.path
        ? tr('activity.searchingForIn', { pattern: String(args.pattern ?? '?'), path: String(args.path) })
        : tr('activity.searchingFor', { pattern: String(args.pattern ?? '?') });
      break;
    case 'command_exec':
      if (workspace && typeof args.command === 'string' && containsExternalPaths(args.command, workspace)) {
        description = tr('activity.ranRestricted');
      } else {
        description = tr('activity.ran', { command: String(args.command ?? name) });
      }
      break;
    case 'web_search':
      description = tr('activity.searched', { query: String(args.query ?? name) });
      break;
    case 'web_fetch':
      description = tr('activity.fetched', { url: String(args.url ?? name) });
      break;
    case 'request_access':
      description = tr('activity.requestedAccess', { path: String(args.path ?? name) });
      break;
    case 'agent_message':
      description = tr('activity.messaged', { handle: String(args.handle ?? args.target ?? name) });
      break;
    case 'browser_automation': {
      const action = args.action as string | undefined;
      switch (action) {
        case 'navigate': description = tr('activity.browser.navigate', { url: String(args.url ?? 'page') }); break;
        case 'search': description = tr('activity.browser.search', { query: String(args.query ?? '?').slice(0, 50) }); break;
        case 'click': description = tr('activity.browser.click', { ref: String(args.ref ?? args.selector ?? '?') }); break;
        case 'screenshot': description = tr('activity.browser.screenshot'); break;
        case 'scroll': description = tr('activity.browser.scroll', { direction: String(args.direction ?? 'down') }); break;
        case 'snapshot': description = tr('activity.browser.snapshot'); break;
        case 'type': description = tr('activity.browser.type', { ref: String(args.ref ?? '?') }); break;
        case 'fill': description = tr('activity.browser.fill'); break;
        case 'search_page': description = tr('activity.browser.searchPage', { text: String(args.text ?? '?').slice(0, 30) }); break;
        case 'fetch': description = tr('activity.browser.fetch', { url: String(args.url ?? '?').slice(0, 60) }); break;
        case 'done': description = tr('activity.browser.done'); break;
        default: description = tr('activity.browser.unknown', { action: String(action ?? 'unknown') }); break;
      }
      break;
    }
    default:
      description = tr('activity.usedTool', { name });
  }

  return {
    id: tc.id,
    category,
    description,
    toolName: name,
    timestamp,
  };
}

// Reasoning, tool_call_row, and empty-content agent_message markers
// may appear inside an agent_run.items array. Tool result content is
// carried on tool_call_row.result_content, paired by tool_call_id.
type CapsuleChild =
  | Extract<DisplayItem, { type: 'reasoning_block' }>
  | Extract<DisplayItem, { type: 'tool_call_row' }>
  | Extract<DisplayItem, { type: 'agent_message' }>;

interface OpenCapsule {
  items: CapsuleChild[];
  startedAtMs: number;
  endedAtMs: number;
  defaultExpanded: boolean;
}

// [Sub-agent] lifecycle markers persisted to JSONL by the lifecycle observer.
// Parsed and surfaced as sub_agent_activity items; without this they are
// silently dropped along with every other non-approval system message.
const SUB_AGENT_STARTED_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+started\b/;
const SUB_AGENT_SENT_RE = /^\[Sub-agent\]\s+Message sent to\s+([\w.-]+):\s*(.*)$/;
const SUB_AGENT_COMPLETED_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+completed\.\s*Summary:\s*([\s\S]*)$/;
const SUB_AGENT_FAILED_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+failed:\s*([\s\S]*)$/;

type SubAgentActivity = Extract<DisplayItem, { type: 'sub_agent_activity' }>;

function parseSubAgentSystemMessage(
  content: string,
  timestamp: string,
): SubAgentActivity | null {
  let m: RegExpMatchArray | null;
  if ((m = content.match(SUB_AGENT_STARTED_RE))) {
    return { type: 'sub_agent_activity', action: 'started', handle: m[1], timestamp };
  }
  if ((m = content.match(SUB_AGENT_SENT_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'sent',
      handle: m[1],
      preview: m[2].trim(),
      timestamp,
    };
  }
  if ((m = content.match(SUB_AGENT_COMPLETED_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'completed',
      handle: m[1],
      summary: m[2].trim(),
      timestamp,
    };
  }
  if ((m = content.match(SUB_AGENT_FAILED_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'failed',
      handle: m[1],
      error: m[2].trim(),
      timestamp,
    };
  }
  return null;
}

type BudgetEventItem = Extract<DisplayItem, { type: 'budget_event' }>;

/**
 * Parse a budget-trip session row into a timeline item, or null if the message
 * isn't a budget event. The row shape (written by the agent loop) is
 * `{role:'system', source:'budget', event:'budget_blocked', payload:{...}}`.
 * `event`/`payload` aren't on the ChatMessage type (they survive as raw JSONL
 * passthrough), so we read them off a narrow record cast. Codes/numbers only —
 * never trusts any daemon-side display string.
 */
function parseBudgetSystemMessage(msg: ChatMessage): BudgetEventItem | null {
  if (msg.source !== 'budget') return null;
  const raw = msg as unknown as {
    event?: unknown;
    payload?: Record<string, unknown>;
  };
  if (raw.event !== 'budget_blocked') return null;
  const p = raw.payload ?? {};
  const action = p.action === 'stop' ? 'stop' : 'pause';
  const window =
    p.window === 'weekly' || p.window === 'monthly' || p.window === 'total'
      ? p.window
      : 'daily';
  const spend = typeof p.spend === 'number' ? p.spend : 0;
  const limit = typeof p.limit === 'number' ? p.limit : null;
  const currency = typeof p.currency === 'string' ? p.currency : 'USD';
  return {
    type: 'budget_event',
    event: 'budget_blocked',
    action,
    window,
    spend,
    limit,
    currency,
    timestamp: msg.timestamp ?? '',
  };
}

export function transformChatHistory(
  messages: ChatMessage[],
  workspace?: string,
  tr: ActivityTranslate = EN_ACTIVITY,
): DisplayItem[] {
  const items: DisplayItem[] = [];
  let i = 0;
  let currentSessionId: string | undefined;
  let capsuleCounter = 0;
  let currentCapsule: OpenCapsule | null = null;

  function tsToMs(ts: string): number {
    const n = Date.parse(ts);
    return Number.isFinite(n) ? n : 0;
  }

  function openCapsuleAt(ts: string, defaultExpanded = false): OpenCapsule {
    const ms = tsToMs(ts);
    return { items: [], startedAtMs: ms, endedAtMs: ms, defaultExpanded };
  }

  function finalizeCapsule(status: 'running' | 'completed' | 'error' | 'stopped' = 'completed'): void {
    if (!currentCapsule || currentCapsule.items.length === 0) {
      currentCapsule = null;
      return;
    }
    const counts: Record<string, number> = {};
    let hasThinking = false;
    for (const it of currentCapsule.items) {
      if (it.type === 'tool_call_row') {
        counts[it.tool_name] = (counts[it.tool_name] ?? 0) + 1;
      } else if (it.type === 'reasoning_block') {
        hasThinking = true;
      }
    }
    items.push({
      type: 'agent_run',
      capsule_id: `cap:${currentCapsule.startedAtMs}:${capsuleCounter++}`,
      status,
      items: currentCapsule.items,
      tool_call_count_by_name: counts,
      has_thinking: hasThinking,
      started_at: currentCapsule.startedAtMs,
      ended_at: status === 'running' ? null : currentCapsule.endedAtMs,
      ...(currentCapsule.defaultExpanded ? { defaultExpanded: true } : {}),
    });
    currentCapsule = null;
  }

  while (i < messages.length) {
    const msg = messages[i];

    if (msg._compaction) {
      i++;
      continue;
    }

    // Synthetic sub-agent run injected by the /chat endpoint after a dispatch
    // marker (source === "sub_agent"). A sub-agent is a first-class agent, so
    // it renders with the SAME display items as the management agent: an
    // agent header, a collapsible `agent_run` tool capsule, and a response
    // bubble. The handle rides on `source` (drives both the name and the icon
    // in ChatMessage). Tool rows carry name + duration only — no args/results
    // are on the wire.
    if (msg.source === 'sub_agent') {
      finalizeCapsule();
      const handle = msg.sub_agent_handle ?? 'sub-agent';
      const startedAtMs = tsToMs(msg.timestamp);
      const toolRows = msg.sub_agent_tool_rows ?? [];

      // 1. Agent header (avatar + "<handle> · HH:MM"), no body.
      items.push({
        type: 'agent_message',
        content: '',
        source: handle,
        timestamp: msg.timestamp,
        isHeaderOnly: true,
      });

      // 2. Tool capsule — identical shape to a management `agent_run` so the
      //    existing capsule renderer (chevron, expand/collapse, tool rows,
      //    summary) works unchanged. Collapsed by default.
      if (toolRows.length > 0) {
        const counts: Record<string, number> = {};
        const capsuleItems: CapsuleChild[] = [];
        for (let r = 0; r < toolRows.length; r++) {
          const row = toolRows[r];
          const name = row.name || 'tool';
          counts[name] = (counts[name] ?? 0) + 1;
          const category = TOOL_NAME_TO_CATEGORY[name.toLowerCase()] ?? 'tool_use';
          capsuleItems.push({
            type: 'tool_call_row',
            tool_name: name,
            // No args/results available; surface the per-tool duration as the
            // row detail (honest: "Write · 1.2s").
            target_description: `${(row.duration_seconds ?? 0).toFixed(1)}s`,
            tool_call_id: `sub:${handle}:${msg.timestamp}:${r}`,
            category,
            timestamp: row.timestamp || msg.timestamp,
            result_content: null,
            result_status: 'received',
          });
        }
        const durationMs = Math.round((msg.sub_agent_duration ?? 0) * 1000);
        items.push({
          type: 'agent_run',
          capsule_id: `sub_agent:${handle}:${msg.timestamp}:${capsuleCounter++}`,
          status: 'completed',
          items: capsuleItems,
          tool_call_count_by_name: counts,
          has_thinking: false,
          started_at: startedAtMs,
          ended_at: startedAtMs + durationMs,
          defaultExpanded: false,
        });
      }

      // 3. Response bubble (same as a management agent_message), if non-empty.
      const respText = (msg.content ?? '').trim();
      if (respText) {
        items.push({
          type: 'agent_message',
          content: msg.content ?? '',
          source: handle,
          timestamp: msg.timestamp,
        });
      }

      i++;
      continue;
    }

    if (msg.role === 'system') {
      // Budget trip row (P3-G): {source:'budget', event:'budget_blocked',
      // payload:{action,window,spend,limit,currency}}. The session endpoint
      // passes the raw JSONL through, so `event`/`payload` ride as extra fields
      // not declared on ChatMessage — read them via a narrow cast.
      const budgetItem = parseBudgetSystemMessage(msg);
      if (budgetItem) {
        finalizeCapsule();
        items.push(budgetItem);
        i++;
        continue;
      }
      if (msg._meta?.approval_request) {
        finalizeCapsule();
        items.push({
          type: 'approval_card',
          what: msg.content ?? '',
          tool_name: (msg._meta.tool_name as string) ?? '',
          tool_call_id: (msg._meta.tool_call_id as string) ?? '',
          tool_args: (msg._meta.tool_args as Record<string, unknown>) ?? {},
          recent_activity: [],
          reasoning: msg._meta.reasoning as string | undefined,
          resolved: msg._meta.resolution as 'approved' | 'denied' | undefined,
        });
        i++;
        continue;
      }
      // [Sub-agent] lifecycle markers — surface them as compact timeline rows.
      // Finalize the open capsule first so the marker appears AFTER the
      // capsule that contains the originating dispatch tool call (chronologic
      // JSONL order), not inside or before it.
      const activity = parseSubAgentSystemMessage(msg.content ?? '', msg.timestamp);
      if (activity) {
        finalizeCapsule();
        items.push(activity);
      }
      // Other system messages (ping-pong guard, etc.) remain dropped — they
      // are not user-facing.
      i++;
      continue;
    }

    // Detect session boundary changes — close any open capsule first.
    if (msg.session_id && currentSessionId && msg.session_id !== currentSessionId) {
      finalizeCapsule();
      items.push({ type: 'session_separator', timestamp: msg.timestamp });
    }
    if (msg.session_id) {
      currentSessionId = msg.session_id;
    }

    if (msg.role === 'user') {
      finalizeCapsule();
      items.push({
        type: 'user_message',
        content: msg.content ?? '',
        timestamp: msg.timestamp,
        ...(msg.target && { target: msg.target }),
      });
      i++;
      continue;
    }

    if (msg.role === 'agent') {
      finalizeCapsule();
      if (msg.chunk_type === 'approval_request') {
        items.push({
          type: 'approval_card',
          what: msg.content ?? '',
          tool_name: (msg._meta?.tool_name as string) ?? '',
          tool_call_id: (msg._meta?.tool_call_id as string) ?? '',
          tool_args: (msg._meta?.tool_args as Record<string, unknown>) ?? {},
          recent_activity: [],
        });
        i++;
        continue;
      }
      const cleaned = (msg.content ?? '').replace(/\x1b\[[0-9;]*m/g, '').trim();
      if (cleaned && cleaned !== '(no response)') {
        items.push({
          type: 'sub_agent_message',
          content: cleaned,
          source: msg.source,
          timestamp: msg.timestamp,
        });
      }
      i++;
      continue;
    }

    if (msg.role === 'assistant') {
      const text = msg.content && msg.content.trim() ? msg.content : null;
      const reasoning = (msg.reasoning_content ?? '').trim();
      const hasTools = !!(msg.tool_calls && msg.tool_calls.length > 0);
      const msTime = tsToMs(msg.timestamp);

      if (text) {
        // Visible text closes any open capsule, then emits inline.
        finalizeCapsule();
        items.push({
          type: 'agent_message',
          content: text,
          source: msg.source,
          timestamp: msg.timestamp,
        });
      } else if (reasoning || hasTools) {
        // Silent assistant turn. If we're already inside a capsule with
        // prior items, insert an empty agent_message marker so the UI
        // can delimit between silent tool batches.
        if (currentCapsule && currentCapsule.items.length > 0) {
          currentCapsule.items.push({
            type: 'agent_message',
            content: '',
            source: msg.source,
            timestamp: msg.timestamp,
          });
          currentCapsule.endedAtMs = msTime;
        }
      } else {
        // Never-vanish guard: an assistant message with empty content AND no
        // reasoning AND no tool_calls would otherwise hit neither branch and
        // emit NOTHING — a silent permanent disappearance. Emit a minimal
        // header-only marker so the turn is always represented in the UI.
        finalizeCapsule();
        items.push({
          type: 'agent_message',
          content: '',
          source: msg.source,
          timestamp: msg.timestamp,
          isHeaderOnly: true,
        });
      }

      // Machinery (reasoning + tool_calls) flows into the capsule that
      // follows the just-emitted visible text, or extends the current one.
      if (reasoning || hasTools) {
        if (!currentCapsule) {
          // FE-A3: a content-null assistant turn would otherwise produce a
          // bare capsule with no agent identity above it, visually attaching
          // to the preceding user message. Emit an `agent_message` header
          // marker (rendered as avatar + "agent · HH:MM" only) so the capsule
          // is anchored to a visible agent turn. Only emitted when we have no
          // visible text to anchor on — `text` already covered that case.
          if (!text) {
            items.push({
              type: 'agent_message',
              content: '',
              source: msg.source,
              timestamp: msg.timestamp,
              isHeaderOnly: true,
            });
          }
          // transformChatHistory only ever runs over PERSISTED (completed)
          // history, so a capsule produced here represents a COMPLETED turn.
          // Per the locked product decision a completed reasoning turn renders
          // COLLAPSED (clean summary); it expands only while actively RUNNING,
          // which is handled at render time in ChatView (running ⇒ force-expand).
          // Therefore never force-expand here — open collapsed.
          currentCapsule = openCapsuleAt(msg.timestamp, false);
        }
        // (No force-expand when appending reasoning to an existing capsule
        // either — see the comment above; completed turns stay collapsed.)
        if (reasoning) {
          currentCapsule.items.push({
            type: 'reasoning_block',
            content: reasoning,
            timestamp: msg.timestamp,
            turn_id: msg.timestamp,
          });
          currentCapsule.endedAtMs = msTime;
        }
        if (hasTools) {
          for (const tc of msg.tool_calls!) {
            const activity = toolCallToActivity(tc, msg.timestamp, msg, workspace, tr);
            currentCapsule.items.push({
              type: 'tool_call_row',
              tool_name: activity.toolName,
              target_description: activity.description,
              tool_call_id: tc.id,
              category: activity.category,
              timestamp: msg.timestamp,
              result_content: null,
              result_status: 'pending',
            });
            currentCapsule.endedAtMs = msTime;
          }
        }
      }

      i++;
      continue;
    }

    if (msg.role === 'tool') {
      if (currentCapsule) {
        const tcId = msg.tool_call_id ?? '';
        const content = typeof msg.content === 'string' ? msg.content : '';
        // Pair by tool_call_id with the matching tool_call_row inside the
        // currently open capsule. Search from the back since out-of-order
        // arrival pairs more recent calls more cheaply.
        for (let k = currentCapsule.items.length - 1; k >= 0; k--) {
          const item = currentCapsule.items[k];
          if (item.type === 'tool_call_row' && item.tool_call_id === tcId) {
            currentCapsule.items[k] = {
              ...item,
              result_content: content,
              result_status: 'received',
            };
            break;
          }
        }
        currentCapsule.endedAtMs = tsToMs(msg.timestamp);
      }
      // Orphan tool results (no open capsule, or no matching call) are
      // dropped — the LLM never associated them with a visible call.
      i++;
      continue;
    }

    i++;
  }

  // End of stream: always finalize the trailing capsule as `completed`. The
  // transform output is now purely a function of persisted history; the
  // "running" status is applied at render time in ChatView (FE-A1) based on
  // the live agentStatus + viewingHolder, so this stays a pure transform.
  finalizeCapsule('completed');

  // Mark items from historical sessions
  let lastSepIndex = -1;
  for (let k = items.length - 1; k >= 0; k--) {
    if (items[k].type === 'session_separator') {
      lastSepIndex = k;
      break;
    }
  }
  if (lastSepIndex >= 0) {
    for (let k = 0; k <= lastSepIndex; k++) {
      const item = items[k];
      if (item.type !== 'session_separator' && item.type !== 'approval_card') {
        (item as { isHistorical?: boolean }).isHistorical = true;
      }
    }
  }

  return items;
}

const RESULT_CHAR_BOUND = 500;
const RESULT_LINE_BOUND = 12;

/**
 * Mechanical truncation for tool result content displayed in the UI.
 * Returns the content unchanged when it fits both bounds; otherwise cuts
 * at whichever bound triggers first and adds a footer noting the totals.
 */
export function truncateResult(
  content: string,
): { text: string; footer: string | null } {
  const totalChars = content.length;
  const lines = content.split('\n');
  const totalLines = lines.length;

  if (totalChars <= RESULT_CHAR_BOUND && totalLines <= RESULT_LINE_BOUND) {
    return { text: content, footer: null };
  }

  // Char count of the first N lines (joined with their newlines). If this
  // is < RESULT_CHAR_BOUND, the line bound triggers earlier in the stream.
  const firstNLinesText = lines.slice(0, RESULT_LINE_BOUND).join('\n');
  const firstNLinesLen = firstNLinesText.length;

  const charBoundFires = totalChars > RESULT_CHAR_BOUND;
  const lineBoundFires = totalLines > RESULT_LINE_BOUND;

  // When both fire, pick whichever produces the shorter (earlier) cut.
  // When only one fires, that one wins outright.
  if (charBoundFires && (!lineBoundFires || RESULT_CHAR_BOUND <= firstNLinesLen)) {
    return {
      text: content.slice(0, RESULT_CHAR_BOUND) + '…',
      footer: `first ${RESULT_CHAR_BOUND} chars · result is ${totalChars} chars total`,
    };
  }
  return {
    text: firstNLinesText + '…',
    footer: `first ${RESULT_LINE_BOUND} lines · result is ${totalLines} lines total`,
  };
}

/**
 * REST-recovery merge for a single recovered assistant row (the text-bubble
 * half of ChatView's /chat?limit=1 fallback).
 *
 * Identity contract (INVESTIGATION-two-speakers): the recovered item's
 * `source` is taken VERBATIM from the persisted row — exactly how
 * transformChatHistory stamps it (`source: msg.source`) — never via a
 * hardcoded default. The old inline version hardcoded source: 'assistant',
 * which rendered one management turn as two speakers ("Assistant" with the
 * Orbit glyph vs raw "assistant" with the grey AS monogram).
 *
 * Merge semantics, in order:
 * 1. Dedup: if an identical-content agent_message exists ANYWHERE in
 *    `prevItems`, return `prevItems` unchanged (the production duplicate
 *    slipped through a last-item-only check because a system message
 *    followed the WS-built bubble).
 * 2. Truncation repair: if the LAST item is an agent_message whose content
 *    is a shorter prefix-era variant (stream cut mid-delta), replace its
 *    content with the full persisted text.
 * 3. Insert: add the recovered message before any trailing user messages so
 *    a recovered reply lands before follow-up questions.
 */
export function mergeRecoveredAssistantMessage(
  prevItems: DisplayItem[],
  row: ChatMessage,
): DisplayItem[] {
  const text = row.content;
  if (!text) return prevItems;

  // 1. Dedup against the whole list, not just the tail.
  const exists = prevItems.some(
    (it) => it.type === 'agent_message' && it.content === text,
  );
  if (exists) return prevItems;

  // 2. Extend a truncated trailing message instead of duplicating it.
  const last = prevItems[prevItems.length - 1];
  if (
    last &&
    last.type === 'agent_message' &&
    text.length > last.content.length &&
    text.startsWith(last.content)
  ) {
    const updated = [...prevItems];
    updated[prevItems.length - 1] = { ...last, content: text };
    return updated;
  }

  // 3. Genuinely missing — insert before trailing user messages, identity
  //    taken verbatim from the row (same as normal ingest).
  const newMsg: DisplayItem = {
    type: 'agent_message',
    content: text,
    source: row.source,
    timestamp: row.timestamp ?? new Date().toISOString(),
  };
  let insertIdx = prevItems.length;
  while (insertIdx > 0 && prevItems[insertIdx - 1].type === 'user_message') {
    insertIdx--;
  }
  const updated = [...prevItems];
  updated.splice(insertIdx, 0, newMsg);
  return updated;
}
