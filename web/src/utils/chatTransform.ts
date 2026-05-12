// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { ChatMessage, ToolCall, ActivityCategory } from '../types';

export interface Activity {
  id: string;
  category: ActivityCategory;
  description: string;
  toolName: string;
  timestamp: string;
}

export type DisplayItem =
  | { type: 'user_message'; content: string; timestamp: string; target?: string; isHistorical?: boolean }
  | { type: 'agent_message'; content: string; source: string; timestamp: string; isHistorical?: boolean }
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
    }
  | { type: 'session_separator'; timestamp: string }
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

function toolCallToActivity(tc: ToolCall, timestamp: string, message?: ChatMessage, workspace?: string): Activity {
  // Check for persisted description first (from JSONL _activity_descriptions)
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
      description = `Read ${args.path ?? args.file_path ?? name}`;
      break;
    case 'file_write':
      description = `Created ${args.path ?? args.file_path ?? name}`;
      break;
    case 'file_edit':
      description = `Edited ${args.path ?? args.file_path ?? name}`;
      break;
    case 'file_search':
      description = `Searching files: ${args.pattern ?? '?'}`;
      break;
    case 'content_search':
      description = `Searching for "${args.pattern ?? '?'}"${args.path ? ` in ${args.path}` : ''}`;
      break;
    case 'command_exec':
      if (workspace && typeof args.command === 'string' && containsExternalPaths(args.command, workspace)) {
        description = 'Ran: shell command (access restricted)';
      } else {
        description = `Ran: ${args.command ?? name}`;
      }
      break;
    case 'web_search':
      description = `Searched: ${args.query ?? name}`;
      break;
    case 'web_fetch':
      description = `Fetched: ${args.url ?? name}`;
      break;
    case 'request_access':
      description = `Requested access to ${args.path ?? name}`;
      break;
    case 'agent_message':
      description = `Messaged: @${args.handle ?? args.target ?? name}`;
      break;
    case 'browser_automation': {
      const action = args.action as string | undefined;
      switch (action) {
        case 'navigate': description = `Navigating to ${args.url ?? 'page'}`; break;
        case 'search': description = `Searching web for '${String(args.query ?? '?').slice(0, 50)}'`; break;
        case 'click': description = `Clicking element ${args.ref ?? args.selector ?? '?'}`; break;
        case 'screenshot': description = 'Taking screenshot'; break;
        case 'scroll': description = `Scrolling ${args.direction ?? 'down'}`; break;
        case 'snapshot': description = 'Reading page content'; break;
        case 'type': description = `Typing into element ${args.ref ?? '?'}`; break;
        case 'fill': description = 'Filling form fields'; break;
        case 'search_page': description = `Searching page for '${String(args.text ?? '?').slice(0, 30)}'`; break;
        case 'fetch': description = `Fetching ${String(args.url ?? '?').slice(0, 60)}`; break;
        case 'done': description = 'Browser task complete'; break;
        default: description = `Browser: ${action ?? 'unknown'}`; break;
      }
      break;
    }
    default:
      description = `Used tool: ${name}`;
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
}

export function transformChatHistory(messages: ChatMessage[], workspace?: string): DisplayItem[] {
  const items: DisplayItem[] = [];
  let i = 0;
  let currentSessionId: string | undefined;
  let capsuleCounter = 0;
  let currentCapsule: OpenCapsule | null = null;

  function tsToMs(ts: string): number {
    const n = Date.parse(ts);
    return Number.isFinite(n) ? n : 0;
  }

  function openCapsuleAt(ts: string): OpenCapsule {
    const ms = tsToMs(ts);
    return { items: [], startedAtMs: ms, endedAtMs: ms };
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
    });
    currentCapsule = null;
  }

  while (i < messages.length) {
    const msg = messages[i];

    if (msg._compaction) {
      i++;
      continue;
    }

    if (msg.role === 'system') {
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
      }
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
      }

      // Machinery (reasoning + tool_calls) flows into the capsule that
      // follows the just-emitted visible text, or extends the current one.
      if (reasoning || hasTools) {
        if (!currentCapsule) {
          currentCapsule = openCapsuleAt(msg.timestamp);
        }
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
            const activity = toolCallToActivity(tc, msg.timestamp, msg, workspace);
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

  // End of stream: any still-open capsule is in-flight.
  finalizeCapsule('running');

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
