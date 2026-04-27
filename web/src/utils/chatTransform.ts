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
  | { type: 'activity_block'; activities: Activity[]; startTime: string; endTime: string; isHistorical?: boolean }
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

export function transformChatHistory(messages: ChatMessage[], workspace?: string): DisplayItem[] {
  const items: DisplayItem[] = [];
  let i = 0;
  let currentSessionId: string | undefined;

  while (i < messages.length) {
    const msg = messages[i];

    if (msg._compaction) {
      i++;
      continue;
    }

    if (msg.role === 'system') {
      if (msg._meta?.approval_request) {
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

    // Detect session boundary changes
    if (msg.session_id && currentSessionId && msg.session_id !== currentSessionId) {
      items.push({ type: 'session_separator', timestamp: msg.timestamp });
    }
    if (msg.session_id) {
      currentSessionId = msg.session_id;
    }

    if (msg.role === 'user') {
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
      // Strip ANSI escape codes and filter empty / "(no response)" content
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
      if (msg.content && (!msg.tool_calls || msg.tool_calls.length === 0)) {
        if (msg.source !== 'management' && msg.source !== 'user') {
          items.push({
            type: 'agent_message',
            content: msg.content,
            source: msg.source,
            timestamp: msg.timestamp,
          });
        } else {
          items.push({
            type: 'agent_message',
            content: msg.content,
            source: msg.source,
            timestamp: msg.timestamp,
          });
        }
        i++;
        continue;
      }

      if (msg.tool_calls && msg.tool_calls.length > 0) {
        const activities: Activity[] = [];
        const startTime = msg.timestamp;

        for (const tc of msg.tool_calls) {
          activities.push(toolCallToActivity(tc, msg.timestamp, msg, workspace));
        }

        let endTime = msg.timestamp;
        let j = i + 1;
        while (j < messages.length && messages[j].role === 'tool') {
          endTime = messages[j].timestamp;
          j++;
        }

        if (msg.content) {
          items.push({
            type: 'agent_message',
            content: msg.content,
            source: msg.source,
            timestamp: msg.timestamp,
          });
        }

        items.push({
          type: 'activity_block',
          activities,
          startTime,
          endTime,
        });

        i = j;
        continue;
      }

      if (msg.content) {
        items.push({
          type: 'agent_message',
          content: msg.content,
          source: msg.source,
          timestamp: msg.timestamp,
        });
      }
      i++;
      continue;
    }

    if (msg.role === 'tool') {
      i++;
      continue;
    }

    i++;
  }

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
