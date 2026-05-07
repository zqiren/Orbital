// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { transformChatHistory } from './chatTransform';
import type { ChatMessage } from '../types';

const TS = '2026-05-08T10:00:00Z';
const TS2 = '2026-05-08T10:00:01Z';
const TS3 = '2026-05-08T10:00:02Z';
const TS4 = '2026-05-08T10:00:03Z';
const TS5 = '2026-05-08T10:00:04Z';
const TS6 = '2026-05-08T10:00:05Z';

function asst(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    role: 'assistant',
    content: null,
    source: 'management',
    timestamp: TS,
    ...overrides,
  };
}

function tool(tool_call_id: string, content: string, timestamp = TS): ChatMessage {
  return {
    role: 'tool',
    content,
    source: 'management',
    timestamp,
    tool_call_id,
  };
}

function user(content: string, timestamp = TS): ChatMessage {
  return {
    role: 'user',
    content,
    source: 'user',
    timestamp,
  };
}

function tc(id: string, name: string, args = '{}') {
  return { id, type: 'function' as const, function: { name, arguments: args } };
}

describe('transformChatHistory — Variant A reasoning + per-call rows + inline results (still works)', () => {
  it('preserves user_message, agent_message, and session_separator behavior for pure text', () => {
    const messages: ChatMessage[] = [
      { ...user('hi'), session_id: 's1' },
      asst({ content: 'hello back', session_id: 's1' }),
      asst({ content: 'next session msg', session_id: 's2' }),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map(i => i.type);

    expect(itemTypes).toEqual([
      'user_message',
      'agent_message',
      'session_separator',
      'agent_message',
    ]);
  });
});

describe('transformChatHistory — agent_run capsule grouping', () => {
  it('1. visible-text boundary: text emits inline, machinery folds into following capsule, then closes at next visible text', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: "I'll read X",
        reasoning_content: 'thinking about it',
        tool_calls: [tc('c1', 'read', '{"path":"X"}')],
        timestamp: TS2,
      }),
      tool('c1', 'contents of X', TS3),
      asst({ content: 'Done', timestamp: TS4 }),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map(i => i.type);

    expect(itemTypes).toEqual([
      'user_message',
      'agent_message',
      'agent_run',
      'agent_message',
    ]);

    const capsule = items[2];
    expect(capsule.type).toBe('agent_run');
    if (capsule.type === 'agent_run') {
      expect(capsule.items.map(x => x.type)).toEqual([
        'reasoning_block',
        'tool_call_row',
        'tool_result_inline',
      ]);
      expect(capsule.status).toBe('completed');
      expect(capsule.has_thinking).toBe(true);
      expect(capsule.tool_call_count_by_name).toEqual({ read: 1 });
    }

    const a1 = items[1];
    if (a1.type === 'agent_message') {
      expect(a1.content).toBe("I'll read X");
    }
    const a2 = items[3];
    if (a2.type === 'agent_message') {
      expect(a2.content).toBe('Done');
    }
  });

  it('2. silent chaining: two empty-content tool turns fold into one capsule with marker between turns', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: null,
        reasoning_content: 'reason 1',
        tool_calls: [tc('c1', 'read')],
        timestamp: TS2,
      }),
      tool('c1', 'r1 result', TS3),
      asst({
        content: null,
        reasoning_content: 'reason 2',
        tool_calls: [tc('c2', 'write')],
        timestamp: TS4,
      }),
      tool('c2', 'r2 result', TS5),
      asst({ content: 'Result is Y', timestamp: TS6 }),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map(i => i.type);

    expect(itemTypes).toEqual([
      'user_message',
      'agent_run',
      'agent_message',
    ]);

    const capsule = items[1];
    expect(capsule.type).toBe('agent_run');
    if (capsule.type === 'agent_run') {
      expect(capsule.items.map(x => x.type)).toEqual([
        'reasoning_block',
        'tool_call_row',
        'tool_result_inline',
        'agent_message',
        'reasoning_block',
        'tool_call_row',
        'tool_result_inline',
      ]);
      expect(capsule.status).toBe('completed');
      expect(capsule.tool_call_count_by_name).toEqual({ read: 1, write: 1 });
      // The middle agent_message marker should have empty content
      const marker = capsule.items[3];
      if (marker.type === 'agent_message') {
        expect(marker.content).toBe('');
      }
    }
  });

  it('3. mixed pattern: visible text closes capsule and machinery extends across silent middle step', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: 'Step 1',
        reasoning_content: 'plan',
        tool_calls: [tc('c1', 'read')],
        timestamp: TS2,
      }),
      tool('c1', 'r1', TS3),
      asst({
        content: null,
        // No reasoning on the silent middle step
        tool_calls: [tc('c2', 'edit')],
        timestamp: TS4,
      }),
      tool('c2', 'r2', TS5),
      asst({ content: 'Step 2 done', timestamp: TS6 }),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map(i => i.type);

    expect(itemTypes).toEqual([
      'user_message',
      'agent_message',
      'agent_run',
      'agent_message',
    ]);

    const capsule = items[2];
    if (capsule.type === 'agent_run') {
      expect(capsule.items.map(x => x.type)).toEqual([
        'reasoning_block',
        'tool_call_row',
        'tool_result_inline',
        'agent_message',
        'tool_call_row',
        'tool_result_inline',
      ]);
      expect(capsule.tool_call_count_by_name).toEqual({ read: 1, edit: 1 });
    }
  });

  it('4. in-flight turn with no closing visible item produces capsule with status:running', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: null,
        reasoning_content: 'r',
        tool_calls: [tc('c1', 'shell')],
        timestamp: TS2,
      }),
      tool('c1', 'output', TS3),
    ];

    const items = transformChatHistory(messages);

    expect(items.map(i => i.type)).toEqual(['user_message', 'agent_run']);
    const capsule = items[1];
    if (capsule.type === 'agent_run') {
      expect(capsule.status).toBe('running');
      expect(capsule.ended_at).toBe(null);
    }
  });

  it('5. pure text exchange emits no agent_run capsule', () => {
    const messages: ChatMessage[] = [
      user('hi', TS),
      asst({ content: 'hello back', timestamp: TS2 }),
    ];

    const items = transformChatHistory(messages);

    expect(items.map(i => i.type)).toEqual(['user_message', 'agent_message']);
  });

  it('6. reasoning-only capsule (no tool calls) emits with has_thinking and empty tool_call_count_by_name', () => {
    const messages: ChatMessage[] = [
      user('hi', TS),
      asst({
        content: 'hello',
        reasoning_content: 'a thought',
        timestamp: TS2,
      }),
      asst({ content: 'done', timestamp: TS3 }),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map(i => i.type);

    expect(itemTypes).toEqual([
      'user_message',
      'agent_message',
      'agent_run',
      'agent_message',
    ]);
    const capsule = items[2];
    if (capsule.type === 'agent_run') {
      expect(capsule.items.map(x => x.type)).toEqual(['reasoning_block']);
      expect(capsule.has_thinking).toBe(true);
      expect(capsule.tool_call_count_by_name).toEqual({});
    }
  });

  it('7. multiple user messages: capsules are bounded; no capsule spans across two user messages', () => {
    const messages: ChatMessage[] = [
      user('q1', TS),
      asst({
        content: null,
        reasoning_content: 'r1',
        tool_calls: [tc('c1', 'read')],
        timestamp: TS2,
      }),
      tool('c1', 'r1 result', TS3),
      user('q2', TS4),
      asst({
        content: null,
        reasoning_content: 'r2',
        tool_calls: [tc('c2', 'write')],
        timestamp: TS5,
      }),
      tool('c2', 'r2 result', TS6),
      asst({ content: 'done', timestamp: TS6 }),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map(i => i.type);

    expect(itemTypes).toEqual([
      'user_message',
      'agent_run',
      'user_message',
      'agent_run',
      'agent_message',
    ]);

    const cap1 = items[1];
    const cap2 = items[3];
    if (cap1.type === 'agent_run' && cap2.type === 'agent_run') {
      expect(cap1.tool_call_count_by_name).toEqual({ read: 1 });
      expect(cap2.tool_call_count_by_name).toEqual({ write: 1 });
      // Capsule ids must be distinct
      expect(cap1.capsule_id).not.toBe(cap2.capsule_id);
    }
  });
});
