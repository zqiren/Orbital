// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { transformChatHistory } from './chatTransform';
import type { ChatMessage } from '../types';

const TS = '2026-05-08T10:00:00Z';

function asst(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    role: 'assistant',
    content: null,
    source: 'management',
    timestamp: TS,
    ...overrides,
  };
}

function tool(tool_call_id: string, content: string): ChatMessage {
  return {
    role: 'tool',
    content,
    source: 'management',
    timestamp: TS,
    tool_call_id,
  };
}

function user(content: string): ChatMessage {
  return {
    role: 'user',
    content,
    source: 'user',
    timestamp: TS,
  };
}

describe('transformChatHistory — Variant A reasoning + per-call rows + inline results', () => {
  it('emits reasoning_block before agent_message for an assistant message with reasoning_content', () => {
    const messages: ChatMessage[] = [
      asst({
        content: 'Final answer text.',
        reasoning_content: 'I considered the options and chose A.',
      }),
    ];

    const items = transformChatHistory(messages);

    const reasoningIdx = items.findIndex(i => i.type === 'reasoning_block');
    const agentMsgIdx = items.findIndex(i => i.type === 'agent_message');

    expect(reasoningIdx).toBeGreaterThanOrEqual(0);
    expect(agentMsgIdx).toBeGreaterThanOrEqual(0);
    expect(reasoningIdx).toBeLessThan(agentMsgIdx);

    const reasoning = items[reasoningIdx];
    expect(reasoning.type).toBe('reasoning_block');
    if (reasoning.type === 'reasoning_block') {
      expect(reasoning.content).toBe('I considered the options and chose A.');
    }
  });

  it('emits one tool_call_row per tool_call followed immediately by its tool_result_inline', () => {
    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [
          { id: 'call-1', type: 'function', function: { name: 'read', arguments: '{"path":"a.txt"}' } },
          { id: 'call-2', type: 'function', function: { name: 'grep', arguments: '{"pattern":"foo"}' } },
        ],
      }),
      tool('call-1', 'contents of a.txt'),
      tool('call-2', 'Found 5 matches'),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map(i => i.type);

    expect(itemTypes).toEqual([
      'tool_call_row',
      'tool_result_inline',
      'tool_call_row',
      'tool_result_inline',
    ]);

    const r1 = items[0];
    const i1 = items[1];
    const r2 = items[2];
    const i2 = items[3];

    if (r1.type === 'tool_call_row') {
      expect(r1.tool_call_id).toBe('call-1');
      expect(r1.tool_name).toBe('read');
    }
    if (i1.type === 'tool_result_inline') {
      expect(i1.tool_call_id).toBe('call-1');
      expect(i1.content).toBe('contents of a.txt');
    }
    if (r2.type === 'tool_call_row') {
      expect(r2.tool_call_id).toBe('call-2');
      expect(r2.tool_name).toBe('grep');
    }
    if (i2.type === 'tool_result_inline') {
      expect(i2.tool_call_id).toBe('call-2');
      expect(i2.content).toBe('Found 5 matches');
    }
  });

  it('emits tool_call_rows with no tool_result_inline when results have not arrived (in-flight turn)', () => {
    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [
          { id: 'call-1', type: 'function', function: { name: 'read', arguments: '{"path":"a.txt"}' } },
        ],
      }),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map(i => i.type);

    expect(itemTypes).toEqual(['tool_call_row']);
    expect(itemTypes).not.toContain('tool_result_inline');
  });

  it('emits reasoning_block, agent_message, tool_call_row+tool_result_inline pairs in order for a full turn', () => {
    const messages: ChatMessage[] = [
      asst({
        content: 'Let me search for that.',
        reasoning_content: 'I should use grep.',
        tool_calls: [
          { id: 'call-1', type: 'function', function: { name: 'grep', arguments: '{"pattern":"foo"}' } },
        ],
      }),
      tool('call-1', 'Found 5 matches'),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map(i => i.type);

    expect(itemTypes).toEqual([
      'reasoning_block',
      'agent_message',
      'tool_call_row',
      'tool_result_inline',
    ]);
  });

  it('preserves user_message, agent_message (text-only assistant), and session_separator behavior', () => {
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
