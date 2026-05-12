// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { transformChatHistory, truncateResult } from './chatTransform';
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

describe('transformChatHistory — pure-text and session boundaries (unchanged)', () => {
  it('preserves user_message, agent_message, and session_separator behavior', () => {
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

  it('pure text exchange emits no agent_run capsule', () => {
    const messages: ChatMessage[] = [
      user('hi', TS),
      asst({ content: 'hello back', timestamp: TS2 }),
    ];

    const items = transformChatHistory(messages);

    expect(items.map(i => i.type)).toEqual(['user_message', 'agent_message']);
  });
});

describe('transformChatHistory — capsule grouping (updated for paired results)', () => {
  it('visible-text boundary closes capsule; tool_call_row carries paired result', () => {
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
      ]);
      const row = capsule.items[1];
      if (row.type === 'tool_call_row') {
        expect(row.result_content).toBe('contents of X');
        expect(row.result_status).toBe('received');
      }
    }
  });

  it('silent chaining: each tool_call_row carries its own paired result', () => {
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
    if (capsule.type === 'agent_run') {
      expect(capsule.items.map(x => x.type)).toEqual([
        'reasoning_block',
        'tool_call_row',
        'agent_message', // silent-turn marker
        'reasoning_block',
        'tool_call_row',
      ]);
      const row1 = capsule.items[1];
      const row2 = capsule.items[4];
      if (row1.type === 'tool_call_row' && row2.type === 'tool_call_row') {
        expect(row1.tool_call_id).toBe('c1');
        expect(row1.result_content).toBe('r1 result');
        expect(row1.result_status).toBe('received');
        expect(row2.tool_call_id).toBe('c2');
        expect(row2.result_content).toBe('r2 result');
        expect(row2.result_status).toBe('received');
      }
    }
  });

  it('multiple user messages: capsules are bounded; rows in each capsule pair correctly', () => {
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
  });
});

describe('transformChatHistory — tool result pairing (NEW)', () => {
  it('pairs tool result into the matching tool_call_row, no sibling tool_result_inline', () => {
    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [tc('c1', 'read', '{"path":"a.txt"}')],
        timestamp: TS,
      }),
      tool('c1', 'the result text', TS2),
    ];

    const items = transformChatHistory(messages);
    expect(items.length).toBe(1);
    const capsule = items[0];
    expect(capsule.type).toBe('agent_run');
    if (capsule.type === 'agent_run') {
      // No tool_result_inline anywhere
      for (const child of capsule.items) {
        expect(child.type).not.toBe('tool_result_inline');
      }
      const row = capsule.items.find(x => x.type === 'tool_call_row');
      expect(row).toBeDefined();
      if (row && row.type === 'tool_call_row') {
        expect(row.result_content).toBe('the result text');
        expect(row.result_status).toBe('received');
      }
    }
  });

  it('in-flight call: tool_call_row has null content and pending status', () => {
    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [tc('c1', 'shell')],
        timestamp: TS,
      }),
      // No following role:tool message — agent still working
    ];

    const items = transformChatHistory(messages);
    expect(items.length).toBe(1);
    const capsule = items[0];
    if (capsule.type === 'agent_run') {
      const row = capsule.items.find(x => x.type === 'tool_call_row');
      if (row && row.type === 'tool_call_row') {
        expect(row.result_content).toBeNull();
        expect(row.result_status).toBe('pending');
      }
    }
  });

  it('three parallel calls return out of order: pairing is by id, not position', () => {
    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [
          tc('a', 'read', '{"path":"a"}'),
          tc('b', 'read', '{"path":"b"}'),
          tc('c', 'read', '{"path":"c"}'),
        ],
        timestamp: TS,
      }),
      tool('b', 'result-B', TS2),
      tool('a', 'result-A', TS3),
      tool('c', 'result-C', TS4),
    ];

    const items = transformChatHistory(messages);
    const capsule = items[0];
    if (capsule.type === 'agent_run') {
      const rows = capsule.items.filter(x => x.type === 'tool_call_row');
      expect(rows.length).toBe(3);
      // Order is the original call order (a, b, c)
      const ids = rows.map(r => r.type === 'tool_call_row' ? r.tool_call_id : '');
      expect(ids).toEqual(['a', 'b', 'c']);
      const contents = rows.map(r => r.type === 'tool_call_row' ? r.result_content : null);
      expect(contents).toEqual(['result-A', 'result-B', 'result-C']);
    }
  });

  it('tool_result_inline DisplayItem type no longer exists', () => {
    // Compile-time + runtime check: no DisplayItem with type='tool_result_inline'
    // can be constructed anywhere in the transformed output.
    const messages: ChatMessage[] = [
      asst({ content: null, tool_calls: [tc('c1', 'read')], timestamp: TS }),
      tool('c1', 'r', TS2),
      asst({ content: 'done', timestamp: TS3 }),
    ];

    const items = transformChatHistory(messages);
    function* walk(arr: typeof items): Generator<{ type: string }> {
      for (const it of arr) {
        yield it;
        if (it.type === 'agent_run') yield* walk(it.items as typeof items);
      }
    }
    for (const it of walk(items)) {
      expect(it.type).not.toBe('tool_result_inline');
    }
  });

  it('empty result content: row gets empty string with received status', () => {
    const messages: ChatMessage[] = [
      asst({ content: null, tool_calls: [tc('c1', 'shell')], timestamp: TS }),
      tool('c1', '', TS2),
    ];

    const items = transformChatHistory(messages);
    const capsule = items[0];
    if (capsule.type === 'agent_run') {
      const row = capsule.items.find(x => x.type === 'tool_call_row');
      if (row && row.type === 'tool_call_row') {
        expect(row.result_content).toBe('');
        expect(row.result_status).toBe('received');
      }
    }
  });
});

describe('truncateResult', () => {
  it('returns short input as-is with no footer', () => {
    const r = truncateResult('hello world');
    expect(r.text).toBe('hello world');
    expect(r.footer).toBeNull();
  });

  it('returns empty input as-is with no footer', () => {
    const r = truncateResult('');
    expect(r.text).toBe('');
    expect(r.footer).toBeNull();
  });

  it('char-bound input: truncates at 500, footer reports total chars', () => {
    const long = 'x'.repeat(700); // 1 line, 700 chars
    const r = truncateResult(long);
    expect(r.text).toBe('x'.repeat(500) + '…');
    expect(r.footer).toBe('first 500 chars · result is 700 chars total');
  });

  it('line-bound input: truncates at 12 lines, footer reports total lines', () => {
    const lines = Array.from({ length: 20 }, (_, i) => `line${i}`).join('\n');
    const r = truncateResult(lines);
    const expectedFirst12 = Array.from({ length: 12 }, (_, i) => `line${i}`).join('\n');
    expect(r.text).toBe(expectedFirst12 + '…');
    expect(r.footer).toBe('first 12 lines · result is 20 lines total');
  });

  it('both bounds: char bound triggers when first 12 lines already exceed 500 chars', () => {
    // 12 lines of 50 chars each = 600 chars (without newlines) + 11 newlines = 611.
    const longLine = 'x'.repeat(50);
    const content = Array.from({ length: 14 }, () => longLine).join('\n');
    const r = truncateResult(content);
    expect(r.text).toBe(content.slice(0, 500) + '…');
    expect(r.footer).toBe(`first 500 chars · result is ${content.length} chars total`);
  });

  it('both bounds: line bound triggers when many short lines fit under 500 chars', () => {
    // 20 lines of 3 chars each = 60 chars + 19 newlines = 79. <500 chars but >12 lines.
    const content = Array.from({ length: 20 }, () => 'abc').join('\n');
    const r = truncateResult(content);
    const first12 = Array.from({ length: 12 }, () => 'abc').join('\n');
    expect(r.text).toBe(first12 + '…');
    expect(r.footer).toBe('first 12 lines · result is 20 lines total');
  });

  it('exactly at the bounds (500 chars, 12 lines): no truncation, no footer', () => {
    const content = 'a'.repeat(500);
    const r = truncateResult(content);
    expect(r.text).toBe(content);
    expect(r.footer).toBeNull();
  });
});
