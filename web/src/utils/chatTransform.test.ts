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

    // FE-A3: a content-null assistant turn now emits an `agent_message`
    // header marker before opening its capsule, giving the capsule a visible
    // agent anchor (avatar + sender · HH:MM) so it does not visually attach
    // to the preceding user message.
    expect(itemTypes).toEqual([
      'user_message',
      'agent_message',
      'agent_run',
      'agent_message',
    ]);
    const header = items[1];
    expect(header.type).toBe('agent_message');
    if (header.type === 'agent_message') {
      expect(header.isHeaderOnly).toBe(true);
      expect(header.content).toBe('');
    }

    const capsule = items[2];
    if (capsule.type === 'agent_run') {
      expect(capsule.items.map(x => x.type)).toEqual([
        'reasoning_block',
        'tool_call_row',
        'agent_message', // silent-turn marker (inside the capsule)
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

    // FE-A3: each content-null assistant turn emits an `agent_message`
    // header marker before its capsule.
    expect(itemTypes).toEqual([
      'user_message',
      'agent_message',
      'agent_run',
      'user_message',
      'agent_message',
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
    // FE-A3: header marker + capsule.
    expect(items.length).toBe(2);
    expect(items[0].type).toBe('agent_message');
    const capsule = items[1];
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
    // FE-A3: header marker + capsule.
    expect(items.length).toBe(2);
    const capsule = items[1];
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
    // FE-A3: header marker precedes the capsule for content-null turns.
    const capsule = items.find(i => i.type === 'agent_run');
    expect(capsule).toBeDefined();
    if (capsule && capsule.type === 'agent_run') {
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
    // FE-A3: header marker precedes the capsule.
    const capsule = items.find(i => i.type === 'agent_run');
    if (capsule && capsule.type === 'agent_run') {
      const row = capsule.items.find(x => x.type === 'tool_call_row');
      if (row && row.type === 'tool_call_row') {
        expect(row.result_content).toBe('');
        expect(row.result_status).toBe('received');
      }
    }
  });
});

function system(content: string, timestamp = TS): ChatMessage {
  return {
    role: 'system',
    content,
    source: 'management',
    timestamp,
  };
}

describe('transformChatHistory — FE-1 transform-once across page boundaries', () => {
  // The FE-1 fix accumulates raw messages across pages and transforms the FULL
  // concatenated list once. These tests assert the transform-once property:
  // a tool-call/tool-result pair split across what would be a page boundary is
  // paired correctly when the full raw list is transformed together.

  it('pairs a tool result whose tool_call is in the prior page (no orphan drop)', () => {
    // Simulate a 100-message conversation. The page boundary at limit=50 would
    // split the assistant tool_call (msg 49) from its tool result (msg 51).
    // page1 = messages 0..49, page2 = messages 50..99.
    const page1: ChatMessage[] = [];
    for (let n = 0; n < 49; n++) {
      page1.push(user(`q${n}`, TS));
      page1.push(asst({ content: `a${n}`, timestamp: TS2 }));
    }
    // msg 49 (end of page 1): assistant emits a tool call, no result yet here.
    page1.push(
      asst({
        content: null,
        reasoning_content: 'reading the file across the seam',
        tool_calls: [tc('seam', 'read', '{"path":"seam.txt"}')],
        timestamp: TS3,
      }),
    );

    const page2: ChatMessage[] = [
      // msg 51 (start of page 2): the matching tool result.
      tool('seam', 'SEAM RESULT CONTENT', TS4),
      asst({ content: 'continuing', timestamp: TS5 }),
    ];

    // Transform-once on the full concatenated list (page1 + page2).
    const items = transformChatHistory([...page1, ...page2]);

    // Walk every capsule child looking for the seam tool_call_row and assert
    // its result was paired in (not dropped, not pending).
    let found: { result_content: string | null; result_status: string } | null = null;
    for (const it of items) {
      if (it.type === 'agent_run') {
        for (const child of it.items) {
          if (child.type === 'tool_call_row' && child.tool_call_id === 'seam') {
            found = { result_content: child.result_content, result_status: child.result_status };
          }
        }
      }
    }
    expect(found).not.toBeNull();
    expect(found!.result_status).toBe('received');
    expect(found!.result_content).toBe('SEAM RESULT CONTENT');
  });

  it('boundary tool_call resolves to received (not stuck pending) once both pages are present', () => {
    // page1 ends with an assistant tool_call; page2 begins with its result.
    const page1: ChatMessage[] = [
      user('start', TS),
      asst({
        content: null,
        tool_calls: [tc('b1', 'shell', '{"command":"ls"}')],
        timestamp: TS2,
      }),
    ];
    const page2: ChatMessage[] = [tool('b1', 'file listing', TS3)];

    // Transform-once on the full concatenated list.
    const items = transformChatHistory([...page1, ...page2]);
    let row: { result_status: string; result_content: string | null } | null = null;
    for (const it of items) {
      if (it.type === 'agent_run') {
        for (const child of it.items) {
          if (child.type === 'tool_call_row' && child.tool_call_id === 'b1') {
            row = { result_status: child.result_status, result_content: child.result_content };
          }
        }
      }
    }
    expect(row).not.toBeNull();
    expect(row!.result_status).toBe('received');
    expect(row!.result_content).toBe('file listing');
  });
});

describe('transformChatHistory — FE-A1 trailing capsule status is render-time, not transform-time', () => {
  // The `isActivelyRunning` transform parameter was removed (FE-A1). The
  // trailing capsule is ALWAYS finalized as `completed`; ChatView upgrades
  // the last capsule to `running` at render time when the viewed session is
  // actively executing. This keeps the transform a pure function of
  // persisted history (no status-flip re-runs that wipe the live overlay).

  it('list ending on a system message finalizes the trailing capsule as completed', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: null,
        reasoning_content: 'reason',
        tool_calls: [tc('c1', 'read')],
        timestamp: TS2,
      }),
      tool('c1', 'r1', TS3),
      system('Repetitive action detected.', TS4),
    ];

    const items = transformChatHistory(messages);
    const capsule = items.find(i => i.type === 'agent_run');
    expect(capsule).toBeDefined();
    if (capsule && capsule.type === 'agent_run') {
      expect(capsule.status).toBe('completed');
      expect(capsule.ended_at).not.toBeNull();
    }
  });

  it('trailing capsule is `completed` regardless of session activity (transform is pure)', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: null,
        reasoning_content: 'reason',
        tool_calls: [tc('c1', 'read')],
        timestamp: TS2,
      }),
      tool('c1', 'r1', TS3),
    ];

    const items = transformChatHistory(messages);
    const capsule = items.find(i => i.type === 'agent_run');
    expect(capsule).toBeDefined();
    if (capsule && capsule.type === 'agent_run') {
      expect(capsule.status).toBe('completed');
      expect(capsule.ended_at).not.toBeNull();
    }
  });

  it('list ending on a tool result finalizes as completed when not actively running', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: null,
        tool_calls: [tc('c1', 'read')],
        timestamp: TS2,
      }),
      tool('c1', 'r1', TS3),
    ];

    const items = transformChatHistory(messages, undefined);
    const capsule = items.find(i => i.type === 'agent_run');
    if (capsule && capsule.type === 'agent_run') {
      expect(capsule.status).toBe('completed');
    }
  });

  it('list ending on a visible assistant text closes the capsule as completed regardless of the flag', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: null,
        reasoning_content: 'reason',
        tool_calls: [tc('c1', 'read')],
        timestamp: TS2,
      }),
      tool('c1', 'r1', TS3),
      asst({ content: 'all done', timestamp: TS4 }),
    ];

    // Even with isActivelyRunning=true, the capsule was already closed by the
    // trailing visible-text assistant message, so it is completed.
    const itemsRunning = transformChatHistory(messages, undefined);
    const capsuleRunning = itemsRunning.find(i => i.type === 'agent_run');
    if (capsuleRunning && capsuleRunning.type === 'agent_run') {
      expect(capsuleRunning.status).toBe('completed');
    }

    const itemsIdle = transformChatHistory(messages, undefined);
    const capsuleIdle = itemsIdle.find(i => i.type === 'agent_run');
    if (capsuleIdle && capsuleIdle.type === 'agent_run') {
      expect(capsuleIdle.status).toBe('completed');
    }
  });

  it('defaults isActivelyRunning to false (idle behavior) when omitted', () => {
    const messages: ChatMessage[] = [
      asst({ content: null, tool_calls: [tc('c1', 'shell')], timestamp: TS }),
    ];
    const items = transformChatHistory(messages);
    const capsule = items.find(i => i.type === 'agent_run');
    if (capsule && capsule.type === 'agent_run') {
      expect(capsule.status).toBe('completed');
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

// --------------------------------------------------------------------------
// FE-A2 / FE-A3 / FE-A1 contract tests (the four cases called out in the
// implementation spec). The capsule-grouping and trailing-status tests above
// already cover Step 1 (no isActivelyRunning) and Step 4 (header marker /
// defaultExpanded); these tests cover Step 2 (sub-agent lifecycle parsing)
// and a few targeted invariants the spec calls out.
// --------------------------------------------------------------------------

function sys(content: string, timestamp = TS): ChatMessage {
  return { role: 'system', content, source: 'management', timestamp };
}

describe('transformChatHistory — FE-A2 sub-agent lifecycle markers', () => {
  it('parses [Sub-agent] started / sent / completed lines into sub_agent_activity items', () => {
    const messages: ChatMessage[] = [
      user('dispatch to claude-code', TS),
      asst({
        content: null,
        tool_calls: [tc('c1', 'agent_message', '{"handle":"claude-code","action":"start"}')],
        timestamp: TS2,
      }),
      tool('c1', 'Started Claude Code', TS3),
      sys('[Sub-agent] claude-code started (claude-code, depth 2)', TS3),
      sys('[Sub-agent] Message sent to claude-code: list primes', TS4),
      sys('[Sub-agent] claude-code completed. Summary: 2, 3, 5, 7, 11, 13', TS5),
      asst({ content: 'The primes from 1-20 are: 2, 3, 5, 7, 11, 13, 17, 19.', timestamp: TS6 }),
    ];

    const items = transformChatHistory(messages);
    const activities = items.filter(i => i.type === 'sub_agent_activity');
    expect(activities.length).toBe(3);

    const actions = activities.map(a => a.type === 'sub_agent_activity' ? a.action : '');
    expect(actions).toEqual(['started', 'sent', 'completed']);

    for (const a of activities) {
      if (a.type === 'sub_agent_activity') {
        expect(a.handle).toBe('claude-code');
      }
    }

    const completed = activities[2];
    if (completed.type === 'sub_agent_activity') {
      expect(completed.action).toBe('completed');
      expect(completed.summary).toBe('2, 3, 5, 7, 11, 13');
    }
    const sent = activities[1];
    if (sent.type === 'sub_agent_activity') {
      expect(sent.preview).toBe('list primes');
    }

    // The final assistant text is still rendered as the agent_message.
    const lastAgent = [...items].reverse().find(i => i.type === 'agent_message' && !('isHeaderOnly' in i && i.isHeaderOnly));
    expect(lastAgent).toBeDefined();
    if (lastAgent && lastAgent.type === 'agent_message') {
      expect(lastAgent.content).toContain('The primes from 1-20');
    }
  });

  it('parses [Sub-agent] failed lines into sub_agent_activity with error field', () => {
    const messages: ChatMessage[] = [
      sys('[Sub-agent] claude-code failed: model timed out after 60s', TS),
    ];
    const items = transformChatHistory(messages);
    expect(items.length).toBe(1);
    const a = items[0];
    expect(a.type).toBe('sub_agent_activity');
    if (a.type === 'sub_agent_activity') {
      expect(a.action).toBe('failed');
      expect(a.handle).toBe('claude-code');
      expect(a.error).toBe('model timed out after 60s');
    }
  });

  it('non-[Sub-agent] system messages (e.g. ping-pong guard) are still dropped', () => {
    const messages: ChatMessage[] = [
      sys('Repetitive action detected. Save your state and try a different approach.', TS),
      sys('Some other internal system note.', TS2),
    ];
    const items = transformChatHistory(messages);
    expect(items.length).toBe(0);
  });

  it('an open capsule is finalized before a sub_agent_activity is emitted (chronological order preserved)', () => {
    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [tc('c1', 'agent_message', '{"handle":"claude-code","action":"send"}')],
        timestamp: TS,
      }),
      tool('c1', 'Dispatched to claude-code. Awaiting completion.', TS2),
      sys('[Sub-agent] Message sent to claude-code: hello', TS3),
    ];
    const items = transformChatHistory(messages);
    // header marker, agent_run, sub_agent_activity — in that order.
    const types = items.map(i => i.type);
    expect(types).toEqual(['agent_message', 'agent_run', 'sub_agent_activity']);
  });
});

describe('transformChatHistory — FE-A3 agent header for content-null turns', () => {
  it('content-null assistant turn with reasoning produces a header + COLLAPSED capsule', () => {
    // LOCKED PRODUCT DECISION (2026-06-03): transformChatHistory only ever runs
    // over PERSISTED (completed) history. A completed reasoning turn renders as a
    // COLLAPSED reasoning capsule (clean summary); it expands only while actively
    // RUNNING, which is handled at render time in ChatView (running ⇒ expand),
    // NOT here. So the transform must NEVER force-expand. (This test previously
    // asserted defaultExpanded === true under the old contract — rewritten to the
    // new invariant.)
    const messages: ChatMessage[] = [
      user('do something', TS),
      asst({
        content: null,
        reasoning_content: 'I should read the file first.',
        tool_calls: [tc('c1', 'read', '{"path":"a.txt"}')],
        timestamp: TS2,
      }),
    ];

    const items = transformChatHistory(messages);
    expect(items.map(i => i.type)).toEqual(['user_message', 'agent_message', 'agent_run']);

    const header = items[1];
    expect(header.type).toBe('agent_message');
    if (header.type === 'agent_message') {
      expect(header.isHeaderOnly).toBe(true);
    }

    const capsule = items[2];
    expect(capsule.type).toBe('agent_run');
    if (capsule.type === 'agent_run') {
      // Completed ⇒ collapsed. Falsy / absent defaultExpanded.
      expect(capsule.defaultExpanded).not.toBe(true);
    }
  });

  it('completed turn WITH both content and reasoning collapses the reasoning capsule', () => {
    // After inline-think separation (e.g. MiniMax-M3), one finalized message
    // carries the answer (content) AND the thinking (reasoning_content). The
    // turn is done, so the reasoning capsule must NOT default-expand — it
    // collapses to a summary the user can click open. (Only content-null,
    // still-thinking turns default-expand; see the test above.)
    const messages: ChatMessage[] = [
      user('explain clouds', TS),
      asst({
        content: 'Clouds are condensed water vapor.',
        reasoning_content: 'The user asked a science question; answer plainly.',
        timestamp: TS2,
      }),
    ];
    const items = transformChatHistory(messages);
    const capsules = items.filter(i => i.type === 'agent_run');
    expect(capsules.length).toBe(1);
    const capsule = capsules[0];
    if (capsule.type === 'agent_run') {
      expect(capsule.has_thinking).toBe(true);
      expect(capsule.defaultExpanded).not.toBe(true);
    }
    const bubbles = items.filter(
      i => i.type === 'agent_message' && !i.isHeaderOnly,
    );
    expect(
      bubbles.some(b => b.type === 'agent_message' && b.content.includes('Clouds are')),
    ).toBe(true);
  });

  it('content-null turn WITHOUT reasoning emits a header but capsule is NOT defaultExpanded', () => {
    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [tc('c1', 'read', '{"path":"a.txt"}')],
        timestamp: TS,
      }),
    ];
    const items = transformChatHistory(messages);
    const header = items[0];
    const capsule = items[1];
    expect(header.type).toBe('agent_message');
    if (header.type === 'agent_message') {
      expect(header.isHeaderOnly).toBe(true);
    }
    expect(capsule.type).toBe('agent_run');
    if (capsule.type === 'agent_run') {
      // Falsy / absent — tool-only capsules stay collapsed by default.
      expect(capsule.defaultExpanded).not.toBe(true);
    }
  });

  it('an assistant turn WITH visible text does NOT emit a header marker', () => {
    const messages: ChatMessage[] = [
      user('hi', TS),
      asst({ content: 'hello back', timestamp: TS2 }),
    ];
    const items = transformChatHistory(messages);
    // Just user_message + agent_message (the real one). No header marker.
    expect(items.length).toBe(2);
    const agent = items[1];
    if (agent.type === 'agent_message') {
      expect(agent.isHeaderOnly).toBeUndefined();
      expect(agent.content).toBe('hello back');
    }
  });
});

describe('transformChatHistory — sub-agent renders as a peer agent (capsule + response)', () => {
  it('Test 1: a source=sub_agent message produces header + agent_run capsule + response', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: '[Sub-agent] claude-code started ...', source: 'daemon', timestamp: TS },
      {
        role: 'assistant',
        content: 'Primes are 2,3,5,7',
        source: 'sub_agent',
        timestamp: TS2,
        sub_agent_handle: 'claude-code',
        sub_agent_tool_rows: [
          { name: 'Read', timestamp: TS2, duration_seconds: 0.5 },
          { name: 'Write', timestamp: TS3, duration_seconds: 1.2 },
        ],
        sub_agent_duration: 3.1,
      },
      { role: 'system', content: '[Sub-agent] claude-code completed. Summary: done', source: 'daemon', timestamp: TS4 },
    ];

    const items = transformChatHistory(messages, '/workspace');

    // Header agent_message carries the sub-agent handle as its source.
    const header = items.find((i) => i.type === 'agent_message' && (i as { isHeaderOnly?: boolean }).isHeaderOnly);
    expect(header).toBeDefined();
    if (header && header.type === 'agent_message') {
      expect(header.source).toBe('claude-code');
    }

    // agent_run capsule with one tool_call_row per tool.
    const capsule = items.find((i) => i.type === 'agent_run');
    expect(capsule).toBeDefined();
    if (capsule && capsule.type === 'agent_run') {
      const rows = capsule.items.filter((c) => c.type === 'tool_call_row');
      expect(rows.map((r) => (r.type === 'tool_call_row' ? r.tool_name : ''))).toEqual(['Read', 'Write']);
    }

    // Response agent_message with the sub-agent's text and handle.
    const resp = items.find((i) => i.type === 'agent_message' && !(i as { isHeaderOnly?: boolean }).isHeaderOnly && i.content);
    expect(resp).toBeDefined();
    if (resp && resp.type === 'agent_message') {
      expect(resp.content).toBe('Primes are 2,3,5,7');
      expect(resp.source).toBe('claude-code');
    }
  });

  it('Test 2: the sub-agent capsule is collapsible (sub_agent_ id, completed, collapsed)', () => {
    const messages: ChatMessage[] = [
      {
        role: 'assistant', content: 'done', source: 'sub_agent', timestamp: TS,
        sub_agent_handle: 'claude-code',
        sub_agent_tool_rows: [{ name: 'Bash', timestamp: TS, duration_seconds: 1.0 }],
        sub_agent_duration: 1.0,
      },
    ];
    const items = transformChatHistory(messages, '/workspace');
    const capsule = items.find((i) => i.type === 'agent_run');
    expect(capsule).toBeDefined();
    if (capsule && capsule.type === 'agent_run') {
      expect(capsule.capsule_id.startsWith('sub_agent:')).toBe(true);
      expect(capsule.status).toBe('completed');
      expect(capsule.defaultExpanded).toBe(false);
    }
  });

  it('Test 3: empty response renders capsule only, no response bubble', () => {
    const messages: ChatMessage[] = [
      {
        role: 'assistant', content: '', source: 'sub_agent', timestamp: TS,
        sub_agent_handle: 'codex',
        sub_agent_tool_rows: [{ name: 'Bash', timestamp: TS, duration_seconds: 1.0 }],
        sub_agent_duration: 1.0,
      },
    ];
    const items = transformChatHistory(messages, '/workspace');
    expect(items.some((i) => i.type === 'agent_run')).toBe(true);
    // header agent_message is allowed (empty content + isHeaderOnly); a NON-header
    // empty-content agent_message response must not be emitted.
    const emptyResponse = items.find(
      (i) => i.type === 'agent_message' && !(i as { isHeaderOnly?: boolean }).isHeaderOnly && i.content === '',
    );
    expect(emptyResponse).toBeUndefined();
  });

  it('a sub-agent with no tools emits header + response but no capsule', () => {
    const messages: ChatMessage[] = [
      {
        role: 'assistant', content: 'just text', source: 'sub_agent', timestamp: TS,
        sub_agent_handle: 'gemini-cli', sub_agent_tool_rows: [], sub_agent_duration: 0,
      },
    ];
    const items = transformChatHistory(messages, '/workspace');
    expect(items.some((i) => i.type === 'agent_run')).toBe(false);
    const resp = items.find((i) => i.type === 'agent_message' && i.content === 'just text');
    expect(resp).toBeDefined();
    if (resp && resp.type === 'agent_message') expect(resp.source).toBe('gemini-cli');
  });

  it('does not treat a normal management assistant message as a sub-agent', () => {
    const messages: ChatMessage[] = [
      { role: 'assistant', content: 'hello from management', source: 'management', timestamp: TS },
    ];
    const items = transformChatHistory(messages, '/workspace');
    expect(items[0].type).toBe('agent_message');
    if (items[0].type === 'agent_message') expect(items[0].source).toBe('management');
  });
});

describe('transformChatHistory — reasoning capsule collapse + never-vanish (sites 7 & 8)', () => {
  it('site 7: a reasoning-only / no-answer COMPLETED turn yields a COLLAPSED reasoning capsule', () => {
    // A persisted assistant turn that thought heavily but emitted no visible
    // answer text (content null/empty) and no tool_calls. Per the locked
    // decision it renders as a COLLAPSED reasoning capsule (defaultExpanded
    // falsy), since transformChatHistory only sees COMPLETED history.
    const messages: ChatMessage[] = [
      user('think hard but say nothing', TS),
      asst({
        content: null,
        reasoning_content: 'A long internal monologue with no visible answer.',
        timestamp: TS2,
      }),
    ];
    const items = transformChatHistory(messages);

    const capsule = items.find((i) => i.type === 'agent_run');
    expect(capsule).toBeDefined();
    if (capsule && capsule.type === 'agent_run') {
      expect(capsule.has_thinking).toBe(true);
      // The reasoning_block is present (capsule is NOT empty).
      expect(capsule.items.some((c) => c.type === 'reasoning_block')).toBe(true);
      // COMPLETED ⇒ collapsed.
      expect(capsule.defaultExpanded).not.toBe(true);
    }
  });

  it('site 7: a reasoning-then-answer COMPLETED turn yields an answer body + a COLLAPSED reasoning capsule', () => {
    const messages: ChatMessage[] = [
      user('explain rain', TS),
      asst({
        content: 'Rain is condensed atmospheric water vapor that falls.',
        reasoning_content: 'The user asked a science question; answer plainly.',
        timestamp: TS2,
      }),
    ];
    const items = transformChatHistory(messages);

    // Answer body is present as a non-header agent_message bubble.
    const bubble = items.find(
      (i) => i.type === 'agent_message' && !i.isHeaderOnly && i.content.includes('Rain is'),
    );
    expect(bubble).toBeDefined();

    // Reasoning capsule is present and COLLAPSED.
    const capsule = items.find((i) => i.type === 'agent_run');
    expect(capsule).toBeDefined();
    if (capsule && capsule.type === 'agent_run') {
      expect(capsule.has_thinking).toBe(true);
      expect(capsule.items.some((c) => c.type === 'reasoning_block')).toBe(true);
      expect(capsule.defaultExpanded).not.toBe(true);
    }
  });

  it('site 8 (never-vanish): an assistant message with empty content, no reasoning, no tool_calls emits a minimal placeholder, NOT nothing', () => {
    const messages: ChatMessage[] = [
      user('hi', TS),
      asst({
        content: '',
        reasoning_content: undefined,
        tool_calls: undefined,
        timestamp: TS2,
      }),
    ];
    const items = transformChatHistory(messages);

    // The assistant turn must NOT silently vanish — at minimum a header-only
    // agent_message marker is emitted so the turn is always represented.
    const assistantItems = items.filter((i) => i.type !== 'user_message');
    expect(assistantItems.length).toBeGreaterThan(0);

    const placeholder = items.find(
      (i) => i.type === 'agent_message' && i.isHeaderOnly === true,
    );
    expect(placeholder).toBeDefined();
  });

  it('site 8 (never-vanish): even a lone empty assistant message (no preceding user) is not dropped', () => {
    const messages: ChatMessage[] = [
      asst({ content: '', reasoning_content: undefined, tool_calls: undefined, timestamp: TS }),
    ];
    const items = transformChatHistory(messages);
    expect(items.length).toBeGreaterThan(0);
    expect(items.some((i) => i.type === 'agent_message' && i.isHeaderOnly === true)).toBe(true);
  });
});
