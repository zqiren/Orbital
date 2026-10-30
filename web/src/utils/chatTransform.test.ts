// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import {
  transformChatHistory,
  truncateResult,
  mergeRecoveredAssistantMessage,
  describeLiveActivity,
  describeWorkerTool,
  dispatchFromToolCall,
} from './chatTransform';
import { translate } from '../i18n/useT';
import type { ActivityTranslate } from './chatTransform';
import type { ChatMessage } from '../types';
import subAgentMarkerFixturesData from './subAgentMarkerFixtures.json';
import toolResultPreviewFixturesData from './toolResultPreviewFixtures.json';

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

describe('activity descriptions are derived locally, never backend prose (China i18n B1)', () => {
  const zhTr: ActivityTranslate = (k, v) => translate('zh', k, v);

  it('ignores persisted _activity_descriptions and renders from the tool call', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: null,
        tool_calls: [tc('c1', 'read', '{"path":"notes.md"}')],
        _activity_descriptions: { c1: 'Reading notes.md — backend English prose' },
        timestamp: TS2,
      }),
      tool('c1', 'x', TS3),
      asst({ content: 'done', timestamp: TS4 }),
    ];
    const items = transformChatHistory(messages);
    const capsule = items.find((i) => i.type === 'agent_run');
    expect(capsule?.type).toBe('agent_run');
    if (capsule?.type === 'agent_run') {
      const row = capsule.items.find((x) => x.type === 'tool_call_row');
      if (row?.type === 'tool_call_row') {
        expect(row.target_description).toBe('Read notes.md');
      } else {
        throw new Error('expected tool_call_row');
      }
    }
  });

  it('renders Chinese with a zh translator even when English prose is persisted', () => {
    const messages: ChatMessage[] = [
      user('go', TS),
      asst({
        content: null,
        tool_calls: [tc('c1', 'read', '{"path":"notes.md"}')],
        _activity_descriptions: { c1: 'Reading notes.md' },
        timestamp: TS2,
      }),
      tool('c1', 'x', TS3),
      asst({ content: 'done', timestamp: TS4 }),
    ];
    const items = transformChatHistory(messages, undefined, zhTr);
    const capsule = items.find((i) => i.type === 'agent_run');
    if (capsule?.type === 'agent_run') {
      const row = capsule.items.find((x) => x.type === 'tool_call_row');
      if (row?.type === 'tool_call_row') {
        expect(row.target_description).toBe('读取 notes.md');
      }
    }
  });
});

describe('describeLiveActivity', () => {
  const zhTr: ActivityTranslate = (k, v) => translate('zh', k, v);

  it('derives a localized description from live event arguments', () => {
    expect(
      describeLiveActivity('shell', { command: 'ls -la' }, undefined, zhTr, 'Running: ls -la'),
    ).toBe('运行：ls -la');
  });

  it('falls back to the wire description when the daemon sends no arguments', () => {
    expect(
      describeLiveActivity('shell', undefined, undefined, zhTr, 'Running: ls -la'),
    ).toBe('Running: ls -la');
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

  it('a pre-cut preview with the full totals: marks the cut and reports the full size', () => {
    const r = truncateResult('x'.repeat(500), undefined, { chars: 9000, lines: 1 });
    expect(r.text).toBe('x'.repeat(500) + '…');
    expect(r.footer).toBe('first 500 chars · result is 9000 chars total');
  });
});

// --------------------------------------------------------------------------
// Live tool-result preview parity. The daemon's live tool_result event
// carries only the part of a result the capsule shows (plus the full totals
// when it cut it). The shared fixture's rows are asserted on the producer
// side by tests/unit/test_activity_tool_result_preview.py; here, rendering
// each preview with its totals must equal rendering the full result the way
// a reload does — so a row expanded mid-turn looks exactly like it will
// after the session is reopened.
// --------------------------------------------------------------------------

interface ToolResultPreviewFixture {
  name: string;
  full: string;
  preview: string;
  total_chars?: number;
  total_lines?: number;
}

describe('truncateResult — live preview renders like the reloaded full result', () => {
  const rows = toolResultPreviewFixturesData as ToolResultPreviewFixture[];
  it.each(rows.map((r) => [r.name, r] as const))('%s', (_name, row) => {
    const totals =
      row.total_chars !== undefined && row.total_lines !== undefined
        ? { chars: row.total_chars, lines: row.total_lines }
        : undefined;
    expect(truncateResult(row.preview, undefined, totals)).toEqual(truncateResult(row.full));
  });
});

describe('agent_message rows carry the dispatched message', () => {
  const brief = 'Refactor the parser.\n\nConstraints:\n- keep the public API\n- add tests\n' + 'detail '.repeat(300);

  function onlyRow(messages: ChatMessage[]) {
    const capsule = transformChatHistory(messages).find((i) => i.type === 'agent_run');
    if (capsule?.type !== 'agent_run') throw new Error('expected agent_run');
    const row = capsule.items.find((x) => x.type === 'tool_call_row');
    if (row?.type !== 'tool_call_row') throw new Error('expected tool_call_row');
    return row;
  }

  it('threads the target agent and the FULL message onto the row, alongside the paired ack', () => {
    const row = onlyRow([
      user('delegate it', TS),
      asst({
        tool_calls: [tc('c1', 'agent_message', JSON.stringify({ action: 'send', agent: 'claude-code', message: brief }))],
        timestamp: TS2,
      }),
      tool('c1', 'Dispatched to claude-code. Awaiting completion.', TS3),
    ]);
    expect(row.target_description).toBe('Messaged: @claude-code');
    expect(row.dispatch).toEqual({ agent: 'claude-code', message: brief });
    expect(row.result_content).toBe('Dispatched to claude-code. Awaiting completion.');
  });

  it('a respond carries its message too', () => {
    const row = onlyRow([
      asst({
        tool_calls: [tc('c1', 'agent_message', JSON.stringify({
          action: 'respond', agent: 'codex', interaction_id: 'i1', message: 'Yes, overwrite it.',
        }))],
      }),
      tool('c1', 'Response delivered to codex.', TS2),
    ]);
    expect(row.dispatch).toEqual({ agent: 'codex', message: 'Yes, overwrite it.' });
  });

  it('list / status / stop send no message, so the row has no dispatch', () => {
    const row = onlyRow([
      asst({ tool_calls: [tc('c1', 'agent_message', '{"action":"list","agent":"claude-code"}')] }),
      tool('c1', '[]', TS2),
    ]);
    expect(row.dispatch).toBeUndefined();
    expect(row.target_description).toBe('Messaged: @claude-code');
  });

  it('older argument shapes (handle) still name the agent', () => {
    const row = onlyRow([
      asst({ tool_calls: [tc('c1', 'agent_message', '{"handle":"claude-code","action":"send","message":"hi"}')] }),
      tool('c1', 'ok', TS2),
    ]);
    expect(row.target_description).toBe('Messaged: @claude-code');
    expect(row.dispatch).toEqual({ agent: 'claude-code', message: 'hi' });
  });

  it('other tools never get a dispatch', () => {
    const row = onlyRow([
      asst({ tool_calls: [tc('c1', 'shell', '{"command":"ls","message":"not a dispatch"}')] }),
      tool('c1', 'a.txt', TS2),
    ]);
    expect(row.dispatch).toBeUndefined();
  });

  it('dispatchFromToolCall reads live event arguments the same way', () => {
    expect(dispatchFromToolCall('agent_message', { action: 'send', agent: 'pi', message: brief }))
      .toEqual({ agent: 'pi', message: brief });
    expect(dispatchFromToolCall('agent_message', { action: 'stop', agent: 'pi' })).toBeUndefined();
    expect(dispatchFromToolCall('agent_message', undefined)).toBeUndefined();
    expect(dispatchFromToolCall('read', { message: 'x' })).toBeUndefined();
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

  it('D2: completed marker uses _meta.display_content — agent-facing guidance never reaches the summary field', () => {
    const full =
      "[Sub-agent] claude-code completed. Summary: done. Transcript: /x/y.jsonl. The user can " +
      "already see this summary in chat as the sub-agent's own message — do NOT repeat or " +
      're-summarize it. Verify the work if needed, then continue or reply only with what\'s new.';
    const display = '[Sub-agent] claude-code completed. Summary: done. Transcript: /x/y.jsonl.';
    const messages: ChatMessage[] = [
      { ...sys(full, TS), _meta: { display_content: display } },
    ];
    const items = transformChatHistory(messages);
    expect(items.length).toBe(1);
    const a = items[0];
    expect(a.type).toBe('sub_agent_activity');
    if (a.type === 'sub_agent_activity') {
      expect(a.action).toBe('completed');
      expect(a.summary).not.toContain('do NOT repeat');
      expect(a.summary).not.toContain('Verify the work');
    }
  });

  it('D2: completed marker without _meta (legacy) falls back to full content', () => {
    const full = '[Sub-agent] claude-code completed. Summary: done. Transcript: /x/y.jsonl.';
    const messages: ChatMessage[] = [sys(full, TS)];
    const items = transformChatHistory(messages);
    const a = items[0];
    expect(a.type).toBe('sub_agent_activity');
    if (a.type === 'sub_agent_activity') {
      expect(a.summary).toContain('done. Transcript: /x/y.jsonl.');
    }
  });

  it('D2: parses [Sub-agent] "stopped with error:" lines into sub_agent_activity with error field', () => {
    // Before this fix the renderer whitelist had no rule for on_error's
    // marker shape (only on_failed's "failed:" was covered), so this durable
    // error row silently rendered as nothing.
    const messages: ChatMessage[] = [
      sys('[Sub-agent] claude-code stopped with error: model timed out after 60s', TS),
    ];
    const items = transformChatHistory(messages);
    expect(items.length).toBe(1);
    const a = items[0];
    expect(a.type).toBe('sub_agent_activity');
    if (a.type === 'sub_agent_activity') {
      expect(a.action).toBe('error');
      expect(a.handle).toBe('claude-code');
      expect(a.error).toBe('model timed out after 60s');
    }
  });

  // backlog #27: four more producer shapes that the whitelist never matched,
  // so each rendered as NOTHING on history reload.
  it('#27: parses a bare "stopped by user." marker as a neutral stopped row', () => {
    const items = transformChatHistory([sys('[Sub-agent] claude-code stopped by user.', TS)]);
    expect(items.length).toBe(1);
    const a = items[0];
    expect(a.type).toBe('sub_agent_activity');
    if (a.type === 'sub_agent_activity') {
      expect(a.action).toBe('stopped');
      expect(a.handle).toBe('claude-code');
      // A plain stop destroyed no background work — no count/detail to show.
      expect(a.count).toBeUndefined();
      expect(a.detail).toBeUndefined();
    }
  });

  it('#27: a user stop that killed background work carries the count and commands', () => {
    const items = transformChatHistory([
      sys(
        '[Sub-agent] claude-code stopped by user. Terminated 2 background ' +
          'process(es): npm run dev; python server.py. This background work did NOT complete.',
        TS,
      ),
    ]);
    const a = items[0];
    expect(a.type).toBe('sub_agent_activity');
    if (a.type === 'sub_agent_activity') {
      expect(a.action).toBe('stopped');
      expect(a.count).toBe(2);
      expect(a.detail).toBe('npm run dev; python server.py');
      // The row's own copy says the work didn't complete; the producer's
      // warning sentence must not be duplicated inside the command list.
      expect(a.detail).not.toContain('did NOT complete');
    }
  });

  it('#27: parses a turn-interrupted marker (not an error — no result, agent alive)', () => {
    const items = transformChatHistory([
      sys(
        '[Sub-agent] claude-code was stopped before completing its current task ' +
          '(turn interrupted — e.g. an approval denied with stop). No result was ' +
          'produced. The agent remains available. Transcript: /x/y.jsonl',
        TS,
      ),
    ]);
    const a = items[0];
    expect(a.type).toBe('sub_agent_activity');
    if (a.type === 'sub_agent_activity') {
      expect(a.action).toBe('interrupted');
      expect(a.handle).toBe('claude-code');
    }
  });

  it('#27: parses a teardown background-work-lost marker with count and commands', () => {
    const items = transformChatHistory([
      sys(
        '[Sub-agent] claude-code teardown terminated 2 live background process(es): ' +
          'npm run dev; python server.py. This background work did NOT complete.',
        TS,
      ),
    ]);
    const a = items[0];
    expect(a.type).toBe('sub_agent_activity');
    if (a.type === 'sub_agent_activity') {
      expect(a.action).toBe('background_lost');
      expect(a.count).toBe(2);
      expect(a.detail).toBe('npm run dev; python server.py');
    }
  });

  it('#27: parses an interaction-required marker, stripping the agent-facing instructions', () => {
    const items = transformChatHistory([
      sys(
        '[Sub-agent] claude-code requires input (question): Which file should I edit? ' +
          'Respond on the same in-flight request with agent_message(action="respond", ' +
          'agent="claude-code", interaction_id="int-1", ...). For a free-form question, ' +
          'put the answer in message. Do not send a new task.',
        TS,
      ),
    ]);
    const a = items[0];
    expect(a.type).toBe('sub_agent_activity');
    if (a.type === 'sub_agent_activity') {
      expect(a.action).toBe('interaction_required');
      expect(a.handle).toBe('claude-code');
      expect(a.kind).toBe('question');
      // Only the sub-agent's actual question survives — the "how to answer"
      // prose addresses the LLM and must never reach the reader.
      expect(a.prompt).toBe('Which file should I edit?');
      expect(a.prompt).not.toContain('agent_message(');
      expect(a.prompt).not.toContain('Do not send a new task');
    }
  });

  it('#27: a stop-by-user marker is never mistaken for the stopped-with-error shape', () => {
    // Both open "[Sub-agent] <handle> stopped …" — the rules must stay
    // disjoint or a user stop would render as a red error row.
    const stopped = transformChatHistory([sys('[Sub-agent] claude-code stopped by user.', TS)])[0];
    const errored = transformChatHistory([
      sys('[Sub-agent] claude-code stopped with error: boom', TS),
    ])[0];
    expect(stopped.type === 'sub_agent_activity' && stopped.action).toBe('stopped');
    expect(errored.type === 'sub_agent_activity' && errored.action).toBe('error');
  });
});

// --------------------------------------------------------------------------
// Producer↔renderer marker parity (backlog #23 D2, extended by #27). The
// fixture at subAgentMarkerFixtures.json is the one source of truth for
// every sub-agent system-marker shape. The paired pytest
// (tests/unit/test_sub_agent_marker_parity.py) drives the REAL
// LifecycleObserver producer methods and asserts their rendered output
// matches this same fixture exactly — and, since #27, discovers the
// producer set from the observer's AST so a brand-new marker with no
// fixture row fails there rather than silently rendering as nothing here.
// Together: a producer shape that changes without updating the fixture
// fails the pytest; a fixture shape with no renderer rule (or that stops
// parsing) fails HERE.
// --------------------------------------------------------------------------

interface SubAgentMarkerFixture {
  shape: string;
  action: string;
  content: string;
}

const subAgentMarkerFixtures = subAgentMarkerFixturesData as SubAgentMarkerFixture[];

describe('transformChatHistory — producer/renderer marker parity fixture', () => {
  it('every fixture shape renders to a sub_agent_activity item with the expected action', () => {
    for (const fixture of subAgentMarkerFixtures) {
      const items = transformChatHistory([sys(fixture.content, TS)]);
      const activity = items.find(i => i.type === 'sub_agent_activity');
      expect(
        activity,
        `shape "${fixture.shape}" must render, not drop silently (got: ${JSON.stringify(items)})`,
      ).toBeDefined();
      if (activity && activity.type === 'sub_agent_activity') {
        expect(activity.action).toBe(fixture.action);
      }
    }
  });

  it('an unlisted [Sub-agent] shape still renders as NOTHING — why the guard exists', () => {
    // Not a wish, a statement of fact: the renderer is a whitelist, so any
    // marker without a rule above is dropped in silence rather than falling
    // back to plain text. That is the #23/#27 bug in one line, and it is why
    // the paired pytest discovers producers from lifecycle_observer.py's AST
    // instead of trusting a hand-kept list — a new producer must fail a test
    // somewhere, and it can never fail HERE.
    const items = transformChatHistory([
      sys('[Sub-agent] claude-code did something nobody wrote a rule for.', TS),
    ]);
    expect(items).toHaveLength(0);
  });

  it('a dropped queued message carries its reason without the trailing period', () => {
    const items = transformChatHistory([
      sys(
        '[Sub-agent] claude-code queued message dropped: transport ended before dispatch.',
        TS,
      ),
    ]);
    const activity = items.find(i => i.type === 'sub_agent_activity');
    expect(activity).toBeDefined();
    if (activity?.type !== 'sub_agent_activity') return;
    expect(activity.action).toBe('queue_dropped');
    expect(activity.handle).toBe('claude-code');
    // The producer's sentence-ending period belongs to the marker, not the
    // reason — the label re-punctuates around it.
    expect(activity.reason).toBe('transport ended before dispatch');
  });

  it('a drop recorded by a pre-#35 build still renders, as the failure it was written as', () => {
    // Sessions on disk keep whatever their build wrote. #26 wrote these
    // through on_failed, and re-labelling that text now would claim the old
    // build said something it did not — so the old rule keeps it.
    const items = transformChatHistory([
      sys(
        '[Sub-agent] claude-code failed: queued message dropped: agent stopped ' +
          'before dispatch. The dispatched task did not complete.',
        TS,
      ),
    ]);
    const activity = items.find(i => i.type === 'sub_agent_activity');
    expect(activity).toBeDefined();
    if (activity?.type !== 'sub_agent_activity') return;
    expect(activity.action).toBe('failed');
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

  it('D1: a sub-agent turn with no tool rows emits exactly ONE agent_message (no header-only anchor, no capsule)', () => {
    // Regression for the duplicate-header bug: an anchor with no capsule to
    // anchor is not a header, it's a second identical bubble.
    const messages: ChatMessage[] = [
      {
        role: 'assistant', content: 'just text', source: 'sub_agent', timestamp: TS,
        sub_agent_handle: 'gemini-cli', sub_agent_tool_rows: [], sub_agent_duration: 0,
      },
    ];
    const items = transformChatHistory(messages, '/workspace');
    expect(items.length).toBe(1);
    expect(items.some((i) => i.type === 'agent_run')).toBe(false);
    expect(
      items.some((i) => i.type === 'agent_message' && (i as { isHeaderOnly?: boolean }).isHeaderOnly),
    ).toBe(false);
    const resp = items.find((i) => i.type === 'agent_message' && i.content === 'just text');
    expect(resp).toBeDefined();
    if (resp && resp.type === 'agent_message') expect(resp.source).toBe('gemini-cli');
  });

  it('D1: a sub-agent turn with no tool rows and no content emits nothing', () => {
    const messages: ChatMessage[] = [
      {
        role: 'assistant', content: '', source: 'sub_agent', timestamp: TS,
        sub_agent_handle: 'codex', sub_agent_tool_rows: [], sub_agent_duration: 0,
      },
    ];
    const items = transformChatHistory(messages, '/workspace');
    expect(items.length).toBe(0);
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

describe('mergeRecoveredAssistantMessage — REST recovery resolves identity like normal ingest', () => {
  // Regression: the REST /chat?limit=1 recovery fallback hardcoded
  // source: 'assistant', splitting one management turn into two speaker
  // bubbles ("Assistant" + ◐ vs raw "assistant" + grey AS monogram).
  // Contract: recovery resolves source the SAME way normal ingest does —
  // verbatim from the persisted row (chatTransform `source: msg.source`),
  // never via a hardcoded default.

  function agentItem(content: string, source = 'management', timestamp = TS) {
    return { type: 'agent_message' as const, content, source, timestamp };
  }
  function userItem(content: string, timestamp = TS2) {
    return { type: 'user_message' as const, content, timestamp };
  }

  it('preserves the row source verbatim (management row)', () => {
    const out = mergeRecoveredAssistantMessage(
      [],
      asst({ content: '[STATUS: Starting Claude Code sub-agent]' }),
    );
    expect(out).toHaveLength(1);
    const item = out[0];
    expect(item.type).toBe('agent_message');
    if (item.type === 'agent_message') {
      expect(item.source).toBe('management');
    }
  });

  it('passes any row source through untouched — no normalization, no default', () => {
    // Same-as-ingest means verbatim: a non-management source must survive
    // exactly, proving there is no hardcoded fallback in either direction.
    const out = mergeRecoveredAssistantMessage(
      [],
      asst({ content: 'sub agent text', source: 'claude-code' }),
    );
    const item = out[0];
    if (item.type === 'agent_message') {
      expect(item.source).toBe('claude-code');
    } else {
      throw new Error('expected agent_message');
    }
  });

  it('dedups against an identical message ANYWHERE in the list, not only the last item', () => {
    // The production bug: the WS-built bubble existed but was not the LAST
    // item (a system message followed), so the old last-only check missed
    // and a duplicate was inserted.
    const prev = [
      agentItem('[STATUS: Starting Claude Code sub-agent]'),
      userItem('follow-up question'),
    ];
    const out = mergeRecoveredAssistantMessage(
      prev,
      asst({ content: '[STATUS: Starting Claude Code sub-agent]' }),
    );
    expect(out).toBe(prev); // unchanged, same reference
  });

  it('extends a truncated trailing message instead of duplicating it', () => {
    const prev = [agentItem('partial answ')];
    const out = mergeRecoveredAssistantMessage(
      prev,
      asst({ content: 'partial answer, now complete' }),
    );
    expect(out).toHaveLength(1);
    const item = out[0];
    if (item.type === 'agent_message') {
      expect(item.content).toBe('partial answer, now complete');
    }
  });

  it('inserts a genuinely missing message before trailing user messages', () => {
    const prev = [agentItem('older turn'), userItem('user typed after')];
    const out = mergeRecoveredAssistantMessage(
      prev,
      asst({ content: 'recovered reply', timestamp: TS3 }),
    );
    expect(out).toHaveLength(3);
    expect(out[1].type).toBe('agent_message');
    if (out[1].type === 'agent_message') {
      expect(out[1].content).toBe('recovered reply');
      expect(out[1].source).toBe('management');
    }
    expect(out[2].type).toBe('user_message');
  });
});

describe('mergeRecoveredAssistantMessage — tagged rows never become a bubble (spec 131 A)', () => {
  // Spec 131 A: the end-of-run memory record turn (spec 112 A6) appends rows
  // tagged `_kind: "memory_record"` AFTER the real reply. The live path is
  // silent for them by design, so the idle catch-up's /chat?limit=1 lands on
  // the record turn's closing text — which deepseek-v4.1-flash writes as a
  // restatement of the reply. Any tagged row must be ignored here: the
  // persisted transform already owns rendering for tagged rows (it keeps the
  // edits as a collapsed capsule and drops the text).
  function agentItem(content: string, source = 'management', timestamp = TS) {
    return { type: 'agent_message' as const, content, source, timestamp };
  }

  it('returns the list unchanged for a memory_record text row that differs from the reply', () => {
    const prev = [agentItem('the real reply')];
    const out = mergeRecoveredAssistantMessage(
      prev,
      asst({ content: 'the real reply, memory updated', _kind: 'memory_record', timestamp: TS3 }),
    );
    expect(out).toBe(prev);
  });

  it('does not extend the last bubble with a memory_record row that merely restates it', () => {
    // The truncation-repair branch would otherwise "complete" the reply with
    // the record turn's longer restatement.
    const prev = [agentItem('the real reply')];
    const out = mergeRecoveredAssistantMessage(
      prev,
      asst({ content: 'the real reply — and memory updated.', _kind: 'memory_record', timestamp: TS3 }),
    );
    expect(out).toBe(prev);
    expect(out[0].type === 'agent_message' && out[0].content).toBe('the real reply');
  });

  it('ignores a memory_record row even when the list is empty', () => {
    const prev: ReturnType<typeof agentItem>[] = [];
    const out = mergeRecoveredAssistantMessage(
      prev,
      asst({ content: 'nothing to record', _kind: 'memory_record' }),
    );
    expect(out).toBe(prev);
  });

  it('ignores a memory_tidy row too — any kind, not just memory_record', () => {
    const prev = [agentItem('the real reply')];
    const out = mergeRecoveredAssistantMessage(
      prev,
      asst({ content: 'tidied 3 files', _kind: 'memory_tidy', timestamp: TS3 }),
    );
    expect(out).toBe(prev);
  });

  it('still recovers an untagged row (the gate is kind-aware, not a blanket drop)', () => {
    const prev = [agentItem('older turn')];
    const out = mergeRecoveredAssistantMessage(
      prev,
      asst({ content: 'recovered reply', timestamp: TS3 }),
    );
    expect(out).toHaveLength(2);
    expect(out[1].type === 'agent_message' && out[1].content).toBe('recovered reply');
  });
});

// ── P3-G: budget-trip timeline row ──────────────────────────────────────────
// The agent loop appends a structured system row on a budget trip:
//   {role:'system', source:'budget', event:'budget_blocked', payload:{...}}
// The session endpoint passes the raw JSONL through, so `event`/`payload` ride
// as extra fields not declared on ChatMessage — construct them via a cast.
function budgetRow(payload: {
  action: string;
  window: string;
  spend: number;
  limit: number | null;
  currency: string;
}): ChatMessage {
  return {
    role: 'system',
    content: null,
    source: 'budget',
    timestamp: TS,
    // event/payload are passthrough extras; the parser reads them off a cast.
    ...({ event: 'budget_blocked', payload } as unknown as Partial<ChatMessage>),
  };
}

describe('transformChatHistory — budget_blocked timeline row', () => {
  it('parses a pause trip into a budget_event item with payload codes/numbers', () => {
    const out = transformChatHistory([
      budgetRow({ action: 'pause', window: 'daily', spend: 1.2, limit: 1.0, currency: 'USD' }),
    ]);
    expect(out).toHaveLength(1);
    const item = out[0];
    expect(item.type).toBe('budget_event');
    if (item.type === 'budget_event') {
      expect(item.event).toBe('budget_blocked');
      expect(item.action).toBe('pause');
      expect(item.window).toBe('daily');
      expect(item.spend).toBeCloseTo(1.2);
      expect(item.limit).toBeCloseTo(1.0);
      expect(item.currency).toBe('USD');
    }
  });

  it('carries action=stop and a non-daily window', () => {
    const out = transformChatHistory([
      budgetRow({ action: 'stop', window: 'weekly', spend: 50, limit: 40, currency: 'CNY' }),
    ]);
    expect(out[0].type).toBe('budget_event');
    if (out[0].type === 'budget_event') {
      expect(out[0].action).toBe('stop');
      expect(out[0].window).toBe('weekly');
      expect(out[0].currency).toBe('CNY');
    }
  });

  it('tolerates a null limit (no-limit trip)', () => {
    const out = transformChatHistory([
      budgetRow({ action: 'pause', window: 'monthly', spend: 7, limit: null, currency: 'USD' }),
    ]);
    expect(out[0].type).toBe('budget_event');
    if (out[0].type === 'budget_event') {
      expect(out[0].limit).toBeNull();
    }
  });

  it('defaults an unknown window to daily and an unknown action to pause', () => {
    const out = transformChatHistory([
      budgetRow({ action: 'frobnicate', window: 'fortnightly', spend: 1, limit: 2, currency: 'USD' }),
    ]);
    expect(out[0].type).toBe('budget_event');
    if (out[0].type === 'budget_event') {
      expect(out[0].action).toBe('pause');
      expect(out[0].window).toBe('daily');
    }
  });

  it('ignores non-budget system rows (no budget_event emitted)', () => {
    const notBudget: ChatMessage = {
      role: 'system',
      content: 'some other system note',
      source: 'management',
      timestamp: TS,
    };
    const out = transformChatHistory([notBudget]);
    expect(out.some((i) => i.type === 'budget_event')).toBe(false);
  });

  it('a budget_threshold event does NOT produce a budget_event item', () => {
    // budget_threshold is WS/relay-only and should never land in a session
    // row — but if one ever does (daemon drift), the parser must not render
    // it as a trip. Only event === 'budget_blocked' qualifies.
    const thresholdRow: ChatMessage = {
      role: 'system',
      content: null,
      source: 'budget',
      timestamp: TS,
      ...({
        event: 'budget_threshold',
        payload: { action: 'pause', window: 'daily', spend: 0.8, limit: 1.0, currency: 'USD' },
      } as unknown as Partial<ChatMessage>),
    };
    const out = transformChatHistory([thresholdRow]);
    expect(out.some((i) => i.type === 'budget_event')).toBe(false);
  });
});

// --------------------------------------------------------------------------
// Task 5 (spec 009 §0.5): the `fanout` tool call becomes a standalone
// `fanout_card` DisplayItem — never a regular tool_call_row inside a capsule.
// The tool call's own arguments only carry the requested task labels (the
// backend hasn't assigned handles yet); the fanout_id is recovered from the
// tool RESULT's ack text ("Fanout <8hex> dispatched..."), and per-task handles
// are synthesized as `worker:<fanout_id>-<index>` to match the wire format
// Task 2's WS events use (`fanout.started` tasks carry that exact shape).
// --------------------------------------------------------------------------

describe('transformChatHistory — fanout tool call -> fanout_card', () => {
  it('a fanout tool call + ack result produces a fanout_card with synthesized handles', () => {
    const messages: ChatMessage[] = [
      user('research these in parallel', TS),
      asst({
        content: null,
        tool_calls: [tc('c1', 'fanout', '{"tasks":["Research A","Research B"]}')],
        timestamp: TS2,
      }),
      tool('c1', 'Fanout a1b2c3d4 dispatched: 2 tasks', TS3),
    ];

    const items = transformChatHistory(messages);
    const itemTypes = items.map((i) => i.type);
    // Anchor header (agent turn had no visible text, no reasoning, only the
    // fanout tool call) then the standalone card — no agent_run capsule.
    expect(itemTypes).toEqual(['user_message', 'agent_message', 'fanout_card']);

    const card = items[2];
    expect(card.type).toBe('fanout_card');
    if (card.type === 'fanout_card') {
      expect(card.fanout_id).toBe('a1b2c3d4');
      expect(card.tasks).toEqual([
        { handle: 'worker:a1b2c3d4-0', label: 'Research A' },
        { handle: 'worker:a1b2c3d4-1', label: 'Research B' },
      ]);
      expect(card.timestamp).toBe(TS2);
    }
  });

  it('supports object-shaped task args ({label: ...}) in addition to bare strings', () => {
    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [tc('c1', 'fanout', '{"tasks":[{"label":"Audit the schema"},{"label":"Write the migration"}]}')],
        timestamp: TS,
      }),
      tool('c1', 'Fanout deadbeef dispatched: 2 tasks', TS2),
    ];
    const items = transformChatHistory(messages);
    const card = items.find((i) => i.type === 'fanout_card');
    expect(card).toBeDefined();
    if (card && card.type === 'fanout_card') {
      expect(card.tasks.map((t) => t.label)).toEqual(['Audit the schema', 'Write the migration']);
      expect(card.tasks.map((t) => t.handle)).toEqual(['worker:deadbeef-0', 'worker:deadbeef-1']);
    }
  });

  it('drops the fanout call silently when the result has no recognizable ack (orphan, mirrors tool-result-drop precedent)', () => {
    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [tc('c1', 'fanout', '{"tasks":["A"]}')],
        timestamp: TS,
      }),
      tool('c1', 'some unrelated error text', TS2),
    ];
    const items = transformChatHistory(messages);
    expect(items.some((i) => i.type === 'fanout_card')).toBe(false);
  });

  // Review round 2 found Critical 2 (tool_calls: [read, fanout], results in
  // that same order) then a residual of the SAME bug class (tool_calls:
  // [fanout, read], results in that same order — the fanout ack processed
  // BEFORE read's result). Both the tool_calls declaration order AND the
  // order results actually arrive in matter independently, so all four
  // combinations are covered explicitly. Every case asserts BOTH: (a) `read`
  // ends up resolved with its real content, never orphaned, and (b) exactly
  // one fanout_card is emitted.
  describe('fanout mixed with a regular tool call in the same turn — all four call/result orderings', () => {
    function assertReadResolvedAndCardEmitted(items: ReturnType<typeof transformChatHistory>, fanoutId: string) {
      const capsule = items.find((i) => i.type === 'agent_run');
      expect(capsule).toBeDefined();
      if (capsule && capsule.type === 'agent_run') {
        const readRow = capsule.items.find((it) => it.type === 'tool_call_row' && it.tool_name === 'read');
        expect(readRow).toMatchObject({
          type: 'tool_call_row',
          tool_name: 'read',
          result_status: 'received',
          result_content: 'contents of a.txt',
        });
      }
      const cards = items.filter((i) => i.type === 'fanout_card');
      expect(cards.length).toBe(1);
      if (cards[0].type === 'fanout_card') {
        expect(cards[0].fanout_id).toBe(fanoutId);
      }
    }

    it('tool_calls [read, fanout], results [read, fanout] (declaration order, the original Critical 2 case)', () => {
      const messages: ChatMessage[] = [
        asst({
          content: null,
          tool_calls: [tc('c1', 'read', '{"path":"a.txt"}'), tc('c2', 'fanout', '{"tasks":["A"]}')],
          timestamp: TS,
        }),
        tool('c1', 'contents of a.txt', TS2),
        tool('c2', 'Fanout cafeba01 dispatched: 1 tasks', TS3),
      ];
      assertReadResolvedAndCardEmitted(transformChatHistory(messages), 'cafeba01');
    });

    it('tool_calls [read, fanout], results [fanout, read] (ack-first, out-of-order)', () => {
      const messages: ChatMessage[] = [
        asst({
          content: null,
          tool_calls: [tc('c1', 'read', '{"path":"a.txt"}'), tc('c2', 'fanout', '{"tasks":["A"]}')],
          timestamp: TS,
        }),
        tool('c2', 'Fanout cafeba02 dispatched: 1 tasks', TS2),
        tool('c1', 'contents of a.txt', TS3),
      ];
      assertReadResolvedAndCardEmitted(transformChatHistory(messages), 'cafeba02');
    });

    it('tool_calls [fanout, read], results [fanout, read] (declaration order — the residual bug case)', () => {
      const messages: ChatMessage[] = [
        asst({
          content: null,
          tool_calls: [tc('c1', 'fanout', '{"tasks":["A"]}'), tc('c2', 'read', '{"path":"a.txt"}')],
          timestamp: TS,
        }),
        tool('c1', 'Fanout cafeba03 dispatched: 1 tasks', TS2),
        tool('c2', 'contents of a.txt', TS3),
      ];
      assertReadResolvedAndCardEmitted(transformChatHistory(messages), 'cafeba03');
    });

    it('tool_calls [fanout, read], results [read, fanout] (ack-second, out-of-order)', () => {
      const messages: ChatMessage[] = [
        asst({
          content: null,
          tool_calls: [tc('c1', 'fanout', '{"tasks":["A"]}'), tc('c2', 'read', '{"path":"a.txt"}')],
          timestamp: TS,
        }),
        tool('c2', 'contents of a.txt', TS2),
        tool('c1', 'Fanout cafeba04 dispatched: 1 tasks', TS3),
      ];
      assertReadResolvedAndCardEmitted(transformChatHistory(messages), 'cafeba04');
    });
  });

  it('marks the fanout_card isHistorical when it precedes the last session_separator, like other items', () => {
    const messages: ChatMessage[] = [
      { ...asst({ content: null, tool_calls: [tc('c1', 'fanout', '{"tasks":["A"]}')], timestamp: TS }), session_id: 's1' },
      { ...tool('c1', 'Fanout 12345678 dispatched: 1 tasks', TS2), session_id: 's1' },
      { ...asst({ content: 'moving on', timestamp: TS3 }), session_id: 's2' },
    ];
    const items = transformChatHistory(messages);
    const card = items.find((i) => i.type === 'fanout_card');
    expect(card).toBeDefined();
    if (card) {
      expect((card as { isHistorical?: boolean }).isHistorical).toBe(true);
    }
  });
});

// --------------------------------------------------------------------------
// Review round 2, Critical 1 + Critical 3: the join-summary system message
// the daemon writes when a fanout batch completes. It must (a) render as a
// visible system row (`fanout_summary`) and (b) backfill the matching
// fanout_card's per-task terminal statuses + a join epoch ms — otherwise a
// reload of an already-COMPLETED fanout shows every row stuck "running"
// forever (no live WS event will ever arrive to correct a done session).
// --------------------------------------------------------------------------

describe('transformChatHistory — fanout join-summary parsing (Critical 1 + 3)', () => {
  it('parses the join summary into a fanout_summary row AND backfills the fanout_card statuses + completedAtMs', () => {
    const summary = [
      '[Fanout a1b2c3d4] 3/4 succeeded.',
      '- [completed] Research A (worker:a1b2c3d4-0): Found 3 sources | transcript: /tmp/a.jsonl',
      '- [completed] Research B (worker:a1b2c3d4-1): Found 5 sources | transcript: /tmp/b.jsonl',
      '- [error] Research C (worker:a1b2c3d4-2): timed out | transcript: /tmp/c.jsonl',
      '- [stalled] Research D (worker:a1b2c3d4-3): no output | transcript: /tmp/d.jsonl',
    ].join('\n');

    const messages: ChatMessage[] = [
      asst({
        content: null,
        tool_calls: [tc('c1', 'fanout', '{"tasks":["Research A","Research B","Research C","Research D"]}')],
        timestamp: TS,
      }),
      tool('c1', 'Fanout a1b2c3d4 dispatched: 4 tasks', TS2),
      sys(summary, TS3),
    ];

    const items = transformChatHistory(messages);
    expect(items.map((i) => i.type)).toEqual(['agent_message', 'fanout_card', 'fanout_summary']);

    const card = items[1];
    expect(card.type).toBe('fanout_card');
    if (card.type === 'fanout_card') {
      expect(card.statuses).toEqual({
        'worker:a1b2c3d4-0': 'completed',
        'worker:a1b2c3d4-1': 'completed',
        'worker:a1b2c3d4-2': 'error',
        'worker:a1b2c3d4-3': 'stalled',
      });
      expect(card.completedAtMs).toBe(Date.parse(TS3));
    }

    const summaryItem = items[2];
    expect(summaryItem.type).toBe('fanout_summary');
    if (summaryItem.type === 'fanout_summary') {
      expect(summaryItem.fanout_id).toBe('a1b2c3d4');
      expect(summaryItem.content).toBe(summary);
    }
  });

  it('ignores a trailing guidance paragraph after the task lines', () => {
    const summary = [
      '[Fanout deadbeef] 1/1 succeeded.',
      '- [completed] Only task (worker:deadbeef-0): done | transcript: /tmp/x.jsonl',
      '',
      'All tasks finished — consider reviewing the transcripts before continuing.',
    ].join('\n');
    const messages: ChatMessage[] = [
      asst({ content: null, tool_calls: [tc('c1', 'fanout', '{"tasks":["Only task"]}')], timestamp: TS }),
      tool('c1', 'Fanout deadbeef dispatched: 1 tasks', TS2),
      sys(summary, TS3),
    ];
    const items = transformChatHistory(messages);
    const card = items.find((i) => i.type === 'fanout_card');
    expect(card).toBeDefined();
    if (card && card.type === 'fanout_card') {
      expect(card.statuses).toEqual({ 'worker:deadbeef-0': 'completed' });
    }
  });

  it('a fanout with NO join summary in history stays with no baked statuses (legitimate "still running" residual)', () => {
    const messages: ChatMessage[] = [
      asst({ content: null, tool_calls: [tc('c1', 'fanout', '{"tasks":["A"]}')], timestamp: TS }),
      tool('c1', 'Fanout 87654321 dispatched: 1 tasks', TS2),
    ];
    const items = transformChatHistory(messages);
    const card = items.find((i) => i.type === 'fanout_card');
    expect(card).toBeDefined();
    if (card && card.type === 'fanout_card') {
      expect(card.statuses).toBeUndefined();
      expect(card.completedAtMs).toBeUndefined();
    }
  });

  it('a non-fanout system message is not mistaken for a join summary', () => {
    const messages: ChatMessage[] = [
      sys('[Fanout not-a-valid-header] something else entirely', TS),
    ];
    const items = transformChatHistory(messages);
    expect(items.some((i) => i.type === 'fanout_summary')).toBe(false);
  });

  it('fanout_summary uses _meta.display_content when present (guidance never rendered)', () => {
    const full =
      '[Fanout deadbee1] 1/1 succeeded.\n' +
      '- [completed] T (worker:deadbee1-0): ok | transcript: /x/y.jsonl\n' +
      '\n' +
      'Synthesize these results into a single reply for the user — unlike a single ' +
      "sub-agent's completion, fanout results are not auto-echoed as chat bubbles, " +
      'so the user has not seen this content yet.';
    const display =
      '[Fanout deadbee1] 1/1 succeeded.\n' +
      '- [completed] T (worker:deadbee1-0): ok | transcript: /x/y.jsonl';
    const messages: ChatMessage[] = [
      { ...sys(full, TS3), _meta: { display_content: display } },
    ];
    const items = transformChatHistory(messages);
    const summary = items.find((i) => i.type === 'fanout_summary');
    expect(summary).toBeTruthy();
    if (summary && summary.type === 'fanout_summary') {
      expect(summary.content).toBe(display);
      expect(summary.content).not.toContain('Synthesize these results');
    }
  });

  it('fanout_summary falls back to full content for legacy messages without _meta', () => {
    const full =
      '[Fanout deadbee2] 1/1 succeeded.\n' +
      '- [completed] T (worker:deadbee2-0): ok | transcript: /x/y.jsonl\n\n' +
      'Synthesize these results into a single reply for the user.';
    const messages: ChatMessage[] = [sys(full, TS3)];
    const items = transformChatHistory(messages);
    const summary = items.find((i) => i.type === 'fanout_summary');
    expect(summary).toBeTruthy();
    if (summary && summary.type === 'fanout_summary') {
      expect(summary.content).toBe(full);
    }
  });
});

describe('transformChatHistory — compaction marker', () => {
  function compaction(timestamp = TS3): ChatMessage {
    return {
      role: 'system',
      content: 'Summary of prior conversation.',
      source: 'management',
      timestamp,
      _compaction: true,
    } as ChatMessage;
  }

  it('surfaces the compaction row instead of dropping it', () => {
    // The daemon has always persisted this marker and the transform has
    // always thrown it away, so an auto-compaction — which costs a real LLM
    // call and discards the middle of the conversation — was invisible.
    const items = transformChatHistory([
      user('first'),
      compaction(),
      { ...asst({ content: 'after', timestamp: TS4 }) },
    ]);
    expect(items.map((i) => i.type)).toEqual([
      'user_message',
      'compaction_marker',
      'agent_message',
    ]);
  });

  it('carries the marker timestamp', () => {
    const items = transformChatHistory([user('x'), compaction(TS5)]);
    const marker = items.find((i) => i.type === 'compaction_marker');
    expect(marker).toMatchObject({ type: 'compaction_marker', timestamp: TS5 });
  });

  it('does NOT drop the summary body into the chat as an agent message', () => {
    // The marker's content is the LLM-written summary meant for the model,
    // not for the reader. Rendering it would dump a wall of recap text in
    // the middle of the conversation.
    const items = transformChatHistory([user('x'), compaction()]);
    const texts = JSON.stringify(items);
    expect(texts).not.toContain('Summary of prior conversation.');
  });

  it('is not a session boundary — it must not grey out the history above it', () => {
    // session_separator drives the isHistorical pass. Reusing that type for
    // compaction would mark every message before the compaction as belonging
    // to an old session, which is a different (and wrong) claim.
    const items = transformChatHistory([
      user('before'),
      compaction(),
      { ...asst({ content: 'after', timestamp: TS4 }) },
    ]);
    expect(items.some((i) => i.type === 'session_separator')).toBe(false);
    for (const item of items) {
      expect((item as { isHistorical?: boolean }).isHistorical).toBeFalsy();
    }
  });

  it('closes an open tool capsule before the marker', () => {
    const items = transformChatHistory([
      user('go'),
      asst({ content: null, tool_calls: [tc('t1', 'read')], timestamp: TS2 }),
      tool('t1', 'ok', TS2),
      compaction(),
    ]);
    const order = items.map((i) => i.type);
    expect(order.indexOf('compaction_marker')).toBeGreaterThan(order.indexOf('agent_run'));
    expect(order[order.length - 1]).toBe('compaction_marker');
  });

  it('still greys history above a real session separator', () => {
    const items = transformChatHistory([
      { ...user('old'), session_id: 's1' },
      { ...asst({ content: 'old reply', timestamp: TS2 }), session_id: 's1' },
      { ...user('new'), session_id: 's2', timestamp: TS3 },
    ]);
    const first = items[0] as { isHistorical?: boolean };
    expect(first.isHistorical).toBe(true);
  });
});

// Spec 100: a sub-agent capsule carries each call's arguments and result
// (when the transcript has them) and the turn's thinking, and a dispatch
// still in flight renders as a running capsule the live events extend.
describe('transformChatHistory — spec 100 enriched sub-agent capsule', () => {
  type Run = Extract<ReturnType<typeof transformChatHistory>[number], { type: 'agent_run' }>;
  function capsuleOf(messages: ChatMessage[]): Run {
    const run = transformChatHistory(messages, '/w').find((i) => i.type === 'agent_run');
    expect(run).toBeDefined();
    return run as Run;
  }
  function sub(overrides: Partial<ChatMessage>): ChatMessage {
    return {
      role: 'assistant',
      content: 'Done.',
      source: 'sub_agent',
      timestamp: TS,
      sub_agent_handle: 'claude-code',
      sub_agent_duration: 4,
      ...overrides,
    };
  }

  it('rows with arguments render the live detail and an expandable result', () => {
    const run = capsuleOf([
      sub({
        sub_agent_tool_rows: [
          {
            name: 'Read', timestamp: TS, duration_seconds: 0.4, tool_call_id: 'tu1',
            arguments: { file_path: '/w/notes.txt' }, result_preview: 'hello', is_error: false,
          },
          {
            name: 'commandExecution', timestamp: TS2, duration_seconds: 2, tool_call_id: 'c1',
            arguments: { command: 'ls -la', cwd: '/w' },
            result_preview: 'a', result_total_chars: 900, result_total_lines: 40,
          },
        ],
      }),
    ]);

    expect(run.source).toBe('claude-code');
    const rows = run.items.filter((c) => c.type === 'tool_call_row');
    expect(rows).toHaveLength(2);
    const [read, cmd] = rows as Array<Extract<(typeof rows)[number], { type: 'tool_call_row' }>>;
    expect(read.tool_call_id).toBe('tu1');
    expect(read.target_description).toBe(
      describeWorkerTool('Read', { file_path: '/w/notes.txt' }, '/w'),
    );
    expect(read.target_description).toContain('notes.txt');
    expect(read.result_content).toBe('hello');
    expect(read.result_status).toBe('received');
    expect(cmd.target_description).toContain('ls -la');
    expect(cmd.category).toBe('command_exec');
    expect(cmd.result_totals).toEqual({ chars: 900, lines: 40 });
  });

  it('legacy rows keep the duration-only detail', () => {
    const run = capsuleOf([
      sub({ sub_agent_tool_rows: [{ name: 'Write', timestamp: TS, duration_seconds: 1.2 }] }),
    ]);
    const row = run.items[0];
    expect(row.type === 'tool_call_row' && row.target_description).toBe('1.2s');
    expect(row.type === 'tool_call_row' && row.result_content).toBeNull();
  });

  it('thinking sits between the rows it came between', () => {
    const run = capsuleOf([
      sub({
        sub_agent_tool_rows: [
          { name: 'Glob', timestamp: TS, duration_seconds: 0, tool_call_id: 'g', arguments: { pattern: '*.md' } },
        ],
        sub_agent_thinking: [
          { content: 'Find the notes first.', after_tool: 0 },
          { content: 'Found them.', after_tool: 1 },
          { content: '   ', after_tool: 1 },
        ],
      }),
    ]);
    expect(run.items.map((c) => c.type)).toEqual(['reasoning_block', 'tool_call_row', 'reasoning_block']);
    expect(run.has_thinking).toBe(true);
  });

  it('a thinking-only turn still gets its capsule', () => {
    const run = capsuleOf([
      sub({ sub_agent_tool_rows: [], sub_agent_thinking: [{ content: 'Just answer.', after_tool: 0 }] }),
    ]);
    expect(run.items.map((c) => c.type)).toEqual(['reasoning_block']);
  });

  it('an in-flight dispatch is a running capsule with pending rows and no answer yet', () => {
    const items = transformChatHistory(
      [
        sub({
          content: '',
          sub_agent_in_flight: true,
          sub_agent_tool_rows: [
            { name: 'Read', timestamp: TS, duration_seconds: 0, tool_call_id: 'a', arguments: { file_path: 'x' }, result_preview: 'ok' },
            { name: 'Bash', timestamp: TS2, duration_seconds: 0, tool_call_id: 'b', arguments: { command: 'sleep 5' } },
          ],
        }),
      ],
      '/w',
    );
    const run = items.find((i) => i.type === 'agent_run') as Run;
    expect(run.status).toBe('running');
    expect(run.ended_at).toBeNull();
    const statuses = run.items.map((c) => (c.type === 'tool_call_row' ? c.result_status : ''));
    expect(statuses).toEqual(['received', 'pending']);
    // No response bubble: the in-flight turn has not answered.
    expect(items.filter((i) => i.type === 'agent_message' && !i.isHeaderOnly)).toHaveLength(0);
    // The capsule is anchored under the worker's own header.
    expect(items[0]).toMatchObject({ type: 'agent_message', source: 'claude-code', isHeaderOnly: true });
  });
});

describe('transformChatHistory — spec 104 sub-agent stream (intermediate bubbles survive reload)', () => {
  type Item = ReturnType<typeof transformChatHistory>[number];
  type Run = Extract<Item, { type: 'agent_run' }>;
  type Msg = Extract<Item, { type: 'agent_message' }>;
  const isHeader = (i: Item): i is Msg => i.type === 'agent_message' && !!i.isHeaderOnly;
  const isBubble = (i: Item): i is Msg => i.type === 'agent_message' && !i.isHeaderOnly;
  const rowNames = (run: Run) =>
    run.items.filter((c) => c.type === 'tool_call_row').map((c) => (c.type === 'tool_call_row' ? c.tool_name : ''));

  const bash = { kind: 'tool' as const, name: 'Bash', timestamp: TS, duration_seconds: 1.0 };
  const write = { kind: 'tool' as const, name: 'Write', timestamp: TS3, duration_seconds: 0.5 };
  const msgA = { kind: 'message' as const, content: 'A: listed the files; now writing.', timestamp: TS2 };
  const msgB = { kind: 'message' as const, content: 'B: wrote it.', timestamp: TS4 };

  function sub(overrides: Partial<ChatMessage>): ChatMessage {
    return {
      role: 'assistant',
      content: 'B: wrote it.',
      source: 'sub_agent',
      timestamp: TS,
      sub_agent_handle: 'claude-code',
      sub_agent_tool_rows: [bash, write],
      sub_agent_duration: 3,
      ...overrides,
    };
  }

  it('renders the stream in order: capsule(Bash), bubble A, capsule(Write), bubble B — final text exactly once', () => {
    const items = transformChatHistory([sub({ sub_agent_stream: [bash, msgA, write, msgB] })], '/w');
    expect(items.map((i) => (isHeader(i) ? 'header' : i.type))).toEqual([
      'header', 'agent_run', 'agent_message', 'agent_run', 'agent_message',
    ]);
    const runs = items.filter((i): i is Run => i.type === 'agent_run');
    expect(runs.map(rowNames)).toEqual([['Bash'], ['Write']]);
    const bubbles = items.filter(isBubble);
    expect(bubbles.map((b) => b.content)).toEqual([msgA.content, msgB.content]);
    expect(bubbles.every((b) => b.source === 'claude-code')).toBe(true);
    expect(bubbles[0].timestamp).toBe(TS2);
    // The final text appears once (the stream's last message IS msg.content).
    expect(items.filter((i) => i.type === 'agent_message' && i.content === 'B: wrote it.')).toHaveLength(1);
    // Exactly one header anchor, before the first capsule, and none directly before a bubble.
    expect(items.filter(isHeader)).toHaveLength(1);
    expect(isHeader(items[0])).toBe(true);
    items.forEach((it, idx) => {
      if (isBubble(it) && idx > 0) expect(isHeader(items[idx - 1])).toBe(false);
    });
  });

  it('each stream capsule is a collapsed, completed sub_agent capsule under the worker handle with synthetic ids', () => {
    const items = transformChatHistory([sub({ sub_agent_stream: [bash, msgA, write, msgB] })], '/w');
    const runs = items.filter((i): i is Run => i.type === 'agent_run');
    for (const run of runs) {
      expect(run.capsule_id.startsWith('sub_agent:claude-code:')).toBe(true);
      expect(run.status).toBe('completed');
      expect(run.defaultExpanded).toBe(false);
      expect(run.source).toBe('claude-code');
      expect(run.ended_at).not.toBeNull();
    }
    expect(new Set(runs.map((r) => r.capsule_id)).size).toBe(2);
    const [r0, r1] = runs.map((r) => r.items[0]);
    expect(r0.type === 'tool_call_row' && r0.tool_call_id).toBe(`sub:claude-code:${TS}:0`);
    expect(r1.type === 'tool_call_row' && r1.tool_call_id).toBe(`sub:claude-code:${TS}:1`);
    expect(r0.type === 'tool_call_row' && r0.target_description).toBe('1.0s');
    expect(r0.type === 'tool_call_row' && r0.result_content).toBeNull();
    expect(r0.type === 'tool_call_row' && r0.result_status).toBe('received');
    // Durations: Bash ran 1.0s from TS.
    expect(runs[0].started_at).toBe(Date.parse(TS));
    expect(runs[0].ended_at).toBe(Date.parse(TS) + 1000);
  });

  it('D2: tool rows after the last message render as a trailing capsule after the final bubble', () => {
    const items = transformChatHistory([sub({ sub_agent_stream: [bash, msgA, write, msgB, bash] })], '/w');
    expect(items.map((i) => (isHeader(i) ? 'header' : i.type))).toEqual([
      'header', 'agent_run', 'agent_message', 'agent_run', 'agent_message', 'agent_run',
    ]);
    const last = items[items.length - 1] as Run;
    expect(rowNames(last)).toEqual(['Bash']);
    expect(last.status).toBe('completed');
  });

  it('a stream that starts with a message renders the bubble first, with no header anchor before it', () => {
    const items = transformChatHistory([sub({ sub_agent_stream: [msgA, bash, msgB] })], '/w');
    expect(items.map((i) => (isHeader(i) ? 'header' : i.type))).toEqual([
      'agent_message', 'agent_run', 'agent_message',
    ]);
    // The capsule after the bubble is anchored by the bubble itself (same handle).
    expect(items.filter(isHeader)).toHaveLength(0);
  });

  it('a message-only stream renders one bubble per message and no capsule (codex before spec 100)', () => {
    const items = transformChatHistory(
      [sub({ sub_agent_tool_rows: [], sub_agent_stream: [msgA, msgB] })],
      '/w',
    );
    expect(items.map((i) => i.type)).toEqual(['agent_message', 'agent_message']);
    expect(items.filter(isHeader)).toHaveLength(0);
    expect(items.map((i) => (i.type === 'agent_message' ? i.content : ''))).toEqual([msgA.content, msgB.content]);
  });

  it('skips empty / "(no response)" / ANSI-only message rows and strips ANSI from the rest', () => {
    const items = transformChatHistory(
      [
        sub({
          sub_agent_tool_rows: [],
          content: 'done',
          sub_agent_stream: [
            { kind: 'message', content: '   ', timestamp: TS },
            { kind: 'message', content: '(no response)', timestamp: TS2 },
            { kind: 'message', content: '\x1b[32mgreen\x1b[0m done', timestamp: TS3 },
          ],
        }),
      ],
      '/w',
    );
    expect(items.map((i) => i.type)).toEqual(['agent_message']);
    expect(items[0].type === 'agent_message' && items[0].content).toBe('green done');
  });

  it('spec 100 enrichment flows into the stream rows (arguments, result, totals) through the shared row builder', () => {
    const read = {
      kind: 'tool' as const, name: 'Read', timestamp: TS, duration_seconds: 0.4, tool_call_id: 'tu1',
      arguments: { file_path: '/w/notes.txt' }, result_preview: 'hello', result_total_chars: 900, result_total_lines: 40,
    };
    const items = transformChatHistory(
      [sub({ sub_agent_tool_rows: [read], sub_agent_stream: [read, msgB] })],
      '/w',
    );
    const run = items.find((i): i is Run => i.type === 'agent_run')!;
    const row = run.items[0];
    expect(row.type).toBe('tool_call_row');
    if (row.type !== 'tool_call_row') return;
    expect(row.tool_call_id).toBe('tu1');
    expect(row.target_description).toBe(describeWorkerTool('Read', { file_path: '/w/notes.txt' }, '/w'));
    expect(row.result_content).toBe('hello');
    expect(row.result_totals).toEqual({ chars: 900, lines: 40 });
    // Identical to what the fallback path builds for the same row.
    const fallback = transformChatHistory([sub({ sub_agent_tool_rows: [read] })], '/w')
      .find((i): i is Run => i.type === 'agent_run')!;
    expect(row).toEqual(fallback.items[0]);
  });

  it('thinking rows land inside the capsule they came in, in order', () => {
    const think1 = { kind: 'thinking' as const, content: 'Find the files first.', timestamp: TS };
    const think2 = { kind: 'thinking' as const, content: 'Now write.', timestamp: TS3 };
    const items = transformChatHistory(
      [
        sub({
          sub_agent_thinking: [{ content: think1.content, after_tool: 0 }, { content: think2.content, after_tool: 1 }],
          sub_agent_stream: [think1, bash, msgA, think2, write, msgB],
        }),
      ],
      '/w',
    );
    const runs = items.filter((i): i is Run => i.type === 'agent_run');
    expect(runs).toHaveLength(2);
    expect(runs[0].items.map((c) => c.type)).toEqual(['reasoning_block', 'tool_call_row']);
    expect(runs[1].items.map((c) => c.type)).toEqual(['reasoning_block', 'tool_call_row']);
    expect(runs.every((r) => r.has_thinking)).toBe(true);
    const t0 = runs[0].items[0];
    expect(t0.type === 'reasoning_block' && t0.content).toBe(think1.content);
  });

  it('a thinking row after the last message becomes a trailing (thinking-only) capsule', () => {
    const think = { kind: 'thinking' as const, content: 'Done thinking.', timestamp: TS5 };
    const items = transformChatHistory([sub({ sub_agent_stream: [bash, msgB, think] })], '/w');
    expect(items.map((i) => (isHeader(i) ? 'header' : i.type))).toEqual([
      'header', 'agent_run', 'agent_message', 'agent_run',
    ]);
    const last = items[items.length - 1] as Run;
    expect(last.items.map((c) => c.type)).toEqual(['reasoning_block']);
  });

  it('in flight: intermediate capsules are completed, the trailing one is running, bubbles so far are shown', () => {
    const pendingWrite = { ...write, tool_call_id: 'w1', arguments: { file_path: 'x' } };
    const items = transformChatHistory(
      [
        sub({
          content: '',
          sub_agent_in_flight: true,
          sub_agent_tool_rows: [bash, pendingWrite],
          sub_agent_stream: [bash, msgA, pendingWrite],
        }),
      ],
      '/w',
    );
    expect(items.map((i) => (isHeader(i) ? 'header' : i.type))).toEqual([
      'header', 'agent_run', 'agent_message', 'agent_run',
    ]);
    const runs = items.filter((i): i is Run => i.type === 'agent_run');
    expect(runs[0].status).toBe('completed');
    expect(runs[1].status).toBe('running');
    expect(runs[1].ended_at).toBeNull();
    const pending = runs[1].items[0];
    expect(pending.type === 'tool_call_row' && pending.result_status).toBe('pending');
    expect(items.filter(isBubble).map((b) => b.content)).toEqual([msgA.content]);
  });

  it('in flight with the stream ending in a message: no running capsule, the bubble is last', () => {
    const items = transformChatHistory(
      [sub({ content: '', sub_agent_in_flight: true, sub_agent_tool_rows: [bash], sub_agent_stream: [bash, msgA] })],
      '/w',
    );
    expect(items.map((i) => (isHeader(i) ? 'header' : i.type))).toEqual(['header', 'agent_run', 'agent_message']);
    expect(items.some((i) => i.type === 'agent_run' && i.status === 'running')).toBe(false);
  });

  it('no sub_agent_stream (old daemon) → byte-identical to the legacy capsule + final-bubble shape', () => {
    const legacy = sub({});
    const withEmpty = sub({ sub_agent_stream: [] });
    const withoutField = transformChatHistory([legacy], '/w');
    expect(transformChatHistory([withEmpty], '/w')).toEqual(withoutField);
    // And the legacy shape itself is still header + ONE capsule of every row + ONE bubble.
    expect(withoutField.map((i) => (isHeader(i) ? 'header' : i.type))).toEqual(['header', 'agent_run', 'agent_message']);
    expect(rowNames(withoutField[1] as Run)).toEqual(['Bash', 'Write']);
  });

  it('a stream turn is finalized like any other: an open management capsule closes before it', () => {
    const messages: ChatMessage[] = [
      { role: 'assistant', content: '', source: 'management', timestamp: TS,
        tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read_file', arguments: '{"path":"a"}' } }] },
      { role: 'tool', content: 'x', tool_call_id: 'c1', source: 'tool', timestamp: TS2 },
      sub({ timestamp: TS3, sub_agent_stream: [bash, msgA, write, msgB] }),
    ];
    const items = transformChatHistory(messages, '/w');
    const mgr = items.findIndex((i) => i.type === 'agent_run' && i.capsule_id.startsWith('cap:'));
    expect(mgr).toBeGreaterThanOrEqual(0);
    // The worker's first capsule gets its own header: the previous item is the manager's capsule.
    expect(isHeader(items[mgr + 1])).toBe(true);
    expect((items[mgr + 1] as Msg).source).toBe('claude-code');
    expect(items[mgr + 2].type).toBe('agent_run');
    expect((items[mgr + 2] as Run).source).toBe('claude-code');
  });
});

describe('describeWorkerTool', () => {
  const en: ActivityTranslate = (k, v) => translate('en', k, v);

  it('renders known worker tools like the manager\'s', () => {
    expect(describeWorkerTool('Bash', { command: 'npm test' }, '/w', en)).toBe(
      describeLiveActivity('shell', { command: 'npm test' }, '/w', en, ''),
    );
    expect(describeWorkerTool('Read', { file_path: '/w/a.ts' }, '/w', en)).toContain('a.ts');
    expect(describeWorkerTool('Grep', { pattern: 'TODO' }, '/w', en)).toContain('TODO');
  });

  it('lists a codex fileChange\'s paths', () => {
    expect(
      describeWorkerTool('fileChange', { changes: [{ path: 'a.py' }, { path: 'b.py' }] }, '/w', en),
    ).toContain('a.py, b.py');
  });

  it('falls back to the main argument, then to "Used X"', () => {
    expect(describeWorkerTool('fs.list', { path: '/w/src' }, '/w', en)).toBe('/w/src');
    expect(describeWorkerTool('webSearch', { query: 'orbital agents' }, '/w', en)).toBe(
      translate('en', 'activity.searched', { query: 'orbital agents' }),
    );
    expect(describeWorkerTool('imageView', { path: '/tmp/x.png' }, '/w', en)).toBe('/tmp/x.png');
    expect(describeWorkerTool('Terminal', undefined, '/w', en)).toBe(
      translate('en', 'activity.usedTool', { name: 'Terminal' }),
    );
  });
});


// --------------------------------------------------------------------------
// Spec 116 D5 — a pinned send refused for an old CLI (b17edb8's
// "could not start" row) renders, and a reason-tagged row is translated.
// --------------------------------------------------------------------------

describe('transformChatHistory — sub_agent_dispatch_failed rows (spec 116)', () => {
  const RAW = '[Sub-agent] claude-code could not start: Error: Claude Code v2.1.100 is older than 2.1.114; open Settings → Sub-agents and click Update';
  const row = (meta?: Record<string, unknown>): ChatMessage => ({
    role: 'system', source: 'daemon', content: RAW, timestamp: TS,
    ...(meta ? { _meta: meta } : {}),
  });
  const failedItem = (items: ReturnType<typeof transformChatHistory>) => {
    const item = items.find(i => i.type === 'sub_agent_activity');
    if (!item || item.type !== 'sub_agent_activity') throw new Error('no activity row');
    return item;
  };

  it('a could-not-start row renders as a failed activity with its raw content', () => {
    const item = failedItem(transformChatHistory([row()]));
    expect(item.action).toBe('failed');
    expect(item.handle).toBe('claude-code');
    expect(item.error).toBe('Error: Claude Code v2.1.100 is older than 2.1.114; open Settings → Sub-agents and click Update');
  });

  it('reason=update_needed renders the translated sentence (zh)', () => {
    const zh: ActivityTranslate = (key, vars) => translate('zh', key, vars);
    const item = failedItem(transformChatHistory([row({
      event: 'sub_agent_dispatch_failed', handle: 'claude-code', suppress_wake: true,
      reason: 'update_needed', version: '2.1.100', min_version: '2.1.114', name: 'Claude Code',
    })], undefined, zh));
    expect(item.action).toBe('failed');
    expect(item.error).toBe('Claude Code 版本过旧（v2.1.100），需要 2.1.114 或更新。请到 设置 → 子Agent 点击更新。');
  });

  it('reason=update_needed renders the English sentence by default, naming the handle when no name is given', () => {
    const item = failedItem(transformChatHistory([row({
      event: 'sub_agent_dispatch_failed', handle: 'claude-code',
      reason: 'update_needed', version: '2.1.100', min_version: '2.1.114',
    })]));
    expect(item.error).toBe('claude-code v2.1.100 is too old; Orbital needs 2.1.114 or newer. Open Settings → Sub-agents and click Update.');
  });

  it('any other reason keeps the raw content', () => {
    const item = failedItem(transformChatHistory([row({
      event: 'sub_agent_dispatch_failed', handle: 'claude-code', reason: 'not_authenticated',
    })]));
    expect(item.error).toContain('older than 2.1.114');
    expect(item.error).toContain('click Update');
  });
});

// ---------------------------------------------------------------------------
// Keyless onboarding + session import: the two new daemon rows, and imported
// assistant turns (sub_agent_terminal completed rows carrying `reply`).
// ---------------------------------------------------------------------------
describe('transformChatHistory — import and worker-onboarding rows', () => {
  const IMPORTED_META = {
    event: 'imported_transcript', agent: 'codex', native_id: 'abc',
    last_activity: '2026-10-20T09:00:00Z', truncated: false,
  };
  const daemon = (content: string, _meta: Record<string, unknown>, timestamp = TS): ChatMessage => ({
    role: 'system', source: 'daemon', content, timestamp, _meta,
  } as ChatMessage);
  const user = (content: string, timestamp: string): ChatMessage => ({
    role: 'user', source: 'user', content, timestamp, _meta: { imported: true },
  } as ChatMessage);
  const reply = (handle: string, text: string, timestamp: string): ChatMessage => daemon(
    `[Sub-agent] ${handle} completed. Summary: ${text.slice(0, 200)}`,
    {
      event: 'sub_agent_terminal', kind: 'completed', handle, reply: text,
      display_content: `[Sub-agent] ${handle} completed. Summary: ${text.slice(0, 200)}`,
      suppress_wake: true, imported: true,
    },
    timestamp,
  );

  it('the imported_transcript row becomes a daemon divider carrying agent + date', () => {
    const items = transformChatHistory([
      daemon('[Imported] Conversation imported from Codex (last active 2026-10-20)', IMPORTED_META),
    ]);
    expect(items).toEqual([{
      type: 'daemon_row', event: 'imported_transcript', handle: 'codex',
      lastActivity: '2026-10-20T09:00:00Z', truncated: false, timestamp: TS,
    }]);
  });

  it('carries the truncated flag', () => {
    const items = transformChatHistory([
      daemon('[Imported] … — earlier messages not imported', { ...IMPORTED_META, truncated: true }),
    ]);
    expect(items[0]).toMatchObject({ type: 'daemon_row', truncated: true });
  });

  it('the worker_onboarding row becomes a daemon divider, never a bubble', () => {
    const items = transformChatHistory([
      daemon('[Onboarding] codex is setting up this project', {
        event: 'worker_onboarding', handle: 'codex', mode: 'scan', suppress_wake: true,
      }),
    ]);
    expect(items).toEqual([{
      type: 'daemon_row', event: 'worker_onboarding', handle: 'codex', timestamp: TS,
    }]);
  });

  it('an imported transcript renders user bubbles and worker bubbles, in order', () => {
    const longReply = 'A'.repeat(250) + '\n\n(tools: Read ×3, Edit ×1)';
    const items = transformChatHistory([
      daemon('[Imported] Conversation imported from Codex', IMPORTED_META, TS),
      user('fix the login bug', TS2),
      reply('codex', longReply, TS3),
      user('thanks, now add a test', TS4),
      reply('codex', 'Added test_login.py.', TS5),
    ]);
    expect(items.map((it) => it.type)).toEqual([
      'daemon_row', 'user_message', 'agent_message', 'user_message', 'agent_message',
    ]);
    // The whole reply, not the 200-char summary, under the worker's identity.
    expect(items[2]).toMatchObject({ type: 'agent_message', source: 'codex', content: longReply, timestamp: TS3 });
    expect(items[4]).toMatchObject({ type: 'agent_message', source: 'codex', content: 'Added test_login.py.' });
    expect(items.some((it) => it.type === 'sub_agent_activity')).toBe(false);
  });

  it('a live completion whose transcript join already drew the bubble keeps today\'s one-liner', () => {
    const items = transformChatHistory([
      user('hi', TS),
      { role: 'assistant', source: 'sub_agent', sub_agent_handle: 'codex', content: 'Hello!', timestamp: TS2 } as ChatMessage,
      reply('codex', 'Hello!', TS3),
    ]);
    expect(items.map((it) => it.type)).toEqual(['user_message', 'agent_message', 'sub_agent_activity']);
  });

  it('a completed row without a reply stays the one-line marker', () => {
    const items = transformChatHistory([
      user('hi', TS),
      daemon('[Sub-agent] codex completed. Summary: done', { event: 'sub_agent_terminal', kind: 'completed', handle: 'codex' }, TS2),
    ]);
    expect(items.map((it) => it.type)).toEqual(['user_message', 'sub_agent_activity']);
  });
});

describe('transformChatHistory — spec 129 continued rows (self-initiated worker turns)', () => {
  const daemon = (content: string, _meta: Record<string, unknown>, timestamp = TS): ChatMessage => ({
    role: 'system', source: 'daemon', content, timestamp, _meta,
  } as ChatMessage);
  const user = (content: string, timestamp: string): ChatMessage => ({
    role: 'user', source: 'user', content, timestamp,
  } as ChatMessage);
  const continued = (handle: string, text: string, timestamp: string): ChatMessage => daemon(
    `[Sub-agent] ${handle} continued. Summary: ${text}. Transcript: /t/x.jsonl. This turn was started by the sub-agent itself.`,
    {
      event: 'sub_agent_terminal', kind: 'continued', handle, reply: text, cause: 'success',
      display_content: `[Sub-agent] ${handle} continued. Summary: ${text}. Transcript: /t/x.jsonl.`,
      suppress_wake: true,
    },
    timestamp,
  );
  const completed = (handle: string, text: string, timestamp: string): ChatMessage => daemon(
    `[Sub-agent] ${handle} completed. Summary: ${text}. Transcript: /t/x.jsonl.`,
    {
      event: 'sub_agent_terminal', kind: 'completed', handle, reply: text,
      display_content: `[Sub-agent] ${handle} completed. Summary: ${text}. Transcript: /t/x.jsonl.`,
      suppress_wake: true,
    },
    timestamp,
  );

  it('a continued row with a reply is the worker\'s own bubble, even after its reply to the user', () => {
    const items = transformChatHistory([
      user('run the render in the background and tell me when it is done', TS),
      { role: 'assistant', source: 'sub_agent', sub_agent_handle: 'claude-code', content: 'Started it; I will report when it finishes.', timestamp: TS2 } as ChatMessage,
      completed('claude-code', 'Started it; I will report when it finishes.', TS3),
      continued('claude-code', 'The render finished: v2 is at out/v2.mp4.', TS4),
    ]);
    expect(items.map((it) => it.type)).toEqual([
      'user_message', 'agent_message', 'sub_agent_activity', 'agent_message',
    ]);
    expect(items[3]).toMatchObject({
      type: 'agent_message', source: 'claude-code',
      content: 'The render finished: v2 is at out/v2.mp4.', timestamp: TS4,
    });
  });

  it('a continued row is never paired as the reply to the preceding user row', () => {
    // The user's prompt is still unanswered (its dispatch is in flight):
    // the continued bubble must not be mistaken for the reply, and the
    // completed row that follows still gets its own one-liner.
    const items = transformChatHistory([
      user('how is it going?', TS),
      continued('claude-code', 'Re-ran ffmpeg with the fixed flags.', TS2),
      { role: 'assistant', source: 'sub_agent', sub_agent_handle: 'claude-code', content: 'Going well, nearly done.', timestamp: TS3 } as ChatMessage,
      completed('claude-code', 'Going well, nearly done.', TS4),
    ]);
    expect(items.map((it) => it.type)).toEqual([
      'user_message', 'agent_message', 'agent_message', 'sub_agent_activity',
    ]);
    expect(items[1]).toMatchObject({ source: 'claude-code', content: 'Re-ran ffmpeg with the fixed flags.' });
    expect(items[2]).toMatchObject({ source: 'claude-code', content: 'Going well, nearly done.' });
    expect(items[3]).toMatchObject({ type: 'sub_agent_activity', action: 'completed' });
  });

  it('a continued row without text is a one-line marker with its own action', () => {
    const items = transformChatHistory([
      user('hi', TS),
      daemon('[Sub-agent] claude-code continued. Summary: (no output). Transcript: /t/x.jsonl.', {
        event: 'sub_agent_terminal', kind: 'continued', handle: 'claude-code',
        reply: '(no output)', cause: 'error', suppress_wake: true,
        display_content: '[Sub-agent] claude-code continued. Summary: (no output). Transcript: /t/x.jsonl.',
      }, TS2),
    ]);
    expect(items.map((it) => it.type)).toEqual(['user_message', 'sub_agent_activity']);
    expect(items[1]).toMatchObject({ type: 'sub_agent_activity', action: 'continued', handle: 'claude-code' });
  });
});

describe('transformChatHistory — worker onboarding hides its setup-instruction marker', () => {
  const onboarding: ChatMessage = {
    role: 'system', source: 'daemon', content: '[Onboarding] codex is setting up this project', timestamp: TS,
    _meta: { event: 'worker_onboarding', handle: 'codex', mode: 'interview', suppress_wake: true },
  } as ChatMessage;
  const marker = (handle: string, timestamp = TS2): ChatMessage => ({
    role: 'system', source: 'daemon', timestamp,
    content: `[Sub-agent] Message sent to ${handle}: "You are ${handle} working directly with the user in Orbital project /tmp/ws. Orbital's own agent is not configured…"`,
    _meta: { dispatch_id: 'd-1', handle, transcript_path: '/tmp/t.jsonl' },
  } as ChatMessage);
  const opener: ChatMessage = {
    role: 'system', source: 'daemon', timestamp: TS3,
    content: '[Sub-agent] codex completed. Summary: Hi! What is this project for?',
    _meta: {
      event: 'sub_agent_terminal', kind: 'completed', handle: 'codex',
      reply: 'Hi! What is this project for?',
      display_content: '[Sub-agent] codex completed. Summary: Hi! What is this project for?',
      suppress_wake: true,
    },
  } as ChatMessage;

  it('onboarding row + marker + terminal → one onboarding capsule and one worker bubble', () => {
    const items = transformChatHistory([onboarding, marker('codex'), opener]);
    expect(items.map((it) => it.type)).toEqual(['daemon_row', 'agent_message']);
    expect(items[0]).toMatchObject({ type: 'daemon_row', event: 'worker_onboarding', handle: 'codex' });
    // No user message yet: the terminal row's reply is the worker's first bubble.
    expect(items[1]).toMatchObject({ type: 'agent_message', source: 'codex', content: 'Hi! What is this project for?' });
  });

  it('a marker for a different worker after the onboarding row still renders', () => {
    const items = transformChatHistory([onboarding, marker('claude-code')]);
    expect(items.map((it) => it.type)).toEqual(['daemon_row', 'sub_agent_activity']);
  });

  it('a marker that does not directly follow the onboarding row still renders', () => {
    const user: ChatMessage = { role: 'user', source: 'user', content: 'hello', timestamp: TS2 } as ChatMessage;
    const items = transformChatHistory([onboarding, user, marker('codex', TS3)]);
    expect(items.map((it) => it.type)).toEqual(['daemon_row', 'user_message', 'sub_agent_activity']);
  });

  it('an ordinary dispatch marker is unchanged', () => {
    const items = transformChatHistory([marker('codex')]);
    expect(items.map((it) => it.type)).toEqual(['sub_agent_activity']);
  });
});

describe('transformChatHistory — the tidy pass row (_kind: memory_tidy)', () => {
  // Spec 124 §3.7 follow-up: the Memory session is hidden; the one trace a
  // tidy pass leaves is a daemon system row in the session that triggered
  // it. It renders as ONE collapsed capsule (no bubble, no agent header)
  // whose children are the files the pass changed.
  function tidyRow(meta: Record<string, unknown>, content: string, timestamp = TS3): ChatMessage {
    return {
      role: 'system',
      content,
      source: 'daemon',
      timestamp,
      _meta: { event: 'memory_tidy', trigger: 'over_budget', ...meta },
      _kind: 'memory_tidy',
    };
  }

  it('renders as one collapsed memory_tidy capsule after the reply, one child per changed file', () => {
    const items = transformChatHistory([
      user('plan it', TS),
      asst({ content: 'Here is the plan.', timestamp: TS2 }),
      tidyRow(
        { outcome: 'restored', archived: 3, restored: 7, files: ['orbital/PROJECT_STATE.md', 'orbital/INDEX.md'], llm_calls: 4, ended: 'text' },
        'Memory tidied: 3 archived, 7 restored',
      ),
      user('now the copy', TS4),
    ]);
    expect(items.map((i) => i.type)).toEqual(['user_message', 'agent_message', 'agent_run', 'user_message']);
    const capsule = items[2];
    expect(capsule.type).toBe('agent_run');
    if (capsule.type === 'agent_run') {
      expect(capsule.memory_record).toBeUndefined();
      expect(capsule.memory_tidy).toEqual({ outcome: 'restored', archived: 3, restored: 7 });
      expect(capsule.status).toBe('completed');
      expect(capsule.defaultExpanded).toBeUndefined();
      expect(capsule.source).toBeUndefined();
      expect(capsule.items.map((x) => x.type)).toEqual(['tool_call_row', 'tool_call_row']);
      const [a, b] = capsule.items;
      if (a.type === 'tool_call_row' && b.type === 'tool_call_row') {
        expect(a.tool_name).toBe('edit');
        expect(a.target_description).toBe('orbital/PROJECT_STATE.md');
        expect(a.category).toBe('file_edit');
        expect(a.result_status).toBe('received');
        expect(b.target_description).toBe('orbital/INDEX.md');
      }
      expect(capsule.tool_call_count_by_name).toEqual({ edit: 2 });
      expect(capsule.started_at).toBe(Date.parse(TS3));
      expect(capsule.ended_at).toBe(Date.parse(TS3));
    }
  });

  it('a pass that changed nothing still shows one capsule with no children; a failed pass carries its error', () => {
    const items = transformChatHistory([
      user('plan it', TS),
      asst({ content: 'Here is the plan.', timestamp: TS2 }),
      tidyRow({ outcome: 'no_change', archived: 0, restored: 0, files: [] }, 'Memory tidied: nothing to change', TS3),
      tidyRow({ outcome: 'failed', archived: 0, restored: 0, files: [], error: 'turn timed out' }, 'Memory tidy failed: turn timed out', TS4),
    ]);
    expect(items.map((i) => i.type)).toEqual(['user_message', 'agent_message', 'agent_run', 'agent_run']);
    const quiet = items[2];
    const failed = items[3];
    if (quiet.type === 'agent_run') {
      expect(quiet.memory_tidy).toEqual({ outcome: 'no_change', archived: 0, restored: 0 });
      expect(quiet.items).toEqual([]);
    }
    if (failed.type === 'agent_run') {
      expect(failed.memory_tidy).toEqual({ outcome: 'failed', archived: 0, restored: 0, error: 'turn timed out' });
      expect(failed.status).toBe('error');
    }
  });

  it('never interrupts an open management capsule: the tidy row closes it first', () => {
    const items = transformChatHistory([
      user('go', TS),
      asst({ content: null, tool_calls: [tc('r1', 'read', '{"path":"README.md"}')], timestamp: TS2 }),
      tool('r1', 'readme text', TS2),
      tidyRow({ outcome: 'edited', archived: 1, restored: 0, files: ['orbital/DECISIONS.md'] }, 'Memory tidied: 1 archived, 0 restored', TS3),
      asst({ content: 'Done.', timestamp: TS4 }),
    ]);
    const types = items.map((i) => i.type);
    expect(types.filter((t) => t === 'agent_run')).toHaveLength(2);
    const tidy = items.find((i) => i.type === 'agent_run' && i.memory_tidy);
    expect(tidy).toBeDefined();
  });
});

describe('transformChatHistory — spec 112 memory record turn (_kind: memory_record)', () => {
  // The daemon's end-of-run record turn: the [MEMORY] ask, the model's edits
  // and its closing text all carry `_kind: "memory_record"`. The reply the
  // user already saw stays the reply; the record turn renders as a collapsed
  // activity capsule only when it edited something, and as nothing otherwise.
  function memoryAsk(timestamp = TS3): ChatMessage {
    return {
      role: 'system',
      content: '[MEMORY] This run is ending. Decide whether anything durable happened in it.',
      source: 'daemon',
      timestamp,
      _meta: { event: 'memory_record', trigger: 'run_end' },
      _kind: 'memory_record',
    };
  }

  it('an edit in the record turn renders as one memory capsule after the reply, with no bubble or header', () => {
    const items = transformChatHistory([
      user('plan it', TS),
      asst({ content: 'Here is the plan.', timestamp: TS2 }),
      memoryAsk(TS3),
      asst({
        content: null,
        tool_calls: [tc('m1', 'edit', '{"path":"orbital/PROJECT_STATE.md"}')],
        timestamp: TS4,
        _kind: 'memory_record',
      }),
      { ...tool('m1', 'Edited orbital/PROJECT_STATE.md', TS5), _kind: 'memory_record' },
      asst({ content: 'Done, updated state.', timestamp: TS6, _kind: 'memory_record' }),
    ]);
    expect(items.map((i) => i.type)).toEqual(['user_message', 'agent_message', 'agent_run']);
    const reply = items[1];
    if (reply.type === 'agent_message') expect(reply.content).toBe('Here is the plan.');
    const capsule = items[2];
    expect(capsule.type).toBe('agent_run');
    if (capsule.type === 'agent_run') {
      expect(capsule.memory_record).toBe(true);
      expect(capsule.items.map((x) => x.type)).toEqual(['tool_call_row']);
      const row = capsule.items[0];
      if (row.type === 'tool_call_row') {
        expect(row.tool_name).toBe('edit');
        expect(row.result_status).toBe('received');
        expect(row.result_content).toBe('Edited orbital/PROJECT_STATE.md');
      }
      expect(capsule.defaultExpanded).toBeUndefined();
    }
  });

  it('a record turn with no tool call renders nothing at all', () => {
    const items = transformChatHistory([
      user('thanks', TS),
      asst({ content: 'You are welcome.', timestamp: TS2 }),
      memoryAsk(TS3),
      asst({ content: 'nothing to record', timestamp: TS4, _kind: 'memory_record' }),
    ]);
    expect(items.map((i) => i.type)).toEqual(['user_message', 'agent_message']);
  });

  it('a record turn that read before editing keeps both calls in one capsule', () => {
    const items = transformChatHistory([
      user('go', TS),
      asst({ content: 'Shipped.', timestamp: TS2 }),
      memoryAsk(TS3),
      asst({ content: null, tool_calls: [tc('r1', 'read', '{"path":"orbital/INDEX.md"}')], timestamp: TS4, _kind: 'memory_record' }),
      { ...tool('r1', 'index text', TS4), _kind: 'memory_record' },
      asst({ content: null, tool_calls: [tc('e1', 'edit', '{"path":"orbital/INDEX.md"}')], timestamp: TS5, _kind: 'memory_record' }),
      { ...tool('e1', 'Edited', TS5), _kind: 'memory_record' },
      asst({ content: 'Done.', timestamp: TS6, _kind: 'memory_record' }),
    ]);
    const capsules = items.filter((i) => i.type === 'agent_run');
    expect(capsules).toHaveLength(1);
    if (capsules[0].type === 'agent_run') {
      expect(capsules[0].items.map((x) => x.type)).toEqual(['tool_call_row', 'tool_call_row']);
      expect(capsules[0].tool_call_count_by_name).toEqual({ read: 1, edit: 1 });
    }
  });

  it('a pinned worker record turn renders its edits as a memory capsule under the worker, never its reminder or reply', () => {
    const items = transformChatHistory([
      user('fix the route', TS, ),
      { ...user('[Orbital memory] Your last turn is ending.', TS3), target: 'codex', source: 'daemon', _kind: 'memory_record' },
      {
        role: 'system', source: 'daemon', timestamp: TS3, _kind: 'memory_record',
        content: '[Sub-agent] Message sent to codex: "[Orbital memory] …". Transcript: /t.jsonl',
        _meta: { dispatch_id: 'd9', handle: 'codex', transcript_path: '/t.jsonl' },
      },
      {
        role: 'assistant', source: 'sub_agent', timestamp: TS4, _kind: 'memory_record',
        content: 'Updated PROJECT_STATE.md.',
        sub_agent_handle: 'codex',
        sub_agent_tool_rows: [{ name: 'Edit', timestamp: TS4, duration_seconds: 0.4, arguments: { file_path: 'orbital/PROJECT_STATE.md' } }],
        sub_agent_duration: 1.2,
      } as ChatMessage,
      {
        role: 'system', source: 'daemon', timestamp: TS5, _kind: 'memory_record',
        content: '[Sub-agent] codex completed. Summary: Updated PROJECT_STATE.md. Transcript: /t.jsonl.',
        _meta: { event: 'sub_agent_terminal', kind: 'completed', handle: 'codex' },
      },
    ]);
    expect(items.map((i) => i.type)).toEqual(['user_message', 'agent_run']);
    const capsule = items[1];
    if (capsule.type === 'agent_run') {
      expect(capsule.memory_record).toBe(true);
      expect(capsule.source).toBe('codex');
      expect(capsule.items.map((x) => x.type)).toEqual(['tool_call_row']);
    }
  });

  it('a pinned worker record turn with no tool rows renders nothing', () => {
    const items = transformChatHistory([
      user('hi', TS),
      { ...user('[Orbital memory] Your last turn is ending.', TS3), target: 'codex', source: 'daemon', _kind: 'memory_record' },
      {
        role: 'assistant', source: 'sub_agent', timestamp: TS4, _kind: 'memory_record',
        content: 'nothing to record', sub_agent_handle: 'codex', sub_agent_tool_rows: [], sub_agent_duration: 0.3,
      } as ChatMessage,
    ]);
    expect(items.map((i) => i.type)).toEqual(['user_message']);
  });
});

describe('spec 146: ask rows', () => {
  it('an ask row becomes one line outside the capsule', () => {
    const items = transformChatHistory([
      { role: 'user', content: 'go', timestamp: 't0' },
      { role: 'system', source: 'daemon', _kind: 'ask_card', content: 'Keep them?', _meta: { event: 'ask_opened', ask_id: 'abc123' }, timestamp: 't1' },
    ] as never);
    expect(items.map((i) => i.type)).toEqual(['user_message', 'ask_line']);
    expect((items[1] as { text: string }).text).toBe('Keep them?');
  });
});
