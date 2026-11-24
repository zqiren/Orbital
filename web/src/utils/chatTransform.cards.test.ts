// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Spec 195: dispatched agents render as one card each on reload.

import { describe, it, expect } from 'vitest';
import { transformChatHistory, type DisplayItem } from './chatTransform';
import type { ChatMessage } from '../types';

const T0 = '2026-11-24T10:00:00Z';
const T1 = '2026-11-24T10:00:01Z';
const T2 = '2026-11-24T10:00:02Z';
const T3 = '2026-11-24T10:00:30Z';
const T4 = '2026-11-24T10:01:00Z';
const T5 = '2026-11-24T10:01:05Z';

type Card = Extract<DisplayItem, { type: 'dispatch_card' }>;

function marker(
  handle: string,
  preview: string,
  ts: string,
  meta: Record<string, unknown> = {},
): ChatMessage {
  return {
    role: 'system',
    content: `[Sub-agent] Message sent to ${handle}: "${preview}". Transcript: /t/${handle}.jsonl`,
    source: 'management',
    timestamp: ts,
    _meta: { dispatch_id: `s1:${handle}${ts.slice(-3, -1)}`, handle, transcript_path: `/t/${handle}.jsonl`, ...meta },
  };
}

function run(handle: string, ts: string, reply: string, inFlight = false): ChatMessage {
  return {
    role: 'assistant',
    content: inFlight ? '' : reply,
    source: 'sub_agent',
    sub_agent_handle: handle,
    timestamp: ts,
    sub_agent_tool_rows: [],
    sub_agent_stream: [
      { kind: 'tool', name: 'Read', timestamp: ts, duration_seconds: 1, tool_call_id: `${handle}-r1`, arguments: { file_path: 'a.md' } },
      { kind: 'thinking', content: 'checking the menu', timestamp: ts },
      ...(inFlight ? [] : [{ kind: 'message' as const, content: reply, timestamp: ts }]),
    ],
    ...(inFlight ? { sub_agent_in_flight: true } : {}),
  };
}

function terminal(handle: string, kind: string, ts: string, extra: Record<string, unknown> = {}, text?: string): ChatMessage {
  const content =
    text ??
    (kind === 'completed'
      ? `[Sub-agent] ${handle} completed. Summary: done. Transcript: /t/${handle}.jsonl.`
      : kind === 'error'
        ? `[Sub-agent] ${handle} stopped with error: boom`
        : `[Sub-agent] ${handle} stopped by user.`);
  return {
    role: 'system',
    content,
    source: 'management',
    timestamp: ts,
    _meta: { event: 'sub_agent_terminal', kind, display_content: content, ...extra },
  };
}

const cards = (items: DisplayItem[]) => items.filter((i): i is Card => i.type === 'dispatch_card');

describe('spec 195 — dispatch cards on reload', () => {
  it('an Orbital dispatch is one card, its reply a bubble right under it, no lifecycle lines', () => {
    const items = transformChatHistory([
      { role: 'user', content: 'compare prices', source: 'user', timestamp: T0 },
      { role: 'assistant', content: '交给 Claude Code。', source: 'management', timestamp: T1 },
      { role: 'system', content: '[Sub-agent] claude-code started (initiated by: management_agent). Transcript: /t/c.jsonl', source: 'management', timestamp: T2, _meta: { initiator: 'management_agent' } },
      marker('claude-code', '整理三家品牌的饮品定价对比表\n写入 第3章.md', T2, { initiator: 'management_agent' }),
      run('claude-code', T2, '表格写好了。'),
      terminal('claude-code', 'completed', T3, { reply: '表格写好了。' }),
    ]);
    const types = items.map((i) => i.type);
    expect(types).toEqual(['user_message', 'agent_message', 'dispatch_card', 'agent_message']);
    const card = cards(items)[0];
    expect(card.handle).toBe('claude-code');
    expect(card.label).toBe('整理三家品牌的饮品定价对比表');
    expect(card.status).toBe('completed');
    expect(card.started_at).toBe(Date.parse(T2));
    expect(card.ended_at).toBe(Date.parse(T3));
    expect(card.folded).toBeFalsy();
    // The steps hold what the inline run rendered: header, capsule, message.
    expect(card.steps.map((s) => s.type)).toEqual(['agent_message', 'agent_run', 'agent_message']);
    const reply = items[3];
    expect(reply.type === 'agent_message' && reply.source).toBe('claude-code');
    expect(reply.type === 'agent_message' && reply.content).toBe('表格写好了。');
    expect(items.some((i) => i.type === 'sub_agent_activity')).toBe(false);
  });

  it('Orbital\'s agent_message sends leave its capsule: each card stands for its send', () => {
    const items = transformChatHistory([
      { role: 'user', content: 'two at once', source: 'user', timestamp: T0 },
      {
        role: 'assistant', content: '分给两个 Agent 同时做。', source: 'management', timestamp: T1,
        tool_calls: [
          { id: 'c1', type: 'function', function: { name: 'agent_message', arguments: JSON.stringify({ action: 'send', agent: 'claude-code', message: 'stores' }) } },
          { id: 'c2', type: 'function', function: { name: 'agent_message', arguments: JSON.stringify({ action: 'send', agent: 'codex', message: 'buzz' }) } },
        ],
      },
      marker('claude-code', 'stores', T2, { initiator: 'management_agent' }),
      { role: 'tool', content: 'Message sent', tool_call_id: 'c1', source: 'management', timestamp: T2 },
      marker('codex', 'buzz', T2, { initiator: 'management_agent' }),
      { role: 'tool', content: 'Message sent', tool_call_id: 'c2', source: 'management', timestamp: T2 },
    ]);
    expect(items.map((i) => i.type)).toEqual(['user_message', 'agent_message', 'dispatch_card', 'dispatch_card']);
  });

  it('two agents at once: each card is followed by its own reply, a running one by nothing', () => {
    const items = transformChatHistory([
      { role: 'user', content: 'three at once', source: 'user', timestamp: T0 },
      marker('claude-code', 'stores', T1, { initiator: 'management_agent' }),
      run('claude-code', T1, 'stores done'),
      marker('codex', 'buzz', T2, { initiator: 'management_agent' }),
      run('codex', T2, '', true),
      terminal('claude-code', 'completed', T3, { reply: 'stores done' }),
    ]);
    const types = items.map((i) => (i.type === 'dispatch_card' ? `card:${i.handle}:${i.status}` : i.type));
    expect(types).toEqual([
      'user_message',
      'card:claude-code:completed',
      'agent_message',
      'card:codex:running',
    ]);
    const codex = cards(items)[1];
    expect(codex.ended_at).toBeNull();
    // Its run so far, the trailing capsule still running.
    const cap = codex.steps.find((s) => s.type === 'agent_run');
    expect(cap && cap.type === 'agent_run' && cap.status).toBe('running');
  });

  it('a chat the user pinned to a worker stays inline exactly as before', () => {
    const items = transformChatHistory([
      { role: 'user', content: 'fix it', source: 'user', timestamp: T0, target: 'codex' },
      marker('codex', 'fix it', T1, { initiator: 'user_pinned', suppress_wake: true }),
      run('codex', T1, 'fixed'),
      terminal('codex', 'completed', T3, { reply: 'fixed', suppress_wake: true }),
    ]);
    expect(cards(items)).toHaveLength(0);
    expect(items.map((i) => i.type)).toContain('agent_run');
    const bubble = items.find((i) => i.type === 'agent_message' && !i.isHeaderOnly);
    expect(bubble && bubble.type === 'agent_message' && bubble.content).toBe('fixed');
  });

  it('a queue item the user assigned stays inline', () => {
    const items = transformChatHistory([
      marker('codex', 'nightly', T1, { initiator: 'queue_item', suppress_wake: true }),
      run('codex', T1, 'ok'),
    ]);
    expect(cards(items)).toHaveLength(0);
  });

  it('a worker\'s own dispatch is a card with the reply folded inside, no bubble', () => {
    const items = transformChatHistory([
      { role: 'user', content: 'ask codex to test', source: 'user', timestamp: T0, target: 'claude-code' },
      marker('codex', 'run the tests', T2, { initiator: 'worker:claude-code', suppress_wake: true }),
      run('codex', T2, 'all green'),
      terminal('codex', 'completed', T3, { reply: 'all green', suppress_wake: true }),
    ]);
    const [card] = cards(items);
    expect(card.folded).toBe(true);
    expect(card.status).toBe('completed');
    expect(items[items.length - 1].type).toBe('dispatch_card');
    expect(items.some((i) => i.type === 'agent_message' && i.source === 'codex' && !i.isHeaderOnly)).toBe(false);
    // The reply is still in the card for the panel.
    expect(card.steps.some((s) => s.type === 'agent_message' && s.content === 'all green')).toBe(true);
  });

  it('rows from before the initiator existed: no suppress_wake → card, suppress_wake → inline', () => {
    const old = transformChatHistory([marker('codex', 'go', T1), run('codex', T1, 'went')]);
    expect(cards(old)).toHaveLength(1);
    const pinned = transformChatHistory([marker('codex', 'go', T1, { suppress_wake: true }), run('codex', T1, 'went')]);
    expect(cards(pinned)).toHaveLength(0);
  });

  it('a marker without a dispatch id (legacy) keeps its one-line row', () => {
    const items = transformChatHistory([
      { role: 'system', content: '[Sub-agent] Message sent to claude-code: list primes', source: 'management', timestamp: T1 },
    ]);
    expect(items.map((i) => i.type)).toEqual(['sub_agent_activity']);
  });

  it('a card with no run summary yet (nothing streamed) is still a running card', () => {
    const items = transformChatHistory([marker('codex', 'go', T1, { initiator: 'management_agent' })]);
    const [card] = cards(items);
    expect(card.status).toBe('running');
    expect(card.steps).toEqual([]);
  });

  it('the terminal row carries the reply when the run summary had none: bubble under the card', () => {
    const items = transformChatHistory([
      marker('codex', 'go', T1, { initiator: 'management_agent' }),
      { role: 'user', content: 'meanwhile', source: 'user', timestamp: T2 },
      terminal('codex', 'completed', T3, { reply: 'went fine' }),
    ]);
    expect(items.map((i) => i.type)).toEqual(['dispatch_card', 'agent_message', 'user_message']);
    expect(items[1].type === 'agent_message' && items[1].content).toBe('went fine');
  });

  it('an error ends the card in error and keeps the error line (its text is the news)', () => {
    const items = transformChatHistory([
      marker('codex', 'go', T1, { initiator: 'management_agent' }),
      run('codex', T1, '', true),
      terminal('codex', 'error', T3),
    ]);
    const [card] = cards(items);
    expect(card.status).toBe('error');
    const row = items.find((i) => i.type === 'sub_agent_activity');
    expect(row && row.type === 'sub_agent_activity' && row.action).toBe('error');
  });

  it('a stopped turn\'s last message is not a reply: no bubble under the card, it stays in the steps', () => {
    const items = transformChatHistory([
      marker('codex', 'go', T1, { initiator: 'management_agent' }),
      run('codex', T1, '我会先列出当前文件夹的文件。'),
      terminal('codex', 'stopped', T3),
    ]);
    expect(items.map((i) => i.type)).toEqual(['dispatch_card']);
    const [card] = cards(items);
    expect(card.status).toBe('stopped');
    expect(card.steps.some((s) => s.type === 'agent_message' && s.content === '我会先列出当前文件夹的文件。')).toBe(true);
  });

  it('a closed turn whose terminal row is not written yet still shows its reply', () => {
    const items = transformChatHistory([
      marker('codex', 'go', T1, { initiator: 'management_agent' }),
      run('codex', T1, 'done'),
    ]);
    expect(items.map((i) => i.type)).toEqual(['dispatch_card', 'agent_message']);
  });

  it('a user stop ends the card stopped; the plain stop line is not repeated', () => {
    const items = transformChatHistory([
      marker('codex', 'go', T1, { initiator: 'management_agent' }),
      terminal('codex', 'stopped', T3),
    ]);
    expect(cards(items)[0].status).toBe('stopped');
    expect(items.some((i) => i.type === 'sub_agent_activity')).toBe(false);
  });

  it('the same agent twice in a row: the second dispatch is a second card', () => {
    const items = transformChatHistory([
      marker('codex', 'first', T1, { initiator: 'management_agent' }),
      run('codex', T1, 'one'),
      terminal('codex', 'completed', T3, { reply: 'one' }),
      marker('codex', 'second', T4, { initiator: 'management_agent' }),
      run('codex', T4, 'two'),
      terminal('codex', 'completed', T5, { reply: 'two' }),
    ]);
    const cs = cards(items);
    expect(cs.map((c) => c.label)).toEqual(['first', 'second']);
    expect(cs.every((c) => c.status === 'completed')).toBe(true);
  });

  it('a spec-129 continued turn after a card still renders inline', () => {
    const items = transformChatHistory([
      marker('claude-code', 'build', T1, { initiator: 'management_agent' }),
      run('claude-code', T1, 'started the build'),
      terminal('claude-code', 'completed', T3, { reply: 'started the build' }),
      terminal('claude-code', 'continued', T4, { reply: 'the build finished', handle: 'claude-code' },
        '[Sub-agent] claude-code continued. Summary: the build finished. Transcript: /t.jsonl.'),
    ]);
    const last = items[items.length - 1];
    expect(last.type === 'agent_message' && last.content).toBe('the build finished');
    expect(cards(items)[0].status).toBe('completed');
  });
});
