// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// The [STATUS: …] fold (variant B): status-only replies are steps of one run.
import { describe, it, expect } from 'vitest';
import { transformChatHistory, mergeRecoveredAssistantMessage } from './chatTransform';
import type { DisplayItem } from './chatTransform';
import { statusSteps, latestStatus, displayAgentText } from './statusTag';
import type { ChatMessage } from '../types';

const T0 = Date.parse('2026-11-25T10:32:00Z');
const iso = (s: number) => new Date(T0 + s * 1000).toISOString();
const user = (s: number, content: string): ChatMessage => ({ role: 'user', content, source: 'user', timestamp: iso(s) });
let n = 0;
function reply(s: number, content: string | null, tools: string[] = []): ChatMessage[] {
  const calls = tools.map((name) => ({ id: `call_${++n}`, type: 'function' as const, function: { name, arguments: '{}' } }));
  const rows: ChatMessage[] = [{
    role: 'assistant', content, source: 'management', timestamp: iso(s),
    ...(calls.length ? { tool_calls: calls } : {}),
  }];
  for (const c of calls) rows.push({ role: 'tool', content: 'ok', source: 'management', timestamp: iso(s + 1), tool_call_id: c.id });
  return rows;
}
type Run = Extract<DisplayItem, { type: 'agent_run' }>;
const runs = (items: DisplayItem[]) => items.filter((i): i is Run => i.type === 'agent_run');
const bubbles = (items: DisplayItem[]) =>
  items.filter((i) => i.type === 'agent_message' && !i.isHeaderOnly) as Extract<DisplayItem, { type: 'agent_message' }>[];

describe('status-only replies fold into one capsule', () => {
  const history: ChatMessage[] = [
    user(0, '扫描观猹'),
    ...reply(1, '[STATUS: 打开观猹 Orbital 页]', ['browser', 'browser']),
    ...reply(8, '[STATUS: 读猹评]', ['browser']),
    ...reply(20, '[STATUS: 扫描完毕 — 写日报文件]', ['write']),
    ...reply(68, '日报已写入。\n\n[STATUS: 完成]'),
  ];
  const items = transformChatHistory(history);

  it('draws one header, one capsule and the final bubble — no bubble per step', () => {
    expect(items.map((i) => i.type)).toEqual(['user_message', 'agent_message', 'agent_run', 'agent_message']);
    expect((items[1] as { isHeaderOnly?: boolean }).isHeaderOnly).toBe(true);
    expect(runs(items)).toHaveLength(1);
  });

  it('labels each step with its status, latest last, with its own count and time', () => {
    const steps = statusSteps(runs(items)[0])!;
    expect(steps).toHaveLength(3);
    expect(steps.map((s) => s.status)).toEqual(['打开观猹 Orbital 页', '读猹评', '扫描完毕 — 写日报文件']);
    expect(latestStatus(steps)).toBe('扫描完毕 — 写日报文件');
    expect(steps[0].tool_call_count_by_name).toEqual({ browser: 2 });
    expect(steps[0].ended_at! - steps[0].started_at).toBe(7000);
    // The capsule spans the run: first reply to the last tool result.
    const cap = runs(items)[0];
    expect(cap.ended_at! - cap.started_at).toBe(20_000);
  });

  it('keeps the stored text on the final bubble and shows it without the tag', () => {
    const last = bubbles(items).at(-1)!;
    expect(last.content).toBe('日报已写入。\n\n[STATUS: 完成]');
    expect(displayAgentText(last.content)).toBe('日报已写入。');
  });
});

describe('a reply with real text ends the fold', () => {
  const items = transformChatHistory([
    user(0, 'go'),
    ...reply(1, '[STATUS: 读版本页]', ['browser']),
    ...reply(5, '[STATUS: 版本页改版了]', ['browser']),
    ...reply(9, '观猹的更新日志页改版了，我先用 GitHub API 读版本。\n[STATUS: 改用 GitHub API 读版本]', ['shell', 'shell']),
    ...reply(14, '[STATUS: GitHub API 返回 30 个版本]', ['shell']),
    ...reply(20, 'Done.'),
  ]);

  it('folds before, keeps the text bubble, starts a new fold after', () => {
    expect(items.map((i) => i.type)).toEqual([
      'user_message', 'agent_message', 'agent_run', 'agent_message', 'agent_run', 'agent_message',
    ]);
    const [first, second] = runs(items);
    expect(statusSteps(first)!.map((s) => s.status)).toEqual(['读版本页', '版本页改版了']);
    // The text reply's own status labels the tools that follow it.
    expect(statusSteps(second)!.map((s) => s.status)).toEqual(['改用 GitHub API 读版本', 'GitHub API 返回 30 个版本']);
    expect(statusSteps(second)![0].tool_call_count_by_name).toEqual({ shell: 2 });
    expect(displayAgentText(bubbles(items)[0].content)).toBe('观猹的更新日志页改版了，我先用 GitHub API 读版本。');
  });
});

describe('replies without a status keep today’s capsule', () => {
  it('silent tool turns still merge with a plain delimiter and no fold', () => {
    const items = transformChatHistory([user(0, 'go'), ...reply(1, null, ['read']), ...reply(3, '', ['grep'])]);
    const [cap] = runs(items);
    expect(statusSteps(cap)).toBeNull();
    expect(cap.items.map((c) => c.type)).toEqual(['tool_call_row', 'agent_message', 'tool_call_row']);
    expect(cap.items.some((c) => c.type === 'agent_message' && c.status)).toBe(false);
  });

  it('a status-only reply with no tools stays a bubble (its status as the words)', () => {
    const items = transformChatHistory([user(0, 'go'), ...reply(1, '[STATUS: 完成]')]);
    const b = bubbles(items);
    expect(b).toHaveLength(1);
    expect(b[0].content).toBe('[STATUS: 完成]');
    expect(displayAgentText(b[0].content)).toBe('完成');
  });

  it('a tag inside a code block is not a status', () => {
    const items = transformChatHistory([user(0, 'go'), ...reply(1, '```\n[STATUS: x]\n```', ['read'])]);
    expect(bubbles(items)).toHaveLength(1);
    expect(statusSteps(runs(items)[0])).toBeNull();
  });
});

describe('REST recovery keeps comparing the stored text', () => {
  it('skips a status-only step with tools (the capsule shows it)', () => {
    const prev: DisplayItem[] = [];
    const row = reply(1, '[STATUS: 读猹评]', ['browser'])[0];
    expect(mergeRecoveredAssistantMessage(prev, row)).toBe(prev);
  });

  it('dedups a reply with text and a tag against the raw live bubble', () => {
    const content = 'Done.\n[STATUS: 完成]';
    const prev: DisplayItem[] = [{ type: 'agent_message', content, source: 'management', timestamp: iso(1) }];
    expect(mergeRecoveredAssistantMessage(prev, reply(1, content)[0])).toBe(prev);
  });
});
