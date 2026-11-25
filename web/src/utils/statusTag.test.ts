// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import {
  stripStatusTags,
  statusOf,
  displayAgentText,
  isStatusOnly,
  statusSteps,
  latestStatus,
} from './statusTag';
import type { DisplayItem } from './chatTransform';

describe('stripStatusTags', () => {
  it('removes a complete tag', () => {
    expect(stripStatusTags('[STATUS: 扫描猹馆]')).toBe('');
  });

  it('keeps the text around a tag, without the tag', () => {
    expect(stripStatusTags('[STATUS: Reading files]\n\nI read the three files.')).toBe('I read the three files.');
    expect(stripStatusTags('Done, the report is written.\n[STATUS: Wrote report]')).toBe('Done, the report is written.');
    expect(stripStatusTags('Hello [STATUS: x] world')).toBe('Hello world');
  });

  it('removes every tag when there are several', () => {
    expect(stripStatusTags('[STATUS: one] text [STATUS: two]\n[STATUS: three]')).toBe('text');
  });

  it('returns the very same string when there is no tag', () => {
    const s = 'A [link](x) and [brackets] stay.\n\n';
    expect(stripStatusTags(s)).toBe(s);
    expect(stripStatusTags(s, { streaming: true })).toBe(s);
  });

  it('leaves a tag inside a fenced code block or inline code alone', () => {
    const fenced = 'Example:\n```\n[STATUS: inside code]\n```\n[STATUS: real]';
    expect(stripStatusTags(fenced)).toBe('Example:\n```\n[STATUS: inside code]\n```');
    expect(stripStatusTags('Write `[STATUS: x]` in replies.')).toBe('Write `[STATUS: x]` in replies.');
  });

  it('hides a tag still arriving while streaming', () => {
    for (const partial of ['[', '[ST', '[STATUS', '[STATUS:', '[STATUS: 补猹', '[STATUS: 补猹馆中段 1100']) {
      expect(stripStatusTags(partial, { streaming: true })).toBe('');
    }
    expect(stripStatusTags('Some text\n[STATUS: half', { streaming: true })).toBe('Some text');
    // After the reply ends, a non-tag bracket is text.
    expect(stripStatusTags('see [1', { streaming: true })).toBe('see [1');
    expect(stripStatusTags('[ST')).toBe('[ST');
  });

  it('does not hide an unclosed code fence while streaming', () => {
    expect(stripStatusTags('```\n[STATUS: x', { streaming: true })).toBe('```\n[STATUS: x');
  });
});

describe('statusOf / isStatusOnly / displayAgentText', () => {
  it('reads the first tag outside code, as the daemon does', () => {
    expect(statusOf('[STATUS: first] [STATUS: second]')).toBe('first');
    expect(statusOf('`[STATUS: code]` text')).toBeNull();
    expect(statusOf('no tag')).toBeNull();
    expect(statusOf(null)).toBeNull();
  });

  it('knows a status-only reply', () => {
    expect(isStatusOnly('[STATUS: 扫描完毕 — 写日报文件]')).toBe(true);
    expect(isStatusOnly('  [STATUS: a]\n[STATUS: b] ')).toBe(true);
    expect(isStatusOnly('Text [STATUS: a]')).toBe(false);
    expect(isStatusOnly('Text')).toBe(false);
    expect(isStatusOnly('')).toBe(false);
  });

  it('gives the copy/display text without the tag', () => {
    expect(displayAgentText('日报已写入。\n\n[STATUS: 完成]')).toBe('日报已写入。');
    expect(displayAgentText('plain')).toBe('plain');
    // Nothing but a tag: its words, never the brackets.
    expect(displayAgentText('[STATUS: 完成]')).toBe('完成');
  });
});

type Run = Extract<DisplayItem, { type: 'agent_run' }>;
const T0 = Date.parse('2026-11-25T10:32:00Z');
const iso = (s: number) => new Date(T0 + s * 1000).toISOString();
const marker = (s: number, status?: string) => ({
  type: 'agent_message' as const, content: '', source: 'management', timestamp: iso(s), ...(status ? { status } : {}),
});
const tool = (s: number, name: string, id: string) => ({
  type: 'tool_call_row' as const, tool_name: name, target_description: id, tool_call_id: id,
  category: 'browser_automation' as const, timestamp: iso(s), result_content: 'ok', result_status: 'received' as const,
});
function run(items: Run['items'], endS: number | null): Run {
  return {
    type: 'agent_run', capsule_id: 'c1', status: 'completed', items,
    tool_call_count_by_name: {}, has_thinking: false, started_at: T0, ended_at: endS === null ? null : T0 + endS * 1000,
  };
}

describe('statusSteps', () => {
  it('is null for a capsule without any status (today’s capsule stays)', () => {
    expect(statusSteps(run([tool(0, 'read', 'a'), marker(2), tool(2, 'read', 'b')], 3))).toBeNull();
  });

  it('splits a capsule into one step per reply with counts and durations', () => {
    const steps = statusSteps(run([
      marker(0, '打开观猹'), tool(0, 'browser', 'a'), tool(1, 'browser', 'b'),
      marker(7, '读猹评'), tool(7, 'browser', 'c'),
      marker(20, '写日报'), tool(20, 'write', 'd'),
    ], 67))!;
    expect(steps.map((s) => s.status)).toEqual(['打开观猹', '读猹评', '写日报']);
    expect(steps[0].tool_call_count_by_name).toEqual({ browser: 2 });
    expect(steps[0].ended_at! - steps[0].started_at).toBe(7000);
    expect(steps[2].ended_at! - steps[2].started_at).toBe(47000);
    expect(latestStatus(steps)).toBe('写日报');
  });

  it('keeps a reply without status as its own unlabelled step', () => {
    const steps = statusSteps(run([tool(0, 'read', 'a'), marker(3, 'B'), tool(3, 'shell', 'b')], 5))!;
    expect(steps).toHaveLength(2);
    expect(steps[0].status).toBeUndefined();
    expect(steps[1].status).toBe('B');
  });

  it('counts the step in progress while running, and drops it once finished', () => {
    const items = [marker(0, 'A'), tool(0, 'read', 'a'), marker(4, 'B')];
    const live = statusSteps(run(items, 4), { running: true })!;
    expect(live.map((s) => s.status)).toEqual(['A', 'B']);
    expect(live[1].ended_at).toBeNull();
    expect(latestStatus(live)).toBe('B');
    const done = statusSteps(run(items, 4))!;
    expect(done.map((s) => s.status)).toEqual(['A']);
    expect(done[0].ended_at! - done[0].started_at).toBe(4000);
  });

  it('ignores the memory capsules', () => {
    const c = run([marker(0, 'A'), tool(0, 'edit', 'a')], 1);
    expect(statusSteps({ ...c, memory_record: true })).toBeNull();
  });
});
