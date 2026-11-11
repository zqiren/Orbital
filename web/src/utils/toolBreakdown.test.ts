// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { translate } from '../i18n/useT';
import { formatCapsuleDuration, formatToolBreakdown } from './toolBreakdown';
import { capsuleSummaryText } from '../components/ChatView';
import type { DisplayItem } from './chatTransform';

const zh = (k: Parameters<typeof translate>[1], v?: Record<string, string | number>) => translate('zh', k, v);

describe('formatToolBreakdown', () => {
  it('keeps the English line byte-identical', () => {
    expect(formatToolBreakdown({ read: 6, write: 4, glob: 1 })).toBe('6 reads, 4 writes, glob');
    expect(formatToolBreakdown({ shell: 3, mcp__x__y: 2, Bash: 1 })).toBe('3 shells, 2 mcp__x__ys, Bash');
    expect(formatToolBreakdown({})).toBe('');
  });

  it('speaks Chinese in the zh UI, tool names it does not know kept as is', () => {
    expect(formatToolBreakdown({ read: 6, write: 4, glob: 1 }, zh)).toBe('读取 6 次、写入 4 次、查找文件');
    expect(formatToolBreakdown({ shell: 3, edit: 1 }, zh)).toBe('运行命令 3 次、编辑');
    expect(formatToolBreakdown({ Bash: 2, grep: 1 }, zh)).toBe('Bash 2 次、搜索内容');
  });
});

describe('capsuleSummaryText in the zh UI (pass 2b: "6 reads, 4 writes" in English)', () => {
  it('says the counts in Chinese, the memory turn included', () => {
    const cap = {
      type: 'agent_run', capsule_id: 'c', status: 'done', items: [],
      tool_call_count_by_name: { read: 6, write: 4, glob: 1 }, has_thinking: false,
      started_at: 0, ended_at: 8000,
    } as unknown as Extract<DisplayItem, { type: 'agent_run' }>;
    expect(capsuleSummaryText(cap, zh)).toBe('读取 6 次、写入 4 次、查找文件 · 8秒');
    expect(capsuleSummaryText(cap)).toBe('6 reads, 4 writes, glob · 8s');
    const mem = { ...cap, memory_record: true, tool_call_count_by_name: { edit: 4 } };
    expect(capsuleSummaryText(mem, zh)).toBe('记忆更新 · 编辑 4 次 · 8秒');
  });
});

describe('formatCapsuleDuration', () => {
  it('keeps English and speaks Chinese', () => {
    expect(formatCapsuleDuration(0, 88000)).toBe('1m 28s');
    expect(formatCapsuleDuration(0, 120000)).toBe('2m');
    expect(formatCapsuleDuration(0, 16000)).toBe('16s');
    expect(formatCapsuleDuration(0, 88000, zh)).toBe('1分28秒');
    expect(formatCapsuleDuration(0, 120000, zh)).toBe('2分');
    expect(formatCapsuleDuration(0, 16000, zh)).toBe('16秒');
    expect(formatCapsuleDuration(0, null, zh)).toBe('<1秒');
  });
});
