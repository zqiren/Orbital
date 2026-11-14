// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { translate } from '../i18n/useT';
import { activityLine } from './useWorkbenchFeed';

const zh = (k: Parameters<typeof translate>[1], v?: Parameters<typeof translate>[2]) => translate('zh', k, v);

// A running card's live line says what the chat says for the same event.
describe('activityLine', () => {
  it('a tool result leaves the line alone, as the chat only marks the call done', () => {
    expect(activityLine({ category: 'tool_result', description: 'Tool result received', tool_name: 'call_1' }, zh)).toBeNull();
    expect(
      activityLine({ category: 'agent_output', worker_event: 'tool_result', description: 'Tool result received', tool_name: 'Read' }, zh),
    ).toBeNull();
  });

  it('a tool call is worded in the locale, like the chat row', () => {
    const line = activityLine(
      { category: 'file_read', description: 'Reading plan.md', tool_name: 'read', arguments: { path: 'plan.md' } },
      zh,
    );
    expect(line).not.toBe('Reading plan.md');
    expect(line).toContain('plan.md');
    expect(
      activityLine({ category: 'agent_output', worker_event: 'tool_call', description: 'Using Read', tool_name: 'Read', arguments: { file_path: 'a.md' } }, zh),
    ).toContain('a.md');
  });

  it('an event with no arguments keeps the daemon text (old daemons)', () => {
    expect(activityLine({ category: 'tool_use', description: 'Thinking', tool_name: 'x' }, zh)).toBe('Thinking');
  });

  it("a read of Orbital's own memory files is not the card's activity", () => {
    const read = (path: string) => activityLine(
      { category: 'file_read', description: `Reading ${path}`, tool_name: 'read', arguments: { path } }, zh);
    expect(read('orbital/PROJECT_STATE.md')).toBeNull();
    expect(read('./orbital/ASKS.md')).toBeNull();
    expect(read('orbital/instructions/project_goals.md')).toBeNull();
    expect(read('orbital/sub_agents/codex/notes.md')).toBeNull();
    // A worker reads by absolute path (claude-code Read, ACP read).
    expect(activityLine({
      category: 'agent_output', worker_event: 'tool_call', description: 'Using Read', tool_name: 'Read',
      arguments: { file_path: '/Users/me/proj/orbital/DECISIONS.md' },
    }, zh)).toBeNull();
    expect(activityLine({
      category: 'agent_output', worker_event: 'tool_call', description: 'Using read', tool_name: 'read_file',
      arguments: { path: 'C:\\work\\proj\\orbital\\LESSONS_ARCHIVE.md' },
    }, zh)).toBeNull();
  });

  it("the user's own files stay on the line, even in a folder named orbital", () => {
    const read = (path: string) => activityLine(
      { category: 'file_read', description: `Reading ${path}`, tool_name: 'read', arguments: { path } }, zh);
    expect(read('src/app.ts')).toContain('src/app.ts');
    expect(read('/Users/me/orbital/README.md')).toContain('README.md');
    expect(read('docs/orbital-notes.md')).toContain('orbital-notes.md');
    // Writing memory is not a read: unchanged (the record turn is its own row).
    expect(activityLine(
      { category: 'file_write', description: 'Writing', tool_name: 'write', arguments: { path: 'orbital/PROJECT_STATE.md', content: 'x' } },
      zh,
    )).not.toBeNull();
  });
});
