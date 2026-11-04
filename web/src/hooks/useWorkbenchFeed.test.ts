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
});
