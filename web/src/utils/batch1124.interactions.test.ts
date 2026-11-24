// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Batch 1124: specs 195 (dispatch cards) × 197 (usage-limit row) on reload.
// A dispatched worker that fails on its usage limit: its card turns red, the
// limit row stays in the conversation, and no reply bubble is drawn under the
// card, even when an older transcript (before 197 dropped Claude Code's
// synthetic error message) still holds that message as the run's text.

import { describe, it, expect } from 'vitest';
import { transformChatHistory, type DisplayItem } from './chatTransform';
import type { ChatMessage } from '../types';

const RESET = '2026-11-24T09:24:00+00:00';
const LIMIT_TEXT =
  "[Sub-agent] claude-code stopped with error: Usage limit reached; back at 17:24 local time (2026-11-24). " +
  "Error: You've hit your session limit · resets 5:24pm. Transcript: /t/claude-code.jsonl";

function rows(runText: string, initiator: string): ChatMessage[] {
  return [
    { role: 'user', content: 'compare prices', source: 'user', timestamp: '2026-11-24T08:00:00Z' },
    {
      role: 'system', source: 'management', timestamp: '2026-11-24T08:00:01Z',
      content: '[Sub-agent] Message sent to claude-code: "整理定价表". Transcript: /t/claude-code.jsonl',
      _meta: { dispatch_id: 's1:d1', handle: 'claude-code', transcript_path: '/t/claude-code.jsonl', initiator },
    },
    {
      role: 'assistant', source: 'sub_agent', sub_agent_handle: 'claude-code', timestamp: '2026-11-24T08:00:02Z',
      content: runText, sub_agent_tool_rows: [],
      sub_agent_stream: runText ? [{ kind: 'message', content: runText, timestamp: '2026-11-24T08:00:02Z' }] : [],
    },
    {
      role: 'system', source: 'management', timestamp: '2026-11-24T08:00:03Z', content: LIMIT_TEXT,
      _meta: { event: 'sub_agent_terminal', kind: 'error', reason: 'usage_limit', resets_at: RESET, display_content: LIMIT_TEXT },
    },
  ] as ChatMessage[];
}

type Card = Extract<DisplayItem, { type: 'dispatch_card' }>;

describe('batch 1124 — a dispatched worker on its usage limit', () => {
  for (const initiator of ['management_agent', 'worker:codex']) {
    for (const runText of ['', "You've hit your session limit · resets 5:24pm"]) {
      it(`${initiator}, run text ${runText ? 'from an older transcript' : 'empty'}: red card, limit row, no reply`, () => {
        const items = transformChatHistory(rows(runText, initiator));
        const cardAt = items.findIndex((i) => i.type === 'dispatch_card');
        expect(cardAt).toBeGreaterThanOrEqual(0);
        expect((items[cardAt] as Card).status).toBe('error');
        // No bubble anywhere in the column carries the worker's text.
        expect(items.filter((i) => i.type === 'agent_message' || i.type === 'sub_agent_message')).toHaveLength(0);
        const limit = items.filter((i) => i.type === 'sub_agent_activity' && i.limit);
        expect(limit).toHaveLength(1);
        const row = limit[0] as Extract<DisplayItem, { type: 'sub_agent_activity' }>;
        expect(row.handle).toBe('claude-code');
        expect(row.limit).toEqual({ resetsAt: RESET });
        expect(row.error).toBe("You've hit your session limit · resets 5:24pm");
        expect(items.indexOf(row)).toBeGreaterThan(cardAt);
      });
    }
  }
});
