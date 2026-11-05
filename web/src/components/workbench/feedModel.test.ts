// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import type { WorkbenchFeedRow } from '../../types';
import type { FeedState } from '../../hooks/useWorkbenchFeed';
import { buildFeedGroups, unreadCounts } from './feedModel';

const NOW = Date.parse('2026-11-05T10:00:00Z');

function row(over: Partial<WorkbenchFeedRow>): WorkbenchFeedRow {
  return {
    project_id: 'p1',
    session_id: 's',
    session_uuid: null,
    name: 'task',
    agent: null,
    status: 'idle',
    worker_running: false,
    trigger_type: null,
    last_user_at: '2026-11-05T09:00:00Z',
    last_reply_at: '2026-11-05T09:30:00Z',
    last_activity_at: '2026-11-05T09:30:00Z',
    checked_reply_at: null,
    ask: null,
    result_line: null,
    stop: null,
    ...over,
  } as WorkbenchFeedRow;
}

function feed(rows: WorkbenchFeedRow[]): FeedState {
  return {
    rows, loaded: true, error: false, pending: [], activity: {},
    held: new Map(), closedAsks: new Set(), composerRestore: null,
  };
}

describe('unreadCounts — the number beside a project is unread sessions', () => {
  it('counts a session whose reply the user has not checked', () => {
    const groups = buildFeedGroups(feed([row({ session_id: 'a' }), row({ session_id: 'b', project_id: 'p2' })]), new Map(), NOW);
    expect(unreadCounts(groups, NOW)).toEqual({ p1: 1, p2: 1 });
  });

  it('does not count an open ask or an approval that is not also an unchecked reply', () => {
    const ask = row({
      session_id: 'ask-only',
      ask: { id: 'abc123', text: 'Keep it?', created: '2026-11-05T09:00:00Z', due: null, overdue: false, options: [], detail: '' },
      last_reply_at: '2026-11-05T09:30:00Z',
      checked_reply_at: '2026-11-05T09:31:00Z',
    });
    const approval = row({ session_id: 'approval', status: 'pending_approval', last_reply_at: null });
    const sessionless = row({ session_id: null, ask: { id: 'def456', text: 'Which?', created: '2026-11-05T09:00:00Z', due: null, overdue: false, options: [], detail: '' } });
    const groups = buildFeedGroups(feed([ask, approval, sessionless]), new Map(), NOW);
    expect(groups.waiting).toHaveLength(3); // all three are on the Workbench
    expect(unreadCounts(groups, NOW)).toEqual({});
  });

  it('a checked mark on this device removes the session from the count', () => {
    const marks = new Map([['p1\u0000a', '2026-11-05T09:30:00Z']]);
    const groups = buildFeedGroups(feed([row({ session_id: 'a' })]), marks, NOW);
    expect(unreadCounts(groups, NOW)).toEqual({});
  });
});

describe('buildFeedGroups — 等你 is newest first, nothing else', () => {
  const ask = (id: string, overdue: boolean) => ({
    id, text: `q ${id}`, created: '2026-11-01T09:00:00Z', due: overdue ? '2026-11-02' : null, overdue, options: [], detail: ''
  });

  it('sorts waiting cards by their time alone: an overdue ask or an amber card does not jump the queue', () => {
    const rows = [
      row({ session_id: 'old-overdue', ask: ask('aaaaaa', true), last_user_at: '2026-11-05T05:00:00Z', last_reply_at: '2026-11-05T06:00:00Z' }),
      row({ session_id: 'old-amber', stop: { cause: 'usage_limit', resets_at: null }, last_user_at: '2026-11-05T05:00:00Z', last_reply_at: '2026-11-05T07:00:00Z' }),
      row({ session_id: 'newest', last_reply_at: '2026-11-05T09:45:00Z' }),
      row({ session_id: 'middle', last_user_at: '2026-11-05T05:00:00Z', last_reply_at: '2026-11-05T08:00:00Z' }),
    ];
    const groups = buildFeedGroups(feed(rows), new Map(), NOW);
    expect(groups.waiting.map((i) => i.row?.session_id)).toEqual(['newest', 'middle', 'old-amber', 'old-overdue']);
  });

  it('keeps 进行中 as before: pending sends first, then newest', () => {
    const f = feed([
      row({ session_id: 'run-old', status: 'running', last_user_at: '2026-11-05T08:00:00Z', last_reply_at: null }),
      row({ session_id: 'run-new', status: 'running', last_user_at: '2026-11-05T09:00:00Z', last_reply_at: null }),
    ]);
    f.pending = [{ nonce: 'n1', project_id: 'p1', session_id: null, text: 'go', agent: 'orbital', created_at: NOW - 1000 } as FeedState['pending'][number]];
    const groups = buildFeedGroups(f, new Map(), NOW);
    expect(groups.running.map((i) => i.pending ? 'pending' : i.row?.session_id)).toEqual(['pending', 'run-new', 'run-old']);
  });
});
