// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import type { WorkbenchFeedRow, WorkbenchRecentRow } from '../../types';
import type { FeedState } from '../../hooks/useWorkbenchFeed';
import { buildFeedGroups } from './feedModel';
import { buildRecent } from './recentModel';

const NOW = Date.parse('2026-11-23T12:00:00Z');
const PROJECTS = [{ project_id: 'p1', name: 'site' }, { project_id: 'p2', name: 'thesis' }];

const recent = (sid: string, at: string, over: Partial<WorkbenchRecentRow> = {}): WorkbenchRecentRow => ({
  project_id: 'p1', project_name: 'site (server)', session_id: sid, title: `t-${sid}`, agent: 'claude-code',
  updated_at: at, ...over,
});

function row(over: Partial<WorkbenchFeedRow>): WorkbenchFeedRow {
  return {
    project_id: 'p1', session_id: 's', session_uuid: null, name: 'live', agent: null, status: 'idle',
    worker_running: false, trigger_type: null, last_user_at: '2026-11-23T11:00:00Z',
    last_reply_at: '2026-11-23T11:30:00Z', last_activity_at: '2026-11-23T11:30:00Z', checked_reply_at: null,
    ask: null, result_line: 'done', stop: null, ...over,
  } as WorkbenchFeedRow;
}

function feed(rows: WorkbenchFeedRow[], rec: WorkbenchRecentRow[]): FeedState {
  return {
    rows, recent: rec, loaded: true, error: false, pending: [], activity: {},
    held: new Map(), closedAsks: new Set(), composerRestore: null, unmarked: new Set(), moves: new Map(),
  };
}

const ids = (xs: { sessionId: string }[]) => xs.map((x) => x.sessionId);

describe('最近 — placement', () => {
  it('first landing: nothing live, the newest five, newest first', () => {
    const rec = ['a', 'b', 'c', 'd', 'e', 'f', 'g'].map((s, i) => recent(s, `2026-11-2${i}T00:00:00Z`));
    const f = feed([], rec);
    const out = buildRecent(f, buildFeedGroups(f, new Map(), NOW), PROJECTS);
    expect(ids(out)).toEqual(['g', 'f', 'e', 'd', 'c']);
    // the project's own (display) name wins over the server's
    expect(out[0].projectName).toBe('site');
  });

  it('a session that becomes running or unread leaves 最近', () => {
    const rec = [recent('a', '2026-11-22T00:00:00Z'), recent('b', '2026-11-21T00:00:00Z')];
    const running = row({ session_id: 'a', status: 'running', last_reply_at: null });
    const f = feed([running], rec);
    const groups = buildFeedGroups(f, new Map(), NOW);
    expect(groups.items.map((i) => i.row?.session_id)).toEqual(['a']);
    expect(ids(buildRecent(f, groups, PROJECTS))).toEqual(['b']);
  });

  it('an optimistic send is live too', () => {
    const rec = [recent('a', '2026-11-22T00:00:00Z')];
    const f: FeedState = {
      ...feed([], rec),
      pending: [{ nonce: 'n', text: 'x', created_at: NOW, project_id: 'p1', session_id: 'a', routing: false } as FeedState['pending'][number]],
    };
    expect(buildRecent(f, buildFeedGroups(f, new Map(), NOW), PROJECTS)).toEqual([]);
  });

  it('an unread card the user opens comes back to 最近 before the feed refetches', () => {
    const unread = row({ session_id: 'u', last_reply_at: '2026-11-23T11:30:00Z' });
    const f = feed([unread], [recent('old', '2026-11-20T00:00:00Z')]);
    const before = buildFeedGroups(f, new Map(), NOW);
    expect(before.items).toHaveLength(1);
    expect(ids(buildRecent(f, before, PROJECTS))).toEqual(['old']);
    // the user opened it: the checked mark lands locally
    const marks = new Map([[`p1\u0000u`, '2026-11-23T11:30:00Z']]);
    const after = buildFeedGroups(f, marks, NOW);
    expect(after.items).toHaveLength(0);
    const out = buildRecent(f, after, PROJECTS);
    expect(ids(out)).toEqual(['u', 'old']);
    expect(out[0]).toMatchObject({ projectName: 'site', title: 'live', time: '2026-11-23T11:30:00Z' });
  });

  it('a row is never drawn twice, automation runs and unknown projects stay out', () => {
    const f = feed(
      [row({ session_id: 'auto', trigger_type: 'schedule', status: 'idle' }), row({ project_id: 'gone', session_id: 'x' })],
      [recent('a', '2026-11-22T00:00:00Z'), recent('a', '2026-11-22T00:00:00Z')],
    );
    const marks = new Map([[`p1\u0000auto`, '2026-11-23T11:30:00Z'], [`gone\u0000x`, '2026-11-23T11:30:00Z']]);
    expect(ids(buildRecent(f, buildFeedGroups(f, marks, NOW), PROJECTS))).toEqual(['a']);
  });

  it('no history: nothing', () => {
    const f = feed([], []);
    expect(buildRecent(f, buildFeedGroups(f, new Map(), NOW), PROJECTS)).toEqual([]);
  });
});
