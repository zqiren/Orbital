// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import type { SessionListEntry } from '../types';
import { buildSessionList, isAutomationSession } from './sessionGroups';

let clock = 0;
/** Sessions are handed over ALREADY sorted (pinned first, then most recent
 * first) — that is SessionSidebar's job — so fixtures are listed in display
 * order and get descending timestamps to match. */
function s(id: string, extra: Partial<SessionListEntry> = {}): SessionListEntry {
  clock += 1;
  return {
    session_id: id,
    status: 'idle',
    last_activity_at: new Date(Date.UTC(2026, 8, 20, 12, 0, 0) - clock * 60_000).toISOString(),
    ...extra,
  } as SessionListEntry;
}
const run = (id: string, trigger: string, extra: Partial<SessionListEntry> = {}) =>
  s(id, { name: `[Triggered by schedule '${trigger}' (Every day)]`, trigger_type: 'schedule', ...extra });

const shape = (nodes: ReturnType<typeof buildSessionList>['nodes']) =>
  nodes.map((n) => (n.type === 'session' ? n.session.session_id : `${n.group.key}[${n.group.runs.map((r) => r.session_id).join(',')}]`));

describe('isAutomationSession', () => {
  it('recognises trigger_type, queue origin, and the legacy name prefix', () => {
    expect(isAutomationSession(s('a', { trigger_type: 'file_watch', name: 'renamed by user' }))).toBe(true);
    expect(isAutomationSession(s('b', { origin: 'queue' }))).toBe(true);
    expect(isAutomationSession(s('c', { name: "[Triggered by schedule 'Daily' (Every day)]" }))).toBe(true);
    expect(isAutomationSession(s('d', { name: 'plan the trip' }))).toBe(false);
  });
});

describe('buildSessionList', () => {
  it("'all': every automation collapses to ONE node, placed at its most recent run", () => {
    const sessions = [
      run('r1', 'Daily issues'), s('chat1'), run('r2', 'Daily issues'),
      s('chat2'), run('r3', 'Daily issues'), run('w1', 'Weekly'), run('w2', 'Weekly'),
    ];
    const { nodes, pinned } = buildSessionList(sessions, 'all');
    expect(pinned).toEqual([]);
    expect(shape(nodes)).toEqual([
      'schedule:Daily issues[r1,r2,r3]', 'chat1', 'chat2', 'schedule:Weekly[w1,w2]',
    ]);
  });

  it('a group of one is just that session — nothing to collapse', () => {
    const { nodes } = buildSessionList([s('chat1'), run('only', 'Once')], 'all');
    expect(shape(nodes)).toEqual(['chat1', 'only']);
  });

  it("'chats' drops automations; 'automations' drops chats", () => {
    const sessions = [run('r1', 'Daily'), s('chat1'), run('r2', 'Daily'), s('chat2')];
    expect(shape(buildSessionList(sessions, 'chats').nodes)).toEqual(['chat1', 'chat2']);
    expect(shape(buildSessionList(sessions, 'automations').nodes)).toEqual(['schedule:Daily[r1,r2]']);
  });

  it('pinned sessions stay individual rows and follow the filter', () => {
    const sessions = [run('pr', 'Daily', { pinned: true }), s('pc', { pinned: true }), run('r1', 'Daily'), run('r2', 'Daily')];
    expect(buildSessionList(sessions, 'all').pinned.map((p) => p.session_id)).toEqual(['pr', 'pc']);
    expect(buildSessionList(sessions, 'chats').pinned.map((p) => p.session_id)).toEqual(['pc']);
    expect(buildSessionList(sessions, 'automations').pinned.map((p) => p.session_id)).toEqual(['pr']);
    // The pinned run is not ALSO counted inside its group.
    expect(shape(buildSessionList(sessions, 'all').nodes)).toEqual(['schedule:Daily[r1,r2]']);
  });

  it('same trigger name under different kinds does not merge; queue items share one group', () => {
    const sessions = [
      run('s1', 'Inbox'), run('s2', 'Inbox'),
      s('f1', { name: "[Triggered by file_watch 'Inbox']", trigger_type: 'file_watch' }),
      s('f2', { name: "[Triggered by file_watch 'Inbox']", trigger_type: 'file_watch' }),
      s('q1', { origin: 'queue' }), s('q2', { origin: 'queue' }),
    ];
    expect(shape(buildSessionList(sessions, 'automations').nodes)).toEqual([
      'schedule:Inbox[s1,s2]', 'file_watch:Inbox[f1,f2]', 'queue:[q1,q2]',
    ]);
  });

  it('runs the user renamed (no trigger name left) group by kind alone', () => {
    const sessions = [
      s('x1', { name: 'my rename', trigger_type: 'schedule' }),
      s('x2', { name: 'other rename', trigger_type: 'schedule' }),
    ];
    const { nodes } = buildSessionList(sessions, 'automations');
    expect(shape(nodes)).toEqual(['schedule:[x1,x2]']);
    expect(nodes[0].type === 'group' && nodes[0].group.name).toBeNull();
  });

  it('counts every segment regardless of the active filter', () => {
    const sessions = [run('r1', 'Daily'), s('chat1'), run('r2', 'Daily'), s('pc', { pinned: true })];
    expect(buildSessionList(sessions, 'chats').counts).toEqual({ all: 4, chats: 2, automations: 2 });
  });
});
