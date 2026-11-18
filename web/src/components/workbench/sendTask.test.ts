// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Batch 1118: what the `/` picker put in the Workbench composer (a worker
 * command, a skill chip, `/automate`) is never handed to the routing model.
 * It goes straight to the recipient in the project whose list was shown: the
 * picked project, or — project chip on 自动 — Quick Tasks.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../../config', () => ({ api: apiMock, isRelayMode: false, BASE_URL: '' }));

import { sendWorkbenchTask, type SendDeps } from './sendTask';
import { __resetWorkbenchFeed } from '../../hooks/useWorkbenchFeed';
import type { SkillCatalogEntry } from '../../lib/skill-catalog';

function deps(pin: string | null = 'claude-code') {
  const injectMessage = vi.fn<SendDeps['injectMessage']>(() => Promise.resolve({}));
  const newSession = vi.fn<SendDeps['newSession']>(() => Promise.resolve({ session_id: 'S1', pinned_target: pin }));
  return { injectMessage, newSession } satisfies SendDeps;
}

const routeCalls = () => apiMock.mock.calls.filter(([p]) => String(p).includes('/workbench/route'));

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockImplementation((path: string) =>
    Promise.resolve(String(path).includes('/workbench/route') ? { project_id: 'routed', by: 'model' } : {}));
  __resetWorkbenchFeed();
  localStorage.clear();
});
afterEach(() => vi.clearAllMocks());

describe('sendWorkbenchTask — commands skip routing', () => {
  it('a command on 自动 goes to Quick Tasks, never to the route', async () => {
    const d = deps();
    const out = await sendWorkbenchTask(
      { text: '/context', projectId: null, agent: null, fallbackProjectId: 'qt' }, d);
    expect(out).toEqual({ ok: true, projectId: 'qt', sessionId: 'S1' });
    expect(routeCalls()).toHaveLength(0);
    expect(d.newSession).toHaveBeenCalledWith('qt');
    // To the minted pin, verbatim.
    expect(d.injectMessage.mock.calls[0]).toEqual(
      ['qt', '/context', 'claude-code', expect.any(String), undefined, 'S1', true,
        undefined, undefined, undefined]);
  });

  it('a command with a picked worker goes to that worker', async () => {
    const d = deps(null);
    await sendWorkbenchTask(
      { text: '/review 12', projectId: null, agent: 'codex', fallbackProjectId: 'qt' }, d);
    expect(routeCalls()).toHaveLength(0);
    expect(d.injectMessage.mock.calls[0][1]).toBe('/review 12');
    expect(d.injectMessage.mock.calls[0][2]).toBe('codex');
  });

  it('a command in a picked project stays there', async () => {
    const d = deps();
    await sendWorkbenchTask({ text: '/compact', projectId: 'p1', agent: null, fallbackProjectId: 'qt' }, d);
    expect(d.newSession).toHaveBeenCalledWith('p1');
  });

  it('a command on 自动 with no Quick Tasks is unrouted: the text comes back, no route call', async () => {
    const d = deps();
    const out = await sendWorkbenchTask({ text: '/compact', projectId: null, agent: null }, d);
    expect(out).toEqual({ ok: false, restored: true, unrouted: true });
    expect(routeCalls()).toHaveLength(0);
    expect(d.newSession).not.toHaveBeenCalled();
  });

  it('a path is not a command: it is routed as text', async () => {
    const d = deps();
    await sendWorkbenchTask(
      { text: '/Users/me/a.txt look at this', projectId: null, agent: null, fallbackProjectId: 'qt' }, d);
    expect(routeCalls()).toHaveLength(1);
    expect(d.newSession).toHaveBeenCalledWith('routed');
  });

  it('a catalog skill chip rides as the inject payload, in the project whose list was shown', async () => {
    const d = deps();
    const skill: SkillCatalogEntry = {
      key: 'task planning', name: 'Task Planning', description: '', source: 'project',
      path: '/qt/orbital/skills/task-planning/SKILL.md', dir_name: 'task-planning',
    };
    await sendWorkbenchTask(
      { text: 'plan my week', projectId: null, agent: null, fallbackProjectId: 'qt', skill }, d);
    expect(routeCalls()).toHaveLength(0);
    const call = d.injectMessage.mock.calls[0];
    expect(call[0]).toBe('qt');
    expect(call[1]).toBe('plan my week');
    expect(call[7]).toEqual({ key: skill.key, path: skill.path });
  });

  it('/automate goes to the manager with its instructions inline, whatever the pin', async () => {
    const d = deps('claude-code');
    const automate: SkillCatalogEntry = {
      key: 'builtin:automate', name: 'automate', description: '', source: 'builtin',
      path: 'orbital:builtin/automate', dir_name: 'automate',
      builtin: { body: 'Create an automation…', prefill: 'every weekday at 9am', managerOnly: true },
    };
    await sendWorkbenchTask(
      { text: 'every weekday at 9am summarise my inbox', projectId: 'p1', agent: null, skill: automate }, d);
    const call = d.injectMessage.mock.calls[0];
    expect(call[2]).toBeUndefined();
    expect(call[6]).toBe(false);
    expect(call[1]).toContain('Create an automation…');
    expect(call[1]).toMatch(/every weekday at 9am summarise my inbox$/);
    expect(call[7]).toBeUndefined();
  });

  it('ordinary text on 自动 is still routed', async () => {
    const d = deps();
    await sendWorkbenchTask({ text: 'write the weekly report', projectId: null, agent: null, fallbackProjectId: 'qt' }, d);
    expect(routeCalls()).toHaveLength(1);
  });
});
