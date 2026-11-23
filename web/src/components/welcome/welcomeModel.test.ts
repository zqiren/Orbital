// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import type { SubAgentListEntry } from '../../hooks/useSubAgentList';
import { NO_CHOICES, folderKey, withTickChange, type ImportableResponse } from '../../utils/onboarding';
import {
  agentStateOf, defaultImportCount, importAgents, importCount, importLineTick, importRequestOf, readyAgentCount,
  toggleImportLine,
} from './welcomeModel';

const agent = (over: Partial<SubAgentListEntry>): SubAgentListEntry => ({
  slug: 'claude-code', name: 'Claude Code', installed: true, credentials_configured: true, ...over,
});

describe('agent state', () => {
  it('ready = installed and signed in; signed out has a login; key-only says key', () => {
    expect(agentStateOf(agent({}))).toBe('ready');
    expect(agentStateOf(agent({ auth_expired: true }))).toBe('out');
    expect(agentStateOf(agent({ credentials_configured: false }))).toBe('out');
    expect(agentStateOf(agent({ credentials_configured: false, supports_login: false }))).toBe('key');
  });

  it("the daemon's ready wins: signed in but blocked is off, not ready", () => {
    expect(agentStateOf(agent({ ready: false }))).toBe('off');
    expect(agentStateOf(agent({ ready: true }))).toBe('ready');
    expect(readyAgentCount([agent({ ready: false }), agent({ slug: 'codex', ready: true })])).toBe(1);
  });

  it('a signed-out or not-installed agent does not count as ready', () => {
    expect(readyAgentCount(null)).toBe(0);
    expect(readyAgentCount([
      agent({}),
      agent({ slug: 'codex', auth_expired: true }),
      agent({ slug: 'pi', installed: false }),
      agent({ slug: 'kimi', credentials_configured: false }),
    ])).toBe(1);
  });
});

const DATA: ImportableResponse = {
  status: 'ready',
  candidates: [
    { source: 'claude-code', name: 'a', path: '/a', session_count: 5, last_activity: null, by_agent: { 'claude-code': 5, codex: 1 } },
    { source: 'codex', name: 'b', path: '/b', session_count: 9, last_activity: null },
    { source: 'claude-code', name: 'm', path: '/m', session_count: 1, last_activity: null, minor: true },
    { source: 'claude-code', name: 'done', path: '/d', session_count: 3, last_activity: null, already_imported: true },
    { source: 'obsidian', name: 'v', path: '/v', session_count: 0, last_activity: null },
  ],
  groups: [
    { id: 'g1', session_count: 2, last_activity: null, by_agent: { codex: 2 }, reasons: [], suggested_folder: '/g1', members: [] },
    { id: 'g2', session_count: 2, last_activity: null, by_agent: {}, reasons: [], suggested_folder: null, members: [] },
  ],
};

describe('the import line', () => {
  it('counts what the checklist would import by default', () => {
    // a, b, v (major folders) + g1 (has a folder); not the minor one, not the imported one, not g2
    expect(importCount(DATA, NO_CHOICES)).toBe(4);
    expect(defaultImportCount(DATA, NO_CHOICES)).toBe(4);
    expect(importLineTick(DATA, NO_CHOICES)).toBe('all');
    expect(importRequestOf(DATA, NO_CHOICES)).toEqual({ folders: ['/a', '/b', '/v'], groups: [{ id: 'g1', workspace: '/g1' }] });
    expect(importCount(null, NO_CHOICES)).toBe(0);
  });

  it('one unticked row makes it partial', () => {
    const c = withTickChange(NO_CHOICES, DATA, folderKey('/a'), false);
    expect(importCount(DATA, c)).toBe(3);
    expect(importLineTick(DATA, c)).toBe('some');
  });

  it('the checkbox clears everything, then brings the defaults back', () => {
    const none = toggleImportLine(NO_CHOICES, DATA);
    expect(importCount(DATA, none)).toBe(0);
    expect(importLineTick(DATA, none)).toBe('none');
    const back = toggleImportLine(none, DATA);
    expect(importRequestOf(DATA, back)).toEqual(importRequestOf(DATA, NO_CHOICES));
    // from partial, a click clears
    const partial = withTickChange(NO_CHOICES, DATA, folderKey('/a'), false);
    expect(importCount(DATA, toggleImportLine(partial, DATA))).toBe(0);
  });

  it('a row that arrives after the user chose is not imported behind their back', () => {
    const none = toggleImportLine(NO_CHOICES, DATA);
    const later: ImportableResponse = {
      ...DATA,
      candidates: [...DATA.candidates, { source: 'codex', name: 'late', path: '/late', session_count: 4, last_activity: null }],
    };
    expect(importCount(later, none)).toBe(0);
  });

  it('agents behind the found projects, most conversations first, no vault', () => {
    expect(importAgents(DATA)).toEqual(['codex', 'claude-code']);
    expect(importAgents(null)).toEqual([]);
  });
});
