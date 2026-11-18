// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * The Workbench composer on its own: the Enter that commits an IME candidate
 * (Safari/WKWebView: `isComposing` false, `keyCode` 229) must not send.
 * Keyless, the agent picker's Orbital row is greyed and opens the model
 * settings; a worker unchecked for the picked project is not offered.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { ComponentProps } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../config', () => ({ api: vi.fn(() => Promise.resolve([])), isRelayMode: false, BASE_URL: '' }));

import WorkbenchComposer from './WorkbenchComposer';
import type { Project } from '../../types';
import type { SendChoice } from './sendTask';
import { api } from '../../config';
import { __clearSkillCatalogCacheForTests } from '../../lib/skill-catalog';

const PROJECTS = [{ project_id: 'p1', name: 'Marketing', workspace: '/ws/a' } as Project];

afterEach(cleanup);

function renderComposer() {
  const onSend = vi.fn((_choice: SendChoice) => Promise.resolve(true));
  render(<WorkbenchComposer projects={PROJECTS} agents={[]} onSend={onSend} />);
  const input = screen.getByTestId('wb-composer-input');
  fireEvent.change(input, { target: { value: '写周报' } });
  return { onSend, input };
}

describe('WorkbenchComposer — IME guard', () => {
  it('an Enter with keyCode 229 (IME commit) does not send', () => {
    const { onSend, input } = renderComposer();
    fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 });
    expect(onSend).not.toHaveBeenCalled();
    expect(input).toHaveValue('写周报');
  });

  it('a plain Enter sends', () => {
    const { onSend } = renderComposer();
    fireEvent.keyDown(screen.getByTestId('wb-composer-input'), { key: 'Enter', keyCode: 13 });
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend.mock.calls[0][0]).toMatchObject({ text: '写周报' });
  });
});

const WORKERS = [{ slug: 'claude-code', name: 'Claude Code' }, { slug: 'pi', name: 'Pi' }];

function openAgentPicker(props: Partial<ComponentProps<typeof WorkbenchComposer>> = {}) {
  const onSend = vi.fn((_choice: SendChoice) => Promise.resolve(true));
  render(<WorkbenchComposer projects={PROJECTS} agents={WORKERS} onSend={onSend} {...props} />);
  fireEvent.click(screen.getByTestId('wb-chip-agent'));
  return { onSend, pop: screen.getByTestId('wb-pop-agent') };
}

const optionLabels = (pop: HTMLElement) =>
  Array.from(pop.querySelectorAll('[role="option"]')).map((o) => o.querySelector('.wb-n1')?.textContent);

describe('WorkbenchComposer — agent picker', () => {
  it('keyless: the Orbital row is greyed with the key note, and a click opens the model settings', () => {
    const { onSend, pop } = openAgentPicker({ keyless: true });
    const orbital = screen.getByTestId('wb-opt-orbital');
    expect(orbital).toHaveAttribute('aria-disabled', 'true');
    expect(orbital.className).toMatch(/\bis-off\b/);
    expect(orbital.textContent).toContain('API');
    expect(optionLabels(pop).filter((n) => n === 'Orbital')).toHaveLength(1);
    const opened = vi.fn();
    window.addEventListener('open-global-settings', opened);
    fireEvent.click(orbital);
    window.removeEventListener('open-global-settings', opened);
    expect(opened).toHaveBeenCalledTimes(1);
    expect((opened.mock.calls[0][0] as CustomEvent).detail).toEqual({ section: 'llm' });
    expect(screen.queryByTestId('wb-pop-agent')).toBeNull();
    // Orbital was not picked: the chip still reads 自动 and a send carries no agent.
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'hi' } });
    fireEvent.keyDown(screen.getByTestId('wb-composer-input'), { key: 'Enter', keyCode: 13 });
    expect(onSend.mock.calls[0][0]).toMatchObject({ agent: null });
  });

  it('keyed: the Orbital row is a plain, selectable row', () => {
    openAgentPicker();
    expect(screen.queryByTestId('wb-opt-orbital')).toBeNull();
    const pop = screen.getByTestId('wb-pop-agent');
    const orbital = Array.from(pop.querySelectorAll('[role="option"]')).find((o) => o.textContent?.includes('Orbital'))!;
    expect(orbital).not.toHaveAttribute('aria-disabled');
  });

  it('a worker unchecked for the picked project is not offered', () => {
    const projects = [{ ...PROJECTS[0], disabled_sub_agents: ['pi'] } as Project];
    const { pop } = openAgentPicker({ projects, keyless: true });
    expect(optionLabels(pop)).toContain('Claude Code');
    expect(optionLabels(pop)).not.toContain('Pi');
  });

  it('with no project picked (自动) every worker is offered', () => {
    const projects = [{ ...PROJECTS[0], disabled_sub_agents: ['pi'] } as Project];
    const { pop } = openAgentPicker({ projects });
    expect(optionLabels(pop)).toEqual(expect.arrayContaining(['Orbital', 'Claude Code', 'Pi']));
  });
});

// ---------------------------------------------------------------------------
// Batch 1118: the `/` picker in the Workbench composer
// ---------------------------------------------------------------------------


describe('WorkbenchComposer — the / picker', () => {
  const REVIEW = {
    key: 'review', name: 'review', description: 'Review a PR', source: 'worker',
    path: '', dir_name: '', invoke: '/review ', hint: '<pr>',
  };
  const IMAGEGEN = {
    key: 'imagegen', name: 'imagegen', description: 'Make images', source: 'worker',
    path: '', dir_name: '', invoke: '$imagegen ', hint: '',
  };
  const TASK = {
    key: 'task planning', name: 'Task Planning', description: 'Plan.', source: 'project',
    path: '/ws/orbital/skills/task-planning/SKILL.md', dir_name: 'task-planning',
  };
  const QT = { project_id: 'qt', name: 'Quick Tasks', workspace: '/ws/qt', is_scratch: true,
    effective_pinned_target: null } as Project;
  const MKT = { project_id: 'p1', name: 'Marketing', workspace: '/ws/a',
    effective_pinned_target: 'codex' } as Project;
  let catalogPaths: string[] = [];

  beforeEach(() => {
    __clearSkillCatalogCacheForTests();
    localStorage.clear();
    catalogPaths = [];
    vi.mocked(api).mockImplementation(((path: string) => {
      if (!path.includes('/skill-catalog')) return Promise.resolve([]);
      catalogPaths.push(path);
      const worker = new URL(path, 'http://x').searchParams.get('worker');
      if (worker === 'claude-code') return Promise.resolve([REVIEW, TASK]);
      if (worker === 'codex') return Promise.resolve([IMAGEGEN, TASK]);
      return Promise.resolve([TASK]);
    }) as never);
  });
  afterEach(() => {
    vi.mocked(api).mockImplementation((() => Promise.resolve([])) as never);
  });

  const names = () =>
    Array.from(document.querySelectorAll('[data-testid="skill-option"]')).map((el) => el.getAttribute('data-skill-name'));
  const input = () => screen.getByTestId('wb-composer-input') as HTMLTextAreaElement;
  async function type(value: string) {
    await act(async () => { fireEvent.change(input(), { target: { value } }); });
    await act(async () => { await Promise.resolve(); });
  }
  async function press(key: string) {
    await act(async () => { fireEvent.keyDown(input(), { key, keyCode: key === 'Enter' ? 13 : 0 }); });
  }
  function mount(props: Partial<ComponentProps<typeof WorkbenchComposer>> = {}) {
    const onSend = vi.fn((_choice: SendChoice) => Promise.resolve(true));
    render(<WorkbenchComposer projects={[QT, MKT]} agents={[{ slug: 'claude-code', name: 'Claude Code' }, { slug: 'codex', name: 'Codex' }]} onSend={onSend} {...props} />);
    return onSend;
  }
  async function pickAgent(name: string) {
    fireEvent.click(screen.getByTestId('wb-chip-agent'));
    const pop = screen.getByTestId('wb-pop-agent');
    const opt = Array.from(pop.querySelectorAll('[role="option"]')).find((o) => o.querySelector('.wb-n1')?.textContent === name)!;
    await act(async () => { fireEvent.click(opt); });
    await act(async () => { await Promise.resolve(); });
  }
  async function pickProject(name: string) {
    fireEvent.click(screen.getByTestId('wb-chip-project'));
    const pop = screen.getByTestId('wb-pop-project');
    const opt = Array.from(pop.querySelectorAll('[role="option"], button')).find((o) => o.textContent?.includes(name))!;
    await act(async () => { fireEvent.click(opt); });
  }

  it('a picked worker lists its own commands; a pick prefills the invocation and Enter sends it as typed', async () => {
    const onSend = mount();
    await pickAgent('Claude Code');
    await type('/rev');
    expect(names()).toEqual(['review']);
    // 自动 project: the list is Quick Tasks' for the picked worker.
    expect(catalogPaths.some((p) => p.includes('/projects/qt/skill-catalog?worker=claude-code'))).toBe(true);
    await press('Enter');
    expect(onSend).not.toHaveBeenCalled();
    expect(input().value).toBe('/review ');
    await type('/review 12');
    await press('Enter');
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend.mock.calls[0][0]).toMatchObject({
      text: '/review 12', projectId: null, agent: 'claude-code', fallbackProjectId: 'qt', skill: null,
    });
  });

  it('自动 agent in a picked project lists the project pin\'s commands', async () => {
    mount();
    await pickProject('Marketing');
    await type('/');
    expect(catalogPaths.some((p) => p.includes('/projects/p1/skill-catalog?worker=codex'))).toBe(true);
    expect(names()).toContain('imagegen');
    expect(names()).not.toContain('review');
  });

  it('自动 agent with no pin lists Orbital\'s own entries (skills and /automate), no worker commands', async () => {
    mount();
    await type('/');
    expect(catalogPaths).toContain('/api/v2/projects/qt/skill-catalog');
    expect(names()).toEqual(expect.arrayContaining(['automate', 'Task Planning']));
    expect(names()).not.toContain('review');
  });

  it('a skill pick is a chip; the send carries it', async () => {
    const onSend = mount();
    await type('/tas');
    await press('Enter');
    expect(screen.getByTestId('wb-skill-chip').textContent).toContain('Task Planning');
    expect(input().value).toBe('');
    await type('plan my week');
    await press('Enter');
    expect(onSend.mock.calls[0][0].skill).toMatchObject({ key: TASK.key, path: TASK.path });
    expect(screen.queryByTestId('wb-skill-chip')).toBeNull();
  });

  it('/automate chips and prefills; a command pick afterwards drops the chip', async () => {
    mount();
    await pickAgent('Claude Code');
    await type('/aut');
    await press('Enter');
    expect(screen.getByTestId('wb-skill-chip').textContent).toContain('automate');
    expect(input().value).not.toBe('');
    await type('/rev');
    await press('Enter');
    expect(screen.queryByTestId('wb-skill-chip')).toBeNull();
    expect(input().value).toBe('/review ');
  });

  it('keyless: no /automate (no manager to run it)', async () => {
    mount({ keyless: true, projects: [{ ...QT, manager_ready: false } as Project, { ...MKT, manager_ready: false } as Project] });
    await type('/');
    expect(names()).toEqual(['Task Planning']);
  });

  it('Esc closes the menu and Enter then sends the text', async () => {
    const onSend = mount();
    await type('/tas');
    expect(names()).toEqual(['Task Planning']);
    await press('Escape');
    expect(names()).toEqual([]);
    await press('Enter');
    expect(onSend.mock.calls[0][0]).toMatchObject({ text: '/tas' });
  });
});
