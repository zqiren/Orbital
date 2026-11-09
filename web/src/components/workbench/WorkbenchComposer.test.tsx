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

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { ComponentProps } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../config', () => ({ api: vi.fn(() => Promise.resolve([])), isRelayMode: false, BASE_URL: '' }));

import WorkbenchComposer from './WorkbenchComposer';
import type { Project } from '../../types';
import type { SendChoice } from './sendTask';

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
