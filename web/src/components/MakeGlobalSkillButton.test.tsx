// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Project settings → Skills: the "use in all projects" icon on a skill row
 * (POST /api/v2/skills/from-project). Success turns it into a quiet check; a
 * 409 says "already exists" and offers Replace (overwrite: true); any other
 * failure says so and leaves the icon usable.
 */

import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const { apiMock, MockApiError } = vi.hoisted(() => {
  class MockApiError extends Error {
    constructor(
      public status: number,
      public detail: string,
    ) {
      super(detail);
      this.name = 'ApiError';
    }
  }
  return { apiMock: vi.fn(), MockApiError };
});
vi.mock('../config', () => ({ api: apiMock, ApiError: MockApiError }));

import MakeGlobalSkillButton from './MakeGlobalSkillButton';

afterEach(() => cleanup());

let onMessage: ReturnType<typeof vi.fn<(message: string) => void>>;

function renderButton(alreadyGlobal = false) {
  render(
    <MakeGlobalSkillButton
      projectId="proj_1"
      dirName="invoice-flow"
      name="Invoice flow"
      onMessage={onMessage}
      alreadyGlobal={alreadyGlobal}
      testId="mg"
    />,
  );
  return screen.queryByTestId('mg');
}

function bodies() {
  return apiMock.mock.calls.map(([p, o]) => [p, o.method, JSON.parse(String(o.body))]);
}

beforeEach(() => {
  apiMock.mockReset();
  onMessage = vi.fn<(message: string) => void>();
});

describe('MakeGlobalSkillButton', () => {
  it('is an icon whose only words are its aria-label and the hover/focus tooltip bubble', () => {
    const btn = renderButton()!;
    expect(btn.getAttribute('aria-label')).toBe('Share with all projects');
    const tip = btn.querySelector('[role="tooltip"]')!;
    expect(tip.textContent).toBe('Share with all projects');
    // Hidden until the button is hovered or keyboard-focused.
    expect(tip.className).toContain('opacity-0');
    expect(tip.className).toContain('group-hover:opacity-100');
    expect(tip.className).toContain('group-focus-visible:opacity-100');
    expect(btn.textContent).toBe('Share with all projects');
  });

  it('starts as the check when the global list already has the skill (survives a reload)', () => {
    expect(renderButton(true)).toBeNull();
    const done = screen.getByTestId('mg-done');
    expect(done.getAttribute('aria-label')).toBe('Shared with all projects');
    expect(done.querySelector('[role="tooltip"]')!.textContent).toBe('Shared with all projects');
    expect(done.tabIndex).toBe(0);
  });

  it('success: copies (not moves) the skill and becomes a quiet check', async () => {
    apiMock.mockResolvedValue({ name: 'Invoice flow', source: 'global', replaced: false });
    fireEvent.click(renderButton()!);
    expect((await screen.findByTestId('mg-done')).getAttribute('aria-label')).toBe('Shared with all projects');
    expect(bodies()).toEqual([
      ['/api/v2/skills/from-project', 'POST', { project_id: 'proj_1', dir_name: 'invoice-flow', overwrite: false }],
    ]);
    expect(screen.queryByTestId('mg')).not.toBeInTheDocument();
    expect(onMessage).toHaveBeenLastCalledWith('');
  });

  it('409: says it already exists, then Replace sends overwrite and ends as the check', async () => {
    apiMock.mockRejectedValueOnce(new MockApiError(409, 'Skill already exists: invoice-flow'));
    fireEvent.click(renderButton()!);
    const replace = await screen.findByTestId('mg-replace');
    expect(onMessage).toHaveBeenLastCalledWith('A skill with this name already exists.');
    apiMock.mockResolvedValueOnce({ replaced: true });
    fireEvent.click(replace);
    await screen.findByTestId('mg-done');
    expect(bodies()[1]).toEqual([
      '/api/v2/skills/from-project',
      'POST',
      { project_id: 'proj_1', dir_name: 'invoice-flow', overwrite: true },
    ]);
    expect(onMessage).toHaveBeenLastCalledWith('');
  });

  it('other failure: says so and keeps the icon usable', async () => {
    apiMock.mockRejectedValueOnce(new MockApiError(500, 'boom'));
    fireEvent.click(renderButton()!);
    await waitFor(() => expect(onMessage).toHaveBeenLastCalledWith("Couldn't share it with all projects."));
    const btn = screen.getByTestId('mg') as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
    expect(screen.queryByTestId('mg-done')).not.toBeInTheDocument();
  });

  it('a 400 (no name + description) reads as an invalid skill', async () => {
    apiMock.mockRejectedValueOnce(new MockApiError(400, 'SKILL.md needs a name'));
    fireEvent.click(renderButton()!);
    await waitFor(() => expect(onMessage).toHaveBeenLastCalledWith("This isn't a valid skill file."));
  });
});
