// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * The Workbench composer on its own: the Enter that commits an IME candidate
 * (Safari/WKWebView: `isComposing` false, `keyCode` 229) must not send.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
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
