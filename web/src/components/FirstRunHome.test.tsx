// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import FirstRunHome from './FirstRunHome';

afterEach(() => cleanup());

describe('FirstRunHome', () => {
  it('says what a project is and offers exactly one action', () => {
    const onNewProject = vi.fn();
    render(<FirstRunHome onNewProject={onNewProject} />);

    expect(screen.getByRole('heading', { name: 'Create your first project' })).toBeTruthy();
    expect(screen.getByText(/A project is a folder on your computer/)).toBeTruthy();

    const buttons = screen.getAllByRole('button');
    expect(buttons).toHaveLength(1);
    fireEvent.click(buttons[0]);
    expect(onNewProject).toHaveBeenCalledTimes(1);
  });

  it('does not mention Quick Tasks — the fork is not offered here', () => {
    render(<FirstRunHome onNewProject={vi.fn()} />);
    expect(screen.queryByText(/Quick Tasks/i)).toBeNull();
  });
});
