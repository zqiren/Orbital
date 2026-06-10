// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import SettingsIcon from './SettingsIcon';

describe('SettingsIcon', () => {
  it('renders a button with aria-label "Project settings"', () => {
    render(<SettingsIcon onClick={vi.fn()} />);
    expect(screen.getByRole('button', { name: /project settings/i })).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const onClick = vi.fn();
    render(<SettingsIcon onClick={onClick} />);
    fireEvent.click(screen.getByRole('button', { name: /project settings/i }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('uses the provided title attribute', () => {
    render(<SettingsIcon onClick={vi.fn()} title="Open settings" />);
    const btn = screen.getByRole('button', { name: /project settings/i });
    expect(btn.getAttribute('title')).toBe('Open settings');
  });

  it('defaults title to "Project settings" when not provided', () => {
    render(<SettingsIcon onClick={vi.fn()} />);
    const btn = screen.getByRole('button', { name: /project settings/i });
    expect(btn.getAttribute('title')).toBe('Project settings');
  });
});
