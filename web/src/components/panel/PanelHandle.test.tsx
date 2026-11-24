// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/** Spec 078 §5.1 — the collapsed state's edge handle. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import PanelHandle from './PanelHandle';

afterEach(() => cleanup());

describe('PanelHandle', () => {
  it('is a labelled button that expands the panel when clicked', () => {
    const onExpand = vi.fn();
    render(<PanelHandle working={false} onExpand={onExpand} />);
    const button = screen.getByRole('button', { name: 'Show workspace' });
    fireEvent.click(button);
    expect(onExpand).toHaveBeenCalledTimes(1);
  });

  it('shows the working dot only while the agent is working', () => {
    const { rerender } = render(<PanelHandle working={false} onExpand={vi.fn()} />);
    expect(screen.queryByTestId('panel-handle-working')).toBeNull();
    rerender(<PanelHandle working onExpand={vi.fn()} />);
    expect(screen.getByTestId('panel-handle-working')).toBeInTheDocument();
  });

  it('is 20px wide (w-5) — the mirror of the left edge strip', () => {
    render(<PanelHandle working={false} onExpand={vi.fn()} />);
    expect(screen.getByTestId('panel-handle').className).toContain('w-5');
  });

  // Design pass D (8.10 B): the panel folded to 20px INSIDE the chat's sheet,
  // in the open panel's tone, with the chevron on a raised tab so it can be
  // found on a white sheet. No edge line: the tone step is the edge.
  it('is the folded panel inside the sheet: panel tone, no edge line, a raised chevron tab', () => {
    render(<PanelHandle working={false} onExpand={vi.fn()} />);
    const handle = screen.getByTestId('panel-handle');
    expect(handle.className).toContain('bg-dock');
    expect(handle.className).not.toMatch(/\bborder-l\b|\bbg-nav\b/);
    expect(screen.getByTestId('panel-handle-tab').className).toContain('panel-handle-tab');
  });
});
