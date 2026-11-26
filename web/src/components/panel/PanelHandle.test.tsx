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
  // in the open panel's tone. No edge line: the tone step is the edge.
  it('is the folded panel inside the sheet: panel tone, no edge line', () => {
    render(<PanelHandle working={false} onExpand={vi.fn()} />);
    const handle = screen.getByTestId('panel-handle');
    expect(handle.className).toContain('bg-dock');
    expect(handle.className).not.toMatch(/\bborder-l\b|\bbg-nav\b/);
  });

  // 2026-11-26: no raised tab (it read as a separate button on a bar that is
  // all one button). The chevron sits on the bar in the left EdgeStrip's
  // look: muted, lighting while the bar is hovered.
  it("puts the chevron on the bar itself, in the edge strip's look, with no raised tab", () => {
    render(<PanelHandle working={false} onExpand={vi.fn()} />);
    expect(screen.queryByTestId('panel-handle-tab')).toBeNull();
    const handle = screen.getByTestId('panel-handle');
    expect(handle.querySelector('.panel-handle-tab')).toBeNull();
    const chevron = screen.getByTestId('panel-handle-chevron');
    expect(chevron.parentElement).toBe(handle);
    expect(chevron.className).toContain('text-secondary');
    expect(chevron.className).toContain('group-hover:text-primary');
    expect(chevron.className).toContain('group-hover:bg-card-hover');
    expect(chevron.className).not.toMatch(/shadow|bg-card(?!-hover)|bg-elevated/);
    expect(handle.className).toContain('group');
  });
});
