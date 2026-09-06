// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, render, screen, cleanup, fireEvent } from '@testing-library/react';
import type { ComponentProps } from 'react';
import type { Project, AgentRunStatus } from '../types';
import EdgeStrip from './EdgeStrip';

afterEach(() => {
  cleanup();
});

function makeProject(id: string): Project {
  return {
    project_id: id,
    name: id,
    workspace: `/tmp/${id}`,
    model: 'claude-3-5-sonnet',
    api_key: '',
    base_url: null,
    autonomy: 'hands_off',
    instructions: '',
  };
}

const projects: Project[] = [makeProject('proj-1'), makeProject('proj-2'), makeProject('proj-3')];

function renderStrip(overrides: Partial<ComponentProps<typeof EdgeStrip>> = {}) {
  const onTogglePin = vi.fn();
  const props: ComponentProps<typeof EdgeStrip> = {
    projects,
    currentProjectId: 'proj-1',
    agentStatuses: {} as Record<string, AgentRunStatus>,
    pendingApprovals: {},
    pinned: false,
    onTogglePin,
    children: (
      <div data-testid="flyout-content">
        <button type="button">Row</button>
      </div>
    ),
    ...overrides,
  };
  const view = render(<EdgeStrip {...props} />);
  return {
    onTogglePin,
    /** Re-render with the owner's new `pinned` value, the way App does after a toggle. */
    setPinned(pinned: boolean) {
      view.rerender(<EdgeStrip {...props} pinned={pinned} />);
    },
  };
}

function getWrapper() {
  return screen.getByTestId('edge-strip-wrapper');
}

function getStrip() {
  return screen.getByRole('button', { name: 'Projects' });
}

function getFlyout() {
  // The flyout wrapper is the direct parent of the children we render into it.
  return screen.getByTestId('flyout-content').parentElement as HTMLElement;
}

describe('EdgeStrip — hover-intent', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('opens the flyout after 120ms of hover on the strip', () => {
    renderStrip();
    const strip = getStrip();
    expect(strip).toHaveAttribute('aria-expanded', 'false');

    fireEvent.mouseEnter(strip);
    act(() => {
      vi.advanceTimersByTime(119);
    });
    expect(strip).toHaveAttribute('aria-expanded', 'false');

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(strip).toHaveAttribute('aria-expanded', 'true');
  });

  it('does not open on a brush-past shorter than the open delay', () => {
    renderStrip();
    const strip = getStrip();

    fireEvent.mouseEnter(strip);
    act(() => {
      vi.advanceTimersByTime(50);
    });
    fireEvent.mouseLeave(strip);
    act(() => {
      vi.advanceTimersByTime(200);
    });

    expect(strip).toHaveAttribute('aria-expanded', 'false');
  });

  it('stays open when the pointer moves from the strip into the flyout', () => {
    renderStrip();
    const strip = getStrip();

    fireEvent.mouseEnter(strip);
    act(() => {
      vi.advanceTimersByTime(120);
    });
    expect(strip).toHaveAttribute('aria-expanded', 'true');

    const flyout = getFlyout();
    fireEvent.mouseLeave(strip);
    fireEvent.mouseEnter(flyout);
    act(() => {
      vi.advanceTimersByTime(300);
    });

    expect(strip).toHaveAttribute('aria-expanded', 'true');
  });

  it('closes 160ms after the pointer has left both the strip and the flyout', () => {
    renderStrip();
    const strip = getStrip();

    fireEvent.mouseEnter(strip);
    act(() => {
      vi.advanceTimersByTime(120);
    });
    const flyout = getFlyout();
    fireEvent.mouseLeave(strip);
    fireEvent.mouseEnter(flyout);
    fireEvent.mouseLeave(flyout);

    act(() => {
      vi.advanceTimersByTime(159);
    });
    expect(strip).toHaveAttribute('aria-expanded', 'true');

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(strip).toHaveAttribute('aria-expanded', 'false');
  });

  it('entering the flyout again cancels a pending close', () => {
    renderStrip();
    const strip = getStrip();

    fireEvent.mouseEnter(strip);
    act(() => {
      vi.advanceTimersByTime(120);
    });
    const flyout = getFlyout();
    fireEvent.mouseLeave(strip);
    act(() => {
      vi.advanceTimersByTime(100); // mid-flight through the 160ms close delay
    });
    fireEvent.mouseEnter(flyout);
    act(() => {
      vi.advanceTimersByTime(200); // past where the original close would have fired
    });

    expect(strip).toHaveAttribute('aria-expanded', 'true');
  });
});

describe('EdgeStrip — activation', () => {
  it('click asks the owner to toggle the pin — it never navigates', () => {
    const { onTogglePin } = renderStrip();
    fireEvent.click(getStrip());
    expect(onTogglePin).toHaveBeenCalledTimes(1);
  });

  it('Enter and Space toggle the pin', () => {
    const { onTogglePin } = renderStrip();
    const strip = getStrip();
    fireEvent.keyDown(strip, { key: 'Enter' });
    fireEvent.keyDown(strip, { key: ' ' });
    expect(onTogglePin).toHaveBeenCalledTimes(2);
  });

  it('Esc closes the open flyout and returns focus to the strip', () => {
    vi.useFakeTimers();
    renderStrip();
    const strip = getStrip();
    fireEvent.mouseEnter(strip);
    act(() => {
      vi.advanceTimersByTime(120);
    });
    expect(strip).toHaveAttribute('aria-expanded', 'true');

    fireEvent.keyDown(window, { key: 'Escape' });

    expect(strip).toHaveAttribute('aria-expanded', 'false');
    expect(document.activeElement).toBe(strip);
    vi.useRealTimers();
  });
});

describe('EdgeStrip — pinned (docked list)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('shows the list without hover, widens the wrapper to the list width, and drops the flyout shadow', () => {
    renderStrip({ pinned: true });
    const strip = getStrip();
    expect(strip).toHaveAttribute('aria-expanded', 'true');
    expect(strip).toHaveAttribute('aria-pressed', 'true');
    // Docked is exactly the 260px column every non-project route draws for
    // the full Sidebar, never 280: the list is 248 at x=12, sharing 8px of
    // its padding under the rail.
    expect(getWrapper()).toHaveClass('w-[260px]');
    expect(getFlyout()).toHaveClass('w-[248px]');
    expect(getWrapper()).toHaveAttribute('data-pinned', 'true');
    const flyout = getFlyout();
    expect(flyout).toHaveClass('opacity-100');
    expect(flyout).not.toHaveClass('shadow-xl');
    expect(flyout).not.toHaveClass('pointer-events-none');
  });

  it('floated, the wrapper is 20px and the strip is not pressed', () => {
    renderStrip();
    expect(getWrapper()).toHaveClass('w-5');
    expect(getStrip()).toHaveAttribute('aria-pressed', 'false');
  });

  it('hover-leave does not close a pinned list', () => {
    renderStrip({ pinned: true });
    const strip = getStrip();
    fireEvent.mouseEnter(strip);
    fireEvent.mouseLeave(strip);
    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(strip).toHaveAttribute('aria-expanded', 'true');
  });

  it('a click inside a pinned list (selecting a project) does not close it', () => {
    renderStrip({ pinned: true });
    fireEvent.click(screen.getByRole('button', { name: 'Row' }));
    expect(getStrip()).toHaveAttribute('aria-expanded', 'true');
  });

  it('Esc never unpins or hides a pinned list', () => {
    renderStrip({ pinned: true });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(getStrip()).toHaveAttribute('aria-expanded', 'true');
  });

  it('the rail hairline exists only while collapsed — floated or docked it is the list\'s gutter', () => {
    vi.useRealTimers();
    const { setPinned } = renderStrip();
    expect(getStrip()).toHaveClass('border-r');
    setPinned(true);
    expect(getStrip()).not.toHaveClass('border-r');
  });

  it('hides the aggregate dot while pinned — the real per-project dots are on screen', () => {
    renderStrip({ pinned: true, pendingApprovals: { 'proj-2': 1 } });
    expect(screen.queryByTestId('strip-dot')).not.toBeInTheDocument();
  });

  it('unpinning leaves the list floated until the pointer leaves the strip', () => {
    const { onTogglePin, setPinned } = renderStrip({ pinned: true });
    const strip = getStrip();
    // The pointer is on the strip when it is clicked.
    fireEvent.mouseEnter(strip);
    fireEvent.click(strip);
    expect(onTogglePin).toHaveBeenCalledTimes(1);
    setPinned(false);

    expect(strip).toHaveAttribute('aria-expanded', 'true');
    expect(getWrapper()).toHaveClass('w-5');
    expect(getFlyout()).toHaveClass('shadow-xl');

    fireEvent.mouseLeave(strip);
    act(() => {
      vi.advanceTimersByTime(160);
    });
    expect(strip).toHaveAttribute('aria-expanded', 'false');
  });

  it('pinning from a hovered flyout keeps it visible with no gap', () => {
    const { setPinned } = renderStrip();
    const strip = getStrip();
    fireEvent.mouseEnter(strip);
    act(() => {
      vi.advanceTimersByTime(120);
    });
    expect(strip).toHaveAttribute('aria-expanded', 'true');

    fireEvent.click(strip);
    setPinned(true);
    expect(strip).toHaveAttribute('aria-expanded', 'true');
    expect(getWrapper()).toHaveClass('w-[260px]');
  });

  it('the list sits beside the rail in every state — it never moves between floated and docked', () => {
    const { setPinned } = renderStrip();
    expect(getFlyout()).toHaveClass('left-3');
    setPinned(true);
    expect(getFlyout()).toHaveClass('left-3');
  });

  it('the strip chevron tooltip flips when pinned', () => {
    const { setPinned } = renderStrip();
    expect(getStrip()).toHaveAttribute('title', 'Hover to peek at projects · Click to keep them open');
    setPinned(true);
    expect(getStrip()).toHaveAttribute('title', 'Click to hide projects');
  });
});

describe('EdgeStrip — aggregate dot', () => {
  it('shows nothing when no other project needs attention', () => {
    renderStrip({
      agentStatuses: { 'proj-2': 'running', 'proj-3': 'idle' },
      pendingApprovals: {},
    });
    expect(screen.queryByTestId('strip-dot')).not.toBeInTheDocument();
  });

  it('shows amber when another project has a pending approval', () => {
    renderStrip({
      agentStatuses: {},
      pendingApprovals: { 'proj-2': 1 },
    });
    expect(screen.getByTestId('strip-dot')).toHaveClass('bg-warning');
  });

  it('prefers red over amber when both are present among other projects', () => {
    renderStrip({
      agentStatuses: { 'proj-2': 'error' },
      pendingApprovals: { 'proj-3': 1 },
    });
    expect(screen.getByTestId('strip-dot')).toHaveClass('bg-error');
  });

  it('excludes the current project from the aggregate', () => {
    renderStrip({
      currentProjectId: 'proj-2',
      agentStatuses: { 'proj-2': 'error' },
      pendingApprovals: {},
    });
    expect(screen.queryByTestId('strip-dot')).not.toBeInTheDocument();
  });

  it('does not surface "running" elsewhere as a dot', () => {
    renderStrip({
      agentStatuses: { 'proj-2': 'running' },
      pendingApprovals: {},
    });
    expect(screen.queryByTestId('strip-dot')).not.toBeInTheDocument();
  });
});

describe('EdgeStrip — accessibility', () => {
  it('exposes aria-label, aria-expanded, and a tooltip on the strip button', () => {
    renderStrip();
    const strip = getStrip();
    expect(strip).toHaveAttribute('aria-expanded', 'false');
    expect(strip).toHaveAttribute(
      'title',
      'Hover to peek at projects · Click to keep them open',
    );
  });
});
