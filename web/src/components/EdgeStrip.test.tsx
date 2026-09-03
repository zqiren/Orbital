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
  const onGoHome = vi.fn();
  const props: ComponentProps<typeof EdgeStrip> = {
    projects,
    currentProjectId: 'proj-1',
    agentStatuses: {} as Record<string, AgentRunStatus>,
    pendingApprovals: {},
    onGoHome,
    children: (
      <div data-testid="flyout-content">
        <button type="button">Row</button>
      </div>
    ),
    ...overrides,
  };
  render(<EdgeStrip {...props} />);
  return { onGoHome };
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
  it('click navigates home', () => {
    const { onGoHome } = renderStrip();
    fireEvent.click(getStrip());
    expect(onGoHome).toHaveBeenCalledTimes(1);
  });

  it('Enter and Space navigate home', () => {
    const { onGoHome } = renderStrip();
    const strip = getStrip();
    fireEvent.keyDown(strip, { key: 'Enter' });
    fireEvent.keyDown(strip, { key: ' ' });
    expect(onGoHome).toHaveBeenCalledTimes(2);
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
      'Hover to peek at projects · Click to go back to projects',
    );
  });
});
