// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import FirstJourneyTour from './FirstJourneyTour';

/** Put `data-tour` anchors on the page. jsdom does no layout, so every element
 * measures 0×0 — which the tour (rightly) reads as "not on screen". Give the
 * listed anchors a real box. */
function mountAnchors(ids: string[]) {
  const host = document.createElement('div');
  host.id = 'anchors';
  for (const id of ids) {
    const el = document.createElement('div');
    el.setAttribute('data-tour', id);
    el.getBoundingClientRect = () =>
      ({ left: 100, top: 100, width: 200, height: 40, right: 300, bottom: 140, x: 100, y: 100, toJSON() {} }) as DOMRect;
    host.appendChild(el);
  }
  document.body.appendChild(host);
}

const ALL = ['project-header', 'composer', 'pin-select', 'workspace-panel', 'tab-files', 'tab-queue', 'project-settings', 'edge-strip'];

beforeEach(() => {
  Object.defineProperty(window, 'innerWidth', { value: 1280, configurable: true });
  Object.defineProperty(window, 'innerHeight', { value: 800, configurable: true });
});
afterEach(() => {
  cleanup();
  document.getElementById('anchors')?.remove();
});

const stepId = () => screen.getByTestId('tour-card').getAttribute('data-step');

describe('FirstJourneyTour', () => {
  it('walks all eight stops in order, then closes on Done', async () => {
    mountAnchors(ALL);
    const onClose = vi.fn();
    render(<FirstJourneyTour folderName="copenhagen" agentNames={['Claude Code', 'Codex']} onClose={onClose} />);

    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    expect(stepId()).toBe('project');
    expect(screen.getByTestId('tour-card').textContent).toContain('Your agent works inside copenhagen');
    expect(screen.getByTestId('tour-counter').textContent).toBe('1 of 8');

    const seen = [stepId()];
    for (let i = 0; i < 7; i++) {
      fireEvent.click(screen.getByTestId('tour-next'));
      seen.push(stepId());
    }
    expect(seen).toEqual(['project', 'composer', 'agents', 'panel', 'files', 'tasks', 'settings', 'quickTasks']);
    // The last stop offers Done and no Skip.
    expect(screen.getByTestId('tour-next').textContent).toBe('Done');
    expect(screen.queryByTestId('tour-skip')).toBeNull();
    expect(onClose).not.toHaveBeenCalled();

    fireEvent.click(screen.getByTestId('tour-next'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('names the detected agents on the agents stop', async () => {
    mountAnchors(ALL);
    render(<FirstJourneyTour folderName="x" agentNames={['Claude Code', 'Codex']} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    fireEvent.click(screen.getByTestId('tour-next'));
    fireEvent.click(screen.getByTestId('tour-next'));
    expect(stepId()).toBe('agents');
    expect(screen.getByTestId('tour-card').textContent).toContain('Orbital found Claude Code, Codex on this computer');
  });

  it('skips stops whose anchor is not on screen — no sub-agents, panel not docked — and counts only the rest', async () => {
    mountAnchors(ALL.filter((id) => id !== 'pin-select' && id !== 'workspace-panel'));
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);

    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    expect(screen.getByTestId('tour-counter').textContent).toBe('1 of 6');
    const seen = [stepId()];
    for (let i = 0; i < 5; i++) {
      fireEvent.click(screen.getByTestId('tour-next'));
      seen.push(stepId());
    }
    expect(seen).toEqual(['project', 'composer', 'files', 'tasks', 'settings', 'quickTasks']);
  });

  it('falls back to the panel handle when the panel is collapsed', async () => {
    mountAnchors(['project-header', 'panel-handle']);
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    fireEvent.click(screen.getByTestId('tour-next'));
    expect(stepId()).toBe('panel');
  });

  it('ignores an anchor that is painted but invisible (the closed project-list flyout keeps its Quick Tasks row at opacity 0)', async () => {
    // EdgeStrip hides its flyout with opacity, not display:none (a WKWebView
    // compositor workaround), so the row inside still measures a real box.
    // Found in a real browser: the last stop spotlighted empty space.
    mountAnchors(['project-header', 'edge-strip']);
    const flyout = document.createElement('div');
    flyout.style.opacity = '0';
    const row = document.createElement('div');
    row.setAttribute('data-tour', 'quick-tasks-row');
    row.getBoundingClientRect = () =>
      ({ left: 500, top: 500, width: 200, height: 30, right: 700, bottom: 530, x: 500, y: 500, toJSON() {} }) as DOMRect;
    flyout.appendChild(row);
    document.getElementById('anchors')!.appendChild(flyout);

    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    fireEvent.click(screen.getByTestId('tour-next'));
    expect(stepId()).toBe('quickTasks');
    // Spotlight sits on the edge strip (100,100), not the hidden row (500,500).
    expect(screen.getByTestId('tour-spotlight').style.left).toBe('94px');
  });

  it('Skip tour and Escape both close it', async () => {
    mountAnchors(ALL);
    const onClose = vi.fn();
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={onClose} />);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());

    fireEvent.click(screen.getByTestId('tour-skip'));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it('spotlights the anchor with a little padding', async () => {
    mountAnchors(ALL);
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('tour-spotlight')).toBeTruthy());
    const s = screen.getByTestId('tour-spotlight').style;
    expect([s.left, s.top, s.width, s.height]).toEqual(['94px', '94px', '212px', '52px']);
  });
});
