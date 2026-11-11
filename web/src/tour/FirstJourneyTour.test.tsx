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
function mountAnchors(ids: string[], leftOffAnswered = true) {
  const host = document.createElement('div');
  host.id = 'anchors';
  // The empty chat's left-off answer: "no cards" unless the cards are mounted.
  if (leftOffAnswered && !ids.includes('left-off')) {
    const marker = document.createElement('span');
    marker.setAttribute('data-left-off-state', 'none');
    host.appendChild(marker);
  }
  for (const id of ids) {
    const el = document.createElement('div');
    el.setAttribute('data-tour', id);
    el.getBoundingClientRect = () =>
      ({ left: 100, top: 100, width: 200, height: 40, right: 300, bottom: 140, x: 100, y: 100, toJSON() {} }) as DOMRect;
    host.appendChild(el);
  }
  document.body.appendChild(host);
}

// Spec 137: no tab bar. Files anchors on the panel's maximize control (panel
// open), Tasks on the status line under the title.
const ALL = ['project-header', 'pin-select', 'workspace-panel', 'panel-maximize', 'status-line', 'project-settings', 'quick-tasks-row'];

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
  it('walks all seven stops in order (no composer stop), then closes on Done', async () => {
    mountAnchors(ALL);
    const onClose = vi.fn();
    render(<FirstJourneyTour folderName="copenhagen" agentNames={['Claude Code', 'Codex']} onClose={onClose} />);

    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    expect(stepId()).toBe('project');
    expect(screen.getByTestId('tour-card').textContent).toContain('Your agent works from the context in copenhagen');
    expect(screen.getByTestId('tour-counter').textContent).toBe('1 of 7');

    const seen = [stepId()];
    for (let i = 0; i < 6; i++) {
      fireEvent.click(screen.getByTestId('tour-next'));
      seen.push(stepId());
    }
    expect(seen).toEqual(['project', 'agents', 'panel', 'files', 'tasks', 'settings', 'quickTasks']);
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
    expect(stepId()).toBe('agents');
    expect(screen.getByTestId('tour-card').textContent).toContain('Orbital found Claude Code, Codex on this computer');
  });

  it('skips stops whose anchor is not on screen — no sub-agents, panel not docked — and counts only the rest', async () => {
    // A narrow window: no docked panel and no handle; the header's panel
    // button is where the files are.
    mountAnchors(['project-header', 'panel-button', 'status-line', 'project-settings', 'quick-tasks-row']);
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);

    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    expect(screen.getByTestId('tour-counter').textContent).toBe('1 of 5');
    const seen = [stepId()];
    for (let i = 0; i < 4; i++) {
      fireEvent.click(screen.getByTestId('tour-next'));
      seen.push(stepId());
    }
    expect(seen).toEqual(['project', 'files', 'tasks', 'settings', 'quickTasks']);
  });

  it('never anchors on the retired tab bar', async () => {
    mountAnchors(['project-header', 'tab-files', 'tab-queue']);
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    expect(screen.getByTestId('tour-counter').textContent).toBe('1 of 1');
  });

  it('the Tasks stop spotlights the status line', async () => {
    mountAnchors(['project-header', 'status-line']);
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    fireEvent.click(screen.getByTestId('tour-next'));
    expect(stepId()).toBe('tasks');
  });

  it('a collapsed panel and a collapsed project list get no stop: nothing to point at (F6, 2026-11-11)', async () => {
    // What a new user lands on: the panel folded to its handle and the
    // project list folded to the edge strip. Stops 4 and 7 used to sit on
    // those empty edges.
    mountAnchors(['project-header', 'panel-handle', 'status-line', 'project-settings', 'edge-strip']);
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    expect(screen.getByTestId('tour-counter').textContent).toBe('1 of 3');
    const seen = [stepId()];
    for (let i = 0; i < 2; i++) {
      fireEvent.click(screen.getByTestId('tour-next'));
      seen.push(stepId());
    }
    expect(seen).toEqual(['project', 'tasks', 'settings']);
  });

  it('with left-off cards showing, opens on them and goes to the agents stop next', async () => {
    mountAnchors(['left-off', ...ALL]);
    render(<FirstJourneyTour folderName="x" agentNames={['Codex']} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    expect(stepId()).toBe('leftOff');
    expect(screen.getByTestId('tour-card').textContent).toContain('Pick up where you left off');
    expect(screen.getByTestId('tour-counter').textContent).toBe('1 of 8');
    fireEvent.click(screen.getByTestId('tour-next'));
    expect(stepId()).toBe('agents');
    fireEvent.click(screen.getByTestId('tour-next'));
    expect(stepId()).toBe('project');
  });

  it('the left-off cards stay clickable on their stop, and opening one moves the tour on', async () => {
    mountAnchors(['left-off', ...ALL]);
    render(<FirstJourneyTour folderName="x" agentNames={['Codex']} onClose={vi.fn()} />);
    await waitFor(() => expect(stepId()).toBe('leftOff'));
    // No full-page blocker over the cards: four blockers frame the hole.
    expect(screen.getAllByTestId('tour-blocker')).toHaveLength(4);
    fireEvent.click(document.querySelector('[data-tour="left-off"]')!);
    await waitFor(() => expect(stepId()).toBe('agents'));
    expect(screen.queryAllByTestId('tour-blocker')).toHaveLength(0);
  });

  it('waits for the left-off answer before fixing its stops', async () => {
    // The header is up but the empty chat has not answered yet (still
    // loading, or not even mounted).
    mountAnchors(ALL, false);
    const marker = document.createElement('div');
    marker.setAttribute('data-left-off-state', 'loading');
    document.getElementById('anchors')!.appendChild(marker);
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);
    await new Promise((r) => setTimeout(r, 300));
    expect(screen.queryByTestId('tour-card')).toBeNull();

    // The answer lands with cards.
    marker.remove();
    mountAnchors(['left-off']);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy());
    expect(stepId()).toBe('leftOff');
    document.querySelectorAll('#anchors').forEach((n) => n.remove());
  });

  it('goes on without the left-off stop when no answer ever comes (a session with messages is open)', async () => {
    mountAnchors(ALL, false);
    render(<FirstJourneyTour folderName="x" agentNames={[]} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('tour-card')).toBeTruthy(), { timeout: 3000 });
    expect(stepId()).toBe('project');
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
    // The row is hidden and the strip is not the row: no Quick Tasks stop.
    expect(screen.getByTestId('tour-counter').textContent).toBe('1 of 1');
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
