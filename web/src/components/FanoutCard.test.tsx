// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Task 5 (spec 009 §0.5): FanoutCard renders one row per dispatched task with
// a live status pill. Status comes from a separate live-overlay prop (Map in
// ChatView, plain Record here) rather than being baked into the task list, so
// re-rendering with an updated `statuses` prop must update the pill in place.
//
// Round 2 (2026-07-05): `statuses` upgraded from a plain
// Record<handle, FanoutTaskStatus> to a rich Record<handle, {status,
// completedAtMs?}> so each row can freeze its OWN duration independently
// (issue 1 — previously every row shared one never-freezing batch countdown).
// All fixtures below were rewritten to the rich record as part of this task.

import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import FanoutCard, { isTerminal } from './FanoutCard';

const tasks = [
  { handle: 'worker:a1b2c3d4-0', label: 'Research A' },
  { handle: 'worker:a1b2c3d4-1', label: 'Research B' },
  { handle: 'worker:a1b2c3d4-2', label: 'Research C' },
];

describe('isTerminal', () => {
  it('is false only for running', () => {
    expect(isTerminal('running')).toBe(false);
    expect(isTerminal('completed')).toBe(true);
    expect(isTerminal('error')).toBe(true);
    expect(isTerminal('stalled')).toBe(true);
    expect(isTerminal('interrupted')).toBe(true);
  });
});

describe('FanoutCard', () => {
  it('renders one row per task from args', () => {
    render(
      <FanoutCard
        fanoutId="a1b2c3d4"
        tasks={tasks}
        statuses={{}}
        startedAtMs={Date.now()}
        onSelectTask={() => {}}
      />,
    );
    for (const task of tasks) {
      expect(screen.getByTestId(`fanout-row-${task.handle}`)).toHaveTextContent(task.label);
    }
    expect(screen.getAllByTestId(/^fanout-row-/)).toHaveLength(3);
  });

  it('defaults every task to the running pill when no status has arrived yet', () => {
    render(
      <FanoutCard
        fanoutId="a1b2c3d4"
        tasks={tasks}
        statuses={{}}
        startedAtMs={Date.now()}
        onSelectTask={() => {}}
      />,
    );
    for (const task of tasks) {
      const pill = screen.getByTestId(`fanout-status-${task.handle}`);
      expect(pill.dataset.status).toBe('running');
      expect(pill).toHaveTextContent('Running');
    }
  });

  it('updates the status pill in place when the statuses prop changes (live overlay)', () => {
    const { rerender } = render(
      <FanoutCard
        fanoutId="a1b2c3d4"
        tasks={tasks}
        statuses={{}}
        startedAtMs={Date.now()}
        onSelectTask={() => {}}
      />,
    );
    const pill = screen.getByTestId('fanout-status-worker:a1b2c3d4-0');
    expect(pill.dataset.status).toBe('running');

    rerender(
      <FanoutCard
        fanoutId="a1b2c3d4"
        tasks={tasks}
        statuses={{ 'worker:a1b2c3d4-0': { status: 'completed' } }}
        startedAtMs={Date.now()}
        onSelectTask={() => {}}
      />,
    );
    expect(screen.getByTestId('fanout-status-worker:a1b2c3d4-0').dataset.status).toBe('completed');
    expect(screen.getByTestId('fanout-status-worker:a1b2c3d4-0')).toHaveTextContent('Completed');
    // Untouched rows keep their prior (default) status.
    expect(screen.getByTestId('fanout-status-worker:a1b2c3d4-1').dataset.status).toBe('running');
  });

  it('clicking a row calls onSelectTask with the handle and label', async () => {
    const { default: userEvent } = await import('@testing-library/user-event');
    const clicks: Array<[string, string]> = [];
    render(
      <FanoutCard
        fanoutId="a1b2c3d4"
        tasks={tasks}
        statuses={{}}
        startedAtMs={Date.now()}
        onSelectTask={(handle, label) => clicks.push([handle, label])}
      />,
    );
    const user = userEvent.setup();
    await user.click(screen.getByTestId('fanout-row-worker:a1b2c3d4-1'));
    expect(clicks).toEqual([['worker:a1b2c3d4-1', 'Research B']]);
  });

  it('renders every documented status pill correctly', () => {
    render(
      <FanoutCard
        fanoutId="a1b2c3d4"
        tasks={tasks}
        statuses={{
          'worker:a1b2c3d4-0': { status: 'error' },
          'worker:a1b2c3d4-1': { status: 'stalled' },
          'worker:a1b2c3d4-2': { status: 'interrupted' },
        }}
        startedAtMs={Date.now()}
        onSelectTask={() => {}}
      />,
    );
    expect(screen.getByTestId('fanout-status-worker:a1b2c3d4-0')).toHaveTextContent('Error');
    expect(screen.getByTestId('fanout-status-worker:a1b2c3d4-1')).toHaveTextContent('Stalled');
    expect(screen.getByTestId('fanout-status-worker:a1b2c3d4-2')).toHaveTextContent('Interrupted');
  });

  // C1: the redesign drops the standalone-card box (rounded-lg/border/bg)
  // in favor of the same visual family as the agent_run capsule / tool rows.
  // data-testids stay stable across the restyle — that's the contract the
  // rest of ChatView (and this test file) depends on.
  it('keeps the fanout-card/fanout-row/fanout-status testids stable after the capsule restyle', () => {
    render(
      <FanoutCard
        fanoutId="a1b2c3d4"
        tasks={tasks}
        statuses={{}}
        startedAtMs={Date.now()}
        onSelectTask={() => {}}
      />,
    );
    expect(screen.getByTestId('fanout-card')).toBeInTheDocument();
    expect(screen.getByTestId('fanout-card').dataset.fanoutId).toBe('a1b2c3d4');
  });

  describe('per-row duration freeze (round 2, issue 1)', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });
    afterEach(() => {
      vi.useRealTimers();
    });

    it('freezes a completed row at its own completedAtMs while a running row keeps ticking', () => {
      const startedAtMs = Date.now();
      const { rerender } = render(
        <FanoutCard
          fanoutId="a1b2c3d4"
          tasks={[tasks[0], tasks[1]]}
          statuses={{ 'worker:a1b2c3d4-0': { status: 'running' }, 'worker:a1b2c3d4-1': { status: 'running' } }}
          startedAtMs={startedAtMs}
          onSelectTask={() => {}}
        />,
      );

      // Advance 5s, then freeze row 0 at exactly this moment.
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      const frozenAtMs = startedAtMs + 5000;
      rerender(
        <FanoutCard
          fanoutId="a1b2c3d4"
          tasks={[tasks[0], tasks[1]]}
          statuses={{
            'worker:a1b2c3d4-0': { status: 'completed', completedAtMs: frozenAtMs },
            'worker:a1b2c3d4-1': { status: 'running' },
          }}
          startedAtMs={startedAtMs}
          onSelectTask={() => {}}
        />,
      );

      const row0DurationBefore = screen.getByTestId('fanout-row-worker:a1b2c3d4-0').textContent;
      const row1DurationBefore = screen.getByTestId('fanout-row-worker:a1b2c3d4-1').textContent;

      // Advance another 10s — row 0 (completed) must stay frozen at 5s,
      // row 1 (still running) must keep ticking against the shared clock.
      act(() => {
        vi.advanceTimersByTime(10000);
      });

      expect(screen.getByTestId('fanout-row-worker:a1b2c3d4-0').textContent).toBe(row0DurationBefore);
      expect(screen.getByTestId('fanout-row-worker:a1b2c3d4-0')).toHaveTextContent('5s');
      expect(screen.getByTestId('fanout-row-worker:a1b2c3d4-1').textContent).not.toBe(row1DurationBefore);
      expect(screen.getByTestId('fanout-row-worker:a1b2c3d4-1')).toHaveTextContent('15s');
    });

    it('stops the shared tick entirely once every row is terminal, even without a batch completedAtMs', () => {
      const startedAtMs = Date.now();
      const { rerender } = render(
        <FanoutCard
          fanoutId="a1b2c3d4"
          tasks={[tasks[0], tasks[1]]}
          statuses={{ 'worker:a1b2c3d4-0': { status: 'running' }, 'worker:a1b2c3d4-1': { status: 'running' } }}
          startedAtMs={startedAtMs}
          onSelectTask={() => {}}
        />,
      );
      act(() => {
        vi.advanceTimersByTime(3000);
      });
      // Both rows go terminal without the batch-level completedAtMs ever
      // arriving (edge case: joins can lag the last per-task update).
      rerender(
        <FanoutCard
          fanoutId="a1b2c3d4"
          tasks={[tasks[0], tasks[1]]}
          statuses={{
            'worker:a1b2c3d4-0': { status: 'completed', completedAtMs: startedAtMs + 3000 },
            'worker:a1b2c3d4-1': { status: 'error', completedAtMs: startedAtMs + 3000 },
          }}
          startedAtMs={startedAtMs}
          onSelectTask={() => {}}
        />,
      );
      const before = screen.getByTestId('fanout-card').textContent;
      act(() => {
        vi.advanceTimersByTime(20000);
      });
      expect(screen.getByTestId('fanout-card').textContent).toBe(before);
    });
  });
});
