// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Task 5 (spec 009 §0.5): FanoutCard renders one row per dispatched task with
// a live status pill. Status comes from a separate live-overlay prop (Map in
// ChatView, plain Record here) rather than being baked into the task list, so
// re-rendering with an updated `statuses` prop must update the pill in place.

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import FanoutCard from './FanoutCard';

const tasks = [
  { handle: 'worker:a1b2c3d4-0', label: 'Research A' },
  { handle: 'worker:a1b2c3d4-1', label: 'Research B' },
  { handle: 'worker:a1b2c3d4-2', label: 'Research C' },
];

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
        statuses={{ 'worker:a1b2c3d4-0': 'completed' }}
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
          'worker:a1b2c3d4-0': 'error',
          'worker:a1b2c3d4-1': 'stalled',
          'worker:a1b2c3d4-2': 'interrupted',
        }}
        startedAtMs={Date.now()}
        onSelectTask={() => {}}
      />,
    );
    expect(screen.getByTestId('fanout-status-worker:a1b2c3d4-0')).toHaveTextContent('Error');
    expect(screen.getByTestId('fanout-status-worker:a1b2c3d4-1')).toHaveTextContent('Stalled');
    expect(screen.getByTestId('fanout-status-worker:a1b2c3d4-2')).toHaveTextContent('Interrupted');
  });
});
