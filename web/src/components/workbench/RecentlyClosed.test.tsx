// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * "Recently closed" (spec 089 §3.6): asks the agent or the memory editor
 * closed in the last 7 days, each with a Reopen action.
 */

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import RecentlyClosed from './RecentlyClosed';
import type { WorkbenchClosedAsk } from './types';

function closed(overrides: Partial<WorkbenchClosedAsk> = {}): WorkbenchClosedAsk {
  return {
    project_id: 'proj-a',
    id: 'c1',
    text: 'Pick the venue',
    due: null,
    created: '2026-07-01',
    closed: '2026-07-20',
    closed_by: 'agent',
    kind: 'done',
    note: '"rooftop, final"',
    ...overrides,
  };
}

describe('RecentlyClosed', () => {
  it('renders nothing when there is nothing to undo', () => {
    const { container } = render(
      <RecentlyClosed items={[]} showProjectChip={false} projectName={() => null} onReopen={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('lists each close with who closed it, the quoted words, and a Reopen action', () => {
    const onReopen = vi.fn();
    render(
      <RecentlyClosed
        items={[
          closed(),
          closed({ id: 'c2', text: 'Book the flights', kind: 'dropped', closed_by: 'editor', note: '' }),
        ]}
        showProjectChip={false}
        projectName={() => null}
        onReopen={onReopen}
      />,
    );
    expect(screen.getByText('Recently closed')).toBeInTheDocument();
    const rows = screen.getAllByTestId('workbench-closed-row');
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent('Pick the venue');
    expect(rows[0]).toHaveTextContent('Marked done by the agent');
    expect(rows[0]).toHaveTextContent('"rooftop, final"');
    expect(rows[0]).toHaveTextContent('2026-07-20');
    expect(rows[1]).toHaveTextContent('Dropped during memory tidy-up');

    fireEvent.click(screen.getAllByTestId('workbench-closed-reopen')[1]);
    expect(onReopen).toHaveBeenCalledWith(expect.objectContaining({ id: 'c2' }));
  });

  it('collapses and expands', () => {
    render(
      <RecentlyClosed items={[closed()]} showProjectChip={false} projectName={() => null} onReopen={vi.fn()} />,
    );
    const toggle = screen.getByTestId('workbench-closed-toggle');
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByTestId('workbench-closed-row')).toBeNull();
  });

  it('shows the project name in the global view', () => {
    render(
      <RecentlyClosed
        items={[closed()]}
        showProjectChip
        projectName={(id) => (id === 'proj-a' ? 'Marketing' : null)}
        onReopen={vi.fn()}
      />,
    );
    expect(screen.getByTestId('workbench-closed-row')).toHaveTextContent('Marketing');
  });
});
