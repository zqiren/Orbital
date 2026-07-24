// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * WorkbenchCard tests (Task 7). Covers the plan's Vitest list at the card
 * level: unconfirmed entries show an auto-expanded receipt (stated entries
 * start collapsed), and the two exit buttons fire the right callback without
 * triggering the whole-card doorway tap.
 */

import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import WorkbenchCard from './WorkbenchCard';
import type { WorkbenchEntry } from './workbench/types';

afterEach(cleanup);

function entry(overrides: Partial<WorkbenchEntry> = {}): WorkbenchEntry {
  return {
    project_id: 'proj-a',
    id: 'e1',
    text: 'Send Simon the invoice draft',
    due: null,
    evidence: 'I said I would send it Friday',
    from_session: 'sess-123',
    confidence: 'stated',
    created: '2026-07-01',
    touched: null,
    age_days: 5,
    overdue: false,
    days_late: null,
    ...overrides,
  };
}

describe('WorkbenchCard — entry, age + project chip', () => {
  it('shows "waiting N days" for a non-overdue entry', () => {
    render(
      <WorkbenchCard
        entry={entry({ age_days: 5, overdue: false })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByText(/waiting 5 days/i)).toBeInTheDocument();
  });

  it('shows "N days late" using the server\'s days_late (not a local Date diff)', () => {
    render(
      <WorkbenchCard
        // due/now would compute a completely different local diff (10+
        // days) — the rendered count must come from days_late, not from
        // recomputing against `due`/`now` in the browser.
        entry={entry({ overdue: true, due: '2026-07-01', age_days: 10, days_late: 4 })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByText(/4 days late/i)).toBeInTheDocument();
  });

  it('shows no late label when days_late is absent, even though the browser clock would disagree', () => {
    render(
      <WorkbenchCard
        // overdue + a due date far enough in the past that a local Date
        // diff would happily compute a "late" count — but days_late is
        // null (server didn't provide one), so no late label may render.
        entry={entry({ overdue: true, due: '2026-01-01', age_days: 10, days_late: null })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.queryByText(/days late/i)).toBeNull();
    // Falls back to the waiting label (age_days) rather than showing nothing.
    expect(screen.getByText(/waiting 10 days/i)).toBeInTheDocument();
  });

  it('shows the project chip only when showProjectChip is true', () => {
    const { rerender } = render(
      <WorkbenchCard
        entry={entry()}
        showProjectChip
        projectName="Marketing"
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByTestId('workbench-card-project-chip')).toHaveTextContent('Marketing');

    rerender(
      <WorkbenchCard
        entry={entry()}
        showProjectChip={false}
        projectName="Marketing"
        onOpen={vi.fn()}
      />,
    );
    expect(screen.queryByTestId('workbench-card-project-chip')).toBeNull();
  });
});

describe('WorkbenchCard — receipt expansion', () => {
  it('is expanded by default when confidence is unconfirmed', () => {
    render(
      <WorkbenchCard
        entry={entry({ confidence: 'unconfirmed' })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByTestId('workbench-card-receipt-toggle')).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    expect(screen.getByText(/I said I would send it Friday/)).toBeInTheDocument();
  });

  it('starts collapsed for a stated entry, expands on toggle click', () => {
    render(
      <WorkbenchCard
        entry={entry({ confidence: 'stated' })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByTestId('workbench-card-receipt-toggle')).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    expect(screen.queryByText(/I said I would send it Friday/)).toBeNull();

    fireEvent.click(screen.getByTestId('workbench-card-receipt-toggle'));
    expect(screen.getByText(/I said I would send it Friday/)).toBeInTheDocument();
  });

  it('renders the from-session reference as inert text, not a link with its own handler', () => {
    // Final-review m14: a "source" reference must not present as a link with
    // a side-effecting handler of its own. Plain text participates in the
    // normal whole-card tap (bubbles once) like any other receipt text.
    const onOpen = vi.fn();
    render(
      <WorkbenchCard
        entry={entry({ confidence: 'unconfirmed' })}
        showProjectChip={false}
        onOpen={onOpen}
      />,
    );
    const ref = screen.getByTestId('workbench-card-from-session');
    expect(ref.tagName).not.toBe('BUTTON');
    fireEvent.click(ref);
    expect(onOpen).toHaveBeenCalledTimes(1); // bubbled card tap, no double-fire
  });
});

describe('WorkbenchCard — entry exits', () => {
  it('Resolved fires onExit("fulfilled") and does NOT trigger the doorway open', () => {
    const onExit = vi.fn();
    const onOpen = vi.fn();
    render(
      <WorkbenchCard
        entry={entry()}
        showProjectChip={false}
        onOpen={onOpen}
        onExit={onExit}
      />,
    );
    fireEvent.click(screen.getByTestId('workbench-card-exit-fulfilled'));
    expect(onExit).toHaveBeenCalledWith('fulfilled');
    expect(onOpen).not.toHaveBeenCalled();
    expect(screen.getByTestId('workbench-card-exit-fulfilled')).toHaveTextContent('Resolved');
  });

  it('Delete fires onExit("irrelevant") and does NOT trigger the doorway open', () => {
    const onExit = vi.fn();
    const onOpen = vi.fn();
    render(
      <WorkbenchCard
        entry={entry()}
        showProjectChip={false}
        onOpen={onOpen}
        onExit={onExit}
      />,
    );
    fireEvent.click(screen.getByTestId('workbench-card-exit-irrelevant'));
    expect(onExit).toHaveBeenCalledWith('irrelevant');
    expect(onOpen).not.toHaveBeenCalled();
    expect(screen.getByTestId('workbench-card-exit-irrelevant')).toHaveTextContent('Delete');
  });

  it('tapping the card body (not a button) triggers the doorway open', () => {
    const onOpen = vi.fn();
    render(
      <WorkbenchCard
        entry={entry()}
        showProjectChip={false}
        onOpen={onOpen}
      />,
    );
    fireEvent.click(screen.getByTestId(`workbench-card-entry-proj-a-e1`));
    expect(onOpen).toHaveBeenCalled();
  });
});
