// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * WorkbenchCard tests (Task 7). Covers the plan's Vitest list at the card
 * level: unconfirmed entries show an auto-expanded receipt (stated entries
 * start collapsed), the two exit buttons fire the right callback without
 * triggering the whole-card doorway tap, and computed cards render Dismiss
 * instead of the two exits.
 */

import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import WorkbenchCard from './WorkbenchCard';
import type { WorkbenchComputedCard, WorkbenchEntry, WorkbenchListItem } from './workbench/types';

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

function computed(overrides: Partial<WorkbenchComputedCard> = {}): WorkbenchComputedCard {
  return {
    type: 'broken_automation',
    project_id: 'proj-a',
    key: 'trigger-1',
    text: 'Automation "nightly sync" has not run on schedule.',
    since: '2026-07-10',
    ...overrides,
  };
}

const NOW = new Date('2026-07-24T12:00:00Z');

describe('WorkbenchCard — entry, age + project chip', () => {
  it('shows "waiting N days" for a non-overdue entry', () => {
    render(
      <WorkbenchCard
        item={{ kind: 'entry', data: entry({ age_days: 5, overdue: false }) }}
        showProjectChip={false}
        now={NOW}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByText(/waiting 5 days/i)).toBeInTheDocument();
  });

  it('shows "N days late" using the server\'s days_late (not a local Date diff)', () => {
    render(
      <WorkbenchCard
        item={{
          // due/now would compute a completely different local diff (10+
          // days) — the rendered count must come from days_late, not from
          // recomputing against `due`/`now` in the browser.
          kind: 'entry',
          data: entry({ overdue: true, due: '2026-07-01', age_days: 10, days_late: 4 }),
        }}
        showProjectChip={false}
        now={NOW}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByText(/4 days late/i)).toBeInTheDocument();
  });

  it('shows no late label when days_late is absent, even though the browser clock would disagree', () => {
    render(
      <WorkbenchCard
        item={{
          // overdue + a due date far enough in the past that a local Date
          // diff would happily compute a "late" count — but days_late is
          // null (server didn't provide one), so no late label may render.
          kind: 'entry',
          data: entry({ overdue: true, due: '2026-01-01', age_days: 10, days_late: null }),
        }}
        showProjectChip={false}
        now={NOW}
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
        item={{ kind: 'entry', data: entry() }}
        showProjectChip
        projectName="Marketing"
        now={NOW}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByTestId('workbench-card-project-chip')).toHaveTextContent('Marketing');

    rerender(
      <WorkbenchCard
        item={{ kind: 'entry', data: entry() }}
        showProjectChip={false}
        projectName="Marketing"
        now={NOW}
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
        item={{ kind: 'entry', data: entry({ confidence: 'unconfirmed' }) }}
        showProjectChip={false}
        now={NOW}
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
        item={{ kind: 'entry', data: entry({ confidence: 'stated' }) }}
        showProjectChip={false}
        now={NOW}
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
        item={{ kind: 'entry', data: entry({ confidence: 'unconfirmed' }) }}
        showProjectChip={false}
        now={NOW}
        onOpen={onOpen}
      />,
    );
    const ref = screen.getByTestId('workbench-card-from-session');
    expect(ref.tagName).not.toBe('BUTTON');
    fireEvent.click(ref);
    expect(onOpen).toHaveBeenCalledTimes(1); // bubbled card tap, no double-fire
  });
});

describe('WorkbenchCard — computed primary verbs (spec §6 alignment)', () => {
  const cases = [
    { type: 'overdue', label: 'Do it now' },
    { type: 'broken_automation', label: 'Repair' },
    { type: 'paused_thread', label: 'Resume' },
  ] as const;
  for (const { type, label } of cases) {
    it(`${type} card leads with "${label}" and it fires the doorway`, () => {
      const onOpen = vi.fn();
      render(
        <WorkbenchCard
          item={{ kind: 'computed', data: { type, project_id: 'p', key: 'k', text: 'x', since: '2026-07-23' } }}
          showProjectChip={false}
          now={NOW}
          onOpen={onOpen}
          onDismiss={vi.fn()}
          onDisable={type === 'broken_automation' ? vi.fn() : undefined}
        />,
      );
      fireEvent.click(screen.getByTestId('workbench-card-computed-primary'));
      expect(onOpen).toHaveBeenCalledTimes(1);
      expect(screen.getByTestId('workbench-card-computed-primary')).toHaveTextContent(label);
      expect(screen.getByTestId('workbench-card-dismiss')).toBeInTheDocument();
    });
  }

  it('Disable appears only on broken_automation and fires without the doorway', () => {
    const onOpen = vi.fn();
    const onDisable = vi.fn();
    render(
      <WorkbenchCard
        item={{ kind: 'computed', data: { type: 'broken_automation', project_id: 'p', key: 'trg', text: 'x', since: '2026-07-23' } }}
        showProjectChip={false}
        now={NOW}
        onOpen={onOpen}
        onDismiss={vi.fn()}
        onDisable={onDisable}
      />,
    );
    fireEvent.click(screen.getByTestId('workbench-card-disable'));
    expect(onDisable).toHaveBeenCalledTimes(1);
    expect(onOpen).not.toHaveBeenCalled();
  });
});

describe('WorkbenchCard — entry exits', () => {
  it('Done fires onExit("fulfilled") and does NOT trigger the doorway open', () => {
    const onExit = vi.fn();
    const onOpen = vi.fn();
    render(
      <WorkbenchCard
        item={{ kind: 'entry', data: entry() }}
        showProjectChip={false}
        now={NOW}
        onOpen={onOpen}
        onExit={onExit}
      />,
    );
    fireEvent.click(screen.getByTestId('workbench-card-exit-fulfilled'));
    expect(onExit).toHaveBeenCalledWith('fulfilled');
    expect(onOpen).not.toHaveBeenCalled();
  });

  it('Not relevant fires onExit("irrelevant") and does NOT trigger the doorway open', () => {
    const onExit = vi.fn();
    const onOpen = vi.fn();
    render(
      <WorkbenchCard
        item={{ kind: 'entry', data: entry() }}
        showProjectChip={false}
        now={NOW}
        onOpen={onOpen}
        onExit={onExit}
      />,
    );
    fireEvent.click(screen.getByTestId('workbench-card-exit-irrelevant'));
    expect(onExit).toHaveBeenCalledWith('irrelevant');
    expect(onOpen).not.toHaveBeenCalled();
  });

  it('tapping the card body (not a button) triggers the doorway open', () => {
    const onOpen = vi.fn();
    render(
      <WorkbenchCard
        item={{ kind: 'entry', data: entry() }}
        showProjectChip={false}
        now={NOW}
        onOpen={onOpen}
      />,
    );
    fireEvent.click(screen.getByTestId(`workbench-card-entry-proj-a-e1`));
    expect(onOpen).toHaveBeenCalled();
  });
});

describe('WorkbenchCard — computed cards', () => {
  it('renders text + Dismiss (no exit buttons)', () => {
    const onDismiss = vi.fn();
    const onOpen = vi.fn();
    const item: WorkbenchListItem = { kind: 'computed', data: computed() };
    render(
      <WorkbenchCard
        item={item}
        showProjectChip={false}
        now={NOW}
        onOpen={onOpen}
        onDismiss={onDismiss}
      />,
    );
    expect(screen.getByText(/nightly sync/)).toBeInTheDocument();
    expect(screen.queryByTestId('workbench-card-exit-fulfilled')).toBeNull();

    fireEvent.click(screen.getByTestId('workbench-card-dismiss'));
    expect(onDismiss).toHaveBeenCalled();
  });

  it('an overdue-type computed card shows "N days late" from the server days_late', () => {
    const item: WorkbenchListItem = {
      kind: 'computed',
      data: computed({ type: 'overdue', since: '2026-07-01', days_late: 6 }),
    };
    render(
      <WorkbenchCard item={item} showProjectChip={false} now={NOW} onOpen={vi.fn()} />,
    );
    expect(screen.getByText(/6 days late/i)).toBeInTheDocument();
  });
});
