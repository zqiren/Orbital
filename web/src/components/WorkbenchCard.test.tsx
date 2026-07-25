// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * WorkbenchCard tests (Task 7; receipt removed 2026-07-24 mid-task
 * amendment). Covers the card-level Vitest list: age/late labels, the
 * project chip, the always-visible inline section label, markdown
 * rendering, and the two exit buttons firing the right callback without
 * triggering the whole-card doorway tap.
 */

import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import WorkbenchCard from './WorkbenchCard';
import { eventColor } from './calendar/color';
import type { WorkbenchEntry } from './workbench/types';

afterEach(cleanup);

function entry(overrides: Partial<WorkbenchEntry> = {}): WorkbenchEntry {
  return {
    project_id: 'proj-a',
    id: 'e1',
    text: 'Send Simon the invoice draft',
    due: null,
    created: '2026-07-01',
    touched: null,
    age_days: 5,
    overdue: false,
    days_late: null,
    section: null,
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

// jsdom's CSSOM normalizes any color it reads back through
// `element.style.<prop>` into `rgb()`/`rgba()`, regardless of the hsl()
// string that was assigned — round-trip the expected value through the same
// normalization (via a scratch element) so the comparison isn't sensitive to
// that serialization quirk.
function normalizedColor(value: string): string {
  const probe = document.createElement('div');
  probe.style.color = value;
  return probe.style.color;
}

// Project identity is still the calendar's eventColor hash, but it is carried
// by a small DOT beside the project name rather than by a tinted+bordered pill
// plus a 3px coloured spine down the card. The invariant these tests guard is
// unchanged — identity is visually encoded, only on the global view, and
// differs per project; only the element carrying it moved.
describe('WorkbenchCard — project prominence: reuses the calendar eventColor hash', () => {
  it('marks the project with a dot coloured by eventColor(project_id) on the global view', () => {
    render(
      <WorkbenchCard
        entry={entry({ project_id: 'proj-a' })}
        showProjectChip
        projectName="Marketing"
        onOpen={vi.fn()}
      />,
    );
    const dot = screen.getByTestId('workbench-card-project-dot');
    const accent = eventColor('proj-a');
    expect(normalizedColor(dot.style.backgroundColor)).toBe(normalizedColor(accent.border));
  });

  it('does NOT apply any accent to the card surface itself', () => {
    render(
      <WorkbenchCard
        entry={entry({ project_id: 'proj-a' })}
        showProjectChip
        projectName="Marketing"
        onOpen={vi.fn()}
      />,
    );
    const card = screen.getByTestId('workbench-card-entry-proj-a-e1');
    expect(card.style.borderLeftColor).toBe('');
  });

  it('renders no project marker at all on the lens view (showProjectChip false)', () => {
    render(
      <WorkbenchCard
        entry={entry({ project_id: 'proj-a' })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.queryByTestId('workbench-card-project-dot')).toBeNull();
    const card = screen.getByTestId('workbench-card-entry-proj-a-e1');
    expect(card.style.borderLeftColor).toBe('');
  });

  it('different projects get different accent hues (hash-based, matches the calendar)', () => {
    const { rerender } = render(
      <WorkbenchCard
        entry={entry({ project_id: 'proj-a' })}
        showProjectChip
        projectName="Marketing"
        onOpen={vi.fn()}
      />,
    );
    const dotA = screen.getByTestId('workbench-card-project-dot').style.backgroundColor;

    rerender(
      <WorkbenchCard
        entry={entry({ project_id: 'proj-b' })}
        showProjectChip
        projectName="Ops"
        onOpen={vi.fn()}
      />,
    );
    const dotB = screen.getByTestId('workbench-card-project-dot').style.backgroundColor;

    expect(dotA).not.toBe(dotB);
  });
});

describe('WorkbenchCard — inline section label (spec 2026-07-24 amendment)', () => {
  it('renders the section as always-visible plain text, with no toggle/expander of any kind', () => {
    render(
      <WorkbenchCard
        entry={entry({ section: 'Blockers' })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    // Visible immediately — no click needed.
    expect(screen.getByTestId('workbench-card-section')).toHaveTextContent('Blockers');
    // No expand/collapse control anywhere on the card (the section label is
    // a plain <span>, not a <button>).
    expect(screen.getByTestId('workbench-card-section').tagName).not.toBe('BUTTON');
    expect(document.querySelectorAll('[aria-expanded]')).toHaveLength(0);
  });

  it('hides the section label entirely when section is null', () => {
    render(
      <WorkbenchCard
        entry={entry({ section: null })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.queryByTestId('workbench-card-section')).toBeNull();
  });
});

describe('WorkbenchCard — markdown rendering (spec 2026-07-24)', () => {
  it('renders a backticked path as an inline <code> element', () => {
    render(
      <WorkbenchCard
        entry={entry({ text: 'Update `agent_os/api/app.py` before merging' })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    const code = screen.getByText('agent_os/api/app.py');
    expect(code.tagName).toBe('CODE');
  });

  it('renders bold markdown as a <strong> element', () => {
    render(
      <WorkbenchCard
        entry={entry({ text: 'This is **urgent** work' })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    const strong = screen.getByText('urgent');
    expect(strong.tagName).toBe('STRONG');
  });

  it('a link click stops propagation — does NOT trigger the card doorway (onOpen)', () => {
    const onOpen = vi.fn();
    render(
      <WorkbenchCard
        entry={entry({ text: 'See [the doc](https://example.com/doc) for context' })}
        showProjectChip={false}
        onOpen={onOpen}
      />,
    );
    const link = screen.getByRole('link', { name: 'the doc' });
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
    fireEvent.click(link);
    expect(onOpen).not.toHaveBeenCalled();
  });

  it('does not render raw HTML embedded in the entry text (react-markdown escapes it)', () => {
    render(
      <WorkbenchCard
        entry={entry({ text: 'Watch out <script>window.__pwned = true</script> here' })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    expect(document.querySelector('script')).toBeNull();
    expect((window as unknown as { __pwned?: boolean }).__pwned).toBeUndefined();
  });

  it('does not introduce block-level margin around the sentence (inline rendering inside the clamp <p>)', () => {
    render(
      <WorkbenchCard
        entry={entry({ text: 'Plain sentence with `a/path.ts`' })}
        showProjectChip={false}
        onOpen={vi.fn()}
      />,
    );
    // The markdown-rendered paragraph must NOT produce a nested <p> (which
    // would carry block margins per index.css's .markdown-content p rule) —
    // only the card's own clamp <p> wrapper.
    const testId = screen.getByTestId('workbench-card-entry-proj-a-e1');
    expect(testId.querySelectorAll('p')).toHaveLength(1);
  });
});

describe('WorkbenchCard — entry exits', () => {
  it('Done fires onExit("fulfilled") and does NOT trigger the doorway open', () => {
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
    expect(screen.getByTestId('workbench-card-exit-fulfilled')).toHaveTextContent('Done');
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
