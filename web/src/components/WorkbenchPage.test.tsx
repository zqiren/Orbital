// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * WorkbenchPage integration tests (Task 7). Covers the plan's Vitest list:
 * sort order rendering, an unconfirmed card auto-expanding its receipt, a
 * "Done" click issuing the right exit POST with optimistic removal, and
 * the empty state showing the migrate CTA (per-project and global). The api
 * client is mocked — no network.
 */

import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import WorkbenchPage from './WorkbenchPage';

afterEach(cleanup);
beforeEach(() => {
  apiMock.mockReset();
});

const PROJECTS = [
  { project_id: 'proj-a', name: 'Marketing', workspace: '/ws/a' },
  { project_id: 'proj-b', name: 'Ops', workspace: '/ws/b' },
];

function mockApi(opts: {
  entries?: unknown[];
  digests?: unknown[];
  projects?: unknown[];
  calendarAvailable?: boolean;
  calendarEvents?: unknown[];
} = {}) {
  const {
    entries = [],
    digests = [],
    projects = PROJECTS,
    calendarAvailable = false,
    calendarEvents = [],
  } = opts;
  apiMock.mockImplementation(async (path: string, init?: RequestInit) => {
    if (path.startsWith('/api/v2/workbench') && (!init || init.method === undefined)) {
      return { entries, digests };
    }
    if (path === '/api/v2/projects') return projects;
    if (path.startsWith('/api/v2/calendar/availability')) {
      return { available: calendarAvailable, sources: [] };
    }
    if (path.startsWith('/api/v2/calendar/events')) return { events: calendarEvents };
    if (path.includes('/exit')) return { status: 'ok' };
    if (path.includes('/open')) return { session_id: 'sess-new' };
    if (path.includes('/migrate')) return { session_id: 'sess-migrate' };
    return {};
  });
}

function entry(overrides: Record<string, unknown> = {}) {
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
    ...overrides,
  };
}

describe('WorkbenchPage — sort order', () => {
  it('renders entries in server order as-is, guarding against reintroducing a client-side (overdue, created) re-sort', async () => {
    // Deliberately NOT in "overdue first, then oldest first" order — the
    // overdue entry sits in the middle. A client-side (overdue, created)
    // re-sort would move it to the front and produce a different order
    // than what's asserted below; only rendering the server's order as-is
    // passes.
    mockApi({
      entries: [
        entry({ id: 'older', created: '2026-06-01', overdue: false }),
        entry({ id: 'overdue-entry', created: '2026-07-15', overdue: true, due: '2026-07-10' }),
        entry({ id: 'newer', created: '2026-07-20', overdue: false }),
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());

    const cards = screen
      .getAllByTestId(/^workbench-card-entry-/)
      .map((el) => el.getAttribute('data-testid'));
    expect(cards).toEqual([
      'workbench-card-entry-proj-a-older',
      'workbench-card-entry-proj-a-overdue-entry',
      'workbench-card-entry-proj-a-newer',
    ]);
  });
});

describe('WorkbenchPage — unconfirmed receipt', () => {
  it('auto-expands the receipt for an unconfirmed entry', async () => {
    mockApi({ entries: [entry({ confidence: 'unconfirmed' })] });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());
    expect(screen.getByTestId('workbench-card-receipt-toggle')).toHaveAttribute(
      'aria-expanded',
      'true',
    );
  });
});

describe('WorkbenchPage — Done exit', () => {
  it('POSTs the fulfilled exit and optimistically removes the card', async () => {
    mockApi({ entries: [entry()] });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-exit-fulfilled')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-exit-fulfilled'));

    await waitFor(() =>
      expect(screen.queryByTestId('workbench-card-entry-proj-a-e1')).toBeNull(),
    );
    const exitCall = apiMock.mock.calls.find(([p]) => (p as string).includes('/exit'));
    expect(exitCall?.[0]).toBe('/api/v2/workbench/proj-a/entries/e1/exit');
    expect(JSON.parse((exitCall?.[1] as { body: string }).body)).toEqual({
      kind: 'fulfilled',
      reason: '',
    });
  });
});

describe('WorkbenchPage — doorway navigation', () => {
  it('tapping the card body POSTs /open and navigates to the project chat', async () => {
    mockApi({ entries: [entry()] });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));

    await waitFor(() =>
      expect(apiMock.mock.calls.some(([p]) => (p as string).includes('/open'))).toBe(true),
    );
    await waitFor(() =>
      expect(setRoute).toHaveBeenCalledWith({
        name: 'project',
        projectId: 'proj-a',
        tab: 'chat',
        sessionId: 'sess-new',
      }),
    );
  });
});

describe('WorkbenchPage — empty state', () => {
  it('per-project lens: shows a single migrate CTA that navigates on success', async () => {
    mockApi({ entries: [] });
    const setRoute = vi.fn();
    render(<WorkbenchPage projectId="proj-a" setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-empty')).toBeInTheDocument());

    const cta = screen.getByTestId('workbench-migrate-cta');
    fireEvent.click(cta);

    await waitFor(() =>
      expect(setRoute).toHaveBeenCalledWith({
        name: 'project',
        projectId: 'proj-a',
        tab: 'chat',
        sessionId: 'sess-migrate',
      }),
    );
  });

  it('global view: dropdown lists non-scratch projects; CTA disabled until a selection', async () => {
    mockApi({
      entries: [],
      projects: [
        ...PROJECTS,
        { project_id: 'proj-scratch', name: 'Quick Tasks', workspace: '/ws/s', is_scratch: true },
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-empty')).toBeInTheDocument());

    const picker = screen.getByTestId('workbench-migrate-picker') as HTMLSelectElement;
    // Placeholder first, then real projects; the scratch project is excluded.
    expect(Array.from(picker.options).map((o) => o.value)).toEqual(['', 'proj-a', 'proj-b']);
    expect(screen.getByTestId('workbench-migrate-cta')).toBeDisabled();
  });

  it('global view: selecting a project enables the CTA; clicking migrates it and navigates', async () => {
    mockApi({ entries: [] });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-empty')).toBeInTheDocument());

    fireEvent.change(screen.getByTestId('workbench-migrate-picker'), {
      target: { value: 'proj-b' },
    });
    const cta = screen.getByTestId('workbench-migrate-cta');
    expect(cta).not.toBeDisabled();
    fireEvent.click(cta);

    await waitFor(() =>
      expect(setRoute).toHaveBeenCalledWith({
        name: 'project',
        projectId: 'proj-b',
        tab: 'chat',
        sessionId: 'sess-migrate',
      }),
    );
    expect(
      apiMock.mock.calls.some(([path]) => path === '/api/v2/workbench/proj-b/migrate'),
    ).toBe(true);
  });
});

describe('WorkbenchPage — in-flight digest strip (global only)', () => {
  it('renders one collapsed row per digest on the global view', async () => {
    mockApi({
      entries: [entry()],
      digests: [
        { project_id: 'proj-a', in_progress: 'Working on X', next_steps: 'Ship Y' },
        { project_id: 'proj-b', in_progress: null, next_steps: 'Draft the RFC' },
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());
    expect(screen.getByTestId('workbench-inflight-proj-a')).toBeInTheDocument();
    expect(screen.getByTestId('workbench-inflight-proj-b')).toBeInTheDocument();
  });

  it('does not render digest rows in the per-project lens', async () => {
    mockApi({
      entries: [entry()],
      digests: [{ project_id: 'proj-a', in_progress: 'Working on X', next_steps: null }],
    });
    render(<WorkbenchPage projectId="proj-a" setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());
    expect(screen.queryByTestId('workbench-inflight-proj-a')).toBeNull();
  });

  it('is collapsed by default, then expanding shows in_progress/next_steps as plain text; hides a null subsection', async () => {
    mockApi({
      entries: [entry()],
      digests: [{ project_id: 'proj-a', in_progress: 'Working on X', next_steps: null }],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() =>
      expect(screen.getByTestId('workbench-inflight-proj-a')).toBeInTheDocument(),
    );
    expect(screen.queryByText('Working on X')).toBeNull();

    fireEvent.click(screen.getByTestId('workbench-inflight-proj-a'));
    expect(screen.getByText('In progress')).toBeInTheDocument();
    expect(screen.getByText('Working on X')).toBeInTheDocument();
    expect(screen.queryByText('Next steps')).toBeNull();
  });

  it('hides the strip entirely when digests is empty', async () => {
    mockApi({ entries: [entry()], digests: [] });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());
    expect(screen.queryAllByTestId(/^workbench-inflight-/)).toHaveLength(0);
  });

  it('is visible alongside the empty state (re-entry context matters most then)', async () => {
    mockApi({
      entries: [],
      digests: [{ project_id: 'proj-a', in_progress: 'Working on X', next_steps: 'Ship Y' }],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-empty')).toBeInTheDocument());
    expect(screen.getByTestId('workbench-inflight-proj-a')).toBeInTheDocument();
  });
});

describe('WorkbenchPage — This week', () => {
  it('hides the This week strip when there are no events', async () => {
    mockApi({ entries: [entry()], calendarAvailable: true, calendarEvents: [] });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());
    expect(screen.queryByTestId('workbench-this-week')).toBeNull();
  });

  it('global This week strip filters events from workbench-excluded projects (privacy toggle)', async () => {
    // Spec §6.5: the global surface must not leak excluded projects' entry
    // sentences through the calendar strip (final-review F2).
    mockApi({
      entries: [entry()],
      projects: [
        { project_id: 'proj-a', name: 'Marketing', workspace: '/ws/a' },
        { project_id: 'proj-x', name: 'Secret', workspace: '/ws/x', workbench_exclude_global: true },
      ],
      calendarAvailable: true,
      calendarEvents: [
        { id: 'memory:proj-a/e9', title: 'Public deadline', start: '2026-07-25', all_day: true, project_id: 'proj-a' },
        { id: 'memory:proj-x/s1', title: 'Secret obligation sentence', start: '2026-07-26', all_day: true, project_id: 'proj-x' },
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-this-week')).toBeInTheDocument());
    expect(screen.getByText('Public deadline')).toBeInTheDocument();
    expect(screen.queryByText('Secret obligation sentence')).toBeNull();
  });

  it('This week strip shows ONE row per automation trigger, not one per daily occurrence', async () => {
    // Real-data regression (2026-07-24 screenshot): daily crons emitted 7
    // occurrences each into the strip — five copies of the same two jobs.
    const occ = (day: number, trig: string, title: string) => ({
      id: `automation:proj-a/${trig}/2026-07-${day}T11:00:00+08:00`,
      title,
      start: `2026-07-${day}T03:00:00Z`,
      all_day: false,
      project_id: 'proj-a',
    });
    mockApi({
      entries: [entry()],
      calendarAvailable: true,
      calendarEvents: [
        occ(25, 'trg_scan', 'Daily Adjacent-Repo Issue Scan'),
        occ(26, 'trg_scan', 'Daily Adjacent-Repo Issue Scan'),
        occ(27, 'trg_scan', 'Daily Adjacent-Repo Issue Scan'),
        occ(25, 'trg_issues', 'Daily Orbital issues check'),
        occ(26, 'trg_issues', 'Daily Orbital issues check'),
        { id: 'memory:proj-a/e9', title: 'Ship the deadline thing', start: '2026-07-26', all_day: true, project_id: 'proj-a' },
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-this-week')).toBeInTheDocument());
    expect(screen.getAllByText('Daily Adjacent-Repo Issue Scan')).toHaveLength(1);
    expect(screen.getAllByText('Daily Orbital issues check')).toHaveLength(1);
    expect(screen.getByText('Ship the deadline thing')).toBeInTheDocument();
  });

  it('per-project lens This week strip still shows that project own events even when excluded globally', async () => {
    mockApi({
      entries: [entry({ project_id: 'proj-x' })],
      projects: [
        { project_id: 'proj-x', name: 'Secret', workspace: '/ws/x', workbench_exclude_global: true },
      ],
      calendarAvailable: true,
      calendarEvents: [
        { id: 'memory:proj-x/s1', title: 'Secret obligation sentence', start: '2026-07-26', all_day: true, project_id: 'proj-x' },
      ],
    });
    render(<WorkbenchPage projectId="proj-x" setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-this-week')).toBeInTheDocument());
    expect(screen.getByText('Secret obligation sentence')).toBeInTheDocument();
  });
});
