// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * WorkbenchPage integration tests (Task 7 + F3 2026-07-24 revision, incl. the
 * mid-task amendment that removed the receipt). Covers sort order rendering,
 * a "Done" click issuing the right exit POST with optimistic removal, the
 * empty state showing the migrate CTA (per-project and global), the prefill
 * doorway (round 3 final revision: mints via the SAME POST /new-session "+
 * new session" already uses, then navigates with that sessionId + composer
 * draft; NO Evidence line), and the Today strip (formerly "This week"). The
 * api client is mocked — no network. The In-flight digest strip is gone
 * entirely.
 */

import { render, screen, waitFor, fireEvent, cleanup, within } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const { apiMock, MockApiError } = vi.hoisted(() => {
  class MockApiError extends Error {
    status: number;
    detail: string;
    constructor(status: number, detail: string) {
      super(detail);
      this.status = status;
      this.detail = detail;
      this.name = 'ApiError';
    }
  }
  return { apiMock: vi.fn(), MockApiError };
});
vi.mock('../config', () => ({ api: apiMock, ApiError: MockApiError }));

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
  projects?: unknown[];
  calendarAvailable?: boolean;
  calendarEvents?: unknown[];
  /** POST /new-session response's session_id (the card-tap doorway mint,
   *  round 3 final revision). null simulates a malformed/empty response. */
  newSessionId?: string | null;
  /** Simulates the mint POST rejecting (network/server failure). */
  newSessionFails?: boolean;
  /** When set, the exit POST rejects with an ApiError of this status (bug #45
   *  non-409 surface test). */
  exitRejectStatus?: number;
} = {}) {
  const {
    entries = [],
    projects = PROJECTS,
    calendarAvailable = false,
    calendarEvents = [],
    newSessionId = 'sess-tap-minted',
    newSessionFails = false,
    exitRejectStatus,
  } = opts;
  apiMock.mockImplementation(async (path: string, init?: RequestInit) => {
    if (path.includes('/new-session')) {
      if (newSessionFails) throw new Error('mint failed');
      return { status: 'ok', session_id: newSessionId };
    }
    if (path.startsWith('/api/v2/workbench') && (!init || init.method === undefined)) {
      return { entries };
    }
    if (path === '/api/v2/projects') return projects;
    if (path.startsWith('/api/v2/calendar/availability')) {
      return { available: calendarAvailable, sources: [] };
    }
    if (path.startsWith('/api/v2/calendar/events')) return { events: calendarEvents };
    if (path.includes('/exit')) {
      if (exitRejectStatus != null) throw new MockApiError(exitRejectStatus, 'boom');
      return { status: 'ok' };
    }
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
    created: '2026-07-01',
    touched: null,
    age_days: 5,
    overdue: false,
    section: null,
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

describe('WorkbenchPage — Delete duplicate-card + error surfacing (bug #45)', () => {
  it('renders id-less entries with unique testids and Delete removes exactly one card (no phantom)', async () => {
    // Two entries with id=null used to collapse onto the same React key /
    // data-testid (`workbench-card-entry-<project>-null`), which is what made a
    // delete mis-reconcile into a duplicate phantom card.
    mockApi({
      entries: [
        entry({ id: null, text: 'First id-less obligation' }),
        entry({ id: null, text: 'Second id-less obligation' }),
        entry({ id: 'keep1', text: 'Has a real id' }),
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());

    const cards = screen.getAllByTestId(/^workbench-card-entry-/);
    expect(cards).toHaveLength(3);
    const testids = cards.map((c) => c.getAttribute('data-testid'));
    // No two rendered cards share a data-testid — the collision is gone.
    expect(new Set(testids).size).toBe(testids.length);

    // Deleting the real-id card removes exactly one card, leaving the two
    // id-less ones — not a phantom-inflated list.
    const realCard = screen.getByTestId('workbench-card-entry-proj-a-keep1');
    fireEvent.click(within(realCard).getByTestId('workbench-card-exit-irrelevant'));
    await waitFor(() =>
      expect(screen.getAllByTestId(/^workbench-card-entry-/)).toHaveLength(2),
    );
  });

  it('surfaces a delete-error notice when the exit POST fails (non-409), instead of reverting silently', async () => {
    mockApi({ entries: [entry({ id: 'e1' })], exitRejectStatus: 500 });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() =>
      expect(screen.getByTestId('workbench-card-exit-irrelevant')).toBeInTheDocument(),
    );

    fireEvent.click(screen.getByTestId('workbench-card-exit-irrelevant'));

    // The failure is now visible AND the optimistic removal is reverted.
    await waitFor(() =>
      expect(screen.getByTestId('workbench-delete-error')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument();
  });

  it('never POSTs /entries/null/exit for an id-less entry — it surfaces the error instead', async () => {
    mockApi({ entries: [entry({ id: null })] });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() =>
      expect(screen.getByTestId('workbench-card-exit-irrelevant')).toBeInTheDocument(),
    );

    fireEvent.click(screen.getByTestId('workbench-card-exit-irrelevant'));

    await waitFor(() =>
      expect(screen.getByTestId('workbench-delete-error')).toBeInTheDocument(),
    );
    const nullExit = apiMock.mock.calls.filter(([p]) =>
      (p as string).includes('/entries/null/exit'),
    );
    expect(nullExit).toHaveLength(0);
  });

  it('keyboard-activating a card exit button does NOT also trigger the card doorway (open)', async () => {
    mockApi({ entries: [entry({ id: 'e1' })] });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() =>
      expect(screen.getByTestId('workbench-card-exit-irrelevant')).toBeInTheDocument(),
    );

    fireEvent.keyDown(screen.getByTestId('workbench-card-exit-irrelevant'), { key: 'Enter' });
    // Flush any microtask the doorway's async mint would have queued.
    await Promise.resolve();
    await Promise.resolve();

    // The doorway must not have fired: no session mint, no navigation.
    const mintCalls = apiMock.mock.calls.filter(([p]) => (p as string).includes('/new-session'));
    expect(mintCalls).toHaveLength(0);
    expect(setRoute).not.toHaveBeenCalled();
  });
});

describe('WorkbenchPage — prefill doorway (spec 2026-07-24, round 3 final revision: mint-at-tap)', () => {
  it('tapping the card body mints a new session via the SAME POST /new-session "+ new session" uses, then navigates with that sessionId + the draft', async () => {
    mockApi({ entries: [entry()], newSessionId: 'sess-tap-minted' });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));

    await waitFor(() => expect(setRoute).toHaveBeenCalledTimes(1));
    const routeArg = setRoute.mock.calls[0][0];
    expect(routeArg).toMatchObject({
      name: 'project',
      projectId: 'proj-a',
      tab: 'chat',
      sessionId: 'sess-tap-minted',
    });
    // composeNewSession (an earlier, reverted design) must not reappear.
    expect(routeArg.composeNewSession).toBeUndefined();

    // The mint POST fired exactly once, with NO session_id in the body
    // (fresh-create) — byte-identical contract to "+ new session"
    // (useAgent().newSession(projectId) with no second argument).
    const mintCalls = apiMock.mock.calls.filter(
      ([p, init]) => (p as string).includes('/new-session') && (init as RequestInit | undefined)?.method === 'POST',
    );
    expect(mintCalls).toHaveLength(1);
    expect(mintCalls[0][0]).toBe('/api/v2/agents/proj-a/new-session');
    expect(JSON.parse((mintCalls[0][1] as { body: string }).body)).toEqual({});
  });

  it('on mint failure, surfaces an error via the existing openError banner and does NOT navigate', async () => {
    mockApi({ entries: [entry()], newSessionFails: true });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));

    await waitFor(() =>
      expect(screen.getByText("Couldn't open the project — please try again.")).toBeInTheDocument(),
    );
    expect(setRoute).not.toHaveBeenCalled();
  });

  it('the prefilled draft matches the exact spec format: section + quoted text, due suffix, two trailing newlines', async () => {
    mockApi({
      entries: [
        entry({
          text: 'Send Simon the invoice draft',
          section: 'Blockers',
          due: '2026-08-01',
        }),
      ],
    });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));
    await waitFor(() => expect(setRoute).toHaveBeenCalledTimes(1));

    const routeArg = setRoute.mock.calls[0][0];
    expect(routeArg.draft).toBe(
      'Workbench · Blockers: "Send Simon the invoice draft" (due 2026-08-01)\n\n',
    );
  });

  it('omits the "{section}: " part when section is null', async () => {
    mockApi({ entries: [entry({ section: null, due: null })] });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));
    await waitFor(() => expect(setRoute).toHaveBeenCalledTimes(1));

    const routeArg = setRoute.mock.calls[0][0];
    expect(routeArg.draft).toBe('Workbench · "Send Simon the invoice draft"\n\n');
  });

  it('omits the due suffix when absent', async () => {
    mockApi({ entries: [entry({ section: 'Blockers', due: null })] });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));
    await waitFor(() => expect(setRoute).toHaveBeenCalledTimes(1));

    const routeArg = setRoute.mock.calls[0][0];
    expect(routeArg.draft).toBe('Workbench · Blockers: "Send Simon the invoice draft"\n\n');
  });

  it('the prefill never contains an Evidence line (receipt/evidence removed 2026-07-24)', async () => {
    mockApi({ entries: [entry({ section: 'Blockers', due: '2026-08-01' })] });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));
    await waitFor(() => expect(setRoute).toHaveBeenCalledTimes(1));

    const routeArg = setRoute.mock.calls[0][0];
    expect(routeArg.draft as string).not.toMatch(/Evidence/);
    // Exactly two trailing newlines, nothing appended after the due suffix.
    expect(routeArg.draft as string).toBe(
      'Workbench · Blockers: "Send Simon the invoice draft" (due 2026-08-01)\n\n',
    );
  });
});

describe('WorkbenchPage — project filter (round 3, item 2)', () => {
  it('hides the filter when there are no entries at all', async () => {
    mockApi({ entries: [] });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-empty')).toBeInTheDocument());
    expect(screen.queryByTestId('workbench-project-filter')).toBeNull();
  });

  it('lists "All projects" plus the projects that currently have entries, in the order they first appear', async () => {
    mockApi({
      entries: [
        entry({ project_id: 'proj-b', id: 'b1' }),
        entry({ project_id: 'proj-a', id: 'a1' }),
        entry({ project_id: 'proj-b', id: 'b2' }),
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-project-filter')).toBeInTheDocument());

    const select = screen.getByTestId('workbench-project-filter') as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual([
      'All projects',
      'Ops',
      'Marketing',
    ]);
    expect(select.value).toBe('');
  });

  it('is absent on the per-project lens view even when entries exist', async () => {
    mockApi({ entries: [entry()] });
    render(<WorkbenchPage projectId="proj-a" setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());
    expect(screen.queryByTestId('workbench-project-filter')).toBeNull();
  });

  it('selecting a project filters the card list to that project only', async () => {
    mockApi({
      entries: [
        entry({ project_id: 'proj-a', id: 'a1' }),
        entry({ project_id: 'proj-b', id: 'b1' }),
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-project-filter')).toBeInTheDocument());

    fireEvent.change(screen.getByTestId('workbench-project-filter'), {
      target: { value: 'proj-b' },
    });

    expect(screen.queryByTestId('workbench-card-entry-proj-a-a1')).toBeNull();
    expect(screen.getByTestId('workbench-card-entry-proj-b-b1')).toBeInTheDocument();
  });

  it('falls back to "All projects" once the selected project has no entries left', async () => {
    mockApi({
      entries: [
        entry({ project_id: 'proj-a', id: 'a1' }),
        entry({ project_id: 'proj-b', id: 'b1' }),
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-project-filter')).toBeInTheDocument());

    fireEvent.change(screen.getByTestId('workbench-project-filter'), {
      target: { value: 'proj-b' },
    });
    await waitFor(() =>
      expect(screen.queryByTestId('workbench-card-entry-proj-a-a1')).toBeNull(),
    );

    fireEvent.click(screen.getByTestId('workbench-card-exit-fulfilled'));

    await waitFor(() => {
      const select = screen.getByTestId('workbench-project-filter') as HTMLSelectElement;
      expect(select.value).toBe('');
    });
    expect(screen.getByTestId('workbench-card-entry-proj-a-a1')).toBeInTheDocument();
  });
});

describe('WorkbenchPage — Today strip past-dimming (round 3, item 4)', () => {
  it('dims a past, non-all-day event and leaves a future one undimmed', async () => {
    const now = Date.now();
    const past = new Date(now - 3600_000).toISOString();
    const future = new Date(now + 3600_000).toISOString();
    mockApi({
      entries: [entry()],
      calendarAvailable: true,
      calendarEvents: [
        { id: 'past-1', title: 'Morning standup', start: past, all_day: false, project_id: 'proj-a' },
        { id: 'future-1', title: 'Afternoon sync', start: future, all_day: false, project_id: 'proj-a' },
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-today')).toBeInTheDocument());

    const rows = screen.getAllByTestId('workbench-today-row');
    const pastRow = rows.find((r) => r.textContent?.includes('Morning standup'));
    const futureRow = rows.find((r) => r.textContent?.includes('Afternoon sync'));
    expect(pastRow).toHaveAttribute('data-past', 'true');
    expect(futureRow).toHaveAttribute('data-past', 'false');
  });

  it('does not dim an all-day event even when its instant is in the past', async () => {
    const past = new Date(Date.now() - 3600_000).toISOString();
    mockApi({
      entries: [entry()],
      calendarAvailable: true,
      calendarEvents: [
        { id: 'allday-1', title: 'Public deadline', start: past, all_day: true, project_id: 'proj-a' },
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-today')).toBeInTheDocument());
    expect(screen.getByTestId('workbench-today-row')).toHaveAttribute('data-past', 'false');
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

describe('WorkbenchPage — per-project "what\'s in progress" strip removed (spec 2026-07-24)', () => {
  it('never renders the removed collapsed-summary strip or its copy on the global surface', async () => {
    mockApi({ entries: [entry()] });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());
    expect(screen.queryByText('In flight')).toBeNull();
    expect(screen.queryByText('In progress')).toBeNull();
    expect(screen.queryByText('Next steps')).toBeNull();
    // Only the known regions render: header, list. No collapsed/expandable
    // rows anywhere — the card's own receipt toggle was also removed
    // (2026-07-24 amendment), so there are zero aria-expanded elements.
    expect(document.querySelectorAll('[aria-expanded]')).toHaveLength(0);
  });
});

describe('WorkbenchPage — Today strip', () => {
  it('hides the Today strip when there are no events', async () => {
    mockApi({ entries: [entry()], calendarAvailable: true, calendarEvents: [] });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-list')).toBeInTheDocument());
    expect(screen.queryByTestId('workbench-today')).toBeNull();
  });

  it('renders the "Today" title and "Open Calendar →" affordance', async () => {
    mockApi({
      entries: [entry()],
      calendarAvailable: true,
      calendarEvents: [
        { id: 'memory:proj-a/e9', title: 'Public deadline', start: '2026-07-24', all_day: true, project_id: 'proj-a' },
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-today')).toBeInTheDocument());
    expect(screen.getByText('Today')).toBeInTheDocument();
    expect(screen.getByTestId('workbench-open-calendar')).toHaveTextContent('Open Calendar →');
  });

  it('global Today strip filters events from workbench-excluded projects (privacy toggle)', async () => {
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
        { id: 'memory:proj-a/e9', title: 'Public deadline', start: '2026-07-24', all_day: true, project_id: 'proj-a' },
        { id: 'memory:proj-x/s1', title: 'Secret obligation sentence', start: '2026-07-24', all_day: true, project_id: 'proj-x' },
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-today')).toBeInTheDocument());
    expect(screen.getByText('Public deadline')).toBeInTheDocument();
    expect(screen.queryByText('Secret obligation sentence')).toBeNull();
  });

  it('Today strip shows ONE row per automation trigger, not one per daily occurrence, capped at 5', async () => {
    const occ = (day: number, trig: string, title: string) => ({
      id: `automation:proj-a/${trig}/2026-07-${day}T11:00:00+08:00`,
      title,
      start: `2026-07-24T03:00:00Z`,
      all_day: false,
      project_id: 'proj-a',
    });
    mockApi({
      entries: [entry()],
      calendarAvailable: true,
      calendarEvents: [
        occ(24, 'trg_scan', 'Daily Adjacent-Repo Issue Scan'),
        occ(24, 'trg_scan', 'Daily Adjacent-Repo Issue Scan'),
        occ(24, 'trg_issues', 'Daily Orbital issues check'),
        { id: 'memory:proj-a/e9', title: 'Ship the deadline thing', start: '2026-07-24', all_day: true, project_id: 'proj-a' },
      ],
    });
    render(<WorkbenchPage setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-today')).toBeInTheDocument());
    expect(screen.getAllByText('Daily Adjacent-Repo Issue Scan')).toHaveLength(1);
    expect(screen.getByText('Daily Orbital issues check')).toBeInTheDocument();
    expect(screen.getByText('Ship the deadline thing')).toBeInTheDocument();
  });

  it('per-project lens Today strip still shows that project own events even when excluded globally', async () => {
    mockApi({
      entries: [entry({ project_id: 'proj-x' })],
      projects: [
        { project_id: 'proj-x', name: 'Secret', workspace: '/ws/x', workbench_exclude_global: true },
      ],
      calendarAvailable: true,
      calendarEvents: [
        { id: 'memory:proj-x/s1', title: 'Secret obligation sentence', start: '2026-07-24', all_day: true, project_id: 'proj-x' },
      ],
    });
    render(<WorkbenchPage projectId="proj-x" setRoute={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId('workbench-today')).toBeInTheDocument());
    expect(screen.getByText('Secret obligation sentence')).toBeInTheDocument();
  });
});
