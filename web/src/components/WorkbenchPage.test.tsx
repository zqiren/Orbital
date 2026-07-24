// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * WorkbenchPage integration tests (Task 7 + F3 2026-07-24 revision, incl. the
 * mid-task amendment that removed the receipt). Covers sort order rendering,
 * a "Done" click issuing the right exit POST with optimistic removal, the
 * empty state showing the migrate CTA (per-project and global), the prefill
 * doorway (navigate + composer draft, NO network call, NO session spawn, NO
 * Evidence line), and the Today strip (formerly "This week"). The api client
 * is mocked — no network. The In-flight digest strip is gone entirely.
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
  projects?: unknown[];
  calendarAvailable?: boolean;
  calendarEvents?: unknown[];
} = {}) {
  const {
    entries = [],
    projects = PROJECTS,
    calendarAvailable = false,
    calendarEvents = [],
  } = opts;
  apiMock.mockImplementation(async (path: string, init?: RequestInit) => {
    if (path.startsWith('/api/v2/workbench') && (!init || init.method === undefined)) {
      return { entries };
    }
    if (path === '/api/v2/projects') return projects;
    if (path.startsWith('/api/v2/calendar/availability')) {
      return { available: calendarAvailable, sources: [] };
    }
    if (path.startsWith('/api/v2/calendar/events')) return { events: calendarEvents };
    if (path.includes('/exit')) return { status: 'ok' };
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

describe('WorkbenchPage — prefill doorway (spec 2026-07-24)', () => {
  it('tapping the card body navigates to the project chat tab WITHOUT any POST and WITHOUT a sessionId (no session spawn)', async () => {
    mockApi({ entries: [entry()] });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));

    expect(setRoute).toHaveBeenCalledTimes(1);
    const routeArg = setRoute.mock.calls[0][0];
    expect(routeArg).toMatchObject({ name: 'project', projectId: 'proj-a', tab: 'chat' });
    expect(routeArg.sessionId).toBeUndefined();
    // No network call was made at all for the tap — the only calls on the
    // mock are the initial GETs (workbench/projects/calendar), never a POST.
    expect(apiMock.mock.calls.every(([, init]) => !init || init.method === undefined)).toBe(true);
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

    const routeArg = setRoute.mock.calls[0][0];
    expect(routeArg.draft).toBe('Workbench · "Send Simon the invoice draft"\n\n');
  });

  it('omits the due suffix when absent', async () => {
    mockApi({ entries: [entry({ section: 'Blockers', due: null })] });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));

    const routeArg = setRoute.mock.calls[0][0];
    expect(routeArg.draft).toBe('Workbench · Blockers: "Send Simon the invoice draft"\n\n');
  });

  it('the prefill never contains an Evidence line (receipt/evidence removed 2026-07-24)', async () => {
    mockApi({ entries: [entry({ section: 'Blockers', due: '2026-08-01' })] });
    const setRoute = vi.fn();
    render(<WorkbenchPage setRoute={setRoute} />);
    await waitFor(() => expect(screen.getByTestId('workbench-card-entry-proj-a-e1')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('workbench-card-entry-proj-a-e1'));

    const routeArg = setRoute.mock.calls[0][0];
    expect(routeArg.draft as string).not.toMatch(/Evidence/);
    // Exactly two trailing newlines, nothing appended after the due suffix.
    expect(routeArg.draft as string).toBe(
      'Workbench · Blockers: "Send Simon the invoice draft" (due 2026-08-01)\n\n',
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
