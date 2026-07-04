// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * CalendarPage integration tests (Task H1). Covers the plan's surface-level
 * Vitest list: unavailable zero-state, desktop grid vs. mobile agenda mode
 * switch, project-lens filter (project_id in the events request), event click →
 * detail panel, empty week, and the manual refresh button. The api client is
 * mocked (no network); useIsMobile reads matchMedia, stubbed per test.
 */

import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { CalendarEvent } from './calendar/types';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import CalendarPage from './CalendarPage';

afterEach(() => {
  cleanup();
  delete (window as { matchMedia?: unknown }).matchMedia;
});
beforeEach(() => {
  apiMock.mockReset();
});

/** Build an event today at the given local hours so it lands in the visible week. */
function todayEvent(overrides: Partial<CalendarEvent> = {}): CalendarEvent {
  const now = new Date();
  const at = (h: number) =>
    new Date(now.getFullYear(), now.getMonth(), now.getDate(), h, 0).toISOString();
  return {
    id: 'eventkit:e1',
    source: 'eventkit',
    source_id: 'e1',
    title: 'Standup',
    start: at(10),
    end: at(11),
    all_day: false,
    timezone: null,
    attendees: ['a@example.com'],
    location: 'Room 1',
    url: null,
    project_id: null,
    ...overrides,
  };
}

function mockApi(opts: { available?: boolean; events?: CalendarEvent[]; projects?: unknown[] } = {}) {
  const { available = true, events = [], projects = [] } = opts;
  apiMock.mockImplementation(async (path: string) => {
    if (path.startsWith('/api/v2/calendar/availability')) return { available, sources: [] };
    if (path.startsWith('/api/v2/calendar/events')) return { events };
    if (path === '/api/v2/projects') return projects;
    if (path === '/api/v2/calendar/refresh') return { status: 'ok' };
    if (path.includes('/link')) return { status: 'ok' };
    return {};
  });
}

function stubMobile() {
  (window as { matchMedia?: unknown }).matchMedia = vi.fn().mockReturnValue({
    matches: true,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
}

describe('CalendarPage — availability gating', () => {
  it('shows the unavailable zero-state pointing to Settings when no source is live', async () => {
    mockApi({ available: false });
    render(<CalendarPage />);
    await waitFor(() =>
      expect(screen.getByTestId('calendar-unavailable')).toBeInTheDocument(),
    );
    expect(screen.getByText(/Settings/i)).toBeInTheDocument();
  });

  it('shows the empty-week state when available but the range has no events', async () => {
    mockApi({ events: [] });
    render(<CalendarPage />);
    await waitFor(() => expect(screen.getByTestId('calendar-empty')).toBeInTheDocument());
  });
});

describe('CalendarPage — desktop grid', () => {
  it('renders the week grid and opens the detail panel on event click', async () => {
    mockApi({ events: [todayEvent()] });
    render(<CalendarPage />);
    await waitFor(() => expect(screen.getByTestId('calendar-week-grid')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('calendar-event-eventkit:e1'));

    expect(screen.getByTestId('calendar-event-detail')).toBeInTheDocument();
    // Location only renders in the detail panel, so it is unambiguous.
    expect(screen.getByText('Room 1')).toBeInTheDocument();
    expect(screen.getByTestId('calendar-event-source').textContent).toBe('eventkit');
  });
});

describe('CalendarPage — mobile agenda', () => {
  it('renders the agenda list on a mobile viewport', async () => {
    stubMobile();
    mockApi({ events: [todayEvent({ title: 'Sync' })] });
    render(<CalendarPage />);
    await waitFor(() => expect(screen.getByTestId('calendar-agenda')).toBeInTheDocument());
    expect(screen.getByText('Sync')).toBeInTheDocument();
    expect(screen.queryByTestId('calendar-week-grid')).toBeNull();
  });
});

describe('CalendarPage — project lens', () => {
  it('requests events filtered by project_id and marks the lens attribute', async () => {
    mockApi({ events: [] });
    render(<CalendarPage projectId="proj-a" />);
    await waitFor(() => {
      const call = apiMock.mock.calls.find(([p]) =>
        (p as string).startsWith('/api/v2/calendar/events'),
      );
      expect((call?.[0] as string) ?? '').toContain('project_id=proj-a');
    });
    expect(screen.getByTestId('calendar-page')).toHaveAttribute('data-project-lens', 'true');
  });

  it('omits the lens attribute on the global surface', async () => {
    mockApi({ events: [] });
    render(<CalendarPage />);
    await waitFor(() => expect(screen.getByTestId('calendar-empty')).toBeInTheDocument());
    expect(screen.getByTestId('calendar-page')).not.toHaveAttribute('data-project-lens');
  });
});

describe('CalendarPage — manual refresh', () => {
  it('POSTs /refresh when the refresh button is clicked', async () => {
    mockApi({ events: [] });
    render(<CalendarPage />);
    await waitFor(() => expect(screen.getByTestId('calendar-empty')).toBeInTheDocument());

    apiMock.mockClear();
    mockApi({ events: [] });
    fireEvent.click(screen.getByTestId('calendar-refresh'));

    await waitFor(() =>
      expect(
        apiMock.mock.calls.some(([p]) => p === '/api/v2/calendar/refresh'),
      ).toBe(true),
    );
  });
});
