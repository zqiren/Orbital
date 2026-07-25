// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Calendar surface (Spec 011 §0.4). Mounted twice with the SAME component:
 *   - no `projectId`  → the global Workspace calendar (all events);
 *   - with `projectId` → the per-project lens (server-filtered to linked events).
 * The prop contract is intentionally minimal (`projectId?: string`) so the two
 * App.tsx mount sites need no changes.
 *
 * Desktop renders a Monday-start week grid; mobile renders an agenda list — both
 * over the same visible week, with Prev/Today/Next range nav and a user-required
 * manual refresh button. Clicking an event opens a detail panel with a
 * link-to-project select. Times render in the browser's local zone (Date + Intl,
 * no tz library). Availability gates a friendly zero-state pointing to Settings.
 */

import { useEffect, useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, RefreshCw, CalendarX } from 'lucide-react';
import { api } from '../config';
import type { Project } from '../types';
import { useT } from '../i18n/useT';
import { useCalendar } from './calendar/useCalendar';
import { useIsMobile } from './calendar/useIsMobile';
import { addWeeks, weekDays, weekRangeISO, weekRangeLabel } from './calendar/range';
import WeekGrid from './calendar/WeekGrid';
import AgendaList from './calendar/AgendaList';
import EventDetail from './calendar/EventDetail';
import type { CalendarEvent } from './calendar/types';

interface CalendarPageProps {
  /**
   * When set, the calendar is mounted as a project lens (the per-project
   * `calendar` tab) filtered to events linked to this project. When absent,
   * this is the global calendar surface (spec 011 §0.4).
   */
  projectId?: string;
}

export default function CalendarPage({ projectId }: CalendarPageProps) {
  const t = useT();
  const isMobile = useIsMobile();

  // A date inside the visible week. Nav shifts it by ±1 week; Today resets it.
  const [anchor, setAnchor] = useState<Date>(() => new Date());
  const [selected, setSelected] = useState<CalendarEvent | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);

  const { start, end } = useMemo(() => weekRangeISO(anchor), [anchor]);
  const rangeLabel = useMemo(() => weekRangeLabel(anchor), [anchor]);
  const days = useMemo(() => weekDays(anchor), [anchor]);

  const { availability, events, loading, refreshing, error, refetch, refresh, linkEvent } =
    useCalendar({ projectId, start, end });

  // Project list for the link-to-project select + global project-name chips.
  useEffect(() => {
    let cancelled = false;
    api<Project[]>('/api/v2/projects')
      .then((data) => {
        if (!cancelled) setProjects(Array.isArray(data) ? data : []);
      })
      .catch(() => {
        if (!cancelled) setProjects([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // The linkage returned by the refetch may replace the event object identity;
  // keep the open panel pointed at the fresh copy (or close if it vanished).
  useEffect(() => {
    if (!selected) return;
    const fresh = events.find((e) => e.id === selected.id);
    if (fresh && fresh !== selected) setSelected(fresh);
  }, [events, selected]);

  const showProjectChip = !projectId; // global surface only

  const rootProps = {
    className: 'flex flex-col flex-1 min-h-0',
    'data-testid': 'calendar-page',
    'data-project-lens': projectId ? ('true' as const) : undefined,
  };

  // Availability still probing: neutral spinner (we don't yet know if a
  // calendar exists, so neither the grid nor the zero-state is honest yet).
  if (availability === null) {
    return (
      <div {...rootProps}>
        <div
          className="flex flex-1 items-center justify-center text-sm text-secondary"
          data-testid="calendar-loading"
        >
          {t('calendar.loading')}
        </div>
      </div>
    );
  }

  // No calendar source at all: friendly zero-state pointing to Settings.
  if (!availability.available) {
    return (
      <div {...rootProps}>
        <div
          className="flex flex-col flex-1 items-center justify-center gap-2 px-6 text-center"
          data-testid="calendar-unavailable"
        >
          <CalendarX size={28} aria-hidden="true" className="text-muted" />
          <h1 className="text-base font-semibold tracking-[-0.01em] text-primary">
            {t('calendar.unavailable.title')}
          </h1>
          <p className="max-w-sm text-sm text-secondary">{t('calendar.unavailable.body')}</p>
        </div>
      </div>
    );
  }

  return (
    <div {...rootProps}>
      {/* Range nav + refresh */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <h1 className="mr-1 text-sm font-semibold text-primary">
          {t('workspace.calendar.title')}
        </h1>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setAnchor((a) => addWeeks(a, -1))}
            aria-label={t('calendar.nav.prev')}
            data-testid="calendar-prev"
            className="rounded p-1 text-secondary hover:bg-card-hover hover:text-primary"
          >
            <ChevronLeft size={16} aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={() => setAnchor(new Date())}
            data-testid="calendar-today"
            className="rounded border border-border px-2 py-0.5 text-xs text-secondary hover:bg-card-hover hover:text-primary"
          >
            {t('calendar.nav.today')}
          </button>
          <button
            type="button"
            onClick={() => setAnchor((a) => addWeeks(a, 1))}
            aria-label={t('calendar.nav.next')}
            data-testid="calendar-next"
            className="rounded p-1 text-secondary hover:bg-card-hover hover:text-primary"
          >
            <ChevronRight size={16} aria-hidden="true" />
          </button>
        </div>
        <span
          className="ml-1 flex-1 truncate text-[13px] text-primary"
          data-testid="calendar-range-label"
        >
          {rangeLabel}
        </span>
        <button
          type="button"
          onClick={refresh}
          disabled={refreshing}
          aria-label={t('calendar.refresh')}
          data-testid="calendar-refresh"
          className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs text-secondary hover:bg-card-hover hover:text-primary disabled:opacity-60"
        >
          <RefreshCw
            size={14}
            aria-hidden="true"
            className={refreshing ? 'animate-spin' : ''}
          />
          <span className="hidden sm:inline">
            {refreshing ? t('calendar.refreshing') : t('calendar.refresh')}
          </span>
        </button>
      </div>

      {/* Body */}
      {error ? (
        <div
          className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center"
          data-testid="calendar-error"
        >
          <p className="text-sm text-error">{t('calendar.error.load')}</p>
          <button
            type="button"
            onClick={refetch}
            className="rounded border border-border px-3 py-1 text-xs text-secondary hover:bg-card-hover hover:text-primary"
          >
            {t('calendar.retry')}
          </button>
        </div>
      ) : loading ? (
        <div
          className="flex flex-1 items-center justify-center text-sm text-secondary"
          data-testid="calendar-loading"
        >
          {t('calendar.loading')}
        </div>
      ) : events.length === 0 ? (
        <div
          className="flex flex-1 flex-col items-center justify-center gap-1 px-6 text-center"
          data-testid="calendar-empty"
        >
          <p className="text-sm text-secondary">{t('calendar.empty.title')}</p>
        </div>
      ) : isMobile ? (
        <AgendaList
          anchor={anchor}
          events={events}
          projects={projects}
          showProjectChip={showProjectChip}
          onEventClick={setSelected}
        />
      ) : (
        <WeekGrid
          days={days}
          events={events}
          projects={projects}
          showProjectChip={showProjectChip}
          onEventClick={setSelected}
        />
      )}

      {selected && (
        <EventDetail
          event={selected}
          projects={projects}
          onClose={() => setSelected(null)}
          onLink={linkEvent}
        />
      )}
    </div>
  );
}
