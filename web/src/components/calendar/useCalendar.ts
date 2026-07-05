// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Data hook for the calendar surface (Spec 011 §0.4).
 *
 * Fetches availability once and the events for the currently-visible week range
 * on every range change. Exposes a manual `refresh()` (POST /refresh then
 * refetch — the user-required refresh button) and `linkEvent()` (link/unlink an
 * event to a project). A monotonic request epoch guards against a slow response
 * from a stale range clobbering the current one.
 *
 * Backend contract (Task G1, live):
 *   GET  /api/v2/calendar/availability
 *   GET  /api/v2/calendar/events?start=ISO&end=ISO[&project_id=...]
 *   POST /api/v2/calendar/refresh
 *   POST /api/v2/calendar/events/{source}/{event_id}/link  {project_id|null}
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../../config';
import type { Availability, CalendarEvent, EventsResponse } from './types';

export interface UseCalendarArgs {
  /** Project lens: filter events to those linked to this project (omit for the
   *  global surface). */
  projectId?: string;
  /** Visible week range as ISO instants (from `weekRangeISO`). */
  start: string;
  end: string;
}

export interface UseCalendarResult {
  availability: Availability | null;
  events: CalendarEvent[];
  /** Initial/range-change event fetch is in flight. */
  loading: boolean;
  /** Manual refresh (POST + refetch) is in flight — drives the spinner. */
  refreshing: boolean;
  /** Non-null when the event fetch failed (inline error + retry). */
  error: string | null;
  refetch: () => void;
  refresh: () => Promise<void>;
  linkEvent: (source: string, sourceId: string, projectId: string | null) => Promise<void>;
}

/** Build the events URL, appending `project_id` only in lens mode. */
function eventsUrl(start: string, end: string, projectId?: string): string {
  const q = new URLSearchParams({ start, end });
  if (projectId) q.set('project_id', projectId);
  return `/api/v2/calendar/events?${q.toString()}`;
}

/**
 * The link route is `/events/{source}/{event_id:path}/link` — `event_id` is a
 * PATH converter that keeps slashes, so we encode each source_id segment but
 * preserve the `/` separators rather than percent-encoding them away.
 */
function linkUrl(source: string, sourceId: string): string {
  const encodedId = sourceId.split('/').map(encodeURIComponent).join('/');
  return `/api/v2/calendar/events/${encodeURIComponent(source)}/${encodedId}/link`;
}

export function useCalendar({ projectId, start, end }: UseCalendarArgs): UseCalendarResult {
  const [availability, setAvailability] = useState<Availability | null>(null);
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Monotonic epoch: only the newest events request may write state.
  const epochRef = useRef(0);

  const fetchAvailability = useCallback(async () => {
    try {
      const a = await api<Availability>('/api/v2/calendar/availability');
      setAvailability(a);
    } catch {
      // Treat a failed probe as "no calendar" — the zero-state is the honest UI.
      setAvailability({ available: false, sources: [] });
    }
  }, []);

  const fetchEvents = useCallback(async () => {
    const epoch = ++epochRef.current;
    setLoading(true);
    setError(null);
    try {
      const res = await api<EventsResponse>(eventsUrl(start, end, projectId));
      if (epochRef.current === epoch) setEvents(res?.events ?? []);
    } catch (e) {
      if (epochRef.current === epoch) {
        setError(e instanceof Error ? e.message : 'error');
        setEvents([]);
      }
    } finally {
      if (epochRef.current === epoch) setLoading(false);
    }
  }, [projectId, start, end]);

  useEffect(() => {
    fetchAvailability();
  }, [fetchAvailability]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await api('/api/v2/calendar/refresh', { method: 'POST' });
      await Promise.all([fetchAvailability(), fetchEvents()]);
    } catch {
      // A failed refresh leaves the current view intact; fetchEvents surfaces
      // its own error if the refetch itself failed.
    } finally {
      setRefreshing(false);
    }
  }, [fetchAvailability, fetchEvents]);

  const linkEvent = useCallback(
    async (source: string, sourceId: string, linkProjectId: string | null) => {
      await api(linkUrl(source, sourceId), {
        method: 'POST',
        body: JSON.stringify({ project_id: linkProjectId }),
      });
      await fetchEvents();
    },
    [fetchEvents],
  );

  return {
    availability,
    events,
    loading,
    refreshing,
    error,
    refetch: fetchEvents,
    refresh,
    linkEvent,
  };
}
