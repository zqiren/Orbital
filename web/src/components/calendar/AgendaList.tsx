// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Mobile agenda list (Spec 011 §0.4): the same visible week as the desktop
 * grid, but events grouped by day in a vertical list. Empty days are omitted;
 * an all-empty week is handled by the surface's zero-state.
 */

import type { Project } from '../../types';
import { useT } from '../../i18n/useT';
import type { CalendarEvent } from './types';
import { agendaDayLabel, formatTimeRange, groupEventsByDay, isSameDay } from './range';
import { eventColor } from './color';

export interface AgendaListProps {
  anchor: Date;
  events: CalendarEvent[];
  projects: Project[];
  showProjectChip: boolean;
  onEventClick: (event: CalendarEvent) => void;
}

export default function AgendaList({
  anchor,
  events,
  projects,
  showProjectChip,
  onEventClick,
}: AgendaListProps) {
  const t = useT();
  const today = new Date();
  const projectName = (id: string | null): string | null =>
    id ? (projects.find((p) => p.project_id === id)?.name ?? null) : null;

  const groups = groupEventsByDay(events, anchor).filter((g) => g.events.length > 0);

  return (
    <div className="flex-1 min-h-0 overflow-auto px-3 py-2" data-testid="calendar-agenda">
      <ul className="space-y-4">
        {groups.map((group) => (
          <li key={group.day.toISOString()}>
            <div
              className={`sticky top-0 bg-background py-1 text-xs font-semibold ${
                isSameDay(group.day, today) ? 'text-accent' : 'text-secondary'
              }`}
            >
              {agendaDayLabel(group.day)}
            </div>
            <ul className="mt-1 space-y-1.5">
              {group.events.map((ev) => {
                const c = eventColor(ev.project_id);
                const pname = showProjectChip ? projectName(ev.project_id) : null;
                return (
                  <li key={ev.id}>
                    <button
                      type="button"
                      onClick={() => onEventClick(ev)}
                      data-testid={`calendar-event-${ev.id}`}
                      className="flex w-full items-stretch gap-2 rounded-md border border-border bg-card p-2 text-left hover:bg-card-hover"
                    >
                      <span
                        aria-hidden="true"
                        className="w-1 shrink-0 rounded"
                        style={{ backgroundColor: c.border }}
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[13px] font-medium text-primary">
                          {ev.title || t('calendar.detail.untitled')}
                        </span>
                        <span className="block text-[11px] text-secondary">
                          {ev.all_day ? t('calendar.allDay') : formatTimeRange(ev.start, ev.end)}
                        </span>
                        <span className="mt-0.5 flex items-center gap-1.5">
                          <span className="text-2xs text-muted">{ev.source}</span>
                          {pname && (
                            <span
                              className="rounded-full px-1.5 py-0.5 text-2xs"
                              style={{ backgroundColor: c.bg, color: c.text }}
                            >
                              {pname}
                            </span>
                          )}
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </li>
        ))}
      </ul>
    </div>
  );
}
