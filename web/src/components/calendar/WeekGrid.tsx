// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Desktop week grid (Spec 011 §0.4): a time gutter + seven Monday-start day
 * columns, an all-day strip on top, and timed events positioned by start/end
 * with equal-width side-by-side splitting for overlaps (see `overlap.ts`).
 * Everything renders in the browser's local zone.
 */

import type { Project } from '../../types';
import { useT } from '../../i18n/useT';
import type { CalendarEvent } from './types';
import {
  dayHeaderLabel,
  formatTime,
  isSameDay,
  minutesFromMidnight,
} from './range';
import { layoutOverlaps } from './overlap';
import { eventColor } from './color';

export interface WeekGridProps {
  days: Date[];
  events: CalendarEvent[];
  projects: Project[];
  /** Global surface shows a project-name chip on linked events; the per-project
   *  lens is already filtered so the chip is redundant there. */
  showProjectChip: boolean;
  onEventClick: (event: CalendarEvent) => void;
}

const HOUR_H = 44; // px per hour row
const DAY_H = HOUR_H * 24;
const MIN_EVENT_H = 18;

function hourLabel(hour: number): string {
  const d = new Date();
  d.setHours(hour, 0, 0, 0);
  return formatTime(d.toISOString());
}

export default function WeekGrid({
  days,
  events,
  projects,
  showProjectChip,
  onEventClick,
}: WeekGridProps) {
  const t = useT();
  const today = new Date();
  const projectName = (id: string | null): string | null =>
    id ? (projects.find((p) => p.project_id === id)?.name ?? null) : null;

  const allDayByDay = days.map((day) =>
    events.filter((ev) => ev.all_day && isSameDay(new Date(ev.start), day)),
  );
  const hasAllDay = allDayByDay.some((list) => list.length > 0);

  const gridCols = '56px repeat(7, minmax(0, 1fr))';

  return (
    <div className="flex-1 min-h-0 overflow-auto" data-testid="calendar-week-grid">
      <div className="min-w-[720px]">
        {/* Day headers */}
        <div className="grid sticky top-0 z-10 bg-background" style={{ gridTemplateColumns: gridCols }}>
          <div className="border-b border-r border-border" />
          {days.map((day) => (
            <div
              key={day.toISOString()}
              className={`border-b border-r border-border px-2 py-1.5 text-center text-xs ${
                isSameDay(day, today) ? 'text-accent font-semibold' : 'text-secondary'
              }`}
            >
              {dayHeaderLabel(day)}
            </div>
          ))}
        </div>

        {/* All-day strip (only when there are all-day events that week) */}
        {hasAllDay && (
          <div className="grid" style={{ gridTemplateColumns: gridCols }}>
            <div className="border-b border-r border-border px-1 py-1 text-right text-2xs uppercase tracking-wide text-muted">
              {t('calendar.allDay')}
            </div>
            {allDayByDay.map((list, i) => (
              <div
                key={days[i].toISOString()}
                className="border-b border-r border-border p-1 space-y-1"
              >
                {list.map((ev) => {
                  const c = eventColor(ev.project_id);
                  return (
                    <button
                      key={ev.id}
                      type="button"
                      onClick={() => onEventClick(ev)}
                      data-testid={`calendar-event-${ev.id}`}
                      title={ev.title}
                      className="block w-full truncate rounded px-1.5 py-0.5 text-left text-[11px]"
                      style={{ backgroundColor: c.bg, color: c.text }}
                    >
                      {ev.title || t('calendar.detail.untitled')}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        )}

        {/* Timed grid body */}
        <div className="grid" style={{ gridTemplateColumns: gridCols }}>
          {/* Time gutter */}
          <div className="relative border-r border-border" style={{ height: DAY_H }}>
            {Array.from({ length: 24 }, (_, h) => (
              <div
                key={h}
                className="absolute right-1 -translate-y-1/2 text-2xs text-muted"
                style={{ top: h * HOUR_H }}
              >
                {h === 0 ? '' : hourLabel(h)}
              </div>
            ))}
          </div>

          {/* Day columns */}
          {days.map((day, dayIdx) => {
            const dayTimed = events.filter(
              (ev) => !ev.all_day && isSameDay(new Date(ev.start), day),
            );
            const positioned = layoutOverlaps(dayTimed);
            return (
              <div
                key={day.toISOString()}
                className="relative border-r border-border"
                style={{ height: DAY_H }}
                data-testid={`calendar-day-col-${dayIdx}`}
              >
                {/* Hour gridlines */}
                {Array.from({ length: 24 }, (_, h) => (
                  <div
                    key={h}
                    className="absolute inset-x-0 border-t border-border/60"
                    style={{ top: h * HOUR_H }}
                  />
                ))}

                {positioned.map(({ event: ev, col, cols }) => {
                  const startMin = minutesFromMidnight(ev.start);
                  const durMin =
                    (new Date(ev.end).getTime() - new Date(ev.start).getTime()) / 60000;
                  const top = (startMin / 60) * HOUR_H;
                  const rawH = (durMin / 60) * HOUR_H;
                  const height = Math.max(MIN_EVENT_H, Math.min(rawH, DAY_H - top));
                  const widthPct = 100 / cols;
                  const c = eventColor(ev.project_id);
                  const pname = showProjectChip ? projectName(ev.project_id) : null;
                  return (
                    <button
                      key={ev.id}
                      type="button"
                      onClick={() => onEventClick(ev)}
                      data-testid={`calendar-event-${ev.id}`}
                      title={ev.title}
                      className="absolute overflow-hidden rounded px-1 py-0.5 text-left text-[11px] leading-tight"
                      style={{
                        top,
                        height,
                        left: `calc(${col * widthPct}% + 1px)`,
                        width: `calc(${widthPct}% - 2px)`,
                        backgroundColor: c.bg,
                        color: c.text,
                        borderLeft: `2px solid ${c.border}`,
                      }}
                    >
                      <span className="block truncate font-medium">
                        {ev.title || t('calendar.detail.untitled')}
                      </span>
                      <span className="block truncate text-2xs opacity-80">
                        {formatTime(ev.start)}
                      </span>
                      {pname && (
                        <span className="mt-0.5 block truncate text-3xs opacity-90">
                          {pname}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
