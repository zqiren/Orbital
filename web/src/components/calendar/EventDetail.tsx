// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Event detail panel (Spec 011 §0.4). A centered dialog over a dim backdrop —
 * deliberately NOT a transform slide-in (WKWebView stale-layer blink, per
 * CLAUDE.md). Shows title, time (respecting all_day), location, the first few
 * attendees, a source badge, and a "Link to project" select. Choosing the blank
 * option unlinks (project_id → null).
 */

import { useEffect, useRef, useState } from 'react';
import { X, MapPin, Users, ExternalLink } from 'lucide-react';
import type { Project } from '../../types';
import { useT } from '../../i18n/useT';
import type { CalendarEvent } from './types';
import { agendaDayLabel, formatTimeRange } from './range';
import { eventColor } from './color';

export interface EventDetailProps {
  event: CalendarEvent;
  /** Projects offered in the link select (all of them — no exclusions). */
  projects: Project[];
  onClose: () => void;
  /** Link/unlink; `projectId` null unlinks. Rejects surface as an inline note. */
  onLink: (source: string, sourceId: string, projectId: string | null) => Promise<void>;
}

const MAX_ATTENDEES = 5;

export default function EventDetail({ event, projects, onClose, onLink }: EventDetailProps) {
  const t = useT();
  const [linking, setLinking] = useState(false);
  const [linkError, setLinkError] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  // Close on Escape.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Move focus to the dialog without the WKWebView visual scroll-jump.
  useEffect(() => {
    closeRef.current?.focus({ preventScroll: true });
  }, []);

  const color = eventColor(event.project_id);
  const shownAttendees = event.attendees.slice(0, MAX_ATTENDEES);
  const extraAttendees = event.attendees.length - shownAttendees.length;
  const linkedProject = projects.find((p) => p.project_id === event.project_id);

  async function handleLinkChange(value: string) {
    setLinkError(false);
    setLinking(true);
    try {
      await onLink(event.source, event.source_id, value || null);
    } catch {
      setLinkError(true);
    } finally {
      setLinking(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4"
      data-testid="calendar-event-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-label={t('calendar.detail.aria')}
        data-testid="calendar-event-detail"
        className="w-full max-w-sm rounded-lg border border-border bg-white shadow-lg"
      >
        {/* Header: accent bar + title + close */}
        <div className="flex items-start gap-2 p-4 border-b border-border">
          <span
            aria-hidden="true"
            className="mt-1 h-4 w-1 shrink-0 rounded"
            style={{ backgroundColor: color.border }}
          />
          <h2 className="flex-1 font-mono text-[15px] font-semibold text-primary break-words">
            {event.title || t('calendar.detail.untitled')}
          </h2>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label={t('calendar.detail.close')}
            data-testid="calendar-event-close"
            className="shrink-0 rounded p-1 text-muted hover:text-primary hover:bg-card-hover"
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>

        <div className="p-4 space-y-3 text-[13px] text-primary">
          {/* Time */}
          <div data-testid="calendar-event-time" className="text-secondary">
            {event.all_day
              ? `${agendaDayLabel(new Date(event.start))} · ${t('calendar.allDay')}`
              : formatTimeRange(event.start, event.end)}
          </div>

          {/* Location */}
          {event.location && (
            <div className="flex items-center gap-1.5 text-secondary">
              <MapPin size={13} aria-hidden="true" className="shrink-0" />
              <span className="break-words">{event.location}</span>
            </div>
          )}

          {/* Attendees */}
          {shownAttendees.length > 0 && (
            <div className="flex items-start gap-1.5 text-secondary">
              <Users size={13} aria-hidden="true" className="shrink-0 mt-0.5" />
              <span className="break-words">
                {shownAttendees.join(', ')}
                {extraAttendees > 0 && ` ${t('calendar.detail.attendeesMore', { n: extraAttendees })}`}
              </span>
            </div>
          )}

          {/* Source badge + external link */}
          <div className="flex items-center gap-2">
            <span
              data-testid="calendar-event-source"
              className="inline-flex items-center rounded-full border border-border bg-card px-2 py-0.5 text-[11px] text-secondary"
            >
              {event.source}
            </span>
            {event.url && (
              <a
                href={event.url}
                target="_blank"
                rel="noreferrer noopener"
                className="inline-flex items-center gap-1 text-[11px] text-accent hover:underline"
              >
                <ExternalLink size={11} aria-hidden="true" />
                {t('calendar.detail.open')}
              </a>
            )}
          </div>

          {/* Link-to-project */}
          <div className="pt-2 border-t border-border">
            <label
              htmlFor="calendar-link-select"
              className="block text-[11px] uppercase tracking-wide text-muted mb-1"
            >
              {t('calendar.detail.link')}
            </label>
            <select
              id="calendar-link-select"
              data-testid="calendar-event-link-select"
              value={event.project_id ?? ''}
              disabled={linking}
              onChange={(e) => handleLinkChange(e.target.value)}
              className="w-full rounded-md border border-border bg-white px-2 py-1.5 text-[13px] text-primary disabled:opacity-50"
            >
              <option value="">{t('calendar.detail.linkNone')}</option>
              {projects.map((p) => (
                <option key={p.project_id} value={p.project_id}>
                  {p.name}
                </option>
              ))}
            </select>
            {linkedProject && (
              <p className="mt-1 text-[11px] text-secondary" data-testid="calendar-event-linked">
                {t('calendar.detail.linkedTo', { name: linkedProject.name })}
              </p>
            )}
            {linkError && (
              <p role="alert" className="mt-1 text-[11px] text-error">
                {t('calendar.detail.linkError')}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
