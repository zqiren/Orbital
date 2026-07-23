// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Workbench surface (spec 2026-07-23 §5.3/§5.4/§6). Mounted TWICE with the
 * SAME component, mirroring CalendarPage's mount pattern (spec-011 §0.4):
 *   - no `projectId`  → the global Workbench (privacy-filtered aggregate);
 *   - with `projectId` → the per-project lens (server-filtered to one project,
 *     and — unlike the global view — shown regardless of the project's
 *     "exclude from global Workbench" toggle).
 *
 * ONE list, no bands (spec §6): flagged `[user]` entries and daemon-computed
 * cards (overdue / broken automation / paused thread) interleave into a
 * single sort via `useWorkbench().items` — overdue first, then oldest first.
 * Tapping a card's body is the doorway (spec §5.3): it spawns (or, for a
 * paused thread, resumes) a session and navigates there via the route model.
 */

import { useEffect, useMemo, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { RefreshCw, Inbox } from 'lucide-react';
import { api } from '../config';
import type { Project } from '../types';
import type { Route } from '../route';
import { useT } from '../i18n/useT';
import { useWorkbench } from './workbench/useWorkbench';
import { useCalendar } from './calendar/useCalendar';
import { formatTime } from './calendar/range';
import WorkbenchCard from './WorkbenchCard';
import type { WorkbenchComputedCard, WorkbenchEntry } from './workbench/types';

export interface WorkbenchPageProps {
  /**
   * When set, the Workbench is mounted as a project lens (the per-project
   * `workbench` tab). When absent, this is the global Workspace surface.
   */
  projectId?: string;
  setRoute: Dispatch<SetStateAction<Route>>;
}

/** [start, end) ISO instants for the next 7 days from local midnight today —
 *  a rolling window, unlike the calendar's Monday-aligned week. */
function sevenDayRangeISO(): { start: string; end: string } {
  const now = new Date();
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const end = new Date(start);
  end.setDate(end.getDate() + 7);
  return { start: start.toISOString(), end: end.toISOString() };
}

export default function WorkbenchPage({ projectId, setRoute }: WorkbenchPageProps) {
  const t = useT();
  const [projects, setProjects] = useState<Project[]>([]);
  const [openError, setOpenError] = useState<string | null>(null);

  const { items, entries, computed, loading, error, conflict, refetch, exitEntry, dismissComputed, openEntry, migrate } =
    useWorkbench({ projectId });

  const { start, end } = useMemo(() => sevenDayRangeISO(), []);
  const { availability: weekAvailability, events: weekEvents } = useCalendar({ projectId, start, end });

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

  const showProjectChip = !projectId; // global surface only
  const projectName = (id: string): string | null =>
    projects.find((p) => p.project_id === id)?.name ?? null;

  function navigateToChat(pid: string, sessionId?: string) {
    setRoute({ name: 'project', projectId: pid, tab: 'chat', sessionId });
  }

  async function handleOpenEntry(entry: WorkbenchEntry) {
    setOpenError(null);
    try {
      const sessionId = await openEntry(entry.project_id, entry.id);
      navigateToChat(entry.project_id, sessionId);
    } catch {
      setOpenError(t('workbench.error.load'));
    }
  }

  // No backend doorway exists for overdue/broken_automation computed cards
  // (only /dismiss). paused_thread is the one computed case where the
  // original session exists (spec §6) — its `key` IS the session uuid, so
  // tapping resumes it directly; the others navigate to the project's chat
  // with no session preselected.
  function handleOpenComputed(card: WorkbenchComputedCard) {
    if (card.type === 'paused_thread') {
      navigateToChat(card.project_id, card.key);
    } else {
      navigateToChat(card.project_id);
    }
  }

  async function handleMigrate(pid: string) {
    setOpenError(null);
    try {
      const sessionId = await migrate(pid);
      navigateToChat(pid, sessionId);
    } catch {
      setOpenError(t('workbench.error.load'));
    }
  }

  function handleOpenCalendar() {
    if (projectId) {
      setRoute({ name: 'project', projectId, tab: 'calendar' });
    } else {
      setRoute({ name: 'calendar' });
    }
  }

  const isEmpty = !loading && !error && entries.length === 0 && computed.length === 0;
  const showThisWeek = weekAvailability?.available === true && weekEvents.length > 0;

  const rootProps = {
    className: 'flex flex-col flex-1 min-h-0',
    'data-testid': 'workbench-page',
    'data-project-lens': projectId ? ('true' as const) : undefined,
  };

  return (
    <div {...rootProps}>
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <h1 className="mr-1 font-mono text-[14px] font-semibold text-primary">
          {t('workbench.title')}
        </h1>
        <span className="flex-1" />
        <button
          type="button"
          onClick={() => refetch()}
          aria-label={t('workbench.retry')}
          data-testid="workbench-refresh"
          className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-[12px] text-secondary hover:bg-card-hover hover:text-primary"
        >
          <RefreshCw size={14} aria-hidden="true" />
        </button>
      </div>

      {conflict && (
        <div
          data-testid="workbench-conflict-notice"
          className="border-b border-warning/20 bg-warning/10 px-3 py-1.5 text-[12px] text-warning"
        >
          {t('workbench.conflict.notice')}
        </div>
      )}
      {openError && (
        <div className="border-b border-error/20 bg-error/5 px-3 py-1.5 text-[12px] text-error">
          {openError}
        </div>
      )}

      {showThisWeek && (
        <div data-testid="workbench-this-week" className="border-b border-border px-3 py-2">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-secondary">
              {t('workbench.thisWeek.title')}
            </span>
            <button
              type="button"
              onClick={handleOpenCalendar}
              data-testid="workbench-open-calendar"
              className="text-[11px] text-secondary hover:text-primary"
            >
              {t('workbench.thisWeek.openCalendar')}
            </button>
          </div>
          <ul className="space-y-1">
            {weekEvents
              .slice()
              .sort((a, b) => new Date(a.start).getTime() - new Date(b.start).getTime())
              .slice(0, 5)
              .map((ev) => (
                <li
                  key={ev.id}
                  className="flex items-center gap-2 text-[12px] text-secondary"
                >
                  <span className="font-mono text-[11px]">
                    {ev.all_day ? t('calendar.allDay') : formatTime(ev.start)}
                  </span>
                  <span className="truncate text-primary">{ev.title}</span>
                </li>
              ))}
          </ul>
        </div>
      )}

      {loading ? (
        <div
          className="flex flex-1 items-center justify-center text-sm text-secondary"
          data-testid="workbench-loading"
        >
          {t('workbench.loading')}
        </div>
      ) : error ? (
        <div
          className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center"
          data-testid="workbench-error"
        >
          <p className="text-sm text-error">{t('workbench.error.load')}</p>
          <button
            type="button"
            onClick={() => refetch()}
            className="rounded border border-border px-3 py-1 text-[12px] text-secondary hover:bg-card-hover hover:text-primary"
          >
            {t('workbench.retry')}
          </button>
        </div>
      ) : isEmpty ? (
        <div
          className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center"
          data-testid="workbench-empty"
        >
          <Inbox size={28} aria-hidden="true" className="text-muted" />
          <h2 className="font-mono text-[15px] font-semibold text-primary">
            {t('workbench.empty.title')}
          </h2>
          <p className="max-w-sm text-sm text-secondary">{t('workbench.empty.body')}</p>
          {projectId ? (
            <button
              type="button"
              onClick={() => handleMigrate(projectId)}
              data-testid="workbench-migrate-cta"
              className="bg-accent text-white text-sm font-medium rounded-lg px-4 py-2 hover:bg-accent/90 transition-all duration-150"
            >
              {t('workbench.empty.migrateCta')}
            </button>
          ) : (
            <div className="flex flex-col gap-1.5">
              {projects
                .filter((p) => p.workspace)
                .map((p) => (
                  <button
                    key={p.project_id}
                    type="button"
                    onClick={() => handleMigrate(p.project_id)}
                    data-testid={`workbench-migrate-cta-${p.project_id}`}
                    className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-primary hover:bg-card-hover"
                  >
                    {t('workbench.empty.migrateCtaFor', { name: p.name })}
                  </button>
                ))}
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-auto px-3 py-2" data-testid="workbench-list">
          <ul className="space-y-2">
            {items.map((item) => {
              const key =
                item.kind === 'entry'
                  ? `entry-${item.data.project_id}-${item.data.id}`
                  : `computed-${item.data.project_id}-${item.data.type}-${item.data.key}`;
              return (
                <li key={key}>
                  <WorkbenchCard
                    item={item}
                    showProjectChip={showProjectChip}
                    projectName={projectName(item.data.project_id)}
                    onOpen={() =>
                      item.kind === 'entry'
                        ? handleOpenEntry(item.data)
                        : handleOpenComputed(item.data)
                    }
                    onExit={
                      item.kind === 'entry'
                        ? (kind) => exitEntry(item.data.project_id, item.data.id, kind)
                        : undefined
                    }
                    onDismiss={
                      item.kind === 'computed'
                        ? () => dismissComputed(item.data.project_id, item.data.type, item.data.key)
                        : undefined
                    }
                  />
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
