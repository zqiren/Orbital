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
 * Flagged `[user]` entries render in server order (spec §6) — overdue first,
 * then oldest first. Tapping a card's body is the doorway (spec §5.3): it
 * spawns a session and navigates there via the route model.
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
import type { WorkbenchEntry } from './workbench/types';

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

  const { entries, loading, error, conflict, refetch, exitEntry, openEntry, migrate } =
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
    setRoute({ name: 'calendar' });
  }

  const isEmpty = !loading && !error && entries.length === 0;
  const showThisWeek = weekAvailability?.available === true && weekEvents.length > 0;

  // This-week strip rows: privacy-filter (global mode), chronological, then
  // ONE row per automation trigger (a daily cron emits 7 occurrences in the
  // window — the strip shows the next one, not seven copies; the full series
  // stays on the Calendar). Memory/external events are never deduped.
  const weekRows = useMemo(() => {
    const seenTriggers = new Set<string>();
    return weekEvents
      .slice()
      .filter(
        (ev) =>
          projectId != null ||
          !ev.project_id ||
          !projects.find(
            (p) => p.project_id === ev.project_id && p.workbench_exclude_global,
          ),
      )
      .sort((a, b) => new Date(a.start).getTime() - new Date(b.start).getTime())
      .filter((ev) => {
        const m = /^automation[:/](.*)[/][^/]*$/.exec(ev.id);
        if (!m) return true;
        if (seenTriggers.has(m[1])) return false;
        seenTriggers.add(m[1]);
        return true;
      })
      .slice(0, 5);
  }, [weekEvents, projects, projectId]);

  const rootProps = {
    className: 'flex flex-col flex-1 min-h-0',
    'data-testid': 'workbench-page',
    'data-project-lens': projectId ? ('true' as const) : undefined,
  };

  return (
    <div {...rootProps}>
      <div className="flex items-center gap-2 border-b border-border/60 bg-card/80 px-4 py-2.5 backdrop-blur-xl">
        <h1 className="mr-1 text-[17px] font-semibold tracking-[-0.01em] text-primary">
          {t('workbench.title')}
        </h1>
        <span className="flex-1" />
        <button
          type="button"
          onClick={() => refetch()}
          aria-label={t('workbench.retry')}
          data-testid="workbench-refresh"
          className="inline-flex items-center justify-center rounded-full p-1.5 text-secondary transition-[transform,background-color,color] duration-100 ease-out hover:bg-card-hover hover:text-primary active:scale-[0.92] motion-reduce:transition-none motion-reduce:active:scale-100"
        >
          <RefreshCw size={15} aria-hidden="true" />
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

      {showThisWeek && weekRows.length > 0 && (
        <div data-testid="workbench-this-week" className="px-4 pt-3">
          <div className="rounded-2xl bg-sidebar/50 px-3.5 py-2.5">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-[12px] font-semibold text-secondary">
                {t('workbench.thisWeek.title')}
              </span>
              <button
                type="button"
                onClick={handleOpenCalendar}
                data-testid="workbench-open-calendar"
                className="text-[12px] font-medium text-accent transition-colors duration-100 hover:text-accent/80"
              >
                {t('workbench.thisWeek.openCalendar')}
              </button>
            </div>
            <ul className="space-y-1">
              {weekRows.map((ev) => (
                <li
                  key={ev.id}
                  className="flex items-center gap-2.5 text-[12.5px] text-secondary"
                >
                  <span className="w-16 shrink-0 font-mono text-[11px] tabular-nums text-muted">
                    {ev.all_day ? t('calendar.allDay') : formatTime(ev.start)}
                  </span>
                  <span className="truncate text-primary">{ev.title}</span>
                </li>
              ))}
            </ul>
          </div>
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
          <h2 className="text-[16px] font-semibold tracking-[-0.01em] text-primary">
            {t('workbench.empty.title')}
          </h2>
          <p className="max-w-sm text-sm text-secondary">{t('workbench.empty.body')}</p>
          {projectId ? (
            <button
              type="button"
              onClick={() => handleMigrate(projectId)}
              data-testid="workbench-migrate-cta"
              className="rounded-full bg-accent px-4 py-2 text-sm font-medium text-white transition-[transform,background-color] duration-100 ease-out hover:bg-accent/90 active:scale-[0.97] motion-reduce:transition-none motion-reduce:active:scale-100"
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
                    className="rounded-full border border-border px-4 py-2 text-sm font-medium text-primary transition-[transform,background-color] duration-100 ease-out hover:bg-card-hover active:scale-[0.97] motion-reduce:transition-none motion-reduce:active:scale-100"
                  >
                    {t('workbench.empty.migrateCtaFor', { name: p.name })}
                  </button>
                ))}
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-auto px-4 py-3" data-testid="workbench-list">
          <ul className="space-y-2.5">
            {entries.map((e) => (
              <li key={`entry-${e.project_id}-${e.id}`}>
                <WorkbenchCard
                  entry={e}
                  showProjectChip={showProjectChip}
                  projectName={projectName(e.project_id)}
                  onOpen={() => handleOpenEntry(e)}
                  onExit={(kind) => exitEntry(e.project_id, e.id, kind)}
                />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
