// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Workbench surface (spec 2026-07-23 §5.3/§5.4/§6; prefill doorway + Today
 * strip revision 2026-07-24). Mounted TWICE with the SAME component,
 * mirroring CalendarPage's mount pattern (spec-011 §0.4):
 *   - no `projectId`  → the global Workbench (privacy-filtered aggregate);
 *   - with `projectId` → the per-project lens (server-filtered to one project,
 *     and — unlike the global view — shown regardless of the project's
 *     "exclude from global Workbench" toggle).
 *
 * Flagged `[user]` entries render in server order (spec §6) — overdue first,
 * then oldest first. Tapping a card's body is the doorway (spec §5.3, round 3
 * revision 2026-07-24): it mints a brand-new session via the SAME
 * `newSession()` primitive the "+ new session" sidebar button already uses
 * (POST `/new-session`, no `session_id` — fresh-create), then navigates to
 * the project's chat tab with that session id and the card's context
 * prefilled into the composer draft, for the user to edit and send
 * themselves. Two earlier iterations of this doorway tried to avoid minting
 * at tap time (a route suppression flag; a mint deferred to the user's own
 * send) to avoid littering an empty session on a casual card peek — both
 * added inference machinery to answer "which session did my action affect?"
 * and were reverted; the product decision is that a tap is a genuine "open
 * this" action, exactly like clicking "+ new session" would be, and litter
 * cost is an acceptable trade for zero inference. See `buildPrefillDraft`
 * below for the exact draft format, and ChatTab.tsx/ChatView.tsx for how
 * `route.draft` is consumed once the composer mounts.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { RefreshCw, Inbox } from 'lucide-react';
import { api } from '../config';
import type { Project } from '../types';
import type { Route } from '../route';
import { useT } from '../i18n/useT';
import { useAgent } from '../hooks/useAgent';
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

/** [start, end) ISO instants for today: local midnight today to local
 *  midnight tomorrow. Narrowed from a 7-day rolling window per the
 *  2026-07-24 user decision (the strip is titled "Today", not "This week"). */
function todayRangeISO(): { start: string; end: string } {
  const now = new Date();
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const end = new Date(start);
  end.setDate(end.getDate() + 1);
  return { start: start.toISOString(), end: end.toISOString() };
}

/**
 * Composer prefill text for the card-tap doorway (spec 2026-07-24 §5.3,
 * design-pinned format — do not reflow; Evidence line dropped in the
 * mid-task amendment that removed the receipt, since `evidence` no longer
 * exists on `WorkbenchEntry`):
 *   `Workbench · {section}: "{text}"` — `{section}: ` omitted when
 *   `section` is null — then ` (due {due})` when `due` is set, then two
 *   trailing newlines (cursor lands on the blank line after, composer
 *   focused).
 */
function buildPrefillDraft(entry: WorkbenchEntry): string {
  const sectionPart = entry.section ? `${entry.section}: ` : '';
  let line = `Workbench · ${sectionPart}"${entry.text}"`;
  if (entry.due) line += ` (due ${entry.due})`;
  return line + '\n\n';
}

export default function WorkbenchPage({ projectId, setRoute }: WorkbenchPageProps) {
  const t = useT();
  const [projects, setProjects] = useState<Project[]>([]);
  const [openError, setOpenError] = useState<string | null>(null);
  // Global empty state: project chosen in the migrate dropdown ('' = none).
  const [migrateTarget, setMigrateTarget] = useState('');
  // Project filter (round 3, item 2), GLOBAL view only. '' = All projects.
  const [filterProjectId, setFilterProjectId] = useState('');
  // Delete/exit failure notice (bug #45): the hook signals a failed exit here
  // (non-409 error, or an id-less entry) so a failed Delete is no longer a
  // silent no-op. Kept as page state so the hook's returned surface is stable.
  const [deleteError, setDeleteError] = useState(false);
  const handleExitError = useCallback(() => setDeleteError(true), []);

  const { entries, loading, error, conflict, refetch, exitEntry, migrate } =
    useWorkbench({ projectId, onExitError: handleExitError });
  const { newSession } = useAgent();

  const { start, end } = useMemo(() => todayRangeISO(), []);
  const { availability: todayAvailability, events: todayEvents } = useCalendar({ projectId, start, end });

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

  // Project filter (round 3, item 2), GLOBAL view only: the projects that
  // currently have entries, deduped, in the order they first appear in
  // `entries` (server order — never re-sorted alphabetically).
  const filterOptions = (() => {
    const seen = new Set<string>();
    const opts: Array<{ id: string; name: string }> = [];
    for (const e of entries) {
      if (seen.has(e.project_id)) continue;
      seen.add(e.project_id);
      opts.push({ id: e.project_id, name: projectName(e.project_id) ?? e.project_id });
    }
    return opts;
  })();

  // If the selected project's entries all get exited (or its filter option
  // otherwise disappears), fall back to "All projects" rather than showing a
  // stuck, silently-empty list.
  useEffect(() => {
    if (filterProjectId && !entries.some((e) => e.project_id === filterProjectId)) {
      setFilterProjectId('');
    }
  }, [entries, filterProjectId]);

  const filteredEntries = filterProjectId
    ? entries.filter((e) => e.project_id === filterProjectId)
    : entries;

  function navigateToChat(pid: string, sessionId?: string) {
    setRoute({ name: 'project', projectId: pid, tab: 'chat', sessionId });
  }

  // Card tap doorway (spec 2026-07-24, round 3 final revision): mint a
  // brand-new session via the SAME `newSession()` primitive the "+ new
  // session" sidebar button uses (POST /new-session, no session_id —
  // fresh-create; ChatTab's handleNewSession is the other caller of this
  // exact function), then navigate to the project's chat tab with that
  // session id and the card's context prefilled into the composer draft. On
  // mint failure, surface it via the existing `openError` banner and do NOT
  // navigate — mirrors `handleMigrate`'s error handling below. See
  // buildPrefillDraft's docstring for the exact draft format.
  async function handleCardTap(entry: WorkbenchEntry) {
    setOpenError(null);
    try {
      const minted = await newSession(entry.project_id);
      if (!minted?.session_id) {
        setOpenError(t('workbench.error.open'));
        return;
      }
      setRoute({
        name: 'project',
        projectId: entry.project_id,
        tab: 'chat',
        sessionId: minted.session_id,
        draft: buildPrefillDraft(entry),
      });
    } catch {
      setOpenError(t('workbench.error.open'));
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
  const showToday = todayAvailability?.available === true && todayEvents.length > 0;

  // Today strip rows: privacy-filter (global mode), chronological, then ONE
  // row per automation trigger (a daily cron can emit multiple occurrences in
  // the window — the strip shows the next one, not several copies; the full
  // series stays on the Calendar). Memory/external events are never deduped.
  const todayRows = useMemo(() => {
    const seenTriggers = new Set<string>();
    return todayEvents
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
  }, [todayEvents, projects, projectId]);

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
        {showProjectChip && entries.length > 0 && (
          <select
            value={filterProjectId}
            onChange={(e) => setFilterProjectId(e.target.value)}
            aria-label={t('workbench.filter.all')}
            data-testid="workbench-project-filter"
            className="rounded-full border border-border bg-card px-3 py-1 text-xs text-secondary transition-all duration-150 focus:border-accent focus:outline-none"
          >
            <option value="">{t('workbench.filter.all')}</option>
            {filterOptions.map((o) => (
              <option key={o.id} value={o.id}>
                {o.name}
              </option>
            ))}
          </select>
        )}
        <button
          type="button"
          onClick={() => {
            setDeleteError(false);
            refetch();
          }}
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
          className="border-b border-warning/20 bg-warning/10 px-3 py-1.5 text-xs text-warning"
        >
          {t('workbench.conflict.notice')}
        </div>
      )}
      {openError && (
        <div className="border-b border-error/20 bg-error/5 px-3 py-1.5 text-xs text-error">
          {openError}
        </div>
      )}
      {deleteError && (
        <div
          data-testid="workbench-delete-error"
          className="border-b border-error/20 bg-error/10 px-3 py-1.5 text-xs text-error"
        >
          {t('workbench.exit.error')}
        </div>
      )}

      {showToday && todayRows.length > 0 && (
        <div data-testid="workbench-today" className="px-4 pt-3">
          {/* Same surface recipe as WorkbenchCard — radius, border, shadow.
              These two sit in the same column, so any difference in the
              container reads as "these are unrelated things"; previously the
              strip was a bare white block while the entries were bordered,
              double-shadowed, hover-lifting cards. */}
          <div className="rounded-xl border border-border/60 bg-card px-4 py-3 shadow-[0_1px_2px_rgb(0_0_0/0.04)]">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-xs font-semibold text-secondary">
                {t('workbench.today.title')}
              </span>
              <button
                type="button"
                onClick={handleOpenCalendar}
                data-testid="workbench-open-calendar"
                className="text-xs font-medium text-accent transition-colors duration-100 hover:text-accent/80"
              >
                {t('workbench.today.openCalendar')}
              </button>
            </div>
            <ul className="space-y-1">
              {todayRows.map((ev) => {
                // Round 3, item 4: the strip is a day OVERVIEW now (the
                // backend emits today's already-fired occurrences too), so a
                // past, non-all-day row dims rather than reading as still
                // upcoming. All-day rows have no meaningful "past" instant
                // within today and are never dimmed.
                const isPast = !ev.all_day && new Date(ev.start).getTime() < Date.now();
                return (
                  <li
                    key={ev.id}
                    data-testid="workbench-today-row"
                    data-past={isPast ? 'true' : 'false'}
                    className={`flex items-center gap-2.5 text-xs text-secondary ${isPast ? 'opacity-50' : ''}`}
                  >
                    <span className="w-16 shrink-0 font-mono text-[11px] tabular-nums text-muted">
                      {ev.all_day ? t('calendar.allDay') : formatTime(ev.start)}
                    </span>
                    <span className={`truncate ${isPast ? 'text-muted' : 'text-primary'}`}>{ev.title}</span>
                  </li>
                );
              })}
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
            className="rounded border border-border px-3 py-1 text-xs text-secondary hover:bg-card-hover hover:text-primary"
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
          <h2 className="text-base font-semibold tracking-[-0.01em] text-primary">
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
            <div className="flex flex-wrap items-center justify-center gap-2">
              <select
                value={migrateTarget}
                onChange={(e) => setMigrateTarget(e.target.value)}
                aria-label={t('workbench.empty.migratePicker')}
                data-testid="workbench-migrate-picker"
                className="rounded-full border border-border bg-card px-3.5 py-2 text-sm text-primary transition-all duration-150 focus:border-accent focus:outline-none"
              >
                <option value="">{t('workbench.empty.migratePicker')}</option>
                {projects
                  .filter((p) => p.workspace && !p.is_scratch)
                  .map((p) => (
                    <option key={p.project_id} value={p.project_id}>
                      {p.name}
                    </option>
                  ))}
              </select>
              <button
                type="button"
                onClick={() => migrateTarget && handleMigrate(migrateTarget)}
                disabled={!migrateTarget}
                data-testid="workbench-migrate-cta"
                className="rounded-full bg-accent px-4 py-2 text-sm font-medium text-white transition-[transform,background-color] duration-100 ease-out hover:bg-accent/90 active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-40 motion-reduce:transition-none motion-reduce:active:scale-100"
              >
                {t('workbench.empty.migrateCta')}
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-auto px-4 py-3" data-testid="workbench-list">
          <ul className="space-y-2.5">
            {filteredEntries.map((e, i) => (
              // Fold the list index into the key: a null/absent id (a stale
              // payload the backend heal didn't reach) would otherwise collapse
              // every id-less row onto `entry-<project>-null`, and the
              // duplicate key made React mis-reconcile a delete into a phantom
              // card (bug #45). The index keeps every key unique.
              <li key={`entry-${e.project_id}-${e.id ?? 'null'}-${i}`}>
                <WorkbenchCard
                  entry={e}
                  showProjectChip={showProjectChip}
                  projectName={projectName(e.project_id)}
                  onOpen={() => handleCardTap(e)}
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
