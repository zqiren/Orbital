// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { ReactNode } from 'react';
import { useEffect, useMemo, useState } from 'react';
import { Calendar, GripVertical, Inbox } from 'lucide-react';
import {
  DndContext,
  MouseSensor,
  TouchSensor,
  closestCenter,
  useSensor,
  useSensors,
  type Announcements,
  type DragEndEvent,
} from '@dnd-kit/core';
import {
  SortableContext,
  arrayMove,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import type { Project, AgentRunStatus } from '../types';
import type { Route } from '../route';
import { api } from '../config';
import { useT } from '../i18n/useT';
import BetaBadge from './BetaBadge';

type ConnectionState = 'connected' | 'reconnecting' | 'disconnected' | 'daemon_offline';

interface SidebarProps {
  projects: Project[];
  agentStatuses: Record<string, AgentRunStatus>;
  statusSummaries: Record<string, string>;
  pendingApprovals: Record<string, number>;
  route: Route;
  connectionState: ConnectionState;
  onSelectProject: (id: string) => void;
  onSelectCalendar: () => void;
  onSelectWorkbench: () => void;
  onNewProject: () => void;
  onSettings: () => void;
  /** Persist a new manual order for the regular-projects block (spec 056).
   *  Receives the COMPLETE ordered id list. The caller applies it
   *  optimistically and reverts if the write fails. */
  onReorderProjects: (orderedIds: string[]) => Promise<unknown>;
}

// Width of the drag-handle gutter. The pinned Quick Tasks row is not
// draggable but reserves the same gutter, so every project name in the list
// still starts on one vertical line.
const HANDLE_GUTTER = 'w-5 max-md:w-7 shrink-0';

/** One draggable project row: the shared row button plus its drag handle. */
function SortableProjectRow({
  id,
  handleLabel,
  children,
}: {
  id: string;
  handleLabel: string;
  children: ReactNode;
}) {
  // `attributes` is deliberately NOT spread. It carries tabIndex=0,
  // role="button" and an aria-describedby pointing at dnd-kit's keyboard
  // instructions — all of which describe a keyboard lift/drop mode this app
  // does not have (spec 056 §6 decision 5 dropped the KeyboardSensor on
  // purpose). Advertising it would put a dead control in the tab order and
  // read the wrong instructions to a screen reader. Only the pointer
  // listeners and the activator ref are wired.
  const { listeners, setNodeRef, setActivatorNodeRef, transform, transition, isDragging } =
    useSortable({ id });
  return (
    <div
      ref={setNodeRef}
      style={{
        // x is zeroed rather than pulled in via @dnd-kit/modifiers'
        // restrictToVerticalAxis — same result for a one-column list, no
        // extra dependency.
        transform: transform ? `translate3d(0, ${Math.round(transform.y)}px, 0)` : undefined,
        transition,
      }}
      className={`group flex items-center ${isDragging ? 'relative z-10 opacity-60' : ''}`}
    >
      <span
        ref={setActivatorNodeRef}
        {...listeners}
        data-testid={`project-drag-handle-${id}`}
        title={handleLabel}
        aria-hidden="true"
        className={`${HANDLE_GUTTER} self-stretch flex items-center justify-center text-secondary cursor-grab active:cursor-grabbing touch-none opacity-0 group-hover:opacity-100 max-md:opacity-100 transition-opacity motion-reduce:transition-none`}
      >
        <GripVertical size={12} />
      </span>
      {children}
    </div>
  );
}

/** Nav-row count badge: total flagged entries across the (privacy-filtered)
 *  global Workbench. Fetched on mount and refetched whenever a
 *  `orbital:workbench-changed` event fires (dispatched by `useWorkbench.ts`
 *  after a successful exit/migrate) — event-driven, no polling. */
function useWorkbenchCount(): number {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let cancelled = false;
    function fetchCount() {
      api<{ entries?: unknown[] }>('/api/v2/workbench')
        .then((d) => {
          if (!cancelled) setCount(d?.entries?.length ?? 0);
        })
        .catch(() => {
          if (!cancelled) setCount(0);
        });
    }
    fetchCount();
    window.addEventListener('orbital:workbench-changed', fetchCount);
    return () => {
      cancelled = true;
      window.removeEventListener('orbital:workbench-changed', fetchCount);
    };
  }, []);
  return count;
}

function getProjectDotColor(
  projectId: string,
  agentStatuses: Record<string, AgentRunStatus>,
  pendingApprovals: Record<string, number>,
): string {
  const approvalCount = pendingApprovals[projectId] ?? 0;
  if (approvalCount > 0) return 'bg-warning';

  const status = agentStatuses[projectId] ?? 'idle';
  switch (status) {
    case 'running':
    case 'waiting':
      return 'bg-success';
    case 'error':
      return 'bg-error';
    default:
      return 'bg-idle';
  }
}

function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max) + '…';
}

export default function Sidebar({
  projects,
  agentStatuses,
  statusSummaries,
  pendingApprovals,
  route,
  connectionState,
  onSelectProject,
  onSelectCalendar,
  onSelectWorkbench,
  onNewProject,
  onSettings,
  onReorderProjects,
}: SidebarProps) {
  const t = useT();
  const selectedProjectId = route.name === 'project' ? route.projectId : null;
  const workbenchCount = useWorkbenchCount();

  // Quick Tasks (is_scratch) is pinned to the top of the Projects list; the
  // rest follow below a hairline so it is never duplicated. It sat in the
  // Workspace zone before, alongside Calendar and Workbench, which framed it
  // as a global surface rather than the project you can start working in
  // without setting anything up.
  const scratchProjects = projects.filter((p) => p.is_scratch);
  const regularProjects = projects.filter((p) => !p.is_scratch);
  const regularIds = regularProjects.map((p) => p.project_id);

  // Two sensors, no KeyboardSensor (spec 056 §6 decision 5).
  //   Mouse: a 5px distance threshold, so a plain click on a project never
  //          becomes a drag.
  //   Touch: a 250ms long-press with an 8px slop, so the full-screen mobile
  //          project list still scrolls under a normal swipe.
  // MouseSensor rather than PointerSensor: PointerSensor also claims touch
  // input, which would beat TouchSensor to the gesture and take the
  // long-press delay with it.
  const sensors = useSensors(
    useSensor(MouseSensor, { activationConstraint: { distance: 5 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 250, tolerance: 8 } }),
  );

  // Silence dnd-kit's built-in live-region announcements. They exist to narrate
  // the keyboard lift/move/drop cycle; with no KeyboardSensor there is no such
  // cycle, and a screen reader has no way to reach this interaction anyway
  // (accepted in §6 decision 5). Announcing a drag nobody can perform is noise.
  const accessibility = useMemo(
    () => ({
      announcements: {
        onDragStart: () => undefined,
        onDragOver: () => undefined,
        onDragEnd: () => undefined,
        onDragCancel: () => undefined,
      } satisfies Announcements,
      screenReaderInstructions: { draggable: '' },
    }),
    [],
  );

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const from = regularIds.indexOf(String(active.id));
    const to = regularIds.indexOf(String(over.id));
    if (from < 0 || to < 0) return;
    Promise.resolve(onReorderProjects(arrayMove(regularIds, from, to))).catch((e) => {
      // The caller restores the pre-drag order, so the snap-back is the
      // user-visible feedback. Caught here only so a failed write is never an
      // unhandled rejection.
      console.error('Failed to reorder projects', e);
    });
  }

  // Shared project-row renderer — identical status dot / summary / active-state
  // behavior for the pinned Quick Tasks row and every project below it. Being
  // pinned changes where the row sits, never how it looks or behaves.
  function renderProjectRow(project: Project) {
    const isActive = project.project_id === selectedProjectId;
    const dotColor = getProjectDotColor(project.project_id, agentStatuses, pendingApprovals);
    const summary = statusSummaries[project.project_id];
    return (
      <button
        key={project.project_id}
        onClick={() => onSelectProject(project.project_id)}
        className={`flex-1 min-w-0 text-left px-3 py-2 rounded-sm flex items-center gap-2.5 transition-all duration-150 max-md:min-h-[44px] ${
          isActive ? 'bg-nav-hover' : 'hover:bg-nav-hover/50'
        }`}
      >
        <span className={`w-2 h-2 rounded-full ${dotColor} shrink-0 mt-1.5`} />
        <div className="min-w-0 flex-1">
          <span className="text-xs font-medium text-primary block truncate">
            {truncate(project.name, 20)}
          </span>
          {summary && (
            <span className="text-2xs text-secondary block truncate mt-0.5">
              {summary}
            </span>
          )}
        </div>
      </button>
    );
  }

  // pt-titlebar: clears the macOS traffic lights, which float over this
  // column's top-left corner when the desktop shell runs frameless. Padding
  // (not margin) so bg-nav still fills the band. 0 off macOS.
  return (
    <aside className="w-[260px] shrink-0 bg-nav border-r border-border flex flex-col h-full max-md:w-full pt-titlebar">
      {/* Wordmark */}
      <div className="px-4 pt-4 pb-3 flex items-center gap-2">
        {/* Served straight from web/public (vite copies it to the dist root, so
            it resolves in the desktop bundle too) — no new asset, no import.
            Decorative: the text beside it already names the app, so alt=""
            keeps screen readers from announcing "Orbital" twice. */}
        <img
          src="/icon-192.png"
          alt=""
          aria-hidden="true"
          className="w-5 h-5 shrink-0"
        />
        {/* text-lg, not text-sm: at text-sm the wordmark was the same size as
            the nav rows below it and only font-weight separated them, so the
            app name read as one more list item. Matches the project title
            across the divider. */}
        <span className="text-lg font-semibold text-primary tracking-tight">
          {t('sidebar.wordmark')}
        </span>
      </div>

      {/* The global "Blocked" row used to sit here. It was removed because it
          read as a status readout rather than a control: it was rendered
          without an onClick, so it was not clickable, and at zero it greyed
          itself out — users took it for a state they were supposed to act on
          and had nowhere to go. Its pending-approval signal is not lost: a
          project awaiting approval already shows an amber dot in the list
          below (see getProjectDotColor). BlockedBadge itself is kept for the
          dedicated 'blocked' route it was always headed for. */}
      {/* Workspace zone — global surfaces (Calendar, Workbench), above Projects */}
      <div className="flex items-center px-4 pt-2 pb-1">
        <span className="text-2xs uppercase tracking-[0.08em] text-secondary font-medium">
          {t('workspace.zone.label')}
        </span>
      </div>
      <div className="px-2 pb-1 space-y-0.5">
        <button
          onClick={onSelectCalendar}
          aria-current={route.name === 'calendar' ? 'page' : undefined}
          className={`w-full text-left px-3 py-2 rounded-sm flex items-center gap-2.5 transition-all duration-150 max-md:min-h-[44px] ${
            route.name === 'calendar' ? 'bg-nav-hover' : 'hover:bg-nav-hover/50'
          }`}
        >
          <Calendar size={14} className="shrink-0 text-secondary" aria-hidden="true" />
          <span className="text-xs font-medium text-primary block truncate">
            {t('workspace.calendar.nav')}
          </span>
          <BetaBadge />
        </button>
        <button
          onClick={onSelectWorkbench}
          aria-current={route.name === 'workbench' ? 'page' : undefined}
          aria-label={t(workbenchCount === 1 ? 'workbench.badge.aria.one' : 'workbench.badge.aria.other', { n: workbenchCount })}
          className={`w-full text-left px-3 py-2 rounded-sm flex items-center gap-2.5 transition-all duration-150 max-md:min-h-[44px] ${
            route.name === 'workbench' ? 'bg-nav-hover' : 'hover:bg-nav-hover/50'
          }`}
        >
          <Inbox size={14} className="shrink-0 text-secondary" aria-hidden="true" />
          <span className="text-xs font-medium text-primary block truncate">
            {t('workspace.workbench.nav')}
          </span>
          <BetaBadge />
          {workbenchCount > 0 && (
            <span
              data-testid="workbench-badge-count"
              className="ml-auto font-mono text-2xs font-medium leading-none px-1.5 py-0.5 rounded-full text-secondary bg-nav-hover"
            >
              {workbenchCount}
            </span>
          )}
        </button>
      </div>

      {/* Projects section header */}
      <div className="flex items-center justify-between px-4 pt-2 pb-1">
        <span className="text-2xs uppercase tracking-[0.08em] text-secondary font-medium">
          {t('sidebar.projects')}
        </span>
        <span className="font-mono text-2xs text-secondary">
          {regularProjects.length}
        </span>
      </div>

      {/* Project list — Quick Tasks pinned above the rest.
          Deliberately the same row as any other project (same dot, same name
          treatment): position above a hairline is what marks it pinned, the
          way pinned items read everywhere else. Giving it its own icon made it
          look like a different kind of thing rather than the project you start
          in. The rule is suppressed when there are no regular projects, so a
          fresh install doesn't get a divider with nothing under it. */}
      <nav className="flex-1 overflow-y-auto px-2">
        {scratchProjects.length > 0 && (
          <div
            data-testid="pinned-projects"
            className={
              regularProjects.length > 0
                ? 'pb-1 mb-1 border-b border-border/60'
                : undefined
            }
          >
            {scratchProjects.map((project) => (
              // Not draggable — scratch-first is an invariant of the list, not
              // a position the user owns. It still reserves the handle gutter
              // so its name lines up with every row below the hairline.
              <div key={project.project_id} className="flex items-center">
                <span className={HANDLE_GUTTER} aria-hidden="true" />
                {renderProjectRow(project)}
              </div>
            ))}
          </div>
        )}
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          accessibility={accessibility}
          onDragEnd={handleDragEnd}
        >
          <SortableContext items={regularIds} strategy={verticalListSortingStrategy}>
            {regularProjects.map((project) => (
              <SortableProjectRow
                key={project.project_id}
                id={project.project_id}
                handleLabel={t('sidebar.reorder.handleAria', { name: project.name })}
              >
                {renderProjectRow(project)}
              </SortableProjectRow>
            ))}
          </SortableContext>
        </DndContext>
      </nav>

      {/* Bottom section */}
      <div className="px-3 pb-3 pt-2 border-t border-border space-y-2">
        <button
          onClick={onNewProject}
          className="w-full rounded-lg border border-border/60 bg-card/55 px-3 py-2 text-sm font-medium text-primary shadow-[0_1px_2px_rgb(0_0_0/0.05)] transition-[box-shadow,transform,background-color] duration-100 ease-out hover:bg-card/80 hover:shadow-[0_2px_5px_rgb(0_0_0/0.07)] active:scale-[0.98] motion-reduce:transition-none motion-reduce:active:scale-100 max-md:min-h-[44px]"
        >
          {t('app.newProject')}
        </button>

        <button
          onClick={onSettings}
          aria-current={route.name === 'settings' ? 'page' : undefined}
          className={`w-full text-sm px-3 py-1.5 text-left rounded-lg transition-all duration-150 max-md:min-h-[44px] ${
            route.name === 'settings'
              ? 'bg-nav-hover text-primary'
              : 'text-secondary hover:text-primary'
          }`}
        >
          {t('sidebar.settings')}
        </button>

        {/* Connection indicator */}
        <div className={`flex items-center gap-1.5 px-3 py-1 max-md:py-2 shrink-0${
          connectionState === 'disconnected' || connectionState === 'daemon_offline' ? ' bg-error/5 rounded-lg' : ''
        }`}>
          <span
            className={`w-1.5 h-1.5 rounded-full shrink-0 ${
              connectionState === 'connected'
                ? 'bg-success'
                : connectionState === 'reconnecting'
                  ? 'bg-warning animate-pulse'
                  : connectionState === 'daemon_offline'
                    ? 'bg-warning'
                    : 'bg-error'
            }`}
          />
          <span className={`text-xs ${
            connectionState === 'disconnected' || connectionState === 'daemon_offline' ? 'text-error' : 'text-secondary'
          }`}>
            {connectionState === 'connected'
              ? t('sidebar.conn.connected')
              : connectionState === 'reconnecting'
                ? t('sidebar.conn.reconnecting')
                : connectionState === 'daemon_offline'
                  ? t('sidebar.conn.daemonOffline')
                  : t('sidebar.conn.disconnected')}
          </span>
        </div>
      </div>
    </aside>
  );
}
