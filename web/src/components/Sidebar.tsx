// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import { Calendar, Inbox } from 'lucide-react';
import type { Project, AgentRunStatus } from '../types';
import type { Route } from '../route';
import { api } from '../config';
import BlockedBadge from './BlockedBadge';
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
}

/** Nav-row count badge: total flagged entries across the (privacy-filtered)
 *  global Workbench. Fetched once on mount — the surface itself refetches
 *  live; the sidebar count is a light, best-effort signal, not required to
 *  stay millisecond-fresh (no WS event exists for it). */
function useWorkbenchCount(): number {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let cancelled = false;
    api<{ entries?: unknown[] }>('/api/v2/workbench')
      .then((d) => {
        if (!cancelled) setCount(d?.entries?.length ?? 0);
      })
      .catch(() => {
        if (!cancelled) setCount(0);
      });
    return () => {
      cancelled = true;
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
}: SidebarProps) {
  const t = useT();
  const selectedProjectId = route.name === 'project' ? route.projectId : null;
  const workbenchCount = useWorkbenchCount();

  // Quick Tasks (is_scratch) is promoted into the Workspace zone; the Projects
  // zone below lists only non-scratch projects so it is never duplicated.
  const scratchProjects = projects.filter((p) => p.is_scratch);
  const regularProjects = projects.filter((p) => !p.is_scratch);

  // Shared project-row renderer — identical status dot / summary / active-state
  // behavior wherever a project appears (Quick Tasks in Workspace, projects in
  // the Projects zone). Keeps Quick Tasks' promotion from changing its behavior.
  function renderProjectRow(project: Project) {
    const isActive = project.project_id === selectedProjectId;
    const dotColor = getProjectDotColor(project.project_id, agentStatuses, pendingApprovals);
    const summary = statusSummaries[project.project_id];
    return (
      <button
        key={project.project_id}
        onClick={() => onSelectProject(project.project_id)}
        className={`w-full text-left px-3 py-2 rounded-[6px] flex items-center gap-2.5 transition-all duration-150 max-md:min-h-[44px] ${
          isActive ? 'bg-card-hover' : 'hover:bg-card-hover/50'
        }`}
      >
        <span className={`w-2 h-2 rounded-full ${dotColor} shrink-0 mt-1.5`} />
        <div className="min-w-0 flex-1">
          <span className="font-mono text-[11.5px] font-medium text-primary block truncate">
            {truncate(project.name, 20)}
          </span>
          {summary && (
            <span className="text-[10px] text-secondary block truncate mt-0.5">
              {summary}
            </span>
          )}
        </div>
      </button>
    );
  }

  return (
    <aside className="w-[260px] shrink-0 bg-sidebar border-r border-border flex flex-col h-full max-md:w-full">
      {/* Wordmark */}
      <div className="px-4 pt-4 pb-3">
        <span className="font-mono text-sm font-semibold text-primary tracking-tight">
          {t('sidebar.wordmark')}
        </span>
      </div>

      {/* Global nav-row block */}
      <div className="px-2 pb-1 space-y-0.5">
        {/* Blocked row — reuses BlockedBadge restyled into V1MNavRow form */}
        <BlockedBadge />
      </div>

      {/* Workspace zone — global surfaces (Calendar) + Quick Tasks, above Projects */}
      <div className="flex items-center px-4 pt-2 pb-1">
        <span className="text-[9.5px] uppercase tracking-[0.08em] text-secondary font-medium">
          {t('workspace.zone.label')}
        </span>
      </div>
      <div className="px-2 pb-1 space-y-0.5">
        <button
          onClick={onSelectCalendar}
          aria-current={route.name === 'calendar' ? 'page' : undefined}
          className={`w-full text-left px-3 py-2 rounded-[6px] flex items-center gap-2.5 transition-all duration-150 max-md:min-h-[44px] ${
            route.name === 'calendar' ? 'bg-card-hover' : 'hover:bg-card-hover/50'
          }`}
        >
          <Calendar size={14} className="shrink-0 text-secondary" aria-hidden="true" />
          <span className="font-mono text-[11.5px] font-medium text-primary block truncate">
            {t('workspace.calendar.nav')}
          </span>
          <BetaBadge />
        </button>
        <button
          onClick={onSelectWorkbench}
          aria-current={route.name === 'workbench' ? 'page' : undefined}
          aria-label={t(workbenchCount === 1 ? 'workbench.badge.aria.one' : 'workbench.badge.aria.other', { n: workbenchCount })}
          className={`w-full text-left px-3 py-2 rounded-[6px] flex items-center gap-2.5 transition-all duration-150 max-md:min-h-[44px] ${
            route.name === 'workbench' ? 'bg-card-hover' : 'hover:bg-card-hover/50'
          }`}
        >
          <Inbox size={14} className="shrink-0 text-secondary" aria-hidden="true" />
          <span className="font-mono text-[11.5px] font-medium text-primary block truncate">
            {t('workspace.workbench.nav')}
          </span>
          <BetaBadge />
          {workbenchCount > 0 && (
            <span
              data-testid="workbench-badge-count"
              className="ml-auto font-mono text-[10px] font-medium leading-none px-1.5 py-0.5 rounded-full text-secondary bg-sidebar"
            >
              {workbenchCount}
            </span>
          )}
        </button>
        {scratchProjects.map(renderProjectRow)}
      </div>

      {/* Projects section header */}
      <div className="flex items-center justify-between px-4 pt-2 pb-1">
        <span className="text-[9.5px] uppercase tracking-[0.08em] text-secondary font-medium">
          {t('sidebar.projects')}
        </span>
        <span className="font-mono text-[9.5px] text-secondary">
          {regularProjects.length}
        </span>
      </div>

      {/* Project list (non-scratch only) */}
      <nav className="flex-1 overflow-y-auto px-2">
        {regularProjects.map(renderProjectRow)}
      </nav>

      {/* Bottom section */}
      <div className="px-3 pb-3 pt-2 border-t border-border space-y-2">
        <button
          onClick={onNewProject}
          className="w-full text-sm font-medium text-primary border border-border rounded-[6px] px-3 py-2 hover:bg-card-hover transition-all duration-150 max-md:min-h-[44px]"
        >
          {t('app.newProject')}
        </button>

        <button
          onClick={onSettings}
          aria-current={route.name === 'settings' ? 'page' : undefined}
          className={`w-full text-sm px-3 py-1.5 text-left rounded-lg transition-all duration-150 max-md:min-h-[44px] ${
            route.name === 'settings'
              ? 'bg-card-hover text-primary'
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
