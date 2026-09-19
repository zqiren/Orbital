// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Shared project status → dot color mapping.
 *
 * Lifted out of Sidebar.tsx (spec 078 §9.3) so EdgeStrip's aggregate dot
 * (§4.4) and Sidebar's per-project rows compute the exact same color for the
 * exact same inputs — byte-for-byte, not two implementations that could
 * drift.
 */

import type { AgentRunStatus } from '../types';

/**
 * `subAgentsRunning` is run-status's `sub_agents_running` per project: a
 * worker has an open turn. It lights the dot for work no management turn
 * brackets — a pinned dispatch, a queue item assigned to a worker (spec 095).
 * It ranks below a manager error, so worker activity never hides a fault.
 */
export function getProjectDotColor(
  projectId: string,
  agentStatuses: Record<string, AgentRunStatus>,
  pendingApprovals: Record<string, number>,
  subAgentsRunning: Record<string, boolean> = {},
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
      return subAgentsRunning[projectId] ? 'bg-success' : 'bg-idle';
  }
}

/**
 * WS events after which a project's `sub_agents_running` may have changed:
 * every worker dispatch and terminal, plus manager status changes (a
 * management Stop tears its workers down without a worker event of its own).
 */
export const SUB_AGENTS_RUNNING_EVENTS = [
  'agent.status',
  'sub_agent.dispatched',
  'sub_agent.started',
  'sub_agent.completed',
  'sub_agent.error',
  'sub_agent.failed',
  'sub_agent.stopped',
  'sub_agent.turn_interrupted',
] as const;

export interface SubAgentsRunningSync {
  /** Refetch after an event. One request in flight per project; a burst
   *  while it runs collapses into one follow-up issued after it settles. */
  refresh: (projectId: string) => void;
  /** Register a run-status request made elsewhere (the hydration pass);
   *  hand its `sub_agents_running` to the returned function. */
  track: (projectId: string) => (running: boolean | undefined) => void;
}

/**
 * Keeps each project's worker flag in step with run-status, which stays the
 * single source of truth. Responses can come back out of order (hydration and
 * event refetches overlap, and the relay tunnel reorders), so a response
 * issued before one that was already applied is dropped. A missing field (an
 * older daemon) reads as false.
 */
export function createSubAgentsRunningSync(
  fetchFlag: (projectId: string) => Promise<boolean | undefined>,
  apply: (projectId: string, running: boolean) => void,
): SubAgentsRunningSync {
  const issued = new Map<string, number>();
  const applied = new Map<string, number>();
  const inFlight = new Set<string>();
  const queued = new Set<string>();

  function track(projectId: string) {
    const token = (issued.get(projectId) ?? 0) + 1;
    issued.set(projectId, token);
    return (running: boolean | undefined) => {
      if (token < (applied.get(projectId) ?? 0)) return;
      applied.set(projectId, token);
      apply(projectId, running === true);
    };
  }

  function refresh(projectId: string) {
    if (inFlight.has(projectId)) {
      queued.add(projectId);
      return;
    }
    inFlight.add(projectId);
    const settle = track(projectId);
    fetchFlag(projectId)
      .then(settle, () => {})
      .finally(() => {
        inFlight.delete(projectId);
        if (queued.delete(projectId)) refresh(projectId);
      });
  }

  return { refresh, track };
}
