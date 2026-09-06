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

export function getProjectDotColor(
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
