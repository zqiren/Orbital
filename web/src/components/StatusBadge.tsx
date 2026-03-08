// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { AgentRunStatus } from '../types';

type BadgeStatus = AgentRunStatus | 'needs_input';

interface StatusBadgeProps {
  status: BadgeStatus;
  size?: 'sm' | 'md';
}

const STATUS_CONFIG: Record<BadgeStatus, { color: string; label: string }> = {
  running: { color: 'bg-success', label: 'Active' },
  waiting: { color: 'bg-success', label: 'Waiting for sub-agents...' },
  idle: { color: 'bg-idle', label: 'Idle' },
  stopped: { color: 'bg-idle', label: 'Idle' },
  error: { color: 'bg-error', label: 'Error' },
  needs_input: { color: 'bg-warning', label: 'Needs Input' },
  new_session: { color: 'bg-idle', label: 'Idle' },
  pending_approval: { color: 'bg-warning', label: 'Awaiting Approval' },
};

export default function StatusBadge({ status, size = 'md' }: StatusBadgeProps) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.idle;
  const dotSize = size === 'sm' ? 'w-2 h-2' : 'w-2.5 h-2.5';

  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`${dotSize} rounded-full ${config.color} shrink-0`} />
      {size === 'md' && (
        <span className="text-xs text-secondary">{config.label}</span>
      )}
    </span>
  );
}
