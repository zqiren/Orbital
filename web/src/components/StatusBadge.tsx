// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { AgentRunStatus } from '../types';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

type BadgeStatus = AgentRunStatus | 'needs_input';

interface StatusBadgeProps {
  status: BadgeStatus;
  size?: 'sm' | 'md';
}

// Readable status tokens, not the bright fills (spec 145: ~2:1 on light).
const STATUS_CONFIG: Record<BadgeStatus, { color: string; labelKey: StringKey }> = {
  running: { color: 'bg-success-text', labelKey: 'status.active' },
  waiting: { color: 'bg-success-text', labelKey: 'status.waiting' },
  // Spec 081: a first message waiting for the project slot. Reached only if
  // a queued session is ever routed through the project-header badge (today
  // it renders the agent's own status); the exhaustive Record demands it.
  queued: { color: 'bg-accent-text', labelKey: 'session.status.queued' },
  idle: { color: 'bg-idle', labelKey: 'status.idle' },
  error: { color: 'bg-error-text', labelKey: 'status.error' },
  needs_input: { color: 'bg-warning-text', labelKey: 'status.needsInput' },
  new_session: { color: 'bg-idle', labelKey: 'status.idle' },
  pending_approval: { color: 'bg-warning-text', labelKey: 'status.awaitingApproval' },
};

export default function StatusBadge({ status, size = 'md' }: StatusBadgeProps) {
  const t = useT();
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.idle;
  const dotSize = size === 'sm' ? 'w-2 h-2' : 'w-2.5 h-2.5';

  if (size === 'sm') {
    // Dot only (spec 137 round 3: the state dot sits before the project name
    // in the column, like the dot before a session). The colour says it; the
    // word is the tooltip and what a screen reader announces.
    const label = t(config.labelKey);
    return (
      <span
        role="img"
        aria-label={label}
        title={label}
        data-testid="status-dot"
        data-status={status}
        className={`${dotSize} rounded-full ${config.color} shrink-0`}
      />
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`${dotSize} rounded-full ${config.color} shrink-0`} />
      <span className="text-xs text-secondary">{t(config.labelKey)}</span>
    </span>
  );
}
