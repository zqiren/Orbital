// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useBlockedCount } from '../hooks/useBlockedCount';

interface BlockedBadgeProps {
  onClick?: () => void;
}

/**
 * BlockedBadge — global cross-project "Blocked N" badge.
 *
 * Shows the total count of pending_approval sessions across all projects.
 * Intended for placement in the app-level Sidebar above the project list.
 *
 * When blockedCount === 0 the count pill is hidden and the label is subdued.
 * When blockedCount > 0 the count pill is shown with warning styling.
 */
export default function BlockedBadge({ onClick }: BlockedBadgeProps) {
  const { blockedCount, loading } = useBlockedCount();

  if (loading) return null;

  const hasBlocked = blockedCount > 0;

  return (
    <div
      role="region"
      aria-label={`${blockedCount} session${blockedCount === 1 ? '' : 's'} blocked across all projects`}
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all duration-150 ${
        onClick ? 'cursor-pointer hover:bg-card-hover/50' : ''
      } ${hasBlocked ? '' : 'opacity-40'}`}
    >
      <span className="text-xs font-medium text-secondary">Blocked</span>
      {hasBlocked && (
        <span
          data-testid="blocked-badge-pill"
          className="font-mono text-[10px] font-semibold leading-none px-1.5 py-0.5 rounded-full text-warning bg-warning/10"
        >
          {blockedCount}
        </span>
      )}
    </div>
  );
}
