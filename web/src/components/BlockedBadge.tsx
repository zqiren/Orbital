// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { AlertTriangle } from 'lucide-react';
import { useBlockedCount } from '../hooks/useBlockedCount';
import { useT } from '../i18n/useT';

interface BlockedBadgeProps {
  onClick?: () => void;
}

/**
 * BlockedBadge — global cross-project "Blocked N" nav-row.
 *
 * Shows the total count of pending_approval sessions across all projects.
 * Intended for placement in the app-level Sidebar nav-row block above the
 * project list.
 *
 * When blockedCount === 0 the count pill is hidden and the row is subdued.
 * When blockedCount > 0 the count pill is shown with warning styling.
 */
export default function BlockedBadge({ onClick }: BlockedBadgeProps) {
  const t = useT();
  const { blockedCount, loading } = useBlockedCount();

  if (loading) return null;

  const hasBlocked = blockedCount > 0;

  return (
    <div
      role="region"
      aria-label={t(blockedCount === 1 ? 'blocked.aria.one' : 'blocked.aria.other', { n: blockedCount })}
      onClick={onClick}
      className={`flex items-center gap-2 px-2.5 py-1.5 rounded-[6px] text-[12.5px] transition-all duration-150 ${
        onClick ? 'cursor-pointer hover:bg-card-hover/50' : ''
      } ${hasBlocked ? 'text-primary' : 'text-secondary opacity-60'}`}
    >
      <AlertTriangle
        size={14}
        className={`shrink-0 ${hasBlocked ? 'text-warning' : 'text-secondary'}`}
        aria-hidden="true"
      />
      <span className="flex-1 font-medium">{t('sidebar.blocked')}</span>
      {hasBlocked && (
        <span
          data-testid="blocked-badge-pill"
          className="font-mono text-[10px] font-medium leading-none px-1.5 py-0.5 rounded-full text-warning bg-warning/10"
        >
          {blockedCount}
        </span>
      )}
    </div>
  );
}
