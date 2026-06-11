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
  const { blockedCount, budgetPausedProjects, loading } = useBlockedCount();

  if (loading) return null;

  const hasBlocked = blockedCount > 0;
  // Budget-paused projects are an ADDITIVE, reason-coded marker — they do NOT
  // fold into blockedCount (which stays pending-approval only, preserving the
  // existing approval aria/pill semantics). Default to [] so older callers /
  // test mocks that omit the field render the approval surface unchanged.
  const budgetCount = budgetPausedProjects?.length ?? 0;
  const hasBudget = budgetCount > 0;
  const active = hasBlocked || hasBudget;

  return (
    <div
      role="region"
      aria-label={t(blockedCount === 1 ? 'blocked.aria.one' : 'blocked.aria.other', { n: blockedCount })}
      onClick={onClick}
      className={`flex items-center gap-2 px-2.5 py-1.5 rounded-[6px] text-[12.5px] transition-all duration-150 ${
        onClick ? 'cursor-pointer hover:bg-card-hover/50' : ''
      } ${active ? 'text-primary' : 'text-secondary opacity-60'}`}
    >
      <AlertTriangle
        size={14}
        className={`shrink-0 ${active ? 'text-warning' : 'text-secondary'}`}
        aria-hidden="true"
      />
      <span className="flex-1 font-medium">{t('sidebar.blocked')}</span>
      {hasBudget && (
        <span
          data-testid="blocked-badge-budget-pill"
          aria-label={t(budgetCount === 1 ? 'blocked.budget.aria' : 'blocked.budget.aria.other', { n: budgetCount })}
          className="font-mono text-[10px] font-medium leading-none px-1.5 py-0.5 rounded-full text-error bg-error/10"
        >
          {t('blocked.budget.marker', { n: budgetCount })}
        </span>
      )}
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
