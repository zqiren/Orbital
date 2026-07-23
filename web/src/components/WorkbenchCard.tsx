// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * One row of the Workbench "ONE list, no bands" surface (spec §6). Renders
 * either a flagged `[user]` entry or a daemon-computed card from the same
 * merged list (`useWorkbench().items`).
 *
 * Entry card: sentence · project chip (global view only) · monospace age ·
 * receipt ("Why I believe this": evidence quote + from-session link,
 * expanded by default when `confidence === 'unconfirmed'`) · two exits
 * (Done/Got it, Not relevant). Computed card: text · age (from `since`) ·
 * Dismiss. Tapping the card body anywhere but a button is the doorway
 * (`onOpen`) — spec §5.3.
 */

import { useState } from 'react';
import { useT } from '../i18n/useT';

import type { WorkbenchListItem } from './workbench/types';

export interface WorkbenchCardProps {
  item: WorkbenchListItem;
  /** Show the project name chip — global (unlensed) view only. */
  showProjectChip: boolean;
  projectName?: string | null;
  /** "now" for age computation — overridable for deterministic tests. */
  now?: Date;
  /** Whole-card tap (doorway): spawn/resume and navigate to the project. */
  onOpen: () => void;
  /** Entry-only: fulfilled ("Done"/"Got it") or irrelevant ("Not relevant"). */
  onExit?: (kind: 'fulfilled' | 'irrelevant') => void;
  /** Computed-only. */
  onDismiss?: () => void;
}

function daysBetween(fromISO: string, to: Date): number {
  const from = new Date(fromISO);
  if (Number.isNaN(from.getTime())) return 0;
  return Math.floor((to.getTime() - from.getTime()) / 86_400_000);
}

/** Age badge text: "N days late" when overdue, else "waiting N days". Returns
 *  null when there's no usable date to compute from. */
function ageLabel(
  t: ReturnType<typeof useT>,
  item: WorkbenchListItem,
  now: Date,
): string | null {
  if (item.kind === 'entry') {
    const { overdue, due, age_days } = item.data;
    if (overdue && due) {
      const n = Math.max(1, daysBetween(due, now));
      return t(n === 1 ? 'workbench.age.late.one' : 'workbench.age.late.other', { n });
    }
    if (age_days != null) {
      return t(age_days === 1 ? 'workbench.age.waiting.one' : 'workbench.age.waiting.other', {
        n: age_days,
      });
    }
    return null;
  }
  const { type, since } = item.data;
  if (!since) return null;
  if (type === 'overdue') {
    const n = Math.max(1, daysBetween(since, now));
    return t(n === 1 ? 'workbench.age.late.one' : 'workbench.age.late.other', { n });
  }
  const n = Math.max(0, daysBetween(since, now));
  return t(n === 1 ? 'workbench.age.waiting.one' : 'workbench.age.waiting.other', { n });
}

export default function WorkbenchCard({
  item,
  showProjectChip,
  projectName,
  now,
  onOpen,
  onExit,
  onDismiss,
}: WorkbenchCardProps) {
  const t = useT();
  const isEntry = item.kind === 'entry';
  const entry = isEntry ? item.data : null;
  const computed = !isEntry ? item.data : null;
  const unconfirmed = isEntry && entry!.confidence === 'unconfirmed';
  const [expanded, setExpanded] = useState(unconfirmed);
  const effectiveNow = now ?? new Date();

  const text = isEntry ? entry!.text : computed!.text;
  const age = ageLabel(t, item, effectiveNow);
  const testId = isEntry
    ? `workbench-card-entry-${entry!.project_id}-${entry!.id}`
    : `workbench-card-computed-${computed!.project_id}-${computed!.type}-${computed!.key}`;

  const hasReceipt = isEntry && (entry!.evidence || entry!.from_session);

  return (
    <div
      role="button"
      tabIndex={0}
      data-testid={testId}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onOpen();
      }}
      className="flex flex-col gap-1.5 rounded-lg border border-border bg-card p-3 text-left transition-all duration-150 hover:border-secondary/40 cursor-pointer"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="min-w-0 flex-1 text-sm text-primary">{text}</p>
        {isEntry && entry!.overdue && (
          <span
            data-testid="workbench-card-overdue-badge"
            className="shrink-0 rounded-full bg-error/10 px-1.5 py-0.5 text-[10px] font-medium text-error"
          >
            {t('workbench.overdue')}
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[11px] text-secondary">
        {showProjectChip && projectName && (
          <span
            data-testid="workbench-card-project-chip"
            className="rounded-full bg-sidebar px-1.5 py-0.5 text-[10px] text-secondary"
          >
            {projectName}
          </span>
        )}
        {age && <span className="font-mono">{age}</span>}
      </div>

      {hasReceipt && (
        <div data-testid="workbench-card-receipt">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setExpanded((v) => !v);
            }}
            aria-expanded={expanded}
            data-testid="workbench-card-receipt-toggle"
            className="text-left text-[11px] font-medium text-secondary hover:text-primary"
          >
            {t('workbench.receipt.title')} {expanded ? '▾' : '▸'}
          </button>
          {expanded && (
            <div className="mt-1 space-y-1 border-l-2 border-border pl-2 text-[11px] text-secondary">
              {entry!.evidence && <p className="italic">&ldquo;{entry!.evidence}&rdquo;</p>}
              {entry!.from_session && (
                <p>
                  {t('workbench.receipt.from')}{' '}
                  <button
                    type="button"
                    data-testid="workbench-card-from-session"
                    onClick={(e) => {
                      e.stopPropagation();
                      onOpen();
                    }}
                    className="underline hover:text-primary"
                  >
                    {entry!.from_session}
                  </button>
                </p>
              )}
              {entry!.due && <p>{t('workbench.receipt.due', { due: entry!.due })}</p>}
            </div>
          )}
        </div>
      )}

      <div className="mt-1 flex items-center gap-2">
        {isEntry ? (
          <>
            <button
              type="button"
              data-testid="workbench-card-exit-fulfilled"
              onClick={(e) => {
                e.stopPropagation();
                onExit?.('fulfilled');
              }}
              className="rounded-md border border-border px-2.5 py-1 text-[12px] font-medium text-secondary hover:bg-card-hover hover:text-primary"
            >
              {t('workbench.exit.fulfilled')}
            </button>
            <button
              type="button"
              data-testid="workbench-card-exit-irrelevant"
              onClick={(e) => {
                e.stopPropagation();
                onExit?.('irrelevant');
              }}
              className="rounded-md border border-border px-2.5 py-1 text-[12px] font-medium text-secondary hover:bg-card-hover hover:text-primary"
            >
              {t('workbench.exit.irrelevant')}
            </button>
          </>
        ) : (
          <button
            type="button"
            data-testid="workbench-card-dismiss"
            onClick={(e) => {
              e.stopPropagation();
              onDismiss?.();
            }}
            className="rounded-md border border-border px-2.5 py-1 text-[12px] font-medium text-secondary hover:bg-card-hover hover:text-primary"
          >
            {t('workbench.dismiss')}
          </button>
        )}
      </div>
    </div>
  );
}
