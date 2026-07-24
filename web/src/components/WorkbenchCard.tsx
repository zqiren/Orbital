// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * One row of the Workbench "ONE list, no bands" surface (spec §6). Renders
 * either a flagged `[user]` entry or a daemon-computed card from the same
 * merged list (`useWorkbench().items`).
 *
 * Entry card: sentence · project chip (global view only) · age · receipt
 * ("Why I believe this": evidence quote + from-session reference, expanded by
 * default when `confidence === 'unconfirmed'`) · two exits (Done/Got it,
 * Not relevant). Computed card: text · age (from `since`) · Dismiss. Tapping
 * the card body anywhere but a button is the doorway (`onOpen`) — spec §5.3.
 *
 * Styling follows the Apple-design pass (2026-07-24): soft elevated cards,
 * feedback on press (scale, 100ms), size-specific tracking, motion respects
 * `prefers-reduced-motion` via Tailwind's `motion-reduce:` variants.
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
  /** broken_automation only: disable the silent trigger. */
  onDisable?: () => void;
}

/** Local day-diff — ONLY for the computed-card "waiting" fallback (broken
 *  automation / paused thread), which the backend has no age_days-equivalent
 *  for. Never used for a "late" count — that's always server-authoritative
 *  `days_late` (spec §7.3: browser tz can disagree with project tz). */
function daysBetween(fromISO: string, to: Date): number {
  const from = new Date(fromISO);
  if (Number.isNaN(from.getTime())) return 0;
  return Math.floor((to.getTime() - from.getTime()) / 86_400_000);
}

function lateLabel(t: ReturnType<typeof useT>, n: number): string {
  return t(n === 1 ? 'workbench.age.late.one' : 'workbench.age.late.other', { n });
}

function waitingLabel(t: ReturnType<typeof useT>, n: number): string {
  return t(n === 1 ? 'workbench.age.waiting.one' : 'workbench.age.waiting.other', { n });
}

/** Age badge text: "N days late" when overdue, else "waiting N days". The
 *  late count always comes from the server's `days_late` — never recomputed
 *  from `due`/`since` client-side. Returns null when there's nothing to show. */
function ageLabel(
  t: ReturnType<typeof useT>,
  item: WorkbenchListItem,
  now: Date,
): string | null {
  if (item.kind === 'entry') {
    const { overdue, age_days, days_late } = item.data;
    if (overdue && days_late != null) {
      return lateLabel(t, days_late);
    }
    if (age_days != null) {
      return waitingLabel(t, age_days);
    }
    return null;
  }
  const { type, since, days_late } = item.data;
  if (type === 'overdue' && days_late != null) {
    return lateLabel(t, days_late);
  }
  if (!since) return null;
  const n = Math.max(0, daysBetween(since, now));
  return waitingLabel(t, n);
}

/** Filled primary pill — the card's ONE emphasized action. */
const PRIMARY_BTN =
  'rounded-full bg-accent px-3.5 py-1.5 text-[12.5px] font-medium text-white ' +
  'transition-[transform,background-color] duration-100 ease-out hover:bg-accent/90 ' +
  'active:scale-[0.96] motion-reduce:transition-none motion-reduce:active:scale-100';

/** Quiet text action (secondary: Not relevant / Dismiss / Disable). */
const QUIET_BTN =
  'rounded-full px-2.5 py-1.5 text-[12.5px] font-medium text-secondary ' +
  'transition-[transform,background-color,color] duration-100 ease-out ' +
  'hover:bg-card-hover hover:text-primary active:scale-[0.96] ' +
  'motion-reduce:transition-none motion-reduce:active:scale-100';

export default function WorkbenchCard({
  item,
  showProjectChip,
  projectName,
  now,
  onOpen,
  onExit,
  onDismiss,
  onDisable,
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
      className="group flex cursor-pointer flex-col gap-2 rounded-2xl border border-border/60 bg-card px-4 py-3.5 text-left shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_16px_rgba(0,0,0,0.03)] transition-[transform,box-shadow,border-color] duration-150 ease-out hover:-translate-y-px hover:border-border hover:shadow-[0_2px_4px_rgba(0,0,0,0.05),0_8px_24px_rgba(0,0,0,0.05)] active:scale-[0.99] motion-reduce:transition-none motion-reduce:transform-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent/60"
    >
      <div className="flex items-start justify-between gap-3">
        <p className="line-clamp-3 min-w-0 flex-1 text-[15px] leading-snug tracking-[-0.01em] text-primary">
          {text}
        </p>
        {isEntry && entry!.overdue && (
          <span
            data-testid="workbench-card-overdue-badge"
            className="shrink-0 rounded-full bg-error/10 px-2 py-0.5 text-[11px] font-medium text-error"
          >
            {t('workbench.overdue')}
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {showProjectChip && projectName && (
          <span
            data-testid="workbench-card-project-chip"
            className="rounded-full bg-sidebar px-2 py-0.5 text-[11px] font-medium text-secondary"
          >
            {projectName}
          </span>
        )}
        {age && (
          <span className="font-mono text-[11px] tabular-nums text-muted">{age}</span>
        )}
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
            className="inline-flex items-center gap-1 text-left text-[12px] font-medium text-secondary transition-colors duration-100 hover:text-primary"
          >
            <span
              aria-hidden="true"
              className={`inline-block text-[9px] transition-transform duration-150 ease-out motion-reduce:transition-none ${expanded ? 'rotate-90' : ''}`}
            >
              ▶
            </span>
            {t('workbench.receipt.title')}
          </button>
          {expanded && (
            <div className="mt-1.5 space-y-1 rounded-xl bg-sidebar/60 px-3 py-2 text-[12px] leading-relaxed text-secondary">
              {entry!.evidence && <p className="italic">&ldquo;{entry!.evidence}&rdquo;</p>}
              {entry!.from_session && (
                <p>
                  {t('workbench.receipt.from')}{' '}
                  {/* Inert reference: no transcript viewer exists yet, and a
                      "source" link must not side-effect a session spawn. */}
                  <span
                    data-testid="workbench-card-from-session"
                    className="font-mono text-[11px]"
                  >
                    {entry!.from_session}
                  </span>
                </p>
              )}
              {entry!.due && <p>{t('workbench.receipt.due', { due: entry!.due })}</p>}
            </div>
          )}
        </div>
      )}

      <div className="mt-0.5 flex items-center gap-2">
        {isEntry ? (
          <>
            <button
              type="button"
              data-testid="workbench-card-exit-fulfilled"
              onClick={(e) => {
                e.stopPropagation();
                onExit?.('fulfilled');
              }}
              className={PRIMARY_BTN}
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
              className={QUIET_BTN}
            >
              {t('workbench.exit.irrelevant')}
            </button>
          </>
        ) : (
          <>
            {/* Spec §6: each computed type leads with its OWN doorway verb —
                the primary button and the whole-card tap do the same thing. */}
            <button
              type="button"
              data-testid="workbench-card-computed-primary"
              onClick={(e) => {
                e.stopPropagation();
                onOpen();
              }}
              className={PRIMARY_BTN}
            >
              {computed!.type === 'overdue'
                ? t('workbench.doItNow')
                : computed!.type === 'broken_automation'
                  ? t('workbench.repair')
                  : t('workbench.resume')}
            </button>
            {computed!.type === 'broken_automation' && onDisable && (
              <button
                type="button"
                data-testid="workbench-card-disable"
                onClick={(e) => {
                  e.stopPropagation();
                  onDisable();
                }}
                className={QUIET_BTN}
              >
                {t('workbench.disable')}
              </button>
            )}
            <button
              type="button"
              data-testid="workbench-card-dismiss"
              onClick={(e) => {
                e.stopPropagation();
                onDismiss?.();
              }}
              className={QUIET_BTN}
            >
              {t('workbench.dismiss')}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
