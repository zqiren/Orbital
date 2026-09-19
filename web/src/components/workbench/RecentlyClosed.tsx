// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * "Recently closed" (spec 089 §3.6): asks the agent or the memory editor
 * closed in the last 7 days — anyone who can quote the user may close an ask,
 * and everything not closed by the user is undoable here. Each row shows the
 * ask, who closed it and how, the quoted words it was closed with, and a
 * Reopen action. The user's own Done/Delete taps are not listed.
 *
 * The ask text and the note are agent-authored content — rendered as-is,
 * never translated. Expanded by default: a card that silently vanished from
 * the list above should be findable at a glance.
 */

import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useT } from '../../i18n/useT';
import { eventColor } from '../calendar/color';
import type { WorkbenchClosedAsk } from './types';

export interface RecentlyClosedProps {
  items: WorkbenchClosedAsk[];
  /** Global (unlensed) view: show each row's project. */
  showProjectChip: boolean;
  projectName: (projectId: string) => string | null;
  onReopen: (item: WorkbenchClosedAsk) => void;
}

type CloseLabelKey =
  | 'workbench.closed.done.agent'
  | 'workbench.closed.done.editor'
  | 'workbench.closed.dropped.agent'
  | 'workbench.closed.dropped.editor';

function closeLabelKey(item: WorkbenchClosedAsk): CloseLabelKey {
  const by = item.closed_by === 'editor' ? 'editor' : 'agent';
  return item.kind === 'dropped' ? `workbench.closed.dropped.${by}` : `workbench.closed.done.${by}`;
}

export default function RecentlyClosed({
  items,
  showProjectChip,
  projectName,
  onReopen,
}: RecentlyClosedProps) {
  const t = useT();
  const [open, setOpen] = useState(true);
  if (items.length === 0) return null;

  return (
    <section data-testid="workbench-recently-closed" className="mt-5">
      <button
        type="button"
        data-testid="workbench-closed-toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-xs font-semibold text-secondary transition-colors duration-100 hover:text-primary"
      >
        {open ? <ChevronDown size={14} aria-hidden="true" /> : <ChevronRight size={14} aria-hidden="true" />}
        <span>{t('workbench.closed.title')}</span>
        <span className="font-normal tabular-nums text-muted">({items.length})</span>
      </button>
      {open && (
        <>
          <p className="mb-2 mt-1 text-[11px] text-muted">{t('workbench.closed.hint')}</p>
          <ul className="space-y-1.5">
            {items.map((item) => {
              const name = showProjectChip ? projectName(item.project_id) : null;
              return (
                <li
                  key={`closed-${item.project_id}-${item.id}`}
                  data-testid="workbench-closed-row"
                  className="flex items-start gap-3 rounded-lg border border-border/40 bg-card/60 px-3 py-2"
                >
                  <div className="min-w-0 flex-1">
                    <p className="line-clamp-2 text-sm leading-snug text-secondary">{item.text}</p>
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted">
                      {name && (
                        <span className="inline-flex items-center gap-1.5 font-medium text-secondary">
                          <span
                            aria-hidden="true"
                            className="h-1.5 w-1.5 shrink-0 rounded-full"
                            style={{ backgroundColor: eventColor(item.project_id).border }}
                          />
                          {name}
                        </span>
                      )}
                      <span>{t(closeLabelKey(item))}</span>
                      {item.closed && <span className="tabular-nums">{item.closed}</span>}
                    </div>
                    {item.note && (
                      <p className="mt-0.5 line-clamp-2 text-[11px] italic text-muted">{item.note}</p>
                    )}
                  </div>
                  <button
                    type="button"
                    data-testid="workbench-closed-reopen"
                    onClick={() => onReopen(item)}
                    className="shrink-0 rounded-full px-2.5 py-1 text-xs font-medium text-accent transition-[transform,background-color] duration-100 ease-out hover:bg-accent/10 active:scale-[0.96] motion-reduce:transition-none motion-reduce:active:scale-100"
                  >
                    {t('workbench.closed.reopen')}
                  </button>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </section>
  );
}
