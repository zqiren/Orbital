// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { ChevronRight, Clock, Play } from 'lucide-react';
import type { Trigger } from '../types';
import { useT } from '../i18n/useT';

export type TasksPane = 'queue' | 'automations';

interface TasksStatusLineProps {
  /** Queue items running now. */
  running: number;
  /** Queue items waiting their turn. */
  queued: number;
  triggers: Trigger[];
  /** Open the Tasks page on this pane. */
  onOpen: (pane: TasksPane) => void;
  /**
   * Where it sits (spec 137, the folder shell):
   * - `row` (default): one rounded row in the project column, between the
   *   state row and the sessions. No lines; it fills like a chosen row while
   *   the Tasks page is open (`selected`).
   * - `band`: the phone, which has no column: a full-width line under the
   *   compact header, as the 1027-C build drew it everywhere.
   */
  variant?: 'row' | 'band';
  /** The Tasks page is open (row variant): the line shows as the chosen row. */
  selected?: boolean;
  /** Show the `/automate` hint when nothing is scheduled. False on a
   *  keyless project (2026-11-09): `/automate` needs Orbital's own agent, so
   *  the empty state keeps only its title. */
  teach?: boolean;
}

/** Stand-in for the `/automate` chip inside the translated teaching line. */
const COMMAND_SLOT = '\u0001';

/**
 * The project's status line — spec 137: the ONLY way into the
 * Tasks page now that the tab bar is gone (the user rejected a Tasks row in
 * the session column and a header icon). It always renders:
 *
 * - something running or queued, or any automation: `▶ N running · M queued ·
 *   ⏱ K automations · J off · Tasks ›`. The off-count stays glanceable (a
 *   paused automation is a silent failure mode).
 * - nothing at all: a quiet teaching line pointing at `/automate` (decision 6),
 *   so the queue stays reachable and the tour has something to anchor on.
 *
 * Click → the Tasks page: the Queue pane when something is running or
 * queued, otherwise Automations. Without the queue half of this gate a
 * project with queued work and no automations had no way to reach its queue.
 */
export default function TasksStatusLine({
  running,
  queued,
  triggers,
  onOpen,
  variant = 'row',
  selected = false,
  teach = true,
}: TasksStatusLineProps) {
  const row = variant === 'row';
  const t = useT();
  const offCount = triggers.filter((tr) => !tr.enabled).length;
  const busy = running + queued > 0;
  const empty = !busy && triggers.length === 0;
  const pane: TasksPane = busy ? 'queue' : 'automations';
  const state = empty ? 'empty' : busy ? 'busy' : 'automations';

  const dot = (
    <span className="text-muted" aria-hidden="true">
      ·
    </span>
  );

  // The facts, in order, as translated text — rendered as spans below and
  // joined for the column row's `title` (the row truncates; the tooltip
  // carries everything).
  const facts: { key: string; text: string; icon?: 'play' | 'clock'; off?: string }[] = [];
  if (running > 0) facts.push({ key: 'running', text: t('statusLine.running', { n: running }), icon: 'play' });
  if (queued > 0) facts.push({ key: 'queued', text: t('statusLine.queued', { n: queued }) });
  if (triggers.length > 0) {
    facts.push({
      key: 'automations',
      text: t(triggers.length === 1 ? 'statusLine.automations.one' : 'statusLine.automations.other', {
        n: triggers.length,
      }),
      icon: 'clock',
      off: offCount > 0 ? t('trigger.summary.off', { n: offCount }) : undefined,
    });
  }

  const factNodes = facts.flatMap((f, i) => {
    const node = (
      <span key={f.key} className="flex items-center gap-1.5 shrink-0">
        {f.icon === 'play' && <Play size={11} aria-hidden="true" className="shrink-0 text-success" fill="currentColor" />}
        {f.icon === 'clock' && <Clock size={13} aria-hidden="true" className="shrink-0 text-muted" />}
        <span className="text-primary">{f.text}</span>
        {f.off && <span className="text-muted">{f.off}</span>}
      </span>
    );
    return i === 0 ? [node] : [<span key={`dot-${i}`}>{dot}</span>, node];
  });

  // The teaching text splits around the `/automate` token so the token can be
  // styled as a key. The phone's full-width band keeps the one sentence; the
  // column (260px, 220 narrow) says it as a two-line item instead — a title
  // and a hint, like a session row — so nothing wraps or ends in an ellipsis.
  const [before, after = ''] = t(row ? 'statusLine.teach.hint' : 'statusLine.teach', {
    command: COMMAND_SLOT,
  }).split(COMMAND_SLOT);
  const keyToken = (extra: string) => (
    <kbd className={`mx-0.5 rounded border border-border px-1 text-secondary ${extra}`}>/automate</kbd>
  );

  if (row) {
    // The facts: ONE line, then the chevron; anything that does not fit ends
    // in an ellipsis and the facts are the tooltip. Nothing scheduled: two
    // deliberate lines (title, then the /automate hint under it). A Float
    // card (the Workbench recipe, index.css): a raised white layer, so the
    // automations read as their own group on the tinted column without a
    // line; hover lifts it; while Tasks is open an accent ring marks it chosen.
    const title = empty
      ? (teach ? `${t('statusLine.teach.title')} · ${before}/automate${after}` : t('statusLine.teach.title'))
      : facts.map((f) => (f.off ? `${f.text} ${f.off}` : f.text)).join(' · ');
    return (
      <button
        type="button"
        onClick={() => onOpen(pane)}
        aria-current={selected ? 'page' : undefined}
        title={title}
        data-testid="trigger-summary"
        data-tour="status-line"
        data-state={state}
        className={`mx-2 mt-1.5 mb-1 flex min-w-0 gap-2 surface-float px-2 py-[7px] text-left text-xs text-secondary whitespace-nowrap focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
          empty ? 'items-start' : 'items-center'
        } ${selected ? 'ring-1 ring-accent/45' : ''}`}
      >
        {empty ? (
          <>
            <Clock size={13} aria-hidden="true" className="mt-px shrink-0 text-muted" />
            <span className="flex min-w-0 flex-1 flex-col gap-0.5" data-testid="status-line-teach">
              <span className="flex items-center gap-2">
                <span className="min-w-0 flex-1 text-secondary" data-testid="status-line-teach-title">
                  {t('statusLine.teach.title')}
                </span>
                <ChevronRight size={13} aria-hidden="true" className="shrink-0 text-muted" />
              </span>
              {teach && (
                <span className="text-[11px] text-muted" data-testid="status-line-teach-hint">
                  {before}
                  {keyToken('font-sans text-[10px]')}
                  {after}
                </span>
              )}
            </span>
          </>
        ) : (
          <>
            <span className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">{factNodes}</span>
            <ChevronRight size={13} aria-hidden="true" className="shrink-0 text-muted" />
          </>
        )}
      </button>
    );
  }

  const content = empty ? (
    <>
      <Clock size={13} aria-hidden="true" className="shrink-0 text-muted" />
      <span className="min-w-0 truncate text-muted" data-testid="status-line-teach">
        {teach ? (
          <>
            {before}
            {keyToken('font-mono text-[11px]')}
            {after}
          </>
        ) : t('statusLine.teach.title')}
      </span>
    </>
  ) : (
    factNodes
  );

  return (
    <div className="border-y border-border">
      <button
        type="button"
        onClick={() => onOpen(pane)}
        data-testid="trigger-summary"
        data-tour="status-line"
        data-state={state}
        className="flex w-full min-w-0 items-center gap-2 px-6 py-1.5 text-left text-xs text-secondary transition-colors duration-100 hover:bg-sidebar/50 max-md:min-h-[44px] max-md:px-4"
      >
        {content}
        <span className="ml-auto flex shrink-0 items-center gap-1 pl-2 text-muted">
          {t('projectDetail.tab.queue')}
          <ChevronRight size={13} aria-hidden="true" className="shrink-0" />
        </span>
      </button>
    </div>
  );
}
