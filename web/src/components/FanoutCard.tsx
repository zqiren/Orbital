// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Task 5 (spec 009 §0.5): renders where the management agent's `fanout` tool
// call appears in the timeline. One row per dispatched task; clicking a row
// drills into that worker's transcript (see SubAgentDrillIn.tsx). Status per
// task is a LIVE OVERLAY passed in by ChatView (a Map<fanoutId,
// Record<handle,status>> fed by fanout.task_update/fanout.completed) rather
// than baked into the card's own item — this component stays a pure render of
// whatever the caller currently knows.

import { useEffect, useState } from 'react';
import { useT } from '../i18n/useT';
import type { FanoutTaskDescriptor, FanoutTaskStatus } from '../types';
import type { StringKey } from '../i18n/strings';

interface FanoutCardProps {
  fanoutId: string;
  tasks: FanoutTaskDescriptor[];
  /** handle -> status. A task with no entry yet defaults to 'running' (it was
   *  just dispatched and no update has arrived). */
  statuses: Record<string, FanoutTaskStatus>;
  /** Wall-clock ms the fanout was dispatched — the batch's shared start time.
   *  The WS contract carries no PER-TASK timestamps, so duration is shown at
   *  the batch level (elapsed since dispatch), not per row. */
  startedAtMs: number;
  /** Wall-clock ms the whole fanout joined (fanout.completed), or null/undefined
   *  while still in flight — duration keeps ticking against Date.now(). */
  completedAtMs?: number | null;
  onSelectTask: (handle: string, label: string) => void;
}

const STATUS_CLASS: Record<FanoutTaskStatus, string> = {
  running: 'bg-blue-500/15 text-blue-400 border-blue-500/40 animate-pulse',
  completed: 'bg-green-500/15 text-green-400 border-green-500/40',
  error: 'bg-red-500/15 text-red-400 border-red-500/40',
  stalled: 'bg-amber-500/15 text-amber-400 border-amber-500/40',
  interrupted: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
};

const STATUS_LABEL_KEY: Record<FanoutTaskStatus, StringKey> = {
  running: 'fanout.status.running',
  completed: 'fanout.status.completed',
  error: 'fanout.status.error',
  stalled: 'fanout.status.stalled',
  interrupted: 'fanout.status.interrupted',
};

function formatElapsed(ms: number): string {
  if (ms < 1000) return '<1s';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  return rs ? `${m}m ${rs}s` : `${m}m`;
}

export default function FanoutCard({
  fanoutId,
  tasks,
  statuses,
  startedAtMs,
  completedAtMs,
  onSelectTask,
}: FanoutCardProps) {
  const t = useT();
  // `Date.now()` is impure, so it can't be called during render (React
  // Compiler purity rule). Track "now" in state instead, ticking once a
  // second only while the batch hasn't joined yet — once completedAtMs is
  // set the duration is frozen and no ticking is needed.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (completedAtMs != null) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [completedAtMs]);
  const duration = formatElapsed(Math.max(0, (completedAtMs ?? now) - startedAtMs));

  return (
    <div
      data-testid="fanout-card"
      data-fanout-id={fanoutId}
      className="ml-9 rounded-lg border border-border bg-sidebar/60 overflow-hidden"
    >
      <div className="px-4 py-1.5 border-b border-border/60">
        <span className="text-[10.5px] uppercase tracking-[0.6px] text-secondary font-semibold">
          {t(tasks.length === 1 ? 'fanout.card.title.one' : 'fanout.card.title.other', { n: tasks.length })}
        </span>
      </div>
      <div className="divide-y divide-border/40">
        {tasks.map((task) => {
          const status = statuses[task.handle] ?? 'running';
          return (
            <button
              key={task.handle}
              type="button"
              data-testid={`fanout-row-${task.handle}`}
              onClick={() => onSelectTask(task.handle, task.label)}
              className="w-full flex items-center gap-2 px-4 py-2 text-left hover:bg-card-hover transition-colors duration-150 cursor-pointer"
            >
              <span className="flex-1 min-w-0 truncate text-sm text-primary">{task.label}</span>
              <span
                data-testid={`fanout-status-${task.handle}`}
                data-status={status}
                className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] leading-4 ${STATUS_CLASS[status]}`}
              >
                {t(STATUS_LABEL_KEY[status])}
              </span>
              <span className="shrink-0 w-14 text-right text-[11px] font-mono text-secondary">
                {duration}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
