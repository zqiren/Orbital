// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Task 5 (spec 009 §0.5): renders where the management agent's `fanout` tool
// call appears in the timeline. One row per dispatched task; clicking a row
// drills into that worker's transcript (see SubAgentDrillIn.tsx). Status per
// task is a LIVE OVERLAY passed in by ChatView (a Map<fanoutId,
// Record<handle, {status, completedAtMs}>> fed by
// fanout.task_update/fanout.completed) rather than baked into the card's own
// item — this component stays a pure render of whatever the caller currently
// knows.
//
// Round 2 (2026-07-05), user-locked redesign: rendered in the CAPSULE design
// language — same visual family as the agent_run capsule (ChatView.tsx
// ~2643-2731) and its tool_call_row rows (ChatView.tsx ~245-291) — rather
// than as a standalone bordered/backgrounded card. "Special parallel tool
// calls," not a separate widget. Class strings below are lifted verbatim from
// those two blocks. data-testids (`fanout-card`, `fanout-row-*`,
// `fanout-status-*`) are unchanged by the restyle — ChatView's render merge
// and this file's own tests key off them.
//
// Round 2 also fixes issue 1 (per-task countdown): previously every row
// shared one batch-level `now - startedAtMs` countdown that never froze once
// an individual task finished — only the whole-batch join (`completedAtMs`)
// stopped the clock, so a completed row kept ticking until its slower
// siblings joined too. Each row now has its OWN optional `completedAtMs`
// (round-tripped from the backend's per-task terminal timestamp, or an
// arrival-time fallback stamped by ChatView for older daemons) and freezes
// independently. The shared 1s tick itself stops once nothing is left to
// tick for (see the `allTerminal` effect dependency below).

import { useEffect, useState } from 'react';
import { api } from '../config';
import { useT } from '../i18n/useT';
import type { FanoutTaskDescriptor, FanoutTaskStatus, JobRef } from '../types';
import type { StringKey } from '../i18n/strings';

/** True for every status except 'running' — the vocabulary has no other
 *  in-flight state (spec 009 §0.5). No such util existed in web/src before
 *  round 2; both FanoutCard (row/tick freeze) and ChatView (arrival-time
 *  completedAtMs fallback) need the same definition. */
export function isTerminal(status: FanoutTaskStatus): boolean {
  return status !== 'running';
}

/** Per-row live-overlay entry. `completedAtMs` is set once the row's own
 *  status goes terminal — either the backend's stamped time or (older
 *  daemons / historical backfill) an arrival-time fallback. Its absence
 *  means the row is still ticking against the shared clock. */
export interface FanoutTaskState {
  status: FanoutTaskStatus;
  completedAtMs?: number;
}

interface FanoutCardProps {
  fanoutId: string;
  tasks: FanoutTaskDescriptor[];
  /** handle -> {status, completedAtMs}. A task with no entry yet defaults to
   *  'running' (it was just dispatched and no update has arrived). */
  statuses: Record<string, FanoutTaskState>;
  /** Wall-clock ms the fanout was dispatched — the batch's shared start time
   *  every row's duration is measured from. */
  startedAtMs: number;
  /** Wall-clock ms the whole fanout joined (fanout.completed), or null/
   *  undefined while still in flight. Used as the header's batch duration
   *  and as the fallback freeze point for any row with no completedAtMs of
   *  its own yet. */
  completedAtMs?: number | null;
  onSelectTask: (handle: string, label: string) => void;
}

const STATUS_CLASS: Record<FanoutTaskStatus, string> = {
  running: 'bg-accent/15 text-primary border-accent/40 animate-pulse',
  completed: 'dark:bg-success/15 text-success-text border-success/40',
  error: 'dark:bg-error/15 text-error-text border-error/40',
  stalled: 'dark:bg-warning/15 text-warning-text border-warning/40',
  interrupted: 'dark:bg-idle/15 text-secondary border-idle/30',
};

const STATUS_LABEL_KEY: Record<FanoutTaskStatus, StringKey> = {
  running: 'fanout.status.running',
  completed: 'fanout.status.completed',
  error: 'fanout.status.error',
  stalled: 'fanout.status.stalled',
  interrupted: 'fanout.status.interrupted',
};

// Spec 134 Part B: background job rows reuse the fanout status vocabulary
// (completed = exit 0, error = failed, interrupted = cancelled / stopped
// because Orbital closed) but read as job states.
const JOB_STATUS_LABEL_KEY: Record<FanoutTaskStatus, StringKey> = {
  running: 'fanout.job.state.running',
  completed: 'fanout.job.state.done',
  error: 'fanout.job.state.failed',
  stalled: 'fanout.job.state.failed',
  interrupted: 'fanout.job.state.stopped',
};

type CancelState = 'idle' | 'pending' | 'failed';

function JobRow({ task, job, status, elapsedMs }: {
  task: FanoutTaskDescriptor;
  job: JobRef;
  status: FanoutTaskStatus;
  elapsedMs: number;
}) {
  const t = useT();
  const [cancel, setCancel] = useState<CancelState>('idle');
  const onCancel = () => {
    setCancel('pending');
    api(`/api/v2/projects/${encodeURIComponent(job.project_id)}/jobs/${encodeURIComponent(job.job_id)}/cancel`,
      { method: 'POST' })
      // The row's status flips when the daemon's task update arrives.
      .catch(() => setCancel('failed'));
  };
  const cancelKey: StringKey = cancel === 'pending'
    ? 'fanout.job.cancelling'
    : cancel === 'failed' ? 'fanout.job.cancelFailed' : 'fanout.job.cancel';
  return (
    <div data-testid={`fanout-row-${task.handle}`}
         className="mb-1 flex items-center gap-2 w-full font-mono text-[11px] text-secondary">
      <span className="flex-1 min-w-0 truncate text-primary font-medium" title={task.label}>
        {job.agent ? <span className="text-secondary">{job.agent} · </span> : null}
        {task.label}
      </span>
      <span
        data-testid={`fanout-status-${task.handle}`}
        data-status={status}
        className={`shrink-0 rounded-full border px-2 py-0.5 text-2xs leading-4 ${STATUS_CLASS[status]}`}
      >
        {t(JOB_STATUS_LABEL_KEY[status])}
      </span>
      <span className="shrink-0 w-14 text-right text-[11px] font-mono text-secondary">
        {formatElapsed(elapsedMs)}
      </span>
      {status === 'running' ? (
        <button
          type="button"
          data-testid={`job-cancel-${task.handle}`}
          onClick={onCancel}
          disabled={cancel === 'pending'}
          className="shrink-0 rounded border border-border px-1.5 py-0.5 text-2xs leading-4 text-secondary hover:text-primary hover:border-primary disabled:opacity-60 cursor-pointer disabled:cursor-default"
        >
          {t(cancelKey)}
        </button>
      ) : null}
    </div>
  );
}

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
  // second only while there's still something unfrozen to show: the batch
  // hasn't joined AND at least one row is still non-terminal. Once every row
  // is terminal (or the batch joins) nothing is left to tick — the tick would
  // otherwise run forever showing a duration nobody reads anymore.
  const [now, setNow] = useState(() => Date.now());
  const allTerminal =
    tasks.length > 0 && tasks.every((task) => isTerminal(statuses[task.handle]?.status ?? 'running'));
  useEffect(() => {
    if (completedAtMs != null || allTerminal) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [completedAtMs, allTerminal]);
  const batchDuration = formatElapsed(Math.max(0, (completedAtMs ?? now) - startedAtMs));
  const isJobGroup = tasks.some((task) => task.job);
  // Spec 134 Part B: a reloaded job card only knows what history rows said.
  // A job that already ended while its group still runs has no such row, so
  // rows not known to be finished take their state from the daemon once.
  const [daemonJobs, setDaemonJobs] = useState<Record<string, { status: FanoutTaskStatus; endedAtMs?: number | null }>>({});
  const jobProjectId = tasks.find((task) => task.job)?.job?.project_id;
  const needsJobStates = isJobGroup && tasks.some(
    (task) => task.job && !isTerminal(statuses[task.handle]?.status ?? 'running'));
  useEffect(() => {
    if (!needsJobStates || !jobProjectId) return;
    let alive = true;
    api<{ jobs?: Array<{ job_id: string; status: FanoutTaskStatus; ended_at_ms?: number | null }> }>(
      `/api/v2/projects/${encodeURIComponent(jobProjectId)}/jobs`)
      .then((body) => {
        if (!alive) return;
        const next: Record<string, { status: FanoutTaskStatus; endedAtMs?: number | null }> = {};
        for (const j of body?.jobs ?? []) next[j.job_id] = { status: j.status, endedAtMs: j.ended_at_ms };
        setDaemonJobs(next);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [needsJobStates, jobProjectId]);

  return (
    <div data-testid="fanout-card" data-fanout-id={fanoutId} className="ml-9 flex gap-[10px]">
      <div className="w-0.5 shrink-0 rounded-sm bg-border" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 w-full font-mono text-[11px] text-secondary">
          <span className="truncate text-primary font-medium">
            {isJobGroup
              ? t(tasks.length === 1 ? 'fanout.jobs.title.one' : 'fanout.jobs.title.other', { n: tasks.length })
              : t(tasks.length === 1 ? 'fanout.card.title.one' : 'fanout.card.title.other', { n: tasks.length })}
          </span>
          <span className="ml-auto shrink-0 text-[11px] font-mono text-secondary">{batchDuration}</span>
        </div>
        <div className="mt-2 pt-2 border-t border-border/30">
          {tasks.map((task) => {
            const entry = statuses[task.handle];
            const status = entry?.status ?? 'running';
            // Per-row freeze (issue 1): this row's own completedAtMs wins;
            // else fall back to the batch join time; else keep ticking.
            const rowElapsed = Math.max(0, (entry?.completedAtMs ?? completedAtMs ?? now) - startedAtMs);
            if (task.job) {
              const fetched = daemonJobs[task.handle];
              const jobStatus = entry && isTerminal(entry.status)
                ? entry.status
                : (fetched?.status ?? entry?.status ?? 'running');
              // A job's own recorded end (from history) beats the batch time
              // a history merge stamps on every row.
              const jobEnd = task.job.ended_at_ms ?? fetched?.endedAtMs ?? entry?.completedAtMs
                ?? completedAtMs ?? now;
              return <JobRow key={task.handle} task={task} job={task.job} status={jobStatus}
                             elapsedMs={Math.max(0, jobEnd - startedAtMs)} />;
            }
            return (
              <div key={task.handle} className="mb-1 font-mono text-[11px] text-secondary">
                <button
                  type="button"
                  data-testid={`fanout-row-${task.handle}`}
                  onClick={() => onSelectTask(task.handle, task.label)}
                  className="flex items-center gap-2 w-full text-left cursor-pointer hover:text-primary"
                >
                  <span className="flex-1 min-w-0 truncate text-primary font-medium">{task.label}</span>
                  <span
                    data-testid={`fanout-status-${task.handle}`}
                    data-status={status}
                    className={`shrink-0 rounded-full border px-2 py-0.5 text-2xs leading-4 ${STATUS_CLASS[status]}`}
                  >
                    {t(STATUS_LABEL_KEY[status])}
                  </span>
                  <span className="shrink-0 w-14 text-right text-[11px] font-mono text-secondary">
                    {formatElapsed(rowElapsed)}
                  </span>
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
