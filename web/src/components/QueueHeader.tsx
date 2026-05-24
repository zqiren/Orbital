// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Pause, Play } from 'lucide-react';
import type { QueueRunState, QueueSnapshot } from '../types';

const QUEUE_STATE_LABEL: Record<QueueRunState, string> = {
  running: 'Running',
  paused: 'Paused',
  idle: 'Idle',
};

interface QueueHeaderProps {
  snapshot: QueueSnapshot | null;
  onStop: () => void | Promise<void>;
  onResume: () => void | Promise<void>;
  disabled?: boolean;
}

export default function QueueHeader({
  snapshot,
  onStop,
  onResume,
  disabled,
}: QueueHeaderProps) {
  if (!snapshot) {
    return null;
  }

  const counts = {
    running: snapshot.items.filter((it) => it.state === 'running').length,
    queued: snapshot.items.filter((it) => it.state === 'queued').length,
    blocked: snapshot.items.filter((it) => it.state === 'blocked').length,
    done: snapshot.items.filter((it) => it.state === 'done').length,
  };

  const isPaused = snapshot.state === 'paused';

  return (
    <div className="flex items-center justify-between gap-3 px-6 py-3 border-b border-border max-md:px-4 max-md:flex-wrap">
      <div className="flex items-center gap-4 text-xs text-secondary flex-wrap">
        <span
          className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${
            isPaused
              ? 'bg-warning/15 text-warning'
              : snapshot.state === 'running'
                ? 'bg-info/15 text-info'
                : 'bg-secondary/15 text-secondary'
          }`}
          data-testid="queue-state-pill"
        >
          {QUEUE_STATE_LABEL[snapshot.state] ?? snapshot.state}
        </span>
        <span data-testid="queue-count-running">{counts.running} running</span>
        <span data-testid="queue-count-queued">{counts.queued} queued</span>
        {counts.blocked > 0 && (
          <span className="text-warning" data-testid="queue-count-blocked">
            {counts.blocked} blocked
          </span>
        )}
        {counts.done > 0 && (
          <span data-testid="queue-count-done">{counts.done} done</span>
        )}
      </div>
      <button
        onClick={() => void (isPaused ? onResume() : onStop())}
        disabled={disabled}
        data-testid={isPaused ? 'queue-resume-btn' : 'queue-stop-btn'}
        className={`flex items-center gap-1.5 text-sm font-medium rounded-lg px-3 py-1.5 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px] ${
          isPaused
            ? 'bg-success text-on-accent hover:bg-success/90'
            : 'border border-border text-secondary hover:text-warning hover:border-warning/40'
        }`}
      >
        {isPaused ? (
          <>
            <Play className="w-3.5 h-3.5" /> Resume
          </>
        ) : (
          <>
            <Pause className="w-3.5 h-3.5" /> Stop
          </>
        )}
      </button>
    </div>
  );
}
