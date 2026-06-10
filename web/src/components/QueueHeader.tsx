// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Pause, Play } from 'lucide-react';
import type { QueueRunState, QueueSnapshot } from '../types';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

const QUEUE_STATE_LABEL_KEY: Record<QueueRunState, StringKey> = {
  running: 'queue.header.running',
  paused: 'queue.header.paused',
  idle: 'queue.header.idle',
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
  const t = useT();
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
                ? 'bg-accent/15 text-accent'
                : 'bg-secondary/15 text-secondary'
          }`}
          data-testid="queue-state-pill"
        >
          {QUEUE_STATE_LABEL_KEY[snapshot.state] ? t(QUEUE_STATE_LABEL_KEY[snapshot.state]) : snapshot.state}
        </span>
        <span data-testid="queue-count-running">{t('queue.count.running', { n: counts.running })}</span>
        <span data-testid="queue-count-queued">{t('queue.count.queued', { n: counts.queued })}</span>
        {counts.blocked > 0 && (
          <span className="text-warning" data-testid="queue-count-blocked">
            {t('queue.count.blocked', { n: counts.blocked })}
          </span>
        )}
        {counts.done > 0 && (
          <span data-testid="queue-count-done">{t('queue.count.done', { n: counts.done })}</span>
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
            <Play className="w-3.5 h-3.5" /> {t('queue.resume')}
          </>
        ) : (
          <>
            <Pause className="w-3.5 h-3.5" /> {t('queue.stop')}
          </>
        )}
      </button>
    </div>
  );
}
