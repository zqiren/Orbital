// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import { Pause, Play } from 'lucide-react';
import type { QueueRunState, QueueSnapshot } from '../types';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

const QUEUE_STATE_LABEL_KEY: Record<QueueRunState, StringKey> = {
  running: 'queue.header.running',
  paused: 'queue.header.paused',
  idle: 'queue.header.idle',
};

/** Seconds until the next 9:00 AM local time ("until tomorrow" snooze). */
function secondsUntilNextMorning(): number {
  const now = new Date();
  const next = new Date(now);
  next.setHours(9, 0, 0, 0);
  if (next <= now) next.setDate(next.getDate() + 1);
  return Math.round((next.getTime() - now.getTime()) / 1000);
}

/** "14:30" for same-day deadlines, "Jun 12, 09:00" otherwise. */
function formatResumeTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

interface QueueHeaderProps {
  snapshot: QueueSnapshot | null;
  onStop: (durationSeconds?: number) => void | Promise<void>;
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
  const [menuOpen, setMenuOpen] = useState(false);

  // Close on Escape (mirrors SessionThreeDotMenu's dropdown pattern).
  useEffect(() => {
    if (!menuOpen) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setMenuOpen(false);
    }
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [menuOpen]);

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
  const isIdle = snapshot.state === 'idle';
  const resumeHint =
    isPaused && snapshot.paused_until ? formatResumeTime(snapshot.paused_until) : '';

  const pick = (duration?: number) => {
    setMenuOpen(false);
    void onStop(duration);
  };

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
      <div className="flex items-center gap-3">
        {resumeHint && (
          <span className="text-xs text-secondary" data-testid="queue-autoresume-hint">
            {t('queue.header.autoResume', { time: resumeHint })}
          </span>
        )}
        {isPaused ? (
          <button
            onClick={() => void onResume()}
            disabled={disabled}
            data-testid="queue-resume-btn"
            className="flex items-center gap-1.5 text-sm font-medium rounded-lg px-3 py-1.5 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px] bg-success text-on-accent hover:bg-success/90"
          >
            <Play className="w-3.5 h-3.5" /> {t('queue.resume')}
          </button>
        ) : isIdle ? (
          counts.queued > 0 ? (
            <button
              onClick={() => void onResume()}
              disabled={disabled}
              data-testid="queue-start-btn"
              className="flex items-center gap-1.5 text-sm font-medium rounded-lg px-3 py-1.5 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px] bg-success text-on-accent hover:bg-success/90"
            >
              <Play className="w-3.5 h-3.5" /> {t('queue.start')}
            </button>
          ) : null
        ) : (
          <div className="relative">
            <button
              onClick={() => setMenuOpen((v) => !v)}
              disabled={disabled}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              data-testid="queue-stop-btn"
              className="flex items-center gap-1.5 text-sm font-medium rounded-lg px-3 py-1.5 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px] border border-border text-secondary hover:text-warning hover:border-warning/40"
            >
              <Pause className="w-3.5 h-3.5" /> {t('queue.stop')}
            </button>
            {menuOpen && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />
                <div
                  role="menu"
                  className="absolute right-0 top-full mt-1 z-50 min-w-[200px] rounded-lg border border-border bg-white shadow-lg py-1"
                  data-testid="queue-pause-menu"
                >
                  <button
                    role="menuitem"
                    onClick={() => pick(undefined)}
                    data-testid="queue-pause-until-resume"
                    className="w-full text-left text-sm px-3 py-2 hover:bg-secondary/10"
                  >
                    {t('queue.pause.menu.untilResume')}
                  </button>
                  <button
                    role="menuitem"
                    onClick={() => pick(3600)}
                    data-testid="queue-pause-1h"
                    className="w-full text-left text-sm px-3 py-2 hover:bg-secondary/10"
                  >
                    {t('queue.pause.menu.oneHour')}
                  </button>
                  <button
                    role="menuitem"
                    onClick={() => pick(secondsUntilNextMorning())}
                    data-testid="queue-pause-tomorrow"
                    className="w-full text-left text-sm px-3 py-2 hover:bg-secondary/10"
                  >
                    {t('queue.pause.menu.untilTomorrow')}
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
