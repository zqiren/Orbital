// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * PendingInputNotice — pending-input queue (spec 006, Path A).
 *
 * Rendered beneath the kept optimistic bubble whenever the user sent a message
 * to session B while session A holds the project's single active-loop slot. The
 * backend returned 202 `queued_pending_slot`: the message was ACCEPTED and will
 * auto-dispatch when the slot frees. Two affordances:
 *
 *   - [Run now] — cancels the HOLDER (A) only (spec §3h F5). No re-inject: the
 *     queued message dispatches itself once the slot frees, so re-injecting
 *     would double-send. The parent wires this to `cancelMessage(holder)`.
 *   - [Stop waiting] — cancels this queued entry (parent wires it to
 *     `cancelPendingInput(projectId, sessionId, nonce)`).
 *
 * Callback-driven: the parent owns the registry + bubble removal; this
 * component only surfaces the wait copy, the two buttons, and an inline error.
 */

import { useState } from 'react';
import { Clock } from 'lucide-react';
import { useT } from '../i18n/useT';

export interface PendingInputNoticeProps {
  /** The slot holder (session A) — a human title when available, else the id. */
  holder: string;
  /** [Run now]: cancel the holder ONLY. No inject (the queued msg self-dispatches). */
  onRunNow: () => Promise<void> | void;
  /** [Stop waiting]: cancel this queued pending entry. */
  onStopWaiting: () => Promise<void> | void;
}

export default function PendingInputNotice({
  holder,
  onRunNow,
  onStopWaiting,
}: PendingInputNoticeProps) {
  const t = useT();
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busy = running || stopping;

  async function handleRunNow() {
    if (busy) return;
    setError(null);
    setRunning(true);
    try {
      await onRunNow();
      // On success the holder's turn is cancelled; the slot frees and the
      // backend auto-dispatches this message. The parent unmounts this notice
      // when `chat.pending_dispatched` clears the nonce — leave `running` set.
    } catch {
      setError(t('pending.cancelError'));
      setRunning(false);
    }
  }

  async function handleStopWaiting() {
    if (busy) return;
    setError(null);
    setStopping(true);
    try {
      await onStopWaiting();
      // On success the parent removes the bubble + this notice. Leave `stopping`.
    } catch {
      setError(t('pending.cancelError'));
      setStopping(false);
    }
  }

  return (
    <div
      data-testid="pending-input-notice"
      className="mb-3 rounded-[6px] border border-border bg-background px-4 py-3 shadow-lg"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-start gap-2">
        <Clock size={14} className="mt-0.5 shrink-0 text-secondary" />
        <p
          className="text-sm text-secondary"
          data-testid="pending-input-notice-body"
        >
          {t('pending.waiting.body', { holder })}
        </p>
      </div>
      {error && (
        <p
          className="text-sm text-error mt-2"
          data-testid="pending-input-notice-error"
        >
          {error}
        </p>
      )}
      <div className="mt-3 flex gap-2">
        <button
          type="button"
          data-testid="pending-input-notice-run-now"
          onClick={handleRunNow}
          disabled={busy}
          className="shrink-0 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide bg-accent text-white hover:bg-accent/85 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150"
        >
          {running ? t('pending.running') : t('pending.runNow')}
        </button>
        <button
          type="button"
          data-testid="pending-input-notice-stop"
          onClick={handleStopWaiting}
          disabled={busy}
          className="shrink-0 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide border border-border bg-background text-secondary hover:bg-card-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150"
        >
          {stopping ? t('pending.stopping') : t('pending.stopWaiting')}
        </button>
      </div>
    </div>
  );
}
