// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * PendingInputNotice — pending-input queue (spec 006, Path A · v3).
 *
 * Rendered beneath the kept optimistic bubble whenever the user sent a message
 * that was ACCEPTED but whose turn hasn't started yet. Two kinds:
 *
 *   - `kind === 'cross'` — the project's single active-loop slot is held by
 *     ANOTHER session; the backend returned 202 `queued_pending_slot`. The
 *     message auto-dispatches when the slot frees. Offers a single [Run now]
 *     button that cancels the HOLDER only (the queued message self-dispatches —
 *     no re-inject, which would double-send; spec §3h F5).
 *   - `kind === 'same'` — THIS session's own turn is mid-flight; the backend
 *     returned 200 `queued` (same-session `session._queue`). No Run-now: the
 *     message drains automatically when the current turn ends.
 *
 * Both kinds are EDITABLE: the notice is tappable → recall the queued message
 * into the composer and dequeue it (the mobile equivalent of the desktop ↑
 * accelerator; spec v3 §11c + §12 R1). This replaces the v2 [Stop waiting]
 * button entirely. Recall is server-authoritative in the parent — if the
 * message already dispatched, the parent no-ops. When the entry carries
 * attachments, recall would half-restore chips, so editing is disabled
 * (`canEdit === false`; spec §12 R4).
 *
 * Callback-driven: the parent owns the registry + bubble removal; this
 * component surfaces the wait copy, the tap-to-edit affordance, the optional
 * Run-now button, and an inline error.
 */

import { useState } from 'react';
import { Clock, Pencil } from 'lucide-react';
import { useT } from '../i18n/useT';

export interface PendingInputNoticeProps {
  /** Cross-session (slot held elsewhere) vs same-session (own turn running). */
  kind: 'cross' | 'same';
  /** [Run now]: cancel the holder ONLY. Provided/rendered only for `'cross'`. */
  onRunNow?: () => Promise<void> | void;
  /** Tap-to-edit (mobile equivalent of ↑): recall this queued message. */
  onEdit: () => Promise<void> | void;
  /** False when the entry has attachments — editing is disabled (no chip half-restore). */
  canEdit?: boolean;
}

export default function PendingInputNotice({
  kind,
  onRunNow,
  onEdit,
  canEdit = true,
}: PendingInputNoticeProps) {
  const t = useT();
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRunNow() {
    if (running) return;
    setError(null);
    setRunning(true);
    try {
      await onRunNow?.();
      // On success the holder's turn is cancelled; the slot frees and the
      // backend auto-dispatches this message. The parent unmounts this notice
      // when `chat.pending_dispatched` clears the nonce — leave `running` set.
    } catch {
      setError(t('pending.cancelError'));
      setRunning(false);
    }
  }

  async function handleEdit() {
    if (!canEdit) return;
    setError(null);
    try {
      await onEdit();
      // On success the parent loads the text into the composer and removes the
      // bubble + this notice. On a server-authoritative no-op (already
      // dispatched) the parent also unmounts this notice — nothing to do here.
    } catch {
      setError(t('pending.cancelError'));
    }
  }

  const body =
    kind === 'cross'
      ? t('pending.waiting.crossSession')
      : t('pending.waiting.sameSession');

  return (
    <div
      data-testid="pending-input-notice"
      className="mb-3 rounded-[6px] border border-border bg-background px-4 py-3 shadow-lg"
      role="status"
      aria-live="polite"
    >
      <button
        type="button"
        data-testid="pending-input-notice-edit"
        aria-label={t('pending.edit')}
        onClick={handleEdit}
        disabled={!canEdit}
        className="group flex w-full items-start gap-2 text-left rounded-[4px] -mx-1 px-1 py-0.5 enabled:hover:bg-card-hover enabled:cursor-pointer disabled:cursor-default transition-colors duration-150"
      >
        <Clock size={14} className="mt-0.5 shrink-0 text-secondary" />
        <span
          className="text-sm text-secondary group-enabled:group-hover:text-primary transition-colors duration-150"
          data-testid="pending-input-notice-body"
        >
          {body}
        </span>
        {canEdit && (
          <Pencil
            size={12}
            className="mt-0.5 ml-auto shrink-0 text-secondary opacity-0 group-hover:opacity-100 transition-opacity duration-150"
            aria-hidden="true"
          />
        )}
      </button>
      {error && (
        <p
          className="text-sm text-error mt-2"
          data-testid="pending-input-notice-error"
        >
          {error}
        </p>
      )}
      {kind === 'cross' && (
        <div className="mt-3 flex gap-2">
          <button
            type="button"
            data-testid="pending-input-notice-run-now"
            onClick={handleRunNow}
            disabled={running}
            className="shrink-0 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide bg-accent text-white hover:bg-accent/85 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150"
          >
            {running ? t('pending.running') : t('pending.runNow')}
          </button>
        </div>
      )}
    </div>
  );
}
