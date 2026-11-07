// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * PendingInputNotice — pending-input queue (spec 006 · v3).
 *
 * Rendered beneath the kept optimistic bubble whenever the user sent a message
 * to a session whose own turn is mid-flight: the backend returned 200
 * `queued_same_session` (`session._queue`) and the message drains when that
 * turn ends. Spec 155 removed the cross-session kind (and its Run now): every
 * session of a project runs when the user sends to it.
 *
 * The notice is tappable → recall the queued message into the composer and
 * dequeue it (the mobile equivalent of the desktop ↑ accelerator; spec v3
 * §11c + §12 R1). Recall is server-authoritative in the parent — if the message
 * already dispatched, the parent no-ops. When the entry carries attachments,
 * recall would half-restore chips, so editing is disabled (`canEdit === false`;
 * spec §12 R4).
 */

import { useState } from 'react';
import { Clock, Pencil } from 'lucide-react';
import { useT } from '../i18n/useT';

export interface PendingInputNoticeProps {
  /** Tap-to-edit (mobile equivalent of ↑): recall this queued message. */
  onEdit: () => Promise<void> | void;
  /** False when the entry has attachments — editing is disabled (no chip half-restore). */
  canEdit?: boolean;
}

export default function PendingInputNotice({
  onEdit,
  canEdit = true,
}: PendingInputNoticeProps) {
  const t = useT();
  const [error, setError] = useState<string | null>(null);

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

  return (
    <div
      data-testid="pending-input-notice"
      className="mb-3 surface-float-static px-4 py-3"
      role="status"
      aria-live="polite"
    >
      <button
        type="button"
        data-testid="pending-input-notice-edit"
        aria-label={t('pending.edit')}
        onClick={handleEdit}
        disabled={!canEdit}
        className="group flex w-full items-start gap-2 text-left rounded-xs -mx-1 px-1 py-0.5 enabled:hover:bg-card-hover enabled:cursor-pointer disabled:cursor-default transition-colors duration-150"
      >
        <Clock size={14} className="mt-0.5 shrink-0 text-secondary" />
        <span
          className="text-sm text-secondary group-enabled:group-hover:text-primary transition-colors duration-150"
          data-testid="pending-input-notice-body"
        >
          {t('pending.waiting.sameSession')}
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
    </div>
  );
}
