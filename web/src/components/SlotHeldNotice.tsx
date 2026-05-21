// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SlotHeldNotice — intermediate UX for Track J Phase 1 slot enforcement.
 *
 * Rendered inline in the chat history when /inject returns 202 with the
 * `slot_held` payload (per TASK/DISPATCH-2026-05-22.md §5.4). Two affordances:
 *
 *   - [Wait] — primary, dismisses the notice. The user's typed message stays
 *     in the composer (the caller decides whether to restore it).
 *   - [Cancel running session and send] — secondary, calls the existing
 *     /cancel endpoint on the holding session's project, then re-injects
 *     the user's message.
 *
 * Explicitly intermediate. When the pending-input queue ships (next track
 * after UI design lands), the 202 response will mean "your message has been
 * accepted and will dispatch when the slot frees" — at that point this
 * notice is replaced by the full "Waiting to respond..." indicator.
 */

import { useState } from 'react';

export interface SlotHeldNoticeProps {
  holdingSessionId: string;
  /** Called when the user clicks [Wait] — dismisses the notice. */
  onWait: () => void;
  /**
   * Called when the user clicks [Cancel running session and send].
   * The implementation should cancel the running turn on the holding
   * session (via the existing /cancel endpoint) and then re-inject the
   * pending content. The component awaits this promise and surfaces any
   * error inline.
   */
  onCancelAndSend: () => Promise<void>;
}

export default function SlotHeldNotice({
  holdingSessionId,
  onWait,
  onCancelAndSend,
}: SlotHeldNoticeProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleCancelAndSend() {
    setBusy(true);
    setError(null);
    try {
      await onCancelAndSend();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to cancel and resend.');
      setBusy(false);
    }
    // On success, the parent will unmount this notice (replacing it with
    // the actual user message). No need to clear `busy`.
  }

  return (
    <div
      data-testid="slot-held-notice"
      className="mb-3 rounded-lg border border-secondary/30 bg-secondary/5 px-4 py-3"
      role="status"
      aria-live="polite"
    >
      <p className="text-sm text-secondary">
        Session{' '}
        <span
          data-testid="slot-held-notice-holder"
          className="font-semibold text-primary"
        >
          {holdingSessionId}
        </span>{' '}
        is currently running. Wait for it to finish, or cancel it to send
        this message now.
      </p>
      {error && (
        <p
          className="text-sm text-error mt-2"
          data-testid="slot-held-notice-error"
        >
          {error}
        </p>
      )}
      <div className="mt-3 flex gap-2">
        <button
          type="button"
          data-testid="slot-held-notice-wait"
          onClick={onWait}
          disabled={busy}
          className="rounded-md bg-accent/20 px-3 py-1.5 text-sm font-medium text-primary hover:bg-accent/30 disabled:opacity-50"
        >
          Wait
        </button>
        <button
          type="button"
          data-testid="slot-held-notice-cancel-and-send"
          onClick={handleCancelAndSend}
          disabled={busy}
          className="rounded-md border border-secondary/40 px-3 py-1.5 text-sm font-medium text-secondary hover:text-primary hover:border-secondary/60 disabled:opacity-50"
        >
          {busy ? 'Cancelling…' : 'Cancel running session and send'}
        </button>
      </div>
    </div>
  );
}
