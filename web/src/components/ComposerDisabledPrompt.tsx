// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import { Loader2 } from 'lucide-react';

interface ComposerDisabledPromptProps {
  onPauseQueue: () => void | Promise<void>;
}

/**
 * Replaces the composer textarea when the queue is active (state === 'draining').
 * Renders an inline notice with a "Pause queue" button that stops the queue and
 * hands control back to the chat composer.
 */
export default function ComposerDisabledPrompt({
  onPauseQueue,
}: ComposerDisabledPromptProps) {
  const [pausing, setPausing] = useState(false);

  async function handlePause() {
    if (pausing) return;
    setPausing(true);
    try {
      await onPauseQueue();
    } finally {
      setPausing(false);
    }
  }

  return (
    <div
      role="region"
      aria-label="Composer unavailable — queue is active"
      className="flex items-center justify-between gap-3 px-3 py-2.5 bg-background border border-border rounded-lg shadow-lg"
      data-testid="composer-disabled-prompt"
    >
      <span className="text-sm text-secondary">
        Pause queue to start chatting.
      </span>
      <button
        type="button"
        onClick={handlePause}
        disabled={pausing}
        data-testid="composer-pause-queue-btn"
        className={`shrink-0 px-3 py-1 rounded-md text-xs font-semibold tracking-wide transition-colors duration-150 ${
          pausing
            ? 'bg-secondary/20 text-secondary/40 cursor-not-allowed'
            : 'bg-accent text-white hover:bg-accent/85 cursor-pointer'
        }`}
      >
        {pausing ? (
          <span className="flex items-center gap-1.5">
            <Loader2 size={12} className="animate-spin" />
            Pausing…
          </span>
        ) : (
          'Pause queue'
        )}
      </button>
    </div>
  );
}
