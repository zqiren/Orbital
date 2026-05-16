// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Send } from 'lucide-react';
import { useState } from 'react';

interface QueueComposerProps {
  onSubmit: (content: string, opts: { priority: number; review: boolean }) => Promise<void> | void;
  disabled?: boolean;
  hint?: string;
}

export default function QueueComposer({
  onSubmit,
  disabled,
  hint,
}: QueueComposerProps) {
  const [value, setValue] = useState('');
  const [pinned, setPinned] = useState(false);
  const [review, setReview] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const isEmpty = value.trim().length === 0;
  const canSubmit = !isEmpty && !disabled && !submitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      await onSubmit(value.trim(), { priority: pinned ? 1 : 0, review });
      setValue('');
      setPinned(false);
      setReview(false);
    } finally {
      setSubmitting(false);
    }
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSubmit();
    }
  };

  return (
    <div className="border-t border-border bg-surface p-3 max-md:p-2">
      {hint && (
        <p className="text-xs text-secondary mb-2 max-md:text-[11px]">{hint}</p>
      )}
      <div className="flex gap-2">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Queue a new task..."
          rows={2}
          disabled={disabled}
          className="flex-1 resize-none rounded-lg border border-border bg-surface px-3 py-2 text-sm text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent disabled:opacity-50"
          data-testid="queue-composer-input"
        />
        <button
          onClick={() => void handleSubmit()}
          disabled={!canSubmit}
          aria-label="Add to queue"
          data-testid="queue-composer-submit"
          className="px-3 self-end mb-0 rounded-lg bg-accent text-on-accent disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent/90 transition-colors h-[42px] flex items-center justify-center max-md:min-w-[44px]"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
      <div className="flex gap-3 mt-2 text-xs text-secondary">
        <label className="flex items-center gap-1 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={pinned}
            onChange={(e) => setPinned(e.target.checked)}
            className="w-3.5 h-3.5"
          />
          Pin to top
        </label>
        <label className="flex items-center gap-1 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={review}
            onChange={(e) => setReview(e.target.checked)}
            className="w-3.5 h-3.5"
          />
          Review before advance
        </label>
      </div>
    </div>
  );
}
