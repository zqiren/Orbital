// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Send } from 'lucide-react';
import { useState } from 'react';
import { useT } from '../i18n/useT';

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
  const t = useT();
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
    // While an IME (e.g. Pinyin) is composing, Enter commits the candidate —
    // it belongs to the input method, not to us. Don't submit mid-composition.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSubmit();
    }
  };

  return (
    <div className="border-t border-border bg-card p-3 max-md:p-2">
      <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 focus-within:border-accent transition-colors max-md:gap-1.5">
        <span className="font-mono text-muted text-sm shrink-0 select-none">+</span>
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          placeholder={t('queue.composer.placeholder')}
          rows={1}
          disabled={disabled}
          className="flex-1 resize-none bg-transparent leading-5 text-[13px] text-primary placeholder:text-secondary/60 focus:outline-none disabled:opacity-50"
          data-testid="queue-composer-input"
        />
        <button
          onClick={() => void handleSubmit()}
          disabled={!canSubmit}
          aria-label={t('queue.composer.submit.aria')}
          data-testid="queue-composer-submit"
          className="shrink-0 inline-flex items-center gap-1.5 rounded-md bg-primary text-white text-[11px] font-medium px-2.5 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-primary/90 transition-colors max-md:min-h-[40px]"
        >
          <Send className="w-3 h-3" /> {t('queue.composer.submit')}
        </button>
      </div>
      <div className="flex items-center gap-3 mt-2 text-xs text-secondary max-md:flex-wrap">
        {hint && (
          <span className="text-2xs text-muted max-md:w-full">{hint}</span>
        )}
        <div className="flex items-center gap-3 ml-auto max-md:ml-0">
          <label className="flex items-center gap-1 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={pinned}
              onChange={(e) => setPinned(e.target.checked)}
              className="w-3.5 h-3.5"
            />
            {t('queue.composer.pinToTop')}
          </label>
          <label className="flex items-center gap-1 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={review}
              onChange={(e) => setReview(e.target.checked)}
              className="w-3.5 h-3.5"
            />
            {t('queue.composer.reviewBeforeAdvance')}
          </label>
        </div>
      </div>
    </div>
  );
}
