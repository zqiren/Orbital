// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Plus, Send } from 'lucide-react';
import { useRef, useState } from 'react';
import { useT } from '../i18n/useT';
import { useAttachments } from '../hooks/useAttachments';
import AttachmentChip from './AttachmentChip';
import PinTargetSelect from './PinTargetSelect';

interface QueueComposerProps {
  /** Owning project — attachments upload into its workspace `uploads/`. */
  projectId: string;
  onSubmit: (
    content: string,
    opts: {
      priority: number;
      review: boolean;
      fileRefs: string[];
      /** Spec 079 — chosen worker slug, or null for Orbital (the manager). */
      agent: string | null;
    },
  ) => Promise<void> | void;
  disabled?: boolean;
  hint?: string;
  /**
   * Spec 079 — installed sub-agents offered as runners for this item. The
   * picker hides itself when the list is empty, so a user with no workers sees
   * exactly the composer they see today.
   */
  agents?: Array<{ slug: string; name: string }>;
}

export default function QueueComposer({
  projectId,
  onSubmit,
  disabled,
  hint,
  agents = [],
}: QueueComposerProps) {
  const t = useT();
  const [value, setValue] = useState('');
  const [pinned, setPinned] = useState(false);
  const [review, setReview] = useState(false);
  // Per item, never sticky — reset alongside pinned/review after each add, the
  // same way the two checkboxes are (spec 079 §6.3).
  const [agent, setAgent] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [attachError, setAttachError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    attachments,
    anyUploading,
    removeAttachment,
    retryAttachment,
    clearAttachments,
    handleFilePickerChange,
    handlePaste,
  } = useAttachments(projectId, { onError: setAttachError });

  const isEmpty = value.trim().length === 0;
  // Unlike chat, a queue item ALWAYS needs text: the backend rejects empty
  // content (400), so an attachment-only item isn't representable.
  const canSubmit = !isEmpty && !disabled && !submitting && !anyUploading;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      // Bare content plus the uploaded paths — the dispatcher builds the
      // <attached_files> block at dispatch time, so prepending one here
      // (the way chat does for its optimistic echo) would duplicate it.
      const fileRefs = attachments
        .filter((a) => a.status === 'done' && a.uploadedPath)
        .map((a) => a.uploadedPath!);
      await onSubmit(value.trim(), {
        priority: pinned ? 1 : 0,
        review,
        fileRefs,
        agent,
      });
      setValue('');
      setPinned(false);
      setReview(false);
      setAgent(null);
      setAttachError(null);
      clearAttachments();
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
      {attachments.length > 0 && (
        <div
          className="flex flex-wrap gap-2 mb-2 max-h-[100px] overflow-y-auto"
          data-testid="queue-chip-strip"
        >
          {attachments.map((a) => (
            <AttachmentChip
              key={a.id}
              filename={a.filename}
              mime={a.mime}
              size={a.size}
              status={a.status}
              thumbnailUrl={a.thumbnailUrl}
              errorMessage={a.errorMessage}
              onRemove={() => removeAttachment(a.id)}
              onRetry={a.status === 'error' ? () => retryAttachment(a.id) : undefined}
            />
          ))}
        </div>
      )}
      <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 focus-within:border-accent transition-colors max-md:gap-1.5">
        <input
          type="file"
          multiple
          ref={fileInputRef}
          className="hidden"
          onChange={handleFilePickerChange}
          data-testid="queue-attachment-file-input"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          aria-label={t('chat.attachFiles')}
          data-testid="queue-composer-attach"
          className="shrink-0 text-muted hover:text-primary rounded disabled:opacity-40 disabled:cursor-not-allowed transition-colors max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
        >
          <Plus className="w-4 h-4" />
        </button>
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          onPaste={handlePaste}
          placeholder={t('queue.composer.placeholder')}
          rows={1}
          disabled={disabled}
          className="flex-1 resize-none bg-transparent leading-5 text-[13px] text-primary placeholder:text-secondary/60 focus:outline-none disabled:opacity-50"
          data-testid="queue-composer-input"
        />
        <button
          onClick={() => void handleSubmit()}
          disabled={!canSubmit}
          title={anyUploading ? t('chat.disabled.waitingUploads') : undefined}
          aria-label={t('queue.composer.submit.aria')}
          data-testid="queue-composer-submit"
          className="shrink-0 inline-flex items-center gap-1.5 rounded-md bg-primary text-white text-[11px] font-medium px-2.5 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-primary/90 transition-colors max-md:min-h-[40px]"
        >
          <Send className="w-3 h-3" /> {t('queue.composer.submit')}
        </button>
      </div>
      {attachError && (
        <p className="mt-1.5 text-2xs text-error" data-testid="queue-attach-error">
          {attachError}
        </p>
      )}
      <div className="flex items-center gap-3 mt-2 text-xs text-secondary max-md:flex-wrap">
        {hint && (
          <span className="text-2xs text-muted max-md:w-full">{hint}</span>
        )}
        <div className="flex items-center gap-3 ml-auto max-md:ml-0">
          {/* Who runs this item. Renders nothing when no workers are
              installed, so the option row is unchanged for those users. */}
          <PinTargetSelect
            agents={agents}
            value={agent}
            onChange={setAgent}
            disabled={disabled}
            variant="standalone"
            managerLabel={t('queue.composer.agent.aria')}
            data-testid="queue-composer-agent"
          />
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
