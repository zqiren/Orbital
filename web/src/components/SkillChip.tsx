// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Sparkles, X } from 'lucide-react';
import { useT } from '../i18n/useT';

interface SkillChipProps {
  name: string;
  /** Tooltip: the description in the composer, the SKILL.md path on a bubble. */
  title?: string;
  /** Composer only: the × that drops the picked skill. */
  onRemove?: () => void;
  /**
   * The composer's inline capsule (2026-10-28): it sits at the start of the
   * input's first line, so it is one text line tall (18px) with a smaller
   * name, and the × keeps a larger touch target through an invisible inset
   * instead of the 44px min box (which would make the capsule 44px tall).
   */
  inline?: boolean;
  'data-testid'?: string;
}

/**
 * The skill picked from the composer's `/` menu (spec 132): in the composer's
 * chip strip before sending, and on the user bubble after (parsed back out of
 * the `<use_skill>` block). Same footprint as AttachmentChip.
 */
export default function SkillChip({ name, title, onRemove, inline = false, 'data-testid': testId }: SkillChipProps) {
  const t = useT();
  return (
    <div
      className={
        inline
          ? 'inline-flex min-w-0 items-center gap-1 rounded-md border border-border bg-card-hover pl-1.5 pr-1 h-[18px] text-micro leading-none max-w-[180px]'
          : 'inline-flex items-center gap-1.5 rounded-md border border-border bg-card-hover px-2 py-1 text-caption max-w-[260px]'
      }
      data-testid={testId}
      title={title ?? name}
      aria-label={t('chat.skill.chipAria', { name })}
    >
      <Sparkles className={`${inline ? 'w-3 h-3' : 'w-3.5 h-3.5'} shrink-0 text-accent`} aria-hidden="true" />
      <span className="truncate font-medium">{name}</span>
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          aria-label={t('chat.skill.removeAria', { name })}
          data-testid={testId ? `${testId}-remove` : undefined}
          className={
            inline
              ? "relative shrink-0 text-secondary hover:text-primary before:absolute before:-inset-2 before:content-['']"
              : 'shrink-0 p-0.5 text-secondary hover:text-primary max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center'
          }
        >
          <X className={inline ? 'w-3 h-3' : 'w-3.5 h-3.5'} />
        </button>
      )}
    </div>
  );
}
