// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Check, Copy } from 'lucide-react';
import { useCopyToClipboard } from '../hooks/useCopyToClipboard';
import { useT } from '../i18n/useT';

interface CopyButtonProps {
  /** The exact text placed on the clipboard. */
  text: string;
  /** Accessible name; defaults to the generic "Copy". */
  ariaLabel?: string;
  /** Extra classes for positioning at the call site. */
  className?: string;
  /** Icon edge length in px. 12 on a message label line, 13 in a code header. */
  size?: number;
  /** Show the word "Copy"/"Copied" beside the icon. */
  withLabel?: boolean;
  'data-testid'?: string;
}

/**
 * The single copy affordance (BACKLOG spec 068): icon → check, reverting after
 * 2s. Renders NOTHING where the clipboard API is unavailable rather than
 * offering a control that silently fails — see `useCopyToClipboard` for which
 * surface that is and why.
 */
export default function CopyButton({
  text,
  ariaLabel,
  className = '',
  size = 12,
  withLabel = false,
  'data-testid': testId = 'copy-button',
}: CopyButtonProps) {
  const t = useT();
  const { copied, copy, supported } = useCopyToClipboard();

  if (!supported) return null;

  const label = copied ? t('chat.message.copied') : t('chat.message.copy');

  return (
    <button
      // type="button" is load-bearing wherever this lands inside a <form>
      // (settings surfaces): a bare button submits it.
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        // First await in the gesture — see the WebKit note in the hook.
        void copy(text);
      }}
      aria-label={ariaLabel ?? label}
      title={label}
      data-testid={testId}
      data-copied={copied ? 'true' : undefined}
      className={`inline-flex items-center gap-1 rounded text-secondary hover:text-primary transition-colors ${className}`}
    >
      {copied ? (
        <Check size={size} aria-hidden="true" />
      ) : (
        <Copy size={size} aria-hidden="true" />
      )}
      {withLabel && <span>{label}</span>}
    </button>
  );
}
