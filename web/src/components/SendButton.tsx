// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The composer's send button, option B (owner, 2026-11-24): an accent square
 * with an 8px radius, a light tint while there is nothing to send and filled
 * once there is. The same button sends in the chat composer and in the
 * Workbench composer, so both are one family (and one size: 28px, the pin
 * tile's radius). `kind="stop"` is the quiet stop square in the same style.
 *
 * The button is the hit area; `.send-vis` is what you see, so a phone gets a
 * 44px target while the square stays 28px. Styles: `.send-hit` in index.css.
 */

import type { MouseEventHandler, TouchEventHandler } from 'react';
import { ArrowUp } from 'lucide-react';

interface SendButtonProps {
  kind?: 'send' | 'stop';
  /** Accessible name (also the hover tooltip), already translated. */
  label: string;
  disabled?: boolean;
  /** A stop that is already stopping: dimmed, not pressable. */
  pending?: boolean;
  title?: string;
  testId?: string;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  onTouchEnd?: TouchEventHandler<HTMLButtonElement>;
}

export default function SendButton({
  kind = 'send', label, disabled = false, pending = false, title, testId, onClick, onTouchEnd,
}: SendButtonProps) {
  return (
    <button
      type="button"
      className={`send-hit${kind === 'stop' ? ' is-stop' : ''}${pending ? ' is-pending' : ''}`}
      aria-label={label}
      aria-disabled={disabled || undefined}
      disabled={disabled}
      title={title || undefined}
      data-testid={testId}
      onClick={onClick}
      onTouchEnd={onTouchEnd}
    >
      <span className="send-vis">
        {kind === 'stop' ? (
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <rect width="10" height="10" rx="2.5" fill="currentColor" />
          </svg>
        ) : (
          <ArrowUp size={16} strokeWidth={2.25} aria-hidden="true" />
        )}
      </span>
    </button>
  );
}
