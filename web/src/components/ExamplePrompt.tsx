// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ExamplePrompt — one "here is something you could ask" row, shared by the
 * teaching empty states (a project's empty chat, the empty Automations pane)
 * so they speak one visual language: a card on the surface ladder, an accent
 * icon tile, the prompt in the user's own voice, and a quiet affordance that
 * says where a click takes it.
 *
 * A click never SENDS anything — callers fill a composer with `text`. Without
 * `onClick` it renders as a plain (non-interactive) card.
 */

import type { LucideIcon } from 'lucide-react';

interface ExamplePromptProps {
  icon: LucideIcon;
  /** Tiny caption above the prompt, e.g. the automation kind. */
  label?: string;
  text: string;
  onClick?: () => void;
  /** What a click does, as an icon + its accessible/tooltip label. */
  actionIcon?: LucideIcon;
  actionLabel?: string;
}

export default function ExamplePrompt({
  icon: Icon,
  label,
  text,
  onClick,
  actionIcon: ActionIcon,
  actionLabel,
}: ExamplePromptProps) {
  const body = (
    <>
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent">
        <Icon size={15} aria-hidden="true" />
      </span>
      <span className="min-w-0 flex-1">
        {label && (
          <span
            className="block text-secondary font-semibold"
            style={{ fontSize: '9.5px', letterSpacing: '0.8px', textTransform: 'uppercase' }}
          >
            {label}
          </span>
        )}
        <span className="block text-[13px] leading-snug text-primary">{text}</span>
      </span>
      {onClick && ActionIcon && (
        <ActionIcon
          size={14}
          aria-hidden="true"
          className="shrink-0 text-secondary/60 transition-colors duration-150 group-hover:text-accent"
        />
      )}
    </>
  );
  const shell =
    'group flex w-full items-center gap-3 rounded-xl border border-border bg-card px-3.5 py-3 text-left shadow-[0_1px_2px_rgb(0_0_0/0.04)]';
  if (!onClick) return <div className={shell}>{body}</div>;
  return (
    <button
      type="button"
      onClick={onClick}
      title={actionLabel}
      className={`${shell} transition-[border-color,box-shadow,transform] duration-150 hover:border-accent/50 hover:shadow-[0_4px_12px_-4px_rgb(0_0_0/0.12)] active:scale-[0.99] motion-reduce:transition-none motion-reduce:active:scale-100 max-md:min-h-[44px]`}
    >
      {body}
    </button>
  );
}
