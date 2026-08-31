// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Spec 074 — the composer pin control. The sticky selection IS the session's
// sub-agent pin: choosing a worker PATCHes `pinned_target`, and while pinned
// every composer send dispatches straight to that worker with zero
// management-agent turns. Selecting Orbital (the manager) back is the unpin.
//
// Design (aligned 2026-08-31 via the "Composer Pin Header" mockup): a
// permanent logo-only mark fused to the input row's left edge — Orbital's own
// mark at rest, the pinned agent's mark while pinned. Colorless chrome: no
// tint, no name; identity comes from the avatar itself, the tooltip, and the
// "Message {agent}…" placeholder. Hidden entirely when no sub-agents are
// installed: zero surface for users without workers.

import { useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import MessageAvatar from './MessageAvatar';
import { useT } from '../i18n/useT';

/** Result of resolving one composer send against the mention + pin rules. */
export interface ResolvedSendTarget {
  /** Slug to dispatch to, or undefined for the management agent. */
  target?: string;
  /** Message text with any leading @mention stripped. */
  content: string;
  /** True when `target` came from the sticky dropdown pin (the backend maps
   *  this to initiator="user_pinned" — the wake-suppressed dispatch class).
   *  A leading @mention is always a one-message override with pinned=false,
   *  keeping today's manager-supervises mention semantics. */
  pinned: boolean;
}

/**
 * Target precedence (spec 074 §3.2): a leading `@slug ` mention wins for that
 * one message; otherwise the sticky pin applies; otherwise management. The
 * reserved `@orbital` mention routes one message down the management branch
 * WITHOUT unpinning — the lightweight aside while pinned.
 */
export function resolveSendTarget(
  text: string,
  pinnedTarget: string | null | undefined,
): ResolvedSendTarget {
  const atMatch = text.match(/^@([\w-]+)\s+([\s\S]*)/);
  if (atMatch) {
    const slug = atMatch[1];
    const content = atMatch[2];
    if (slug.toLowerCase() === 'orbital') {
      return { target: undefined, content, pinned: false };
    }
    return { target: slug, content, pinned: false };
  }
  if (pinnedTarget) {
    return { target: pinnedTarget, content: text, pinned: true };
  }
  return { target: undefined, content: text, pinned: false };
}

interface PinTargetSelectProps {
  /** Installed sub-agents — same enumeration + filter as the mention menu. */
  agents: Array<{ slug: string; name: string }>;
  /** Currently pinned slug, or null when talking to Orbital (the manager). */
  value: string | null;
  /** Fired with the new slug, or null when Orbital is selected (unpin). */
  onChange: (slug: string | null) => void;
  disabled?: boolean;
}

const MENU_ROW =
  'w-full flex items-center gap-2.5 px-3 py-2 text-[13px] text-primary text-left ' +
  'hover:bg-card-hover/50 max-md:min-h-[44px]';

export default function PinTargetSelect({ agents, value, onChange, disabled }: PinTargetSelectProps) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDocMouseDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDocMouseDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  // No sub-agents installed → no pin surface at all.
  if (agents.length === 0) return null;

  // A stale pin (agent uninstalled since) still renders so the user can see
  // and clear it — the avatar lookup falls back to a monogram badge, and the
  // menu appends a bare-slug row instead of silently showing Orbital while
  // the session is actually still pinned.
  const known = agents.some((a) => a.slug === value);
  const pinnedName = value ? (agents.find((a) => a.slug === value)?.name ?? value) : null;

  const pick = (slug: string | null) => {
    setOpen(false);
    onChange(slug);
  };

  return (
    // -ml-3/-my-2 pull the control through the composer card's padding so the
    // mark fuses to the card's left edge, full row height (self-stretch), per
    // the aligned mockup. rounded-l matches the card's inner radius.
    <div
      ref={rootRef}
      className="relative self-stretch flex shrink-0 -ml-3 -my-2"
      data-testid="pin-target-select"
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        disabled={disabled}
        aria-label={t('pinAgent.aria')}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={pinnedName
          ? t('pinAgent.tooltipPinned', { name: pinnedName })
          : t('pinAgent.tooltipManager')}
        className="flex items-center gap-1 pl-3 pr-1.5 rounded-l-[7px] border-r border-border
                   hover:bg-card-hover/50 disabled:opacity-50 focus-visible:outline-none
                   focus-visible:bg-card-hover/50 max-md:min-w-[48px]"
      >
        <MessageAvatar variant="agent" agentHandle={value ?? undefined} />
        <ChevronDown size={10} className="text-muted shrink-0" />
      </button>

      {open && (
        <div
          role="listbox"
          aria-label={t('pinAgent.aria')}
          className="absolute bottom-full left-0 mb-2 min-w-[220px] bg-card border border-border
                     rounded-lg shadow-lg overflow-hidden z-50"
        >
          <button
            type="button"
            role="option"
            aria-selected={value === null}
            onClick={() => pick(null)}
            className={MENU_ROW}
          >
            <MessageAvatar variant="agent" />
            <span className="font-medium">{t('pinAgent.orbital')}</span>
            <span className="ml-auto text-xs text-muted">{t('pinAgent.managerRole')}</span>
          </button>
          <div className="h-px bg-border/60" aria-hidden />
          {agents.map((a) => (
            <button
              key={a.slug}
              type="button"
              role="option"
              aria-selected={value === a.slug}
              onClick={() => pick(a.slug)}
              className={MENU_ROW}
            >
              <MessageAvatar variant="agent" agentHandle={a.slug} />
              <span className="font-medium">{a.name}</span>
              {value === a.slug && <span className="ml-auto text-xs text-muted">✓</span>}
            </button>
          ))}
          {value && !known && (
            <button
              type="button"
              role="option"
              aria-selected
              onClick={() => pick(value)}
              className={MENU_ROW}
            >
              <MessageAvatar variant="agent" agentHandle={value} />
              <span className="font-medium">{value}</span>
              <span className="ml-auto text-xs text-muted">✓</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
