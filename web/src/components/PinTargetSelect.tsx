// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Spec 074 — the composer "Talking to" selector. The sticky selection IS the
// session's sub-agent pin: choosing a worker PATCHes `pinned_target`, and
// while pinned every composer send dispatches straight to that worker with
// zero management-agent turns. `Orbital` (the default) is the management
// agent — selecting it back is the unpin. Hidden entirely when no sub-agents
// are installed: zero surface for users without workers.

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

export default function PinTargetSelect({ agents, value, onChange, disabled }: PinTargetSelectProps) {
  const t = useT();
  // No sub-agents installed → no pin surface at all.
  if (agents.length === 0) return null;
  // A stale pin (agent uninstalled since) still renders so the user can see
  // and clear it — append it as a bare-slug option instead of blanking the
  // select to Orbital while the session is actually still pinned.
  const known = agents.some((a) => a.slug === value);
  return (
    <label
      className="flex items-center gap-1.5 text-xs text-secondary shrink-0"
      data-testid="pin-target-select"
    >
      <span>{t('pinAgent.talkingTo')}</span>
      <select
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value === '' ? null : e.target.value)}
        disabled={disabled}
        aria-label={t('pinAgent.aria')}
        className="bg-transparent border border-border rounded-md px-1.5 py-0.5 text-xs text-primary focus:outline-none focus:border-accent cursor-pointer disabled:opacity-50 max-md:min-h-[32px]"
      >
        <option value="">{t('pinAgent.orbital')}</option>
        {agents.map((a) => (
          <option key={a.slug} value={a.slug}>
            {a.name}
          </option>
        ))}
        {value && !known && (
          <option value={value}>{value}</option>
        )}
      </select>
    </label>
  );
}
