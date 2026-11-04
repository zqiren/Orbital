// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 065 — an agent's mark at 13 px, the 154 rule: logo, not name. The name
 * is only the tooltip and the accessible label. No tile in light mode, a small
 * light tile in dark mode; Orbital's own mark already carries one.
 */
import { useState } from 'react';
import { getAgentIcon } from '../../utils/agentIcons';
import { useT } from '../../i18n/useT';
import { agentName } from './historyModel';

export function AgentLogo({ slug, title }: { slug: string; title?: string }) {
  const [failed, setFailed] = useState(false);
  const icon = getAgentIcon(slug);
  const name = title ?? agentName(slug);
  const orbital = slug === 'orbital';
  if (!icon.src || failed) {
    return (
      <span
        role="img"
        aria-label={name}
        title={name}
        data-agent={slug}
        className="inline-flex h-[13px] w-[13px] shrink-0 items-center justify-center rounded-[3px] text-[7px] font-semibold text-white"
        style={{ backgroundColor: icon.color }}
      >
        {icon.monogram.slice(0, 1)}
      </span>
    );
  }
  return (
    <span
      role="img"
      aria-label={name}
      title={name}
      data-agent={slug}
      className={`inline-flex h-[13px] w-[13px] shrink-0 overflow-hidden rounded-[3px] ${
        orbital ? '' : 'dark:-m-px dark:h-[15px] dark:w-[15px] dark:bg-logo-tile dark:p-[1.5px]'
      }`}
    >
      <img
        src={icon.src}
        alt=""
        onError={() => setFailed(true)}
        className="block h-full w-full object-contain"
      />
    </span>
  );
}

/** The user's own mark: a small grey person glyph in a circle. */
export function YouGlyph({ title }: { title?: string }) {
  const t = useT();
  const name = title ?? t('history.you');
  return (
    <span
      role="img"
      aria-label={name}
      title={name}
      data-agent="you"
      className="inline-flex h-[13px] w-[13px] shrink-0 items-center justify-center rounded-full bg-muted/20 text-secondary"
    >
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
        <circle cx="12" cy="8" r="4.5" />
        <path d="M3.5 21a8.5 8.5 0 0 1 17 0z" />
      </svg>
    </span>
  );
}

/** Logos of one author: one per agent, or the user glyph. */
export function AuthorLogos({
  agents,
  you,
  title,
}: {
  agents: string[];
  you?: boolean;
  title?: string | null;
}) {
  if (you) return <YouGlyph />;
  if (agents.length === 0) return null;
  return (
    <span className="inline-flex shrink-0 items-center gap-0.5" title={title ?? undefined}>
      {agents.map((a) => (
        <AgentLogo key={a} slug={a} title={title ?? undefined} />
      ))}
    </span>
  );
}
