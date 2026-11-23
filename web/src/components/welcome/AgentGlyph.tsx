// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * An agent's mark at any size, the logo-not-word rule: the name is only the
 * tooltip and the accessible label. Orbital's own mark is the app icon; every
 * other mark sits on the same logo tile so black marks stay visible on dark.
 * A missing image falls back to the brand-coloured monogram.
 */
import { useState } from 'react';
import { getAgentIcon, MAIN_AGENT_HANDLE } from '../../utils/agentIcons';

export default function AgentGlyph({
  slug,
  size = 16,
  title,
  dim = false,
}: {
  slug: string;
  size?: number;
  /** Tooltip + accessible name; omitted = decorative. */
  title?: string;
  dim?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const icon = getAgentIcon(slug || MAIN_AGENT_HANDLE);
  const orbital = (slug || MAIN_AGENT_HANDLE) === MAIN_AGENT_HANDLE;
  const a11y = title ? { role: 'img' as const, 'aria-label': title, title } : { 'aria-hidden': true as const };
  const box = { width: size, height: size };
  if (!icon.src || failed) {
    return (
      <span
        {...a11y}
        data-agent={slug}
        className={`inline-flex shrink-0 items-center justify-center rounded-[4px] font-semibold text-white ${dim ? 'opacity-45' : ''}`}
        style={{ ...box, backgroundColor: icon.color, fontSize: Math.max(7, Math.round(size * 0.45)) }}
      >
        {icon.monogram.slice(0, size < 18 ? 1 : 2)}
      </span>
    );
  }
  return (
    <span
      {...a11y}
      data-agent={slug}
      className={`inline-flex shrink-0 overflow-hidden rounded-[4px] ${orbital ? '' : 'bg-logo-tile'} ${dim ? 'opacity-45 grayscale' : ''}`}
      style={{ ...box, padding: orbital ? 0 : Math.max(1, Math.round(size * 0.12)) }}
    >
      <img src={icon.src} alt="" onError={() => setFailed(true)} className="block h-full w-full object-contain" />
    </span>
  );
}
