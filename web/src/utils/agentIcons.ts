// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Resolve a recognizable icon for an agent handle.
 *
 * Known agents render their vendor mark from `web/public/agents/<slug>.svg`
 * (see that directory's README for sources and licensing). Every entry also
 * carries a brand-coloured 2-letter monogram, which is what unknown handles get
 * and what `MessageAvatar` falls back to if an image fails to load — so a
 * deleted or broken asset degrades instead of leaving a blank row.
 *
 * Matching is exact on the normalized handle, then on successively shorter
 * `-`-delimited prefixes. `codex` and `claude-code` hit directly; `gemini-cli`
 * misses, then matches on `gemini`. This deliberately replaces the older
 * `handle.includes('claude')` substring test, which would false-positive an
 * unrelated handle that merely contained a known name.
 */
export interface AgentIcon {
  src?: string;
  monogram: string;
  color: string;
}

/**
 * The handle used for Orbital's own management agent. It has no sub-agent
 * handle of its own on the wire, so `MessageAvatar` substitutes this to route
 * the main agent through the same lookup as its peers.
 */
export const MAIN_AGENT_HANDLE = 'orbital';

const ICONS: Record<string, AgentIcon> = {
  // Orbital's own mark — the app icon, already shipped for the PWA manifest.
  [MAIN_AGENT_HANDLE]: { src: '/icon-192.png', monogram: 'OR', color: '#D9694A' },
  'claude-code': { src: '/agents/claude-code.svg', monogram: 'CC', color: '#D97757' },
  claude: { src: '/agents/claude-code.svg', monogram: 'CC', color: '#D97757' },
  codex: { src: '/agents/codex.svg', monogram: 'CX', color: '#10A37F' },
  cursor: { src: '/agents/cursor.svg', monogram: 'CU', color: '#1A1A1A' },
  gemini: { src: '/agents/gemini.svg', monogram: 'GM', color: '#4285F4' },
  grok: { src: '/agents/grok.svg', monogram: 'GK', color: '#000000' },
};

export function getAgentIcon(handle: string): AgentIcon {
  const h = (handle || '').trim().toLowerCase();

  // Exact slug first, then drop trailing `-`-delimited segments so suffixed
  // variants ("gemini-cli", "codex-cli") land on their base agent.
  const parts = h.split('-');
  for (let end = parts.length; end > 0; end--) {
    const hit = ICONS[parts.slice(0, end).join('-')];
    if (hit) return hit;
  }

  return { monogram: (h || '?').slice(0, 2).toUpperCase(), color: '#6B7280' };
}
