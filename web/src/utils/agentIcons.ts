// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Resolve a recognizable icon for a sub-agent handle. We use styled monogram
 * badges (brand colour + 2-letter mark) rather than bundling vendor SVGs —
 * the monograms are dependency-free, license-clean, and the fallback path for
 * unknown handles is the same code, so there is no hard-fail on new agents.
 *
 * `src` is part of the return shape for forward-compat (a future build may add
 * vendor SVGs); today it is always undefined and the renderer draws the
 * monogram badge.
 */
export interface AgentIcon {
  src?: string;
  monogram: string;
  color: string;
}

export function getAgentIcon(handle: string): AgentIcon {
  const h = (handle || '').toLowerCase();
  if (h.includes('claude')) return { monogram: 'CC', color: '#D97757' }; // Anthropic coral
  if (h.includes('codex')) return { monogram: 'CX', color: '#10A37F' };  // OpenAI green
  if (h.includes('gemini')) return { monogram: 'GM', color: '#4285F4' }; // Google blue
  return { monogram: (handle || '?').slice(0, 2).toUpperCase(), color: '#6B7280' };
}
