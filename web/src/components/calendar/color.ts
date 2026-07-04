// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Deterministic per-project accent color for calendar event chips. Tailwind
 * can't express a dynamic-from-id class, so this returns inline HSL strings
 * (border + tinted background) keyed by a stable string hash of the project id.
 * Unlinked events (project_id === null) get a neutral slate accent.
 */

export interface EventColor {
  /** Solid accent (left border / dot). */
  border: string;
  /** Low-alpha fill behind the chip. */
  bg: string;
  /** Readable text over `bg`. */
  text: string;
}

const NEUTRAL: EventColor = {
  border: 'hsl(220 9% 60%)',
  bg: 'hsl(220 9% 60% / 0.14)',
  text: 'hsl(220 12% 30%)',
};

// A small fixed hue set spread around the wheel — enough variety without a
// full palette, and stable across reloads since we index by hash.
const HUES = [210, 265, 330, 12, 150, 40, 190, 300];

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export function eventColor(projectId: string | null): EventColor {
  if (!projectId) return NEUTRAL;
  const hue = HUES[hashString(projectId) % HUES.length];
  return {
    border: `hsl(${hue} 65% 48%)`,
    bg: `hsl(${hue} 65% 48% / 0.15)`,
    text: `hsl(${hue} 55% 32%)`,
  };
}
