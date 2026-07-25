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
  border: 'hsl(220 9% 58%)',
  bg: 'hsl(220 9% 58% / 0.10)',
  text: 'hsl(220 12% 32%)',
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

/**
 * Saturation is deliberately low (38%, was 65%). At 65% these hues read as
 * primary colours — a wall of vivid blue/red/green chips and accent borders
 * that dominated both the Workbench list and the Calendar grid. Hue identity
 * survives desaturation perfectly well; the chips still tell projects apart,
 * they just stop shouting. The fill alpha dropped with it (0.15 -> 0.09) so a
 * chip reads as a tint rather than a block of colour.
 */
export function eventColor(projectId: string | null): EventColor {
  if (!projectId) return NEUTRAL;
  const hue = HUES[hashString(projectId) % HUES.length];
  return {
    border: `hsl(${hue} 38% 52%)`,
    bg: `hsl(${hue} 38% 52% / 0.09)`,
    text: `hsl(${hue} 30% 34%)`,
  };
}
