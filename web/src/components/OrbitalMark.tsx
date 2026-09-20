// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * OrbitalMark — the logo as inline SVG, for places that need it as a scalable
 * part of a composition rather than as an <img> of the app icon.
 *
 * The geometry is copied VERBATIM from the locked master
 * (`assets/logo/orbital-logo.svg`, produced by `scripts/gen-logo-assets.py`):
 * a 512 canvas with the orange disc at its centre, a quarter-arc stem, the
 * quarter-disc navy leaf and the capped red bud. It is a fixed construction —
 * do not nudge, restyle or approximate it here. If the logo changes, it
 * changes in the generator, and these four shapes are re-copied from its
 * output.
 */
export default function OrbitalMark({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 512 512"
      aria-hidden="true"
      className={className}
    >
      <circle cx="256" cy="256" r="138" fill="#f4a38c" />
      <path
        d="M 385.000 127.000 A 252.000 252.000 0 0 1 133.000 379.000"
        fill="none"
        stroke="#000000"
        strokeWidth="12"
      />
      <path
        d="M 79.000 325.000 L 79.000 379.000 A 54.000 54.000 0 0 0 133.000 433.000 L 187.000 433.000 A 108.000 108.000 0 0 0 79.000 325.000 Z"
        fill="#1a1a2e"
      />
      <path
        d="M 385.000 86.500 A 40.500 40.500 0 1 0 425.500 127.000 A 64.036 64.036 0 0 0 385.000 86.500 Z"
        fill="#c1292e"
      />
    </svg>
  );
}
