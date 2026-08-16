// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The one description of what a form control looks like.
 *
 * Before this, the same class string
 * (`text-sm bg-sidebar border border-border rounded-lg … focus:border-accent`)
 * was written out by hand at ~19 call sites across 13 files, with the padding
 * step drifting between `px-2`, `px-2.5` and `px-3` depending on which screen
 * copied which. Retuning the field look meant finding all of them.
 */
const SURFACE =
  'bg-sidebar border border-border rounded-lg text-primary ' +
  'placeholder:text-secondary/60 focus:outline-none focus:border-accent ' +
  'transition-all duration-150';

/**
 * Single-line controls (text inputs, `type="time"`, `type="number"`).
 *
 * `h-9` is not decoration. A native `<input type="time">` and a native
 * `<select>` carry different intrinsic heights in WebKit, so a row of them laid
 * out with `items-end` came out visibly ragged (the TIME box stood taller than
 * the REPEAT and TIME ZONE menus beside it). Pinning one height is what makes a
 * row of mixed control types read as a row.
 *
 * `appearance-none` matters for the same reason it does on `<select>`: WebKit
 * paints `type="time"` and `type="number"` with its own inner spin buttons and
 * ignores parts of the box we set here.
 */
export const FIELD = `h-9 appearance-none text-sm px-2.5 ${SURFACE}`;

/**
 * Multi-line controls. Same surface, no fixed height, no `appearance-none`
 * (a textarea has no native widget to suppress and needs its resize grip).
 */
export const FIELD_MULTILINE = `text-sm px-2.5 py-2 ${SURFACE}`;
