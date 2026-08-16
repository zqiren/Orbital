// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { SelectHTMLAttributes } from 'react';
import { ChevronDown } from 'lucide-react';

/** Room for the chevron, as an inline style rather than a `pr-7` class.
 *  Call sites set their own horizontal padding (`px-2`, `px-2.5`, `px-3`),
 *  and whether a `pr-*` utility beats a `px-*` utility depends on the order
 *  Tailwind happens to emit `padding-right` and `padding-inline` in — a
 *  coin-flip this component must not depend on. An inline style beats both. */
const CHEVRON_CLEARANCE = '1.75rem';

/**
 * The app's `<select>`.
 *
 * A bare `<select>` renders as the native macOS menu button — grey gradient,
 * double-arrow stepper, square corners — and WebKit discards the background,
 * border and radius set on it. Every menu in the app was therefore a system
 * control sitting next to custom-styled text inputs, which is what made forms
 * built from both read as two different applications. `appearance-none` is what
 * hands the box back to us; the chevron below replaces the stepper we suppress.
 *
 * `className` passes through to the `<select>` untouched, so this is a drop-in
 * at every existing call site: each screen keeps the exact metrics it had, and
 * only the native-widget look changes. Deliberately no `size` prop — the app's
 * menus range from a full-width settings field to a text-xs chip inside a
 * sentence, and inventing two or three canonical sizes would mean restyling
 * screens this change has no business touching.
 *
 * This stays a real `<select>` on purpose. A div-and-listbox reimplementation
 * would have to re-earn keyboard navigation, type-ahead, the native picker on
 * iOS/Android, and form semantics — all of which a styled native element keeps
 * for free.
 */
export default function Select({
  className = '',
  style,
  children,
  ...rest
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    // `block`, so the wrapper fills a flex-column label the way the select used
    // to, and sizes to its content when it is itself a flex item in a row.
    // `min-w-0` so a long option (a timezone name) can't push a layout wide.
    <span className="relative block min-w-0">
      <select
        {...rest}
        style={{ paddingRight: CHEVRON_CLEARANCE, ...style }}
        className={`appearance-none cursor-pointer ${className}`}
      >
        {children}
      </select>
      <ChevronDown
        size={13}
        aria-hidden="true"
        // pointer-events-none: the chevron sits over the control, and a click
        // on it must still open the menu.
        className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-secondary"
      />
    </span>
  );
}
