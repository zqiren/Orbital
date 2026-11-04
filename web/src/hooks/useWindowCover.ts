// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * useWindowCover — is a maximized document covering the window? (2026-11-04)
 *
 * Maximize takes the whole window except the 20px edge strip, in a project
 * and on the Workbench alike. A project list the user pinned open would
 * otherwise sit on top of the document's left 260px, so App shows it as the
 * strip (unpinned) while anything covers the window and pins it again when
 * that ends; hovering the strip still floats the list above the document.
 *
 * Everything that maximizes registers with `useCoverWindow(active)`; App reads
 * `useWindowCovered()`. A counter, so two at once (a project panel and the
 * pane's document) cannot end each other's cover. Module state, no context.
 */

import { useEffect, useSyncExternalStore } from 'react';

let count = 0;
const listeners = new Set<() => void>();

function emit(): void {
  for (const fn of listeners) fn();
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Anything covering the window right now (also for tests and non-React code). */
export function getWindowCovered(): boolean {
  return count > 0;
}

/** Register a cover while `active`. */
export function useCoverWindow(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    count += 1;
    emit();
    return () => {
      count -= 1;
      emit();
    };
  }, [active]);
}

export function useWindowCovered(): boolean {
  return useSyncExternalStore(subscribe, getWindowCovered, getWindowCovered);
}

/** The list is drawn pinned only while nothing covers the window; the
 *  user's pin itself is untouched, so it comes back when the cover ends. */
export function effectivePinned(pinned: boolean, covered: boolean): boolean {
  return pinned && !covered;
}
