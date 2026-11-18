// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * sessionColumn — is the session column folded to its rail? (2026-11-18)
 *
 * One value for every project, per device: folding the column is about the
 * window, not about a project. Same `orbital:sessionSidebar.*` family as the
 * filter keys in SessionSidebar; every access is guarded, since storage can
 * be absent or throw (private window, locked-down webview).
 */

const COLLAPSED_KEY = 'orbital:sessionSidebar.collapsed';

export function readSessionColumnCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSED_KEY) === '1';
  } catch {
    return false;
  }
}

export function writeSessionColumnCollapsed(collapsed: boolean): void {
  try {
    localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0');
  } catch {
    /* storage unavailable — still collapses for this visit */
  }
}
