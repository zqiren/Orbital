// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * In-memory route model for Orbital V1 UI.
 *
 * No URL-based router — navigation is driven entirely by React state.
 * The single `route: Route` object replaces the old `view`/`tab`/`selectedProjectId` triad.
 */

export type Route =
  | { name: 'list' }
  | { name: 'create' }
  | { name: 'blocked' }
  | { name: 'settings' }
  | {
      name: 'project';
      projectId: string;
      tab: 'queue' | 'chat' | 'files';
      sessionId?: string;
      settings?: boolean;
      /**
       * Optional intent to scroll the settings overlay to a specific section on
       * open (P3-G: the budget corner deep-links here). Only meaningful when
       * `settings` is true; consumed (scrolled-to) once by SettingsView.
       */
      settingsAnchor?: 'budget';
      /**
       * Open the pricing-table editor overlay (P3-I). Reached from the Budget
       * section's "Edit pricing table" link; takes precedence over the settings
       * surface in App's content switch. Back returns to settings.
       */
      pricing?: boolean;
      /**
       * Workspace-relative path of the file to preview in the slide-out
       * FilePreviewDrawer (spec 002). Set when a user clicks a clickable path
       * in chat; the drawer is open iff this is non-null. Does NOT change the
       * active `tab` — the chat stays mounted underneath. In-memory only (not
       * persisted across reload), like the rest of the route model.
       */
      previewPath?: string;
    };
