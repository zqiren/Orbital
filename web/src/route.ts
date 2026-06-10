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
  | { name: 'project'; projectId: string; tab: 'queue' | 'chat' | 'files'; sessionId?: string; settings?: boolean };
