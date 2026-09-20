// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * One-shot handoff from "the user just created this project" to the chat view
 * that opens next, so the agent can speak first (POST /start-onboarding).
 *
 * Deliberately in-memory and consumed exactly once: opening a project must
 * stay pure navigation (the open-time auto-start was removed in 0722d5fa
 * because agents with goals invented their own work). A reload, a second tab
 * or a later visit never carries the request — and the backend refuses anyway
 * unless the project has no goals and no sessions.
 */
const requested = new Set<string>();

export function requestOnboardingKickoff(projectId: string): void {
  requested.add(projectId);
}

/** True at most once per request; clears it. */
export function consumeOnboardingKickoff(projectId: string): boolean {
  return requested.delete(projectId);
}
