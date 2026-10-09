// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * One-shot handoff from "this tab just minted this session id" to the chat
 * view that lands on it next (spec 107 FE-1).
 *
 * `POST /new-session` writes no file: the session materializes on its first
 * message. So the history of a freshly minted id is empty by construction —
 * and the tab that minted it knows that. Without this mark, ChatView's load
 * effect fetches `/chat?session_id=<fresh>` anyway and shows a skeleton until
 * the backend has looked through every session log for a session that cannot
 * exist (seconds on a project with hundreds of MB of logs).
 *
 * Deliberately in-memory and consumed exactly once, like
 * `onboardingKickoff.ts`: a reload, a second tab, or a later visit to the same
 * id goes through the normal fetch — by then the file may exist. Keyed by
 * project AND session id so an id minted for one project can never short-
 * circuit a load in another.
 */
const fresh = new Set<string>();

function key(projectId: string, sessionId: string): string {
  return `${projectId}:${sessionId}`;
}

/** Call BEFORE navigating to the minted id, so the load effect sees it. */
export function markFreshSession(projectId: string, sessionId: string): void {
  fresh.add(key(projectId, sessionId));
}

/** True at most once per mark; clears it. */
export function consumeFreshSession(projectId: string, sessionId: string): boolean {
  return fresh.delete(key(projectId, sessionId));
}
