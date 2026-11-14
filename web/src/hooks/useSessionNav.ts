// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * useSessionNav — the project's session navigation: open a session, mint a
 * new one, step off a deleted one, follow a moved one.
 *
 * Spec 137 (the folder shell) moved the session list out of the chat and into
 * the project column, which stays on screen while a page (Tasks, Settings)
 * fills the right area. So "open a session" now also means "back to chat":
 * every action here clears `route.page`. The chat itself (ChatView's links to
 * another session) uses the same `selectSession`, so the two can never
 * disagree about what opening a session does.
 *
 * Selection is a route change (`route.sessionId` is the source of truth) plus
 * the per-project persisted choice (`useSession`), exactly as ChatTab did it
 * before the move.
 */

import { useCallback } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Route } from '../route';
import type { SessionListEntry } from '../types';
import { useSession } from './useSession';
import { useAgent } from './useAgent';
import { markFreshSession } from '../utils/freshSession';

export interface SessionNav {
  /** Open `sessionId` in the chat (leaves any page). */
  selectSession: (sessionId: string) => void;
  /**
   * "+ new session": mint a genuinely BLANK session on the backend and open
   * it. Never clears `route.sessionId` to undefined — ChatTab's default
   * resolution would re-open the most recent existing session (the old bug).
   * Resolves once the route has moved (or the mint failed and was logged).
   */
  newSession: () => Promise<void>;
  /**
   * After a delete: if the deleted session was the one being viewed, move to
   * the most recent remaining one (`remaining` arrives sorted, most recent
   * first), else leave the route alone.
   */
  sessionDeleted: (deletedId: string, remaining: SessionListEntry[], viewedId: string | undefined) => void;
  /**
   * Spec 180: `sessionId` now lives in `toProjectId` (moved from the sidebar,
   * or a send the daemon delivered there). If it is the one open, the view
   * follows it; else the route is left alone.
   */
  sessionMoved: (sessionId: string, toProjectId: string) => void;
}

export function useSessionNav(
  projectId: string,
  setRoute: Dispatch<SetStateAction<Route>>,
): SessionNav {
  const { setActiveSessionId } = useSession(projectId);
  const { newSession: mintSession } = useAgent();

  const goTo = useCallback(
    (sessionId: string | undefined) => {
      setRoute((prev) =>
        prev.name === 'project' && prev.projectId === projectId
          ? { ...prev, sessionId, page: undefined }
          : prev,
      );
    },
    [projectId, setRoute],
  );

  const selectSession = useCallback(
    (sessionId: string) => {
      goTo(sessionId);
      setActiveSessionId(sessionId);
    },
    [goTo, setActiveSessionId],
  );

  const newSession = useCallback(async () => {
    try {
      const result = await mintSession(projectId);
      const newId = result.session_id;
      if (!newId) {
        console.error('newSession returned no session_id', result);
        return;
      }
      // Spec 107 FE-1: tell ChatView's load effect this id was minted here
      // and has no history yet — BEFORE the route flips, so the effect skips
      // the /chat fetch (and its skeleton) for a session that cannot exist
      // on disk until the first message.
      markFreshSession(projectId, newId);
      goTo(newId);
      setActiveSessionId(newId);
    } catch (err) {
      console.error('Failed to create a new session', err);
    }
  }, [mintSession, projectId, goTo, setActiveSessionId]);

  const sessionDeleted = useCallback(
    (deletedId: string, remaining: SessionListEntry[], viewedId: string | undefined) => {
      if (viewedId !== deletedId) return; // a non-viewed row was deleted
      const nextId = remaining[0]?.session_id;
      setRoute((prev) =>
        prev.name === 'project' && prev.projectId === projectId
          ? { ...prev, sessionId: nextId }
          : prev,
      );
      setActiveSessionId(nextId ?? null);
    },
    [projectId, setRoute, setActiveSessionId],
  );

  const sessionMoved = useCallback(
    (sessionId: string, toProjectId: string) => {
      setRoute((prev) =>
        prev.name === 'project' && prev.projectId === projectId && prev.sessionId === sessionId
          ? { name: 'project', projectId: toProjectId, sessionId }
          : prev,
      );
    },
    [projectId, setRoute],
  );

  return { selectSession, newSession, sessionDeleted, sessionMoved };
}
