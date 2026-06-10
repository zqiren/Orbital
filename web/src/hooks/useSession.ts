// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';

/**
 * useSession — per-project active-session tracker.
 *
 * Manages which session_id is "active" (i.e. currently open in the UI) for a
 * given project. Persists the last-active session to localStorage under the
 * key `orbital:lastSession:<projectId>` so that returning to a project (e.g.
 * after switching tabs or reloading) restores the previous selection.
 *
 * This hook does NOT fetch sessions from the backend — that is useSession's
 * caller's responsibility (combine with useSessions for the full list).
 *
 * @returns `{ activeSessionId, setActiveSessionId }`
 *   - `activeSessionId`: the currently active session_id string, or null if
 *     none has been set yet for this project.
 *   - `setActiveSessionId`: call to change the active session; persists to
 *     localStorage immediately.
 */
export function useSession(projectId: string | null): {
  activeSessionId: string | null;
  setActiveSessionId: (sessionId: string | null) => void;
} {
  const storageKey = projectId ? `orbital:lastSession:${projectId}` : null;

  const [activeSessionId, setActiveSessionIdState] = useState<string | null>(() => {
    if (!storageKey) return null;
    try {
      return localStorage.getItem(storageKey);
    } catch {
      // localStorage unavailable (e.g., in tests without a DOM stub)
      return null;
    }
  });

  // When the projectId changes, load that project's persisted session.
  useEffect(() => {
    if (!storageKey) {
      setActiveSessionIdState(null);
      return;
    }
    try {
      const persisted = localStorage.getItem(storageKey);
      setActiveSessionIdState(persisted);
    } catch {
      setActiveSessionIdState(null);
    }
  }, [storageKey]);

  const setActiveSessionId = useCallback(
    (sessionId: string | null) => {
      setActiveSessionIdState(sessionId);
      if (!storageKey) return;
      try {
        if (sessionId === null) {
          localStorage.removeItem(storageKey);
        } else {
          localStorage.setItem(storageKey, sessionId);
        }
      } catch {
        // localStorage unavailable — silently ignore persistence failure
      }
    },
    [storageKey],
  );

  return { activeSessionId, setActiveSessionId };
}
