// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ChatTab — the Chat-tab 2-column layout (V1 Phase 1B).
 *
 *   [ SessionSidebar 260px ][ ChatView 1fr ]
 *
 * The Chat tab is session-aware: the LEFT column lists the project's sessions
 * (SessionSidebar) and the RIGHT column renders the conversation for the
 * ACTIVE session. Selecting a session is a route change — it sets
 * `route.sessionId` (the source of truth for the active session) and the
 * sidebar persists it via useSession.
 *
 * Active-session resolution (§2 of the T5 brief):
 *   route.sessionId is the source of truth. When undefined (project just
 *   opened, or a stale route after a tab switch), resolve a default:
 *     1. useSession's localStorage-persisted activeSessionId (if it still
 *        names a live session),
 *     2. else the most-recently-active session (max last_activity_at),
 *     3. else (no sessions) leave it undefined → ChatView renders empty state.
 *   The resolved id is reflected back into the route via setRoute so the
 *   route and UI agree.
 */

import { useEffect } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { AgentRunStatus, Project, SessionListEntry } from '../types';
import type { Route } from '../route';
import { useSessions } from '../hooks/useSessions';
import { useSession } from '../hooks/useSession';
import { useAgent } from '../hooks/useAgent';
import { SessionSidebar } from './SessionSidebar';
import ChatView from './ChatView';

interface ChatTabProps {
  project: Project;
  agentStatus: AgentRunStatus;
  statusTick?: number;
  mentionAgents: Array<{ slug: string; name: string }>;
  route: Extract<Route, { name: 'project' }>;
  setRoute: Dispatch<SetStateAction<Route>>;
  /** Re-fetch this project's runtime fields (e.g. budget) after a turn ends. */
  onRefreshProject?: (id: string) => void;
}

/**
 * Pick the default active session for a project when the route hasn't named
 * one yet. Prefers the persisted selection (if it still names a live session),
 * then the most-recently-active session, else undefined.
 */
function resolveDefaultSession(
  sessions: SessionListEntry[],
  persisted: string | null,
): string | undefined {
  if (sessions.length === 0) return undefined;
  if (persisted && sessions.some((s) => s.session_id === persisted)) {
    return persisted;
  }
  // Most-recently-active: max last_activity_at (null sorts last). Falls back
  // to the first entry when no timestamps are present.
  let best: SessionListEntry | undefined;
  let bestTs = -Infinity;
  for (const s of sessions) {
    const ts = s.last_activity_at ? Date.parse(s.last_activity_at) : NaN;
    const v = Number.isNaN(ts) ? -Infinity : ts;
    if (best === undefined || v > bestTs) {
      best = s;
      bestTs = v;
    }
  }
  return best?.session_id;
}

export default function ChatTab({
  project,
  agentStatus,
  statusTick,
  mentionAgents,
  route,
  setRoute,
  onRefreshProject,
}: ChatTabProps) {
  const projectId = project.project_id;
  const { sessions } = useSessions(projectId);
  const { activeSessionId, setActiveSessionId } = useSession(projectId);
  const { newSession } = useAgent();

  const routeSessionId = route.sessionId;

  // Default-resolution: when the route hasn't named a session yet, resolve one
  // and reflect it into the route (and persist it). Runs whenever the session
  // list or persisted id changes while route.sessionId is still undefined.
  useEffect(() => {
    if (routeSessionId !== undefined) return;
    const resolved = resolveDefaultSession(sessions, activeSessionId);
    if (resolved === undefined) return;
    // Reflect into the route AND persist so route + UI agree.
    setRoute((prev) =>
      prev.name === 'project' && prev.projectId === projectId
        ? { ...prev, sessionId: resolved }
        : prev,
    );
    if (activeSessionId !== resolved) {
      setActiveSessionId(resolved);
    }
  }, [routeSessionId, sessions, activeSessionId, projectId, setRoute, setActiveSessionId]);

  // Selecting a session in the sidebar is a route change. SessionSidebar is
  // controlled (it does NOT persist internally), so ChatTab — the single
  // source of truth — both updates route.sessionId AND persists the selection.
  function handleSessionSelect(sessionId: string) {
    setRoute((prev) =>
      prev.name === 'project' && prev.projectId === projectId
        ? { ...prev, sessionId }
        : prev,
    );
    setActiveSessionId(sessionId);
  }

  // After a session is deleted: if it was the one being viewed, navigate to the
  // most recent remaining session, else to a blank state (sessionId undefined,
  // which lets the resolution effect pick a default or render the empty state).
  // The sidebar passes the already-pruned remaining list (sorted most-recent
  // first), so [0] is the most recent.
  function handleSessionDeleted(deletedId: string, remaining: SessionListEntry[]) {
    if (routeSessionId !== deletedId) return; // a non-viewed row was deleted
    const nextId = remaining[0]?.session_id;
    setRoute((prev) =>
      prev.name === 'project' && prev.projectId === projectId
        ? { ...prev, sessionId: nextId }
        : prev,
    );
    setActiveSessionId(nextId ?? null);
  }

  // "+ new session": mint a genuinely BLANK session on the backend and
  // navigate to it. We must NOT clear route.sessionId to undefined — that
  // would let the resolution effect re-open the most-recent existing session
  // (the bug). Instead we POST /new-session with NO session_id (fresh-create),
  // read the minted `session_id` from the response, and set it on the route.
  //
  // Because the minted id is DEFINED, the resolution effect above early-returns
  // (`if (routeSessionId !== undefined) return;`) and cannot clobber it. The
  // fresh session has no messages and is not yet in the `sessions` list (it
  // materializes server-side only on first inject); ChatView tolerates an
  // unknown id and renders a blank composer.
  //
  // This is a click handler — fire-and-forget the async mint; do not block the
  // UI. Log on failure (minimal error handling).
  function handleNewSession() {
    void newSession(projectId)
      .then((result) => {
        const newId = result.session_id;
        if (!newId) {
          console.error('newSession returned no session_id', result);
          return;
        }
        setRoute((prev) =>
          prev.name === 'project' && prev.projectId === projectId
            ? { ...prev, sessionId: newId }
            : prev,
        );
        setActiveSessionId(newId);
      })
      .catch((err) => {
        console.error('Failed to create a new session', err);
      });
  }

  return (
    <div className="flex h-full min-h-0" data-testid="chat-tab">
      <div className="shrink-0 max-md:hidden">
        <SessionSidebar
          projectId={projectId}
          selectedSessionId={routeSessionId}
          onSessionSelect={handleSessionSelect}
          onNewSession={handleNewSession}
          onSessionDeleted={handleSessionDeleted}
        />
      </div>
      <div className="flex-1 min-w-0 min-h-0">
        <ChatView
          key={projectId}
          projectId={projectId}
          project={project}
          agentStatus={agentStatus}
          statusTick={statusTick}
          mentionAgents={mentionAgents}
          sessionId={routeSessionId}
          onRefreshProject={onRefreshProject}
        />
      </div>
    </div>
  );
}
