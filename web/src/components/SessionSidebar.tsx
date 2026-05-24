// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SessionSidebar — 260px left column of the Chat tab (V1 layout).
 *
 * Header: "SESSIONS" uppercase + count of active sessions.
 * Body: SessionListItem rows for ACTIVE sessions (running/waiting/pending_approval/idle).
 * Footer: collapsible "archived (N)" section for stopped sessions.
 * Bottom: "+ new session" button — calls onNewSession prop (no implementation here).
 *
 * ALL sessions are listed regardless of origin (manual + queue-dispatched).
 * Queue-dispatched sessions get a subtle hue variation on their status dot,
 * handled by SessionListItem.
 *
 * Selection is CONTROLLED: the highlighted (active) session is driven by the
 * `selectedSessionId` prop, NOT an internal hook. ChatTab owns the single
 * source of truth (route.sessionId, resolved from route → persisted →
 * most-recent) and passes it down, so the sidebar highlight can never disagree
 * with the conversation being shown. Persistence (useSession.setActiveSessionId)
 * is owned by ChatTab too — the sidebar only reports selections via
 * onSessionSelect.
 *
 * Props:
 *   projectId         — passed to useSessions for the session list.
 *   selectedSessionId — the active session_id to highlight (from ChatTab's
 *                       route resolution). null/undefined → no row highlighted.
 *   onNewSession      — called when the user clicks "+ new session".
 *   onSessionSelect   — called with the clicked session_id; ChatTab updates the
 *                       route and persists.
 */

import { useState } from 'react';
import { useSessions } from '../hooks/useSessions';
import type { SessionListEntry } from '../types';
import { SessionListItem } from './SessionListItem';
import { isActiveStatus, isArchivedStatus } from './sessionStatus';

export interface SessionSidebarProps {
  projectId: string | null;
  selectedSessionId?: string | null;
  onNewSession?: () => void;
  onSessionSelect?: (sessionId: string) => void;
}

export function SessionSidebar({
  projectId,
  selectedSessionId,
  onNewSession,
  onSessionSelect,
}: SessionSidebarProps) {
  const { sessions, loading } = useSessions(projectId);
  const [archivedOpen, setArchivedOpen] = useState(false);

  const activeSessions: SessionListEntry[] = sessions.filter((s) => isActiveStatus(s.status));
  const archivedSessions: SessionListEntry[] = sessions.filter((s) => isArchivedStatus(s.status));

  function handleSelect(session: SessionListEntry) {
    onSessionSelect?.(session.session_id);
  }

  return (
    <aside
      data-testid="session-sidebar"
      style={{ width: 260, minWidth: 260, maxWidth: 260 }}
      className="flex flex-col h-full bg-sidebar border-r border-border"
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 pt-3 pb-1.5 shrink-0"
        data-testid="session-sidebar-header"
      >
        <span
          className="font-mono text-secondary font-semibold"
          style={{
            fontSize: '9.5px',
            letterSpacing: '0.8px',
            textTransform: 'uppercase',
          }}
        >
          Sessions
        </span>
        <span
          className="font-mono text-secondary text-xs"
          data-testid="session-active-count"
          style={{ fontSize: '9.5px' }}
        >
          {activeSessions.length}
        </span>
      </div>

      {/* Active session list */}
      <div
        className="flex-1 overflow-y-auto px-1.5 py-1 flex flex-col gap-0.5"
        data-testid="session-list"
      >
        {loading && activeSessions.length === 0 && (
          <p className="text-xs text-secondary px-2 py-2" data-testid="session-loading">
            Loading…
          </p>
        )}
        {!loading && activeSessions.length === 0 && (
          <p className="text-xs text-secondary px-2 py-2" data-testid="session-empty">
            No active sessions
          </p>
        )}
        {activeSessions.map((session) => (
          <SessionListItem
            key={session.session_uuid ?? session.session_id}
            session={session}
            selected={selectedSessionId === session.session_id}
            onSelect={() => handleSelect(session)}
          />
        ))}
      </div>

      {/* Archived section */}
      {archivedSessions.length > 0 && (
        <div className="shrink-0 border-t border-border px-1.5 pb-1" data-testid="session-archived-section">
          <button
            type="button"
            data-testid="session-archived-toggle"
            onClick={() => setArchivedOpen((v) => !v)}
            className="flex items-center gap-1 w-full px-2 py-1.5 text-secondary hover:text-primary transition-colors"
            aria-expanded={archivedOpen}
          >
            <span style={{ fontSize: '10px' }} aria-hidden="true">
              {archivedOpen ? '▾' : '▸'}
            </span>
            <span
              className="font-mono font-semibold"
              style={{
                fontSize: '9.5px',
                letterSpacing: '0.8px',
                textTransform: 'uppercase',
              }}
            >
              archived ({archivedSessions.length})
            </span>
          </button>
          {archivedOpen && (
            <div className="flex flex-col gap-0.5 mt-0.5" data-testid="session-archived-list">
              {archivedSessions.map((session) => (
                <SessionListItem
                  key={session.session_uuid ?? session.session_id}
                  session={session}
                  selected={selectedSessionId === session.session_id}
                  onSelect={() => handleSelect(session)}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* New session button */}
      <div className="shrink-0 px-3 pb-3 pt-2">
        <button
          type="button"
          data-testid="session-new-button"
          onClick={onNewSession}
          className="w-full font-mono rounded-md border border-border text-secondary hover:text-primary hover:border-secondary/60 hover:bg-card-hover transition-colors text-sm py-1.5"
          style={{ fontSize: '12px' }}
        >
          + new session
        </button>
      </div>
    </aside>
  );
}
