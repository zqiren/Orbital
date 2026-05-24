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
 * Props:
 *   projectId       — passed to useSessions / useSession hooks.
 *   onNewSession    — called when the user clicks "+ new session".
 *   onSessionSelect — optional callback when active session changes;
 *                     the sidebar also persists via useSession internally.
 */

import { useState } from 'react';
import { useSessions } from '../hooks/useSessions';
import { useSession } from '../hooks/useSession';
import type { SessionListEntry } from '../types';
import { SessionListItem } from './SessionListItem';
import { isActiveStatus, isArchivedStatus } from './sessionStatus';

export interface SessionSidebarProps {
  projectId: string | null;
  onNewSession?: () => void;
  onSessionSelect?: (sessionId: string) => void;
}

export function SessionSidebar({ projectId, onNewSession, onSessionSelect }: SessionSidebarProps) {
  const { sessions, loading } = useSessions(projectId);
  const { activeSessionId, setActiveSessionId } = useSession(projectId);
  const [archivedOpen, setArchivedOpen] = useState(false);

  const activeSessions: SessionListEntry[] = sessions.filter((s) => isActiveStatus(s.status));
  const archivedSessions: SessionListEntry[] = sessions.filter((s) => isArchivedStatus(s.status));

  function handleSelect(session: SessionListEntry) {
    setActiveSessionId(session.session_id);
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
          className="text-secondary font-semibold"
          style={{
            fontSize: '9.5px',
            letterSpacing: '0.8px',
            textTransform: 'uppercase',
            fontFamily: 'var(--font-mono)',
          }}
        >
          Sessions
        </span>
        <span
          className="text-secondary text-xs"
          data-testid="session-active-count"
          style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px' }}
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
            selected={activeSessionId === session.session_id}
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
              style={{
                fontSize: '9.5px',
                letterSpacing: '0.8px',
                textTransform: 'uppercase',
                fontFamily: 'var(--font-mono)',
              }}
              className="font-semibold"
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
                  selected={activeSessionId === session.session_id}
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
          className="w-full rounded-md border border-border text-secondary hover:text-primary hover:border-secondary/60 hover:bg-card-hover transition-colors text-sm py-1.5"
          style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}
        >
          + new session
        </button>
      </div>
    </aside>
  );
}
