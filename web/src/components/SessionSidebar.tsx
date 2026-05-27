// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SessionSidebar — 260px left column of the Chat tab (V1 layout).
 *
 * Header: "SESSIONS" uppercase + total session count.
 * Body: one unified list of SessionListItem rows for ALL sessions, sorted by
 *   last-activity descending (most recent first). A session's state is derived
 *   purely from its current condition (running/waiting/idle/error/
 *   pending_approval/new_session) — there is no separate bucket or grouping.
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

import { useSessions } from '../hooks/useSessions';
import type { SessionListEntry } from '../types';
import { SessionListItem } from './SessionListItem';

export interface SessionSidebarProps {
  projectId: string | null;
  selectedSessionId?: string | null;
  onNewSession?: () => void;
  onSessionSelect?: (sessionId: string) => void;
  /**
   * Called after a session is successfully deleted. ChatTab uses this to
   * navigate away when the deleted session was the one being viewed. Receives
   * the deleted session_id and the list of session_ids that REMAIN (already
   * pruned), so the caller can navigate to the most-recent remaining session.
   */
  onSessionDeleted?: (deletedId: string, remaining: SessionListEntry[]) => void;
}

export function SessionSidebar({
  projectId,
  selectedSessionId,
  onNewSession,
  onSessionSelect,
  onSessionDeleted,
}: SessionSidebarProps) {
  const { sessions, loading, renameSession, deleteSession } = useSessions(projectId);

  // One unified list of ALL sessions, sorted by last-activity descending
  // (most recent first). Null/missing timestamps sort last. Sort a copy so we
  // never mutate the hook's array.
  const sortedSessions: SessionListEntry[] = [...sessions].sort((a, b) => {
    const ta = a.last_activity_at ? Date.parse(a.last_activity_at) : NaN;
    const tb = b.last_activity_at ? Date.parse(b.last_activity_at) : NaN;
    const va = Number.isNaN(ta) ? -Infinity : ta;
    const vb = Number.isNaN(tb) ? -Infinity : tb;
    return vb - va;
  });

  function handleSelect(session: SessionListEntry) {
    onSessionSelect?.(session.session_id);
  }

  async function handleRename(sessionId: string, name: string) {
    try {
      await renameSession(sessionId, name);
    } catch (e) {
      console.error('Failed to rename session', e);
    }
  }

  async function handleDelete(sessionId: string) {
    try {
      await deleteSession(sessionId);
      // Compute the remaining set (sorted, most-recent first) so ChatTab can
      // navigate to the most recent remaining session if the deleted one was
      // being viewed.
      const remaining = sortedSessions.filter((s) => s.session_id !== sessionId);
      onSessionDeleted?.(sessionId, remaining);
    } catch (e) {
      // 409 (running session) or network error — surface to console; the row
      // stays put because deleteSession only prunes on success.
      console.error('Failed to delete session', e);
    }
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
          ({sortedSessions.length})
        </span>
      </div>

      {/* Unified session list (all sessions, most-recent first) */}
      <div
        className="flex-1 overflow-y-auto px-1.5 py-1 flex flex-col gap-0.5"
        data-testid="session-list"
      >
        {loading && sortedSessions.length === 0 && (
          <p className="text-xs text-secondary px-2 py-2" data-testid="session-loading">
            Loading…
          </p>
        )}
        {!loading && sortedSessions.length === 0 && (
          <p className="text-xs text-secondary px-2 py-2" data-testid="session-empty">
            No sessions
          </p>
        )}
        {sortedSessions.map((session) => (
          <SessionListItem
            key={session.session_uuid ?? session.session_id}
            session={session}
            selected={selectedSessionId === session.session_id}
            onSelect={() => handleSelect(session)}
            onRename={handleRename}
            onDelete={handleDelete}
          />
        ))}
      </div>

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
