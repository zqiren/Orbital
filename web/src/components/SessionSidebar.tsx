// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SessionSidebar — 260px left column of the Chat tab (V1 layout).
 *
 * Header: "SESSIONS" uppercase + the count for the active filter, then an
 *   All | Chats | Automations filter (remembered per device).
 * Body: pinned rows first (spec 067), then everything else most recent first.
 *   Sessions a schedule, a file watch or the queue fired are folded into ONE
 *   row per automation, sitting where its latest run was (see
 *   utils/sessionGroups). Spec 066 4b had parked them in a section below the
 *   chats so hundreds of identical runs stopped burying the conversations —
 *   but in a project with hundreds of chats that put them a long scroll away.
 *   One row per automation un-buries the chats without hiding the
 *   automations, and the filter makes either half one click away. A session's
 *   state is derived purely from its current condition
 *   (running/waiting/idle/error/pending_approval/new_session).
 * Bottom: "+ new session" button — calls onNewSession prop (no implementation here).
 *
 * ALL sessions are listed regardless of origin. A collapsed automation group
 * still shows its active runs and the selected one, so nothing live or open
 * is ever hidden.
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

import { useCallback, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Clock, FolderSearch, ListOrdered } from 'lucide-react';
import { useSessions } from '../hooks/useSessions';
import type { SessionListEntry } from '../types';
import { SessionListItem } from './SessionListItem';
import { formatRelativeTime } from '../utils/relativeTime';
import { getStatusDisplay } from './sessionStatus';
import {
  buildSessionList,
  type AutomationGroup,
  type SessionFilter,
} from '../utils/sessionGroups';
import type { StringKey } from '../i18n/strings';
import { useT } from '../i18n/useT';

const FILTER_KEY = 'orbital:sessionSidebar.filter';

const FILTERS: { key: SessionFilter; labelKey: StringKey }[] = [
  { key: 'all', labelKey: 'sessionSidebar.filter.all' },
  { key: 'chats', labelKey: 'sessionSidebar.filter.chats' },
  { key: 'automations', labelKey: 'sessionSidebar.automations' },
];

const GROUP_KIND_KEY: Record<AutomationGroup['kind'], StringKey> = {
  queue: 'sessionItem.kind.queue',
  schedule: 'sessionItem.kind.schedule',
  file_watch: 'sessionItem.kind.fileWatch',
};

/** The group header states its kind with an icon (labelled by the same
 * strings the row chips use) rather than a text chip: at 260px the chip cost
 * the automation's NAME most of its room. */
const GROUP_KIND_ICON = {
  queue: ListOrdered,
  schedule: Clock,
  file_watch: FolderSearch,
} as const;

function readFilter(): SessionFilter {
  try {
    const v = localStorage.getItem(FILTER_KEY);
    return v === 'chats' || v === 'automations' ? v : 'all';
  } catch {
    return 'all'; // storage unavailable (private/locked-down webview)
  }
}

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
  const { sessions, loading, renameSession, pinSession, deleteSession } =
    useSessions(projectId);
  const t = useT();

  // One unified list of ALL sessions: pinned rows first (spec 067), then
  // last-activity descending (most recent first) WITHIN each group. Null/
  // missing timestamps sort last. Sort a copy so we never mutate the hook's
  // array. Bug #48 (fix D): memoized (with stable useCallback handlers below)
  // so an agent.status-triggered refresh doesn't re-sort and re-render every
  // row on unrelated parent renders.
  //
  // A pin outranks liveness deliberately: a running-but-unpinned session stays
  // below the pins and is found by its status glyph, not by its position. The
  // alternative (float running rows above pins) would let the top of the list
  // move on its own, which is half of what pinning exists to stop.
  const sortedSessions: SessionListEntry[] = useMemo(
    () =>
      [...sessions].sort((a, b) => {
        if (!!a.pinned !== !!b.pinned) return a.pinned ? -1 : 1;
        const ta = a.last_activity_at ? Date.parse(a.last_activity_at) : NaN;
        const tb = b.last_activity_at ? Date.parse(b.last_activity_at) : NaN;
        const va = Number.isNaN(ta) ? -Infinity : ta;
        const vb = Number.isNaN(tb) ? -Infinity : tb;
        return vb - va;
      }),
    [sessions],
  );

  const [filter, setFilter] = useState<SessionFilter>(readFilter);
  const chooseFilter = useCallback((next: SessionFilter) => {
    setFilter(next);
    try {
      localStorage.setItem(FILTER_KEY, next);
    } catch {
      /* storage unavailable — the filter still works for this visit */
    }
  }, []);

  // Pinned rows (any kind — a pin is the user saying "keep this at the top"),
  // then the rest in the order above, with each automation's runs folded into
  // one group node.
  const { pinned, nodes, counts } = useMemo(
    () => buildSessionList(sortedSessions, filter),
    [sortedSessions, filter],
  );

  // Which automation groups are open. Not persisted: a group is opened to look
  // something up, and a sidebar that remembers every expansion drifts back to
  // the wall of runs this grouping exists to remove.
  const [openGroups, setOpenGroups] = useState<ReadonlySet<string>>(() => new Set());
  const toggleGroup = useCallback((key: string) => {
    setOpenGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const handleSelect = useCallback(
    (sessionId: string) => {
      onSessionSelect?.(sessionId);
    },
    [onSessionSelect],
  );

  const handleRename = useCallback(
    async (sessionId: string, name: string) => {
      try {
        await renameSession(sessionId, name);
      } catch (e) {
        console.error('Failed to rename session', e);
      }
    },
    [renameSession],
  );

  const handlePin = useCallback(
    async (sessionId: string, pinned: boolean) => {
      try {
        await pinSession(sessionId, pinned);
      } catch (e) {
        console.error('Failed to pin session', e);
      }
    },
    [pinSession],
  );

  const handleDelete = useCallback(
    async (sessionId: string) => {
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
    },
    [deleteSession, onSessionDeleted, sortedSessions],
  );

  function renderRow(session: SessionListEntry, inGroup = false) {
    return (
      <SessionListItem
        key={session.session_uuid ?? session.session_id}
        session={session}
        hideKindChip={inGroup}
        selected={selectedSessionId === session.session_id}
        onSelect={handleSelect}
        onRename={handleRename}
        onPin={handlePin}
        onDelete={handleDelete}
      />
    );
  }

  function renderGroup(group: AutomationGroup) {
    const open = openGroups.has(group.key);
    // Collapsed hides only resting runs: a running/waiting/blocked run and the
    // session being viewed always stay visible.
    const visibleRuns = open
      ? group.runs
      : group.runs.filter(
          (r) => r.session_id === selectedSessionId || !getStatusDisplay(r.status).resting,
        );
    const kindLabel = t(GROUP_KIND_KEY[group.kind]);
    const KindIcon = GROUP_KIND_ICON[group.kind];
    return (
      <div key={`group:${group.key}`} data-testid="session-automation-group">
        <button
          type="button"
          data-testid="session-automation-group-toggle"
          aria-expanded={open}
          onClick={() => toggleGroup(group.key)}
          className="w-full flex items-center gap-1.5 px-2.5 py-[7px] rounded-md text-left select-none transition-colors hover:bg-card-hover"
        >
          <span className="w-[14px] shrink-0 flex items-center justify-center text-secondary">
            {open ? (
              <ChevronDown size={12} aria-hidden="true" />
            ) : (
              <ChevronRight size={12} aria-hidden="true" />
            )}
          </span>
          <span className="flex-1 min-w-0 flex items-baseline gap-1.5">
            <KindIcon
              size={12}
              role="img"
              aria-label={kindLabel}
              className="shrink-0 self-center text-secondary"
            />
            {/* A group with no trigger name left (queue items, renamed runs)
                is named by its kind. */}
            <span
              className="flex-1 min-w-0 truncate font-medium text-primary"
              style={{ fontSize: '11.5px' }}
              title={group.name ?? kindLabel}
            >
              {group.name ?? kindLabel}
            </span>
            <span className="text-2xs text-secondary shrink-0">
              {t('sessionSidebar.group.runs', { n: group.runs.length })}
              {' · '}
              {formatRelativeTime(group.runs[0].last_activity_at, t)}
            </span>
          </span>
        </button>
        {visibleRuns.length > 0 && (
          <div className="ml-[13px] pl-1.5 border-l border-border/60 flex flex-col gap-0.5">
            {visibleRuns.map((r) => renderRow(r, true))}
          </div>
        )}
      </div>
    );
  }

  const visibleCount = counts[filter];
  const emptyKey: StringKey =
    filter === 'chats'
      ? 'sessionSidebar.emptyChats'
      : filter === 'automations'
        ? 'sessionSidebar.emptyAutomations'
        : 'sessionSidebar.empty';

  return (
    <aside
      data-testid="session-sidebar"
      style={{ width: 260, minWidth: 260, maxWidth: 260 }}
      className="flex flex-col h-full bg-sidebar border-r border-border/60 shadow-well"
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
          }}
        >
          {t('sessionSidebar.header')}
        </span>
        <span
          className="font-mono text-secondary text-xs"
          data-testid="session-active-count"
          style={{ fontSize: '9.5px' }}
        >
          ({visibleCount})
        </span>
      </div>

      {/* All | Chats | Automations — a segmented control on a track, the same
          pattern as the Tasks tab's Queue | Automations switch. */}
      <div className="px-3 pb-1.5 shrink-0">
        <div
          role="tablist"
          aria-label={t('sessionSidebar.filter.label')}
          className="flex gap-0.5 rounded-lg border border-border bg-nav p-0.5"
        >
          {FILTERS.map((f) => {
            const active = filter === f.key;
            return (
              <button
                key={f.key}
                type="button"
                role="tab"
                aria-selected={active}
                data-testid={`session-filter-${f.key}`}
                onClick={() => chooseFilter(f.key)}
                className={`flex-auto whitespace-nowrap text-[11px] font-medium px-1.5 py-1 rounded-md transition-colors duration-150 ${
                  active
                    ? 'bg-card text-primary shadow-[0_1px_2px_rgb(0_0_0/0.06)]'
                    : 'text-secondary hover:text-primary'
                }`}
              >
                {t(f.labelKey)}
              </button>
            );
          })}
        </div>
      </div>

      {/* Session list: pinned, then the rest most-recent first, one row per automation */}
      <div
        className="flex-1 overflow-y-auto px-1.5 py-1 flex flex-col gap-0.5"
        data-testid="session-list"
      >
        {loading && sortedSessions.length === 0 && (
          <p className="text-xs text-secondary px-2 py-2" data-testid="session-loading">
            {t('sessionSidebar.loading')}
          </p>
        )}
        {!loading && visibleCount === 0 && (
          <p className="text-xs text-secondary px-2 py-2" data-testid="session-empty">
            {t(emptyKey)}
          </p>
        )}
        {pinned.map((p) => renderRow(p))}
        {/* Hairline closing the pinned block — the same "pinned above a rule"
            pattern the nav column uses for Quick Tasks. Only rendered when
            there is both a pin AND something under it — a rule at the very
            bottom of the list would separate nothing. */}
        {pinned.length > 0 && nodes.length > 0 && (
          <div
            role="separator"
            data-testid="session-pin-divider"
            className="my-1 border-t border-border/60"
          />
        )}
        {nodes.map((node) =>
          node.type === 'session' ? renderRow(node.session) : renderGroup(node.group),
        )}
      </div>

      {/* New session button */}
      <div className="shrink-0 px-3 pb-3 pt-2">
        <button
          type="button"
          data-testid="session-new-button"
          onClick={onNewSession}
          className="w-full rounded-lg border border-border/60 bg-card/70 py-1.5 text-sm font-medium text-primary shadow-[0_1px_2px_rgb(0_0_0/0.04)] transition-[box-shadow,transform,background-color] duration-100 ease-out hover:bg-card hover:shadow-[0_2px_5px_rgb(0_0_0/0.06)] active:scale-[0.98] motion-reduce:transition-none motion-reduce:active:scale-100"
          style={{ fontSize: '12px' }}
        >
          {t('sessionSidebar.newSession')}
        </button>
      </div>
    </aside>
  );
}
