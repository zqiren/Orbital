// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SessionSidebar — the session half of the project column (spec 137, the
 * folder shell: name, state, status line, then this). It fills whatever room
 * the column gives it; the column owns the width, the tint and the absence of
 * any border. It sits directly on the column tint: the status line above it is
 * the column's ONE white card (spec 137 round 4 — a full-height white session
 * card read as a second panel and clashed with the pinned project list's
 * square corner).
 *
 * Header (item 157, 2026-11-04 — reverses the 2026-10-28 "always shown"
 *   call): "会话 · N" with a bare chevron at its right (tooltip 筛选). The
 *   chevron expands the All | Chats | Automations control inline under the
 *   header; open/closed and the choice are remembered per device. A filter
 *   other than All while the control is closed reads in the header itself
 *   ("会话 · 自动任务 · 4") with the chevron in accent — no pill, no row.
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
 * Cap (session import, 2026-10-21): only the 50 most recent rows below the
 *   pins render, then a "Show older (n)" row that reveals the rest for this
 *   project until the sidebar moves to another one. Importing a heavy Claude
 *   Code / Codex history can add hundreds of rows; the cap applies to the whole
 *   list, imported or not. An automation group counts as one row; n counts
 *   the sessions behind the hidden rows.
 * First row (item 157): "+ 新建会话", styled like a session row on a white
 *   tint (--card / --elevated), sticky at the top while the list scrolls. The
 *   bottom button is gone. Calls onNewSession (no implementation here).
 *
 * ALL sessions are listed regardless of origin. A collapsed automation group
 * still shows its active runs and the selected one, so nothing live or open
 * is ever hidden.
 *
 * Unread badge (spec 108, moved into the daemon by spec 154): a row whose
 * agent replied after the user's last message, and whose reply the user has
 * not checked on any device, carries a dot (`hasUnseenReply` = the one
 * `isUnchecked` predicate, with its 14-day window). The row being viewed
 * never shows it — its reply is checked the moment the list learns of it —
 * and the user's own newer message clears it without a write. A collapsed
 * automation group shows the same dot while any hidden run waits on the user
 * (spec 150, folded into 154).
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

import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Clock, FolderSearch, ListOrdered, Plus } from 'lucide-react';
import { useSessions } from '../hooks/useSessions';
import { useSeenReplies } from '../hooks/useSeenReplies';
import { hasUnseenReply, seenKey } from '../utils/sessionUnread';
import { isWaiting } from '../utils/waitingOnYou';
import type { SessionListEntry } from '../types';
import { SessionListItem } from './SessionListItem';
import { formatRelativeTime } from '../utils/relativeTime';
import { getStatusDisplay, rowDisplayStatus } from './sessionStatus';
import {
  buildSessionList,
  type AutomationGroup,
  type SessionFilter,
} from '../utils/sessionGroups';
import type { StringKey } from '../i18n/strings';
import { useT } from '../i18n/useT';

const FILTER_KEY = 'orbital:sessionSidebar.filter';
const FILTER_OPEN_KEY = 'orbital:sessionSidebar.filterOpen';

/** Rows shown below the pins before "Show older (n)". */
const SESSION_ROW_CAP = 50;

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

function readFilterOpen(): boolean {
  try {
    return localStorage.getItem(FILTER_OPEN_KEY) === '1';
  } catch {
    return false;
  }
}

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
  /**
   * Spec 137: a page (Tasks, Settings) fills the right area. The selected
   * session is then only where the user came from, so its row drops to the
   * faint "was here" mark instead of the selected fill.
   */
  pageOpen?: boolean;
}

export function SessionSidebar({
  projectId,
  selectedSessionId,
  onNewSession,
  onSessionSelect,
  onSessionDeleted,
  pageOpen = false,
}: SessionSidebarProps) {
  const { sessions, loading, renameSession, pinSession, deleteSession } =
    useSessions(projectId);
  const t = useT();

  // Spec 108: the per-device "which reply have I shown" map. The row being
  // viewed is marked seen whenever the list carries a newer reply for it, so
  // a reply that lands while the user is looking never badges later.
  const { seenAt, markSeen } = useSeenReplies(projectId, sessions);
  useEffect(() => {
    if (!selectedSessionId) return;
    const row = sessions.find((s) => s.session_id === selectedSessionId);
    if (!row?.last_reply_at) return;
    const key = seenKey(row);
    const seen = seenAt(key);
    if (seen === undefined || Date.parse(row.last_reply_at) > Date.parse(seen)) {
      markSeen(key, row.last_reply_at);
    }
  }, [selectedSessionId, sessions, seenAt, markSeen]);

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
  const [filterOpen, setFilterOpen] = useState<boolean>(readFilterOpen);
  const toggleFilterOpen = useCallback(() => {
    setFilterOpen((open) => {
      const next = !open;
      try {
        localStorage.setItem(FILTER_OPEN_KEY, next ? '1' : '0');
      } catch {
        /* storage unavailable — still toggles for this visit */
      }
      return next;
    });
  }, []);
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

  // The cap. Revealed per project and in memory only: the reveal belongs to
  // the project it was clicked in, so switching projects starts capped again.
  const [showOlderFor, setShowOlderFor] = useState<string | null>(null);
  const showAll = projectId !== null && showOlderFor === projectId;
  const { shownNodes, olderCount } = useMemo(() => {
    if (showAll || nodes.length <= SESSION_ROW_CAP) return { shownNodes: nodes, olderCount: 0 };
    const hidden = nodes.slice(SESSION_ROW_CAP);
    return {
      shownNodes: nodes.slice(0, SESSION_ROW_CAP),
      olderCount: hidden.reduce(
        (n, node) => n + (node.type === 'session' ? 1 : node.group.runs.length),
        0,
      ),
    };
  }, [nodes, showAll]);

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
    const current = selectedSessionId === session.session_id;
    const selected = current && !pageOpen;
    return (
      <SessionListItem
        key={session.session_uuid ?? session.session_id}
        session={session}
        hideKindChip={inGroup}
        selected={selected}
        wasHere={current && pageOpen}
        // The viewed row is never unread (its marker is written by the
        // effect above, one commit later — this keeps the dot from flashing
        // in between).
        unread={!current && hasUnseenReply(session, seenAt(seenKey(session)))}
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
    // session being viewed always stay visible. A run whose worker is working
    // (spec 102) counts as running, exactly like a manager-running run.
    const visibleRuns = open
      ? group.runs
      : group.runs.filter(
          (r) =>
            r.session_id === selectedSessionId ||
            !getStatusDisplay(rowDisplayStatus(r)).resting,
        );
    const kindLabel = t(GROUP_KIND_KEY[group.kind]);
    const KindIcon = GROUP_KIND_ICON[group.kind];
    // Spec 150 (folded into 154): a collapsed group shows the row's unread
    // dot while any run in it waits on the user. Open, the runs show their own.
    const waitingRun = !open && group.runs.some(
      (r) => r.session_id !== selectedSessionId &&
        isWaiting({ ...r, checked_reply_at: seenAt(seenKey(r)) ?? null }),
    );
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
            {waitingRun && (
              <span
                data-testid="session-group-unread-dot"
                role="img"
                aria-label={t('sessionItem.unread')}
                title={t('sessionItem.unread')}
                className="w-1.5 h-1.5 rounded-full bg-accent shrink-0 self-center"
              />
            )}
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
    <section
      data-testid="session-sidebar"
      aria-label={t('sessionSidebar.header')}
      className="flex flex-col flex-1 min-h-0"
    >
      {/* Header (item 157): "会话 · 16" and a bare chevron that expands the
          filter inline. A filter left on while closed reads in the header:
          "会话 · 自动任务 · 4", chevron in accent. */}
      <div
        className="flex items-center gap-1.5 px-4 pt-2.5 pb-0.5 min-h-[32px] shrink-0 text-xs text-secondary"
        data-testid="session-sidebar-header"
      >
        <span className="font-semibold">{t('sessionSidebar.header')}</span>
        <span className="text-muted" aria-hidden="true">·</span>
        {filter !== 'all' && !filterOpen && (
          <>
            <span data-testid="session-header-filter">{t(FILTERS.find((f) => f.key === filter)!.labelKey)}</span>
            <span className="text-muted" aria-hidden="true">·</span>
          </>
        )}
        <span data-testid="session-active-count">{visibleCount}</span>
        <button
          type="button"
          data-testid="session-filter-toggle"
          aria-expanded={filterOpen}
          aria-label={t('sessionSidebar.filterToggle')}
          title={t('sessionSidebar.filterToggle')}
          onClick={toggleFilterOpen}
          className={`ml-auto -mr-[5px] inline-flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-md transition-colors hover:bg-card-hover hover:text-primary ${
            filterOpen ? 'bg-card-hover text-primary' : filter !== 'all' ? 'text-accent' : 'text-muted'
          }`}
        >
          <ChevronDown
            size={14}
            aria-hidden="true"
            className={`transition-transform duration-150 ${filterOpen ? 'rotate-180' : ''}`}
          />
        </button>
      </div>

      {/* All | Chats | Automations — the segmented control, shown only while
          the header chevron has it open. */}
      {filterOpen && (
        <div className="px-4 pt-0.5 pb-1.5 shrink-0">
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
                      ? 'bg-segment-on text-segment-on-text shadow-[0_1px_2px_rgb(0_0_0/0.06)]'
                      : 'text-secondary hover:text-primary'
                  }`}
                >
                  {t(f.labelKey)}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Session list: pinned, then the rest most-recent first, one row per automation */}
      <div
        className="flex-1 overflow-y-auto px-1.5 pb-1.5 flex flex-col gap-0.5"
        data-testid="session-list"
      >
        {/* Item 157: New session is the first row, on a white tint, sticky
            while the list scrolls; 6px below it before the first session. */}
        <div
          className="sticky top-0 z-[2] -mx-1.5 shrink-0 bg-sidebar px-1.5 pt-0.5 pb-1.5"
          data-testid="session-new-row"
        >
          <button
            type="button"
            data-testid="session-new-button"
            onClick={onNewSession}
            className="w-full flex items-center gap-1.5 px-2.5 py-[7px] rounded-md text-left text-primary bg-card dark:bg-elevated transition-colors"
            style={{ fontSize: '11.5px' }}
          >
            <span className="w-[14px] shrink-0 flex items-center justify-center text-secondary">
              <Plus size={13} strokeWidth={2} aria-hidden="true" />
            </span>
            <span className="flex-1 min-w-0 truncate">{t('sessionSidebar.newSessionRow')}</span>
          </button>
        </div>
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
        {shownNodes.map((node) =>
          node.type === 'session' ? renderRow(node.session) : renderGroup(node.group),
        )}
        {olderCount > 0 && (
          <button
            type="button"
            data-testid="session-show-older"
            onClick={() => setShowOlderFor(projectId)}
            className="w-full px-2.5 py-[7px] rounded-md text-left text-secondary hover:text-primary hover:bg-card-hover transition-colors"
            style={{ fontSize: '11.5px' }}
          >
            {t('session.showOlder', { n: olderCount })}
          </button>
        )}
      </div>

    </section>
  );
}
