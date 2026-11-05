// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The Workbench — the app's home page (spec 154). "The Workbench is the
 * inbox, the session is the desk": the user types a task once and Orbital
 * routes it to its project; the list then holds exactly two kinds of task,
 * 等你 / Waiting on you and 进行中 / Running, with no headers and one muted
 * summary line. They are ONE list by recency (2026-11-05): a task just sent
 * is the first card, the dot tells the state (see feedModel).
 *
 * Every row comes from the App-level feed store (`useWorkbenchFeed`); the
 * page itself holds no copy. With nothing waiting or running the page is the
 * greeting and the composer alone, with the composer's usual placeholder
 * (2026-11-05: the greeting already says what the old empty-state sentence
 * did) — no illustration, no "All clear".
 *
 * Layout (2026-11-05): on desktop one short greeting (picked at random on
 * mount, three variants) sits above the composer — the logo mark tried there
 * first was dropped — and the composer's top edge lands at the golden section of the first
 * screen (38.2% down); the summary line and the cards follow straight under
 * it and the whole column scrolls as one, so the newest card waiting on the
 * user is the first thing under the input. The phone keeps the composer
 * under its header (a low composer would jump with the keyboard).
 *
 * Card actions (§5.8, amended 2026-11-04):
 *  - the title opens the session (viewing it checks its reply on every
 *    device); a session-less ask keeps the old doorway (mint + ask quoted);
 *  - a click on the card body selects it and opens its session in the
 *    reading pane (WorkbenchPane: a right docked panel, a bottom sheet on a
 *    phone); a second click, or ×, closes it. Nothing selected, the page is
 *    unchanged;
 *  - ✓ closes an ask by:user and/or checks the reply; the session itself is
 *    untouched;
 *  - a reply (from the card's field or the pane) goes into that session
 *    exactly as the chat sends it (an ask's answer closes it by:user); the
 *    card moves to 进行中 at once and comes back to 等你 as finished and
 *    unchecked when the agent's turn ends.
 *
 * Removed with spec 154 (their routes stay for cached phone SPAs): the page
 * header and project filter, the Today strip, Recently closed, the migrate
 * button and the per-project lens.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { AgentRunStatus, Project } from '../types';
import type { Route } from '../route';
import { api } from '../config';
import { useT } from '../i18n/useT';
import { useAgent } from '../hooks/useAgent';
import { markChecked, useCheckedMarks } from '../hooks/useSeenReplies';
import {
  activityText, hideClosedAsk, holdCard, refreshFeed, useWorkbenchFeed,
} from '../hooks/useWorkbenchFeed';
import { projectDisplayName } from '../utils/projectDisplayName';
import { formatAskAnswer } from '../utils/askAnswer';
import { buildFeedGroups, type FeedItem } from './workbench/feedModel';
import { classifySessionName } from '../lib/sessionLabel';
import { recoverDraft, sendWorkbenchTask, type SendChoice } from './workbench/sendTask';
import TaskCard from './workbench/TaskCard';
import WorkbenchComposer, { rememberLastProject } from './workbench/WorkbenchComposer';
import WorkbenchPane from './workbench/WorkbenchPane';
import type { StringKey } from '../i18n/strings';
import './workbench/workbench.css';

/** The line above the composer (2026-11-05): one, at random per mount. */
const GREETINGS: StringKey[] = ['wb.greeting.1', 'wb.greeting.2', 'wb.greeting.3'];

export interface WorkbenchPageProps {
  setRoute: Dispatch<SetStateAction<Route>>;
  projects: Project[];
  /** Installed workers (App-level availability), built-in excluded. */
  agents: Array<{ slug: string; name: string }>;
  phone?: boolean;
  /** Orbital's own model line in the agent picker. */
  managerModel?: string;
  /** Per project: the manager's run status and its tick (the pane's chat). */
  agentStatuses?: Record<string, AgentRunStatus>;
  statusTicks?: Record<string, number>;
}

function newNonce(): string {
  const c = globalThis.crypto as Crypto | undefined;
  return c && typeof c.randomUUID === 'function'
    ? c.randomUUID()
    : `wb-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export default function WorkbenchPage({
  setRoute, projects, agents, phone = false, managerModel,
  agentStatuses = {}, statusTicks = {},
}: WorkbenchPageProps) {
  const t = useT();
  const [greeting] = useState(() => Math.floor(Math.random() * GREETINGS.length));
  const feed = useWorkbenchFeed();
  const marks = useCheckedMarks();
  const { newSession, injectMessage } = useAgent();
  const [openError, setOpenError] = useState<string | null>(null);

  // A draft the last page load left unsent comes back into the composer.
  useEffect(() => {
    void recoverDraft();
  }, []);

  const groups = useMemo(() => buildFeedGroups(feed, marks), [feed, marks]);
  const items = groups.items;

  // The reading pane: the selected card, by key. A card that leaves the list
  // (✓, checked elsewhere) closes it.
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const selected = selectedKey ? items.find((i) => i.key === selectedKey) ?? null : null;
  useEffect(() => {
    if (selectedKey && feed.loaded && !selected) setSelectedKey(null);
  }, [selectedKey, selected, feed.loaded]);
  const handleSelect = useCallback((item: FeedItem) => {
    setSelectedKey((cur) => (cur === item.key ? null : item.key));
  }, []);

  // Sidebar order, minus projects excluded from the Workbench.
  const pickable = useMemo(
    () => projects.filter((p) => !p.workbench_exclude_global),
    [projects],
  );
  const projectName = useCallback(
    (pid: string | null) => {
      if (!pid) return null;
      const p = projects.find((x) => x.project_id === pid);
      return p ? projectDisplayName(p, t) : null;
    },
    [projects, t],
  );
  const agentName = useCallback(
    (slug: string) => agents.find((a) => a.slug === slug)?.name ?? slug,
    [agents],
  );

  // No API key (2026-11-05): Quick Tasks' manager is not ready, so 自动 has
  // nothing to route with. Read from the loaded project rows, so adding a key
  // takes effect when the list refreshes.
  const keyless = projects.find((p) => p.is_scratch)?.manager_ready === false;

  const handleSend = useCallback(
    async (choice: SendChoice) => {
      setOpenError(null);
      const outcome = await sendWorkbenchTask(choice, { newSession, injectMessage });
      if (!outcome.ok) setOpenError(t('workbench.error.open'));
      // A routed send too: the next keyless default is where the last task went.
      else rememberLastProject(outcome.projectId);
      return outcome.ok;
    },
    [newSession, injectMessage, t],
  );

  // A session-less ask: mint a session and open it with the ask quoted (the
  // pre-154 doorway), so what the user sends answers it.
  async function openAskDoorway(projectId: string, ask: { id: string; text: string }) {
    setOpenError(null);
    try {
      const minted = await newSession(projectId);
      if (!minted?.session_id) {
        setOpenError(t('workbench.error.open'));
        return;
      }
      setRoute({ name: 'project', projectId, sessionId: minted.session_id, askReply: ask });
    } catch {
      setOpenError(t('workbench.error.open'));
    }
  }

  function handleOpen(item: FeedItem) {
    if (item.pending) {
      const { project_id: pid, session_id: sid } = item.pending;
      if (pid) setRoute({ name: 'project', projectId: pid, sessionId: sid ?? undefined });
      return;
    }
    const row = item.row;
    if (!row) return;
    if (!row.session_id) {
      if (row.ask) void openAskDoorway(row.project_id, { id: row.ask.id, text: row.ask.text });
      return;
    }
    setRoute({ name: 'project', projectId: row.project_id, sessionId: row.session_id });
  }

  function handleConfirm(item: FeedItem) {
    const row = item.row;
    if (!row) return;
    if (row.ask) {
      hideClosedAsk(row.project_id, row.ask.id);
      api(`/api/v2/workbench/${encodeURIComponent(row.project_id)}/entries/${encodeURIComponent(row.ask.id)}/exit`, {
        method: 'POST',
        body: JSON.stringify({ kind: 'fulfilled' }),
      }).catch(() => void refreshFeed());
    }
    const key = row.session_uuid ?? row.session_id;
    if (key && row.last_reply_at) void markChecked(row.project_id, key, row.last_reply_at);
  }

  async function handleReply(item: FeedItem, words: string): Promise<boolean> {
    const row = item.row;
    if (!row) return false;
    setOpenError(null);
    try {
      let sessionId = row.session_id;
      if (!sessionId) {
        const minted = await newSession(row.project_id);
        if (!minted?.session_id) throw new Error('mint failed');
        sessionId = minted.session_id;
      }
      const ask = row.ask;
      const content = ask ? formatAskAnswer(ask.id, ask.text, words) : words;
      const target = row.agent ?? undefined;
      holdCard({ ...row, session_id: sessionId });
      await injectMessage(
        row.project_id, content, target, newNonce(), undefined, sessionId, !!target,
        undefined, ask ? { id: ask.id, words } : undefined,
      );
      void refreshFeed();
      return true;
    } catch {
      setOpenError(t('workbench.error.open'));
      void refreshFeed();
      return false;
    }
  }

  // The pane's subject: a real session in a known project.
  const paneRow = selected?.row && selected.row.session_id ? selected.row : null;
  const paneProject = paneRow ? projects.find((p) => p.project_id === paneRow.project_id) : undefined;
  const paneTitle = paneRow
    ? (paneRow.trigger_type ? classifySessionName(paneRow.name).displayName : null)
      ?? paneRow.name ?? t('workbench.card.untitled')
    : '';
  const handlePaneSend = useCallback(() => {
    if (!paneRow) return;
    holdCard(paneRow);
    void refreshFeed();
  }, [paneRow]);

  const summary = [
    groups.waiting.length ? t('workbench.summary.waiting', { n: groups.waiting.length }) : '',
    groups.running.length ? t('workbench.summary.running', { n: groups.running.length }) : '',
  ].filter(Boolean).join(' · ');

  return (
    <div
      className={`wb-page relative flex-1 min-h-0 flex ${phone ? 'wb-phone' : ''}`}
      data-testid="workbench-page"
    >
      <div className="flex-1 min-w-0 min-h-0 overflow-y-auto" data-testid="workbench-scroller">
        {/* A direct child of the scroller, so its % height is of the
            scroller's (definite) height: the golden section of the screen. */}
        {!phone && <div className="wb-golden" aria-hidden="true" />}
        <div className="wb-col">
          {/* No greeting while keyless (it promises routing that cannot
              run); its empty slot keeps the composer at the golden section. */}
          {!phone && (keyless
            ? <div className="wb-greeting" aria-hidden="true" />
            : <p className="wb-greeting" data-testid="workbench-greeting">{t(GREETINGS[greeting])}</p>)}
          {openError && (
            <div
              role="alert"
              data-testid="workbench-open-error"
              className="mb-4 rounded-lg bg-warning-surface px-4 py-2 text-xs text-warning-text"
            >
              {openError}
            </div>
          )}
          <WorkbenchComposer
            projects={pickable}
            keyless={keyless}
            agents={agents}
            managerModel={managerModel}
            onSend={handleSend}
          />
          {items.length > 0 && (
            <>
              <div className="wb-summary" data-testid="workbench-summary">{summary}</div>
              <div className="wb-cards" data-testid="workbench-cards">
                {items.map((item) => (
                  <TaskCard
                    key={item.key}
                    item={item}
                    phone={phone}
                    fresh={!!item.pending}
                    projectName={projectName}
                    agentName={agentName}
                    activity={item.row ? activityText(feed, item.row.project_id, item.row.session_id) : null}
                    onOpen={handleOpen}
                    onConfirm={handleConfirm}
                    onReply={handleReply}
                    selected={item.key === selectedKey}
                    onSelect={item.row?.session_id ? handleSelect : undefined}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
      {paneRow && paneProject && selected && (
        <WorkbenchPane
          project={paneProject}
          sessionId={paneRow.session_id!}
          title={paneTitle}
          dot={selected.dot}
          agentStatus={agentStatuses[paneRow.project_id] ?? 'idle'}
          statusTick={statusTicks[paneRow.project_id]}
          agents={agents}
          phone={phone}
          onClose={() => setSelectedKey(null)}
          onOpenSession={(sid) => setRoute({ name: 'project', projectId: paneRow.project_id, sessionId: sid })}
          onUserSend={handlePaneSend}
        />
      )}
    </div>
  );
}
