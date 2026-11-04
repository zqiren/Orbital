// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The Workbench — the app's home page (spec 154). "The Workbench is the
 * inbox, the session is the desk": the user types a task once and Orbital
 * routes it to its project; the list then holds exactly two kinds of task,
 * 等你 / Waiting on you (asks first, then finished work not yet checked) and
 * 进行中 / Running, with no headers and one muted summary line.
 *
 * Every row comes from the App-level feed store (`useWorkbenchFeed`); the
 * page itself holds no copy. With nothing waiting or running the page is the
 * composer alone and its placeholder carries the one sentence the empty state
 * has (§5.7) — no illustration, no "All clear".
 *
 * Card actions (§5.8):
 *  - the title or → opens the session (viewing it checks its reply on every
 *    device); a session-less ask keeps the old doorway (mint + ask quoted);
 *  - ✓ closes an ask by:user and/or checks the reply; the session itself is
 *    untouched;
 *  - a reply goes into that session exactly as the chat sends it (an ask's
 *    answer closes it by:user) and the card keeps its place in 等你 until the
 *    agent's turn ends.
 *
 * Removed with spec 154 (their routes stay for cached phone SPAs): the page
 * header and project filter, the Today strip, Recently closed, the migrate
 * button and the per-project lens.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Project } from '../types';
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
import { recoverDraft, sendWorkbenchTask, type SendChoice } from './workbench/sendTask';
import TaskCard from './workbench/TaskCard';
import WorkbenchComposer from './workbench/WorkbenchComposer';
import './workbench/workbench.css';

export interface WorkbenchPageProps {
  setRoute: Dispatch<SetStateAction<Route>>;
  projects: Project[];
  /** Installed workers (App-level availability), built-in excluded. */
  agents: Array<{ slug: string; name: string }>;
  phone?: boolean;
  /** Orbital's own model line in the agent picker. */
  managerModel?: string;
}

function newNonce(): string {
  const c = globalThis.crypto as Crypto | undefined;
  return c && typeof c.randomUUID === 'function'
    ? c.randomUUID()
    : `wb-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export default function WorkbenchPage({
  setRoute, projects, agents, phone = false, managerModel,
}: WorkbenchPageProps) {
  const t = useT();
  const feed = useWorkbenchFeed();
  const marks = useCheckedMarks();
  const { newSession, injectMessage } = useAgent();
  const [openError, setOpenError] = useState<string | null>(null);

  // A draft the last page load left unsent comes back into the composer.
  useEffect(() => {
    void recoverDraft();
  }, []);

  const groups = useMemo(() => buildFeedGroups(feed, marks), [feed, marks]);
  const items = useMemo(() => [...groups.waiting, ...groups.running], [groups]);
  const empty = feed.loaded && items.length === 0;

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

  const handleSend = useCallback(
    async (choice: SendChoice) => {
      setOpenError(null);
      const outcome = await sendWorkbenchTask(choice, { newSession, injectMessage });
      if (!outcome.ok) setOpenError(t('workbench.error.open'));
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

  const summary = [
    groups.waiting.length ? t('workbench.summary.waiting', { n: groups.waiting.length }) : '',
    groups.running.length ? t('workbench.summary.running', { n: groups.running.length }) : '',
  ].filter(Boolean).join(' · ');

  return (
    <div
      className={`wb-page flex-1 min-h-0 flex flex-col ${phone ? 'wb-phone' : ''}`}
      data-testid="workbench-page"
    >
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="wb-col">
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
            agents={agents}
            managerModel={managerModel}
            empty={empty || (!feed.loaded && items.length === 0)}
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
                  />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
