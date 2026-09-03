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
 *
 * Composer prefill (Workbench card-tap doorway, spec 2026-07-24 §5.3):
 *   `route.draft`, when set, is threaded to ChatView as `initialDraft` — a
 *   one-shot text to load into the composer (never auto-sent, never a
 *   session spawn). `handleDraftConsumed` clears it back to undefined once
 *   ChatView reports the draft applied.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type {
  ActivityEvent,
  AgentRunStatus,
  Project,
  SessionListEntry,
  WebSocketEvent,
} from '../types';
import type { Route } from '../route';
import { useSessions } from '../hooks/useSessions';
import { useSession } from '../hooks/useSession';
import { useAgent } from '../hooks/useAgent';
import { useFiles } from '../hooks/useFiles';
import { useChatHistory } from '../hooks/useChatHistory';
import { useWebSocket } from '../hooks/useWebSocket';
import { useAnnotations } from '../hooks/useAnnotations';
import { usePanelDockable, usePanelState } from '../hooks/usePanelState';
import {
  eventAsMessage,
  latestScreenshot,
  pathForEvent,
  touchedFiles,
  viewForEvent,
} from '../utils/panelSelectors';
import { SessionSidebar } from './SessionSidebar';
import ChatView from './ChatView';
import ScopeChip from './ScopeChip';
import FilePreviewDrawer from './FilePreviewDrawer';
import WorkspacePanel from './panel/WorkspacePanel';
import FilesView from './panel/FilesView';
import BrowserView from './panel/BrowserView';
import PanelHandle from './panel/PanelHandle';

/**
 * Statuses that mean "a turn is in flight". `pending_approval` counts: the
 * agent is mid-turn waiting on the user, which is exactly the moment the panel
 * exists for (D6 — intercept or assist), so it must not collapse there.
 */
const IN_RUN: ReadonlySet<AgentRunStatus> = new Set<AgentRunStatus>([
  'running',
  'waiting',
  'pending_approval',
]);

/** Keep the live-event tail bounded; only the touched set is derived from it. */
const MAX_LIVE_EVENTS = 200;

/** Stable empty tail, so a session switch doesn't churn the touched memo. */
const NO_EVENTS: ActivityEvent[] = [];

interface ChatTabProps {
  project: Project;
  agentStatus: AgentRunStatus;
  statusTick?: number;
  mentionAgents: Array<{ slug: string; name: string }>;
  route: Extract<Route, { name: 'project' }>;
  setRoute: Dispatch<SetStateAction<Route>>;
  /** Re-fetch this project's runtime fields (e.g. budget) after a turn ends. */
  onRefreshProject?: (id: string) => void;
  /**
   * App-level projects list, consumed by the Quick Tasks ScopeChip (the
   * cross-project read-scope multi-select — Spec 012 §2c). Only read when
   * `project.is_scratch`; callers rendering normal projects may omit it.
   */
  projects?: Project[];
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
  projects,
}: ChatTabProps) {
  const projectId = project.project_id;
  const { sessions } = useSessions(projectId);
  const { activeSessionId, setActiveSessionId } = useSession(projectId);
  const { newSession } = useAgent();

  const routeSessionId = route.sessionId;

  // ── Spec 078: the workspace panel (Files · Browser) ─────────────────────
  // Docked only above the push threshold and never on mobile; below it the
  // FilePreviewDrawer stays the overlay ProjectDetail owns.
  const docked = usePanelDockable();
  const {
    expanded: panelExpanded,
    view: panelView,
    file: panelFile,
    expand: expandPanel,
    collapse: collapsePanel,
    setView: setPanelView,
    setFile: setPanelFile,
    expandForEvent,
    onRunStart,
    onRunEnd,
  } = usePanelState(projectId, routeSessionId);
  const { annotating, setAnnotating, annotations, add: addAnnotation } =
    useAnnotations(routeSessionId);
  const { messages, loadHistory, clearMessages } = useChatHistory({
    sessionId: routeSessionId ?? null,
  });
  const { saveFileContent } = useFiles();
  const { on, off } = useWebSocket();

  // Live activity for the viewed session. Kept separately from `messages` so
  // the touched list stays current mid-run without refetching the transcript:
  // each event is replayed through the same `touchedFiles` selector as a
  // synthetic tool-call row. The session key is stored WITH the events so a
  // session switch discards them by derivation, not by a reset effect.
  const sessionKey = `${projectId}:${routeSessionId ?? ''}`;
  const [liveBySession, setLiveBySession] = useState<{ key: string; events: ActivityEvent[] }>(
    () => ({ key: sessionKey, events: [] }),
  );
  const liveEvents = liveBySession.key === sessionKey ? liveBySession.events : NO_EVENTS;

  // History is loaded lazily — the first time the panel is actually opened for
  // a session, and again after a turn ends (new tool calls, new screenshots).
  const loadedKeyRef = useRef<string | null>(null);
  const historyStaleRef = useRef(false);
  const sessionKeyRef = useRef(sessionKey);

  useEffect(() => {
    if (sessionKeyRef.current === sessionKey) return;
    sessionKeyRef.current = sessionKey;
    loadedKeyRef.current = null;
    historyStaleRef.current = false;
    clearMessages(); // the previous session's transcript must not badge this one
  }, [sessionKey, clearMessages]);

  useEffect(() => {
    if (!docked || !panelExpanded || !routeSessionId) return;
    if (loadedKeyRef.current === sessionKey && !historyStaleRef.current) return;
    loadedKeyRef.current = sessionKey;
    historyStaleRef.current = false;
    void loadHistory(projectId);
  }, [docked, panelExpanded, projectId, routeSessionId, sessionKey, loadHistory]);

  // Lifecycle (D8) driven by the live activity stream. Handled synchronously
  // in the socket callback rather than from a rendered `lastEvent`, so a burst
  // of events in one tick cannot swallow the file path of all but the last.
  const handleActivityRef = useRef<(event: WebSocketEvent) => void>(() => {});
  // Written in an effect, not during render: render must stay pure, and the
  // socket can only reach the ref after the first commit anyway.
  useEffect(() => {
    handleActivityRef.current = (event: WebSocketEvent) => {
      if (event.type !== 'agent.activity') return;
      const activity = event as ActivityEvent;
      if (activity.project_id !== projectId) return;
      if (activity.session_id && routeSessionId && activity.session_id !== routeSessionId) return;

      if (eventAsMessage(activity)) {
        setLiveBySession((prev) => ({
          key: sessionKey,
          events: [...(prev.key === sessionKey ? prev.events : []), activity].slice(
            -MAX_LIVE_EVENTS,
          ),
        }));
      }
      const view = viewForEvent(activity);
      if (view === null) return; // command-only turns do not move the panel (§13.3)
      expandForEvent(view);
      if (view === 'files') {
        const path = pathForEvent(activity);
        if (path) setPanelFile(path);
      }
    };
  });

  useEffect(() => {
    if (!docked) return;
    const listener = (event: WebSocketEvent) => handleActivityRef.current(event);
    on('agent.activity', listener);
    return () => off('agent.activity', listener);
  }, [docked, on, off]);

  // Run start / end. `agentStatus` is the same value the header badge reads,
  // so the panel can never disagree with what the UI says the agent is doing.
  const wasRunningRef = useRef(false);
  useEffect(() => {
    const running = IN_RUN.has(agentStatus);
    if (running && !wasRunningRef.current) onRunStart();
    if (!running && wasRunningRef.current) {
      historyStaleRef.current = true;
      onRunEnd();
    }
    wasRunningRef.current = running;
  }, [agentStatus, onRunStart, onRunEnd]);

  // A chat path click (route.previewPath) opens the panel's Files preview
  // instead of the overlay drawer while the panel is docked (§9.10).
  const previewPath = route.previewPath;
  useEffect(() => {
    if (!docked || !previewPath) return;
    setPanelView('files');
    setPanelFile(previewPath);
    expandPanel();
  }, [docked, previewPath, setPanelView, setPanelFile, expandPanel]);

  const clearPreviewPath = useCallback(() => {
    setRoute((prev) =>
      prev.name === 'project' && prev.projectId === projectId && prev.previewPath !== undefined
        ? { ...prev, previewPath: undefined }
        : prev,
    );
  }, [projectId, setRoute]);

  const handlePanelSelectFile = useCallback(
    (path: string | null) => {
      setPanelFile(path);
      // The panel owns the selection from here; a stale route.previewPath
      // would otherwise re-open the file it pointed at.
      clearPreviewPath();
    },
    [setPanelFile, clearPreviewPath],
  );

  const handleOpenInFiles = useCallback(
    (path: string) => {
      void path; // the Files tab keeps its own selection; this only navigates
      setRoute((prev) =>
        prev.name === 'project' && prev.projectId === projectId
          ? { ...prev, tab: 'files', settings: false, previewPath: undefined }
          : prev,
      );
    },
    [projectId, setRoute],
  );

  const panelMessages = useMemo(() => {
    const synthetic = liveEvents
      .map(eventAsMessage)
      .filter((m): m is NonNullable<ReturnType<typeof eventAsMessage>> => m !== null);
    return synthetic.length === 0 ? messages : [...messages, ...synthetic];
  }, [messages, liveEvents]);

  const touched = useMemo(() => touchedFiles(panelMessages), [panelMessages]);
  const screenshot = useMemo(() => latestScreenshot(messages), [messages]);
  const handleSave = useCallback(
    (path: string, content: string) => saveFileContent(projectId, path, content),
    [saveFileContent, projectId],
  );

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

  // Workbench card-tap doorway (spec 2026-07-24 §5.3): route.draft is a
  // one-shot composer prefill. ChatView applies it once and calls this back
  // so the consumed copy is cleared from the route — otherwise it would
  // reapply on a later remount (e.g. switching to the Queue tab and back).
  function handleDraftConsumed() {
    setRoute((prev) =>
      prev.name === 'project' && prev.projectId === projectId
        ? { ...prev, draft: undefined }
        : prev,
    );
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
      <div
        className={`flex-1 min-w-0 min-h-0 flex flex-col bg-card ${
          docked ? 'md:min-w-[520px]' : ''
        }`}
      >
        {/* Quick Tasks scope chip (Spec 012 §2c) — chat-header area, scratch
            project only. ScopeChip also self-gates on is_scratch. */}
        {project.is_scratch && (
          <div className="shrink-0 flex items-center px-4 pt-2">
            <ScopeChip
              project={project}
              projects={projects ?? []}
              sessionId={routeSessionId}
            />
          </div>
        )}
        <div className="flex-1 min-w-0 min-h-0">
          <ChatView
            key={projectId}
            projectId={projectId}
            project={project}
            agentStatus={agentStatus}
            statusTick={statusTick}
            mentionAgents={mentionAgents}
            sessionId={routeSessionId}
            initialDraft={route.draft}
            onDraftConsumed={handleDraftConsumed}
            onRefreshProject={onRefreshProject}
          />
        </div>
      </div>
      {/* Spec 078 §5.1 — the third column: the panel when expanded, the 20px
          edge handle when collapsed. Only above the push threshold. */}
      {docked &&
        (panelExpanded ? (
          <FilePreviewDrawer
            docked
            open
            selectedPath={null}
            fileContent={null}
            loading={false}
            onClose={collapsePanel}
          >
            <WorkspacePanel
              view={panelView}
              onViewChange={setPanelView}
              annotating={annotating}
              onToggleAnnotate={() => setAnnotating(!annotating)}
              browser={
                <BrowserView
                  projectId={projectId}
                  active={panelExpanded && panelView === 'browser'}
                  fallback={
                    screenshot ? { path: screenshot.path, title: screenshot.title } : null
                  }
                  annotating={annotating}
                  annotations={annotations}
                  onAddAnnotation={addAnnotation}
                />
              }
              files={
                <FilesView
                  projectId={projectId}
                  touched={touched}
                  file={panelFile}
                  onSelectFile={handlePanelSelectFile}
                  onOpenInFiles={handleOpenInFiles}
                  annotating={annotating}
                  onAddAnnotation={addAnnotation}
                  onSave={handleSave}
                />
              }
            />
          </FilePreviewDrawer>
        ) : (
          <PanelHandle working={agentStatus === 'running'} onExpand={expandPanel} />
        ))}
    </div>
  );
}
