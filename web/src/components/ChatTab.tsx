// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ChatTab — the project's chat home: the conversation for the ACTIVE session
 * and, beside it, the workspace panel (spec 078).
 *
 *   [ ChatView 1fr ][ panel | handle ]
 *
 * Spec 137 (the folder shell): the session list moved out of here into the
 * project column ProjectDetail draws, because that column stays on screen
 * while a page (Tasks, Settings) replaces the chat. Selecting a session is
 * still a route change — `route.sessionId` is the source of truth — made
 * through `useSessionNav`, which the column and ChatView's session links share.
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
import { useSessionNav } from '../hooks/useSessionNav';
import { useFiles } from '../hooks/useFiles';
import { useChatHistory } from '../hooks/useChatHistory';
import { useSessionMessages } from '../utils/sessionMessagesStore';
import { useWebSocket } from '../hooks/useWebSocket';
import { useAnnotations } from '../hooks/useAnnotations';
import { usePanelDockable, usePanelMobile, usePanelState } from '../hooks/usePanelState';
import SessionHeading from './SessionHeading';
import { useWorkspaceUpload } from '../hooks/useWorkspaceUpload';
import {
  eventAsMessage,
  latestScreenshot,
  pathForEvent,
  touchedFiles,
  viewForEvent,
} from '../utils/panelSelectors';
import ChatView from './ChatView';
import ScopeChip from './ScopeChip';
import FilePreviewDrawer from './FilePreviewDrawer';
import WorkspacePanel, { WorkspacePanelBar } from './panel/WorkspacePanel';
import FilesView from './panel/FilesView';
import BrowserView from './panel/BrowserView';
import PanelHandle from './panel/PanelHandle';
import FileExplorer from './FileExplorer';

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
  agents: Array<{ slug: string; name: string }>;
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
  allSessions: SessionListEntry[],
  persisted: string | null,
): string | undefined {
  // Session import: a native conversation not converted yet is never opened
  // by default. Opening it materializes the import, and landing there on
  // project entry would skip the cold-start card and the onboarding kickoff
  // (both wait for "no session"). The user opens one by clicking its row.
  const sessions = allSessions.filter((s) => s.imported?.materialized !== false);
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
  agents,
  route,
  setRoute,
  onRefreshProject,
  projects,
}: ChatTabProps) {
  const projectId = project.project_id;
  const { sessions } = useSessions(projectId);
  const { activeSessionId, setActiveSessionId } = useSession(projectId);
  const { selectSession } = useSessionNav(projectId, setRoute);

  const routeSessionId = route.sessionId;

  // ── Spec 078: the workspace panel (Files · Browser) ─────────────────────
  // Docked only above the push threshold and never on mobile. Below it a
  // chat path click keeps the FilePreviewDrawer overlay ProjectDetail owns,
  // and (spec 137) the header button opens the whole panel as an overlay —
  // a bottom sheet on a phone.
  const docked = usePanelDockable();
  const mobile = usePanelMobile();
  const {
    expanded: panelExpanded,
    view: panelView,
    file: panelFile,
    maximized: panelMaximized,
    overlayOpen,
    expand: expandPanel,
    collapse: collapsePanel,
    maximize: maximizePanel,
    restore: restorePanel,
    closeOverlay,
    setView: setPanelView,
    setFile: setPanelFile,
    expandForEvent,
    onRunStart,
    onRunEnd,
  } = usePanelState(projectId, routeSessionId);
  // Where the panel is showing right now, if anywhere.
  const panelOpen = docked ? panelExpanded : overlayOpen;
  // Spec 137: maximize exists docked and in the narrow overlay; the phone's
  // bottom sheet is already the whole screen.
  const canMaximize = docked || !mobile;
  const maximized = panelOpen && canMaximize && panelMaximized;
  // The docked panel maximized takes the whole right area: the conversation
  // hides (CSS only — the chat stays mounted, composer draft and stream
  // intact). The project column beside it stays (spec 137 item 12).
  const panelFillsRow = docked && maximized;
  const { annotating, setAnnotating, annotations, add: addAnnotation } =
    useAnnotations(routeSessionId);
  // Spec 078 §11.5: ChatView owns the session transcript and publishes it by
  // reference; the panel reads that same array. useChatHistory remains only
  // as a fallback for the window before ChatView has published anything.
  const sharedMessages = useSessionMessages(`${projectId}:${routeSessionId ?? ''}`);
  const { messages: fetchedMessages, loadHistory, clearMessages } = useChatHistory({
    sessionId: routeSessionId ?? null,
  });
  const messages = sharedMessages ?? fetchedMessages;
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
    if (!panelOpen || !routeSessionId) return;
    if (sharedMessages !== null) return; // ChatView already holds the transcript
    if (loadedKeyRef.current === sessionKey && !historyStaleRef.current) return;
    loadedKeyRef.current = sessionKey;
    historyStaleRef.current = false;
    void loadHistory(projectId);
  }, [panelOpen, projectId, routeSessionId, sessionKey, loadHistory, sharedMessages]);

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

  const clearPreviewPath = useCallback(() => {
    setRoute((prev) =>
      prev.name === 'project' && prev.projectId === projectId && prev.previewPath !== undefined
        ? { ...prev, previewPath: undefined }
        : prev,
    );
  }, [projectId, setRoute]);

  // A chat path click (route.previewPath) opens the panel's Files preview
  // instead of the overlay drawer while the panel is docked (§9.10).
  // The path is a request, not panel state: consume it once it has acted
  // (spec 110). Otherwise a repeat click on the same card writes the same
  // string, the deps never change, and the card is dead after the first
  // open once the panel has been hidden or switched to Browser.
  const previewPath = route.previewPath;
  useEffect(() => {
    if (!docked || !previewPath) return;
    setPanelView('files');
    setPanelFile(previewPath);
    expandPanel();
    clearPreviewPath();
  }, [docked, previewPath, setPanelView, setPanelFile, expandPanel, clearPreviewPath]);

  const handlePanelSelectFile = useCallback(
    (path: string | null) => {
      setPanelFile(path);
      // The panel owns the selection from here; a stale route.previewPath
      // would otherwise re-open the file it pointed at.
      clearPreviewPath();
    },
    [setPanelFile, clearPreviewPath],
  );

  // Spec 137: "Open full width" (was "Open in Files", which routed to the
  // Files tab) maximizes the panel on this file — the tree on the left, the
  // file on the right, as the tab had it.
  const handleOpenInFiles = useCallback(
    (path: string) => {
      setPanelView('files');
      setPanelFile(path);
      clearPreviewPath();
      maximizePanel();
    },
    [setPanelView, setPanelFile, clearPreviewPath, maximizePanel],
  );

  const toggleMaximize = useCallback(() => {
    if (panelMaximized) restorePanel();
    else maximizePanel();
  }, [panelMaximized, restorePanel, maximizePanel]);

  // Esc restores a maximized docked panel. The overlay handles its own Esc
  // (restore first, then close); Annotate owns Esc while it is on.
  useEffect(() => {
    if (!docked || !maximized || annotating) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !e.defaultPrevented) restorePanel();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [docked, maximized, annotating, restorePanel]);

  // Spec 137 gap 2: upload from the Files view, the old Files tab's
  // affordance. A landed file bumps `filesRefresh`, which FilesView re-lists on.
  const { upload, uploading, error: uploadError, clearError: clearUploadError } =
    useWorkspaceUpload(projectId);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [filesRefresh, setFilesRefresh] = useState(0);
  const handleUploadClick = useCallback(() => {
    clearUploadError();
    uploadInputRef.current?.click();
  }, [clearUploadError]);
  const handleUploadPicked = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      e.target.value = ''; // the same file can be picked again
      if (await upload(file)) setFilesRefresh((n) => n + 1);
    },
    [upload],
  );
  useEffect(() => {
    if (!uploadError) return;
    const id = setTimeout(clearUploadError, 4000);
    return () => clearTimeout(id);
  }, [uploadError, clearUploadError]);

  // The overlay is always mounted (it slides), but its views are only built
  // once it has been opened: a narrow window that never opens it must not
  // list the workspace or connect the browser view.
  const [overlayUsed, setOverlayUsed] = useState(false);
  if (overlayOpen && !overlayUsed) setOverlayUsed(true);

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

  // Workbench card-tap doorway (spec 2026-07-24 §5.3): route.draft is a
  // one-shot composer prefill. ChatView applies it once and calls this back
  // so the consumed copy is cleared from the route — otherwise it would
  // reapply on a later remount (e.g. switching to the Queue tab and back).
  function handleDraftConsumed() {
    setRoute((prev) =>
      prev.name === 'project' && prev.projectId === projectId
        ? { ...prev, draft: undefined, draftSkill: undefined, askReply: undefined }
        : prev,
    );
  }

  // The panel's chrome and body, shared by the docked column and the
  // narrow overlay / phone sheet.
  const panelBar = (
    <WorkspacePanelBar
      view={panelView}
      onViewChange={setPanelView}
      annotating={annotating}
      onToggleAnnotate={() => setAnnotating(!annotating)}
      onUpload={maximized ? undefined : handleUploadClick}
      uploading={uploading}
      onToggleMaximize={canMaximize ? toggleMaximize : undefined}
      maximized={maximized}
    />
  );
  const panelBody = (
    <>
      {uploadError && (
        <p role="status" className="shrink-0 mx-3 mt-2 px-3 py-2 text-xs text-error bg-error/10 border border-error/30 rounded-md">
          {uploadError}
        </p>
      )}
      <input
        ref={uploadInputRef}
        type="file"
        className="hidden"
        data-testid="panel-upload-input"
        onChange={handleUploadPicked}
      />
      <WorkspacePanel
        view={panelView}
        browser={
          <BrowserView
            projectId={projectId}
            active={panelOpen && panelView === 'browser'}
            fallback={screenshot ? { path: screenshot.path, title: screenshot.title } : null}
            annotating={annotating}
            annotations={annotations}
            onAddAnnotation={addAnnotation}
          />
        }
        files={
          maximized ? (
            // Maximized Files = the old Files tab: tree left, file right.
            // The explorer is moved, not rebuilt (upload + reveal included).
            <FileExplorer
              projectId={projectId}
              initialPath={panelFile ?? undefined}
              onSelectFile={setPanelFile}
            />
          ) : (
            <FilesView
              projectId={projectId}
              touched={touched}
              file={panelFile}
              onSelectFile={handlePanelSelectFile}
              onOpenInFiles={canMaximize ? handleOpenInFiles : undefined}
              onAddAnnotation={addAnnotation}
              onSave={handleSave}
              refreshKey={filesRefresh}
            />
          )
        }
      />
    </>
  );

  return (
    <div className="flex h-full min-h-0" data-testid="chat-tab">
      <div
        data-testid="chat-tab-conversation"
        className={`flex-1 min-w-0 min-h-0 flex flex-col bg-card ${
          docked ? 'md:min-w-[520px]' : ''
        } ${panelFillsRow ? 'hidden' : ''}`}
      >
        {/* Spec 137 follow-up: the chat's heading, aligned with the project
            name at the top of the column. The phone keeps its own header. */}
        {!mobile && <SessionHeading session={sessions.find((s) => s.session_id === routeSessionId)} />}
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
            agents={agents}
            sessionId={routeSessionId}
            initialDraft={route.draft}
            initialSkill={route.draftSkill}
            initialAskReply={route.askReply}
            onDraftConsumed={handleDraftConsumed}
            onRefreshProject={onRefreshProject}
            onOpenSession={selectSession}
          />
        </div>
      </div>
      {/* Spec 078 §5.1 — beside the chat: the panel when expanded, the 20px
          edge handle when collapsed. Only above the push threshold. Spec
          137: a full-height sibling with a left edge only; maximized, it
          takes the whole right area (the project column stays). */}
      {docked &&
        (panelExpanded ? (
          <FilePreviewDrawer
            docked
            open
            maximized={maximized}
            selectedPath={null}
            fileContent={null}
            loading={false}
            onClose={collapsePanel}
            header={panelBar}
          >
            {panelBody}
          </FilePreviewDrawer>
        ) : (
          <PanelHandle working={agentStatus === 'running'} onExpand={expandPanel} />
        ))}
      {/* Spec 137 gap 3 — below the dock threshold: the same panel as an
          overlay over the chat (maximize grows it to the full width right of
          the strip/list), and on a phone the drawer's bottom sheet. Opened
          from the header button; Esc restores a maximized overlay, then
          closes it. */}
      {!docked && (
        <FilePreviewDrawer
          open={overlayOpen}
          maximized={maximized}
          selectedPath={null}
          fileContent={null}
          loading={false}
          onClose={closeOverlay}
          onEscape={maximized ? restorePanel : closeOverlay}
          header={panelBar}
        >
          {overlayUsed ? panelBody : <div />}
        </FilePreviewDrawer>
      )}
    </div>
  );
}
