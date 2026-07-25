// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { usePlatform } from './hooks/usePlatform';
import { useProjects } from './hooks/useProjects';
import { useTriggers } from './hooks/useTriggers';
import { useWebSocket, type ConnectionState } from './hooks/useWebSocket';
import type {
  AgentRunStatus,
  ApprovalRequestEvent,
  ApprovalResolvedEvent,
  AgentStatusEvent,
  DeviceStatusEvent,
  StatusSummaryEvent,
  ProjectCreateRequest,
  ProjectUpdateRequest,
  WebSocketEvent,
} from './types';
import type { Route } from './route';
import SetupGate from './components/SetupGate';
import SetupWizard from './components/SetupWizard';
import Sidebar from './components/Sidebar';
import CreateProject from './components/CreateProject';
import ProjectDetail from './components/ProjectDetail';
import QueueTab from './components/QueueTab';
import SettingsModalPage from './components/SettingsModalPage';
import PricingEditorPage from './components/PricingEditorPage';
import ChatTab from './components/ChatTab';
import ErrorBoundary from './components/ErrorBoundary';
import FileExplorer from './components/FileExplorer';
import GlobalSettings from './components/GlobalSettings';
import CalendarPage from './components/CalendarPage';
import WorkbenchPage from './components/WorkbenchPage';
import { useT } from './i18n/useT';
import { api, isRelayMode } from './config';

function mapConnectionState(
  state: ConnectionState,
  daemonOnline: boolean,
): 'connected' | 'reconnecting' | 'disconnected' | 'daemon_offline' {
  if (state !== 'connected') {
    return state === 'connecting' ? 'reconnecting' : 'disconnected';
  }
  // WebSocket is connected; in remote mode, also check daemon tunnel status
  if (isRelayMode && !daemonOnline) return 'daemon_offline';
  return 'connected';
}

export default function App() {
  const t = useT();
  const platform = usePlatform();
  const {
    projects,
    listProjects,
    createProject,
    refreshProject,
    updateProject,
    deleteProject,
  } = useProjects();
  const ws = useWebSocket();

  const [setupComplete, setSetupComplete] = useState<boolean | null>(null);
  const [needsWizard, setNeedsWizard] = useState<boolean | null>(null);
  const [route, setRoute] = useState<Route>({ name: 'list' });

  // Bumped after a pricing-table edit (PUT /pricing/overrides). Threaded into
  // the settings Budget meter's useCost so historical cost visibly recomputes
  // against the new rates without polling. See PricingEditor.onSaved.
  const [costRefreshKey, setCostRefreshKey] = useState(0);

  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentRunStatus>>({});
  const [statusTicks, setStatusTicks] = useState<Record<string, number>>({});
  const [statusSummaries, setStatusSummaries] = useState<Record<string, string>>({});
  const [pendingApprovals, setPendingApprovals] = useState<Record<string, number>>({});
  const [daemonOnline, setDaemonOnline] = useState(true);
  // Project-level agent availability. Lifted out of ChatView so a Files->Chat
  // tab switch doesn't block on the daemon's /agents/available cache miss
  // (cold call can take up to 8s). null = not yet loaded.
  const [agentsAvailable, setAgentsAvailable] = useState<Array<{ slug: string; name: string }> | null>(null);
  // Global/default LLM model (from app settings) — header fallback when a
  // project has no per-project model pinned. Fetched once on mount.
  const [defaultModel, setDefaultModel] = useState<string>('');
  useEffect(() => {
    api<{ llm?: { model?: string } }>('/api/v2/settings')
      .then((d) => setDefaultModel(d?.llm?.model || ''))
      .catch(() => {});
  }, []);

  // Calendar availability (EventKit hub / connected calendar source). Fetched
  // once on mount; any error (incl. the endpoint not existing until the backend
  // lands this wave) is treated as unavailable. Gates the per-project calendar
  // lens tab in ProjectDetail.
  // Sandbox-repair affordance: the wizard's old sandbox step carried the only
  // Retry Setup button; with that step removed (spec 011 §0.7) the warning
  // banner hosts it instead. triggerSetup re-fetches status on success, which
  // hides the banner when setup completes.
  const [sandboxRetrying, setSandboxRetrying] = useState(false);
  // Derive selected project ID from route for convenience
  const selectedProjectId = route.name === 'project' ? route.projectId : null;

  const { triggers, fetchTriggers, toggleTrigger, deleteTrigger } = useTriggers(selectedProjectId ?? '');

  // Fetch triggers when selected project changes or agent goes idle
  const selectedStatus = selectedProjectId ? agentStatuses[selectedProjectId] : undefined;
  useEffect(() => {
    if (selectedProjectId) {
      fetchTriggers();
    }
  }, [selectedProjectId, fetchTriggers]);
  useEffect(() => {
    if (selectedProjectId && selectedStatus === 'idle') {
      fetchTriggers();
    }
  }, [selectedProjectId, selectedStatus, fetchTriggers]);

  // Fetch agent availability when a project is selected. Pre-filtered to the
  // shape the @-mention dropdown actually consumes: installed sub-agents, no
  // 'built-in'. The fetch happens once per project select, off the chat-tab
  // mount path, so tab switches don't block on this network call.
  useEffect(() => {
    if (!selectedProjectId) return;
    let cancelled = false;
    setAgentsAvailable(null);
    api<Array<{ slug: string; name: string; installed: boolean }>>(
      '/api/v2/agents/available',
    )
      .then((agents) => {
        if (cancelled) return;
        setAgentsAvailable(
          agents.filter((a) => a.slug !== 'built-in' && a.installed)
            .map((a) => ({ slug: a.slug, name: a.name })),
        );
      })
      .catch(() => {
        if (cancelled) return;
        // Leave list empty on failure; @-dropdown simply shows no matches.
        setAgentsAvailable([]);
      });
    return () => { cancelled = true; };
  }, [selectedProjectId]);

  // Mobile panel swap navigation
  const [isMobile, setIsMobile] = useState(false);
  const [mobileView, setMobileView] = useState<'sidebar' | 'content'>('sidebar');

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 767px)');
    setIsMobile(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // Check platform status on mount
  // Remote (phone) clients skip platform setup — it's a desktop-only concern
  useEffect(() => {
    if (isRelayMode || (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1')) {
      setSetupComplete(true);
      setNeedsWizard(false);
      // Fetch projects with retry — daemon tunnel may not be ready on page load
      let retryTimer: ReturnType<typeof setTimeout>;
      let retryCount = 0;
      const maxRetries = 3;
      const fetchWithRetry = async () => {
        const result = await listProjects();
        if (result.length === 0 && retryCount < maxRetries) {
          retryCount++;
          retryTimer = setTimeout(fetchWithRetry, retryCount * 2000);
        }
      };
      fetchWithRetry();
      return () => clearTimeout(retryTimer);
    }
    let cancelled = false;
    async function check() {
      const status = await platform.getStatus();
      if (cancelled) return;
      if (status) {
        setSetupComplete(true);
        listProjects();
      } else {
        setSetupComplete(false);
      }
      // Check if first-run wizard is needed (API key missing)
      const settingsData = await api<{ llm: { api_key_set: boolean } }>('/api/v2/settings').catch(() => null);
      if (!cancelled && settingsData) {
        const needsApiKey = !settingsData.llm.api_key_set;
        setNeedsWizard(needsApiKey);
      } else if (!cancelled) {
        setNeedsWizard(true);
      }
    }
    check();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Subscribe to all project IDs when projects change
  useEffect(() => {
    if (!setupComplete || projects.length === 0) return;
    ws.subscribe(projects.map((p) => p.project_id));
  }, [projects, setupComplete]); // eslint-disable-line react-hooks/exhaustive-deps

  // Hydrate agent statuses via REST when projects load or WS reconnects.
  // WS events are the primary source, but can be missed on initial load or
  // after a relay reconnect. This ensures we never show stale 'idle' status.
  useEffect(() => {
    if (!setupComplete || projects.length === 0) return;
    if (ws.connectionState !== 'connected') return;
    let cancelled = false;
    for (const p of projects) {
      api<{ project_id: string; status: AgentRunStatus }>(
        `/api/v2/agents/${encodeURIComponent(p.project_id)}/run-status`,
      )
        .then((result) => {
          if (cancelled) return;
          setAgentStatuses((prev) => {
            if (prev[result.project_id] === result.status) return prev;
            return { ...prev, [result.project_id]: result.status };
          });
        })
        .catch(() => {});
    }
    return () => { cancelled = true; };
  }, [setupComplete, projects, ws.connectionState]);

  // WebSocket event handlers
  const handleAgentStatus = useCallback((event: WebSocketEvent) => {
    const e = event as AgentStatusEvent;
    setAgentStatuses((prev) => ({ ...prev, [e.project_id]: e.status }));
    setStatusTicks((prev) => ({ ...prev, [e.project_id]: (prev[e.project_id] ?? 0) + 1 }));
  }, []);

  const handleStatusSummary = useCallback((event: WebSocketEvent) => {
    const e = event as StatusSummaryEvent;
    setStatusSummaries((prev) => ({ ...prev, [e.project_id]: e.summary }));
  }, []);

  const handleApprovalRequest = useCallback(
    (event: WebSocketEvent) => {
      const e = event as ApprovalRequestEvent;
      setPendingApprovals((prev) => ({
        ...prev,
        [e.project_id]: (prev[e.project_id] ?? 0) + 1,
      }));
      // Browser notification when tab not focused
      if (!document.hasFocus()) {
        const project = projects.find((p) => p.project_id === e.project_id);
        const name = project?.name ?? t('app.notify.fallbackName');
        if (Notification.permission === 'granted') {
          new Notification(t('app.notify.approval', { name }));
        } else if (Notification.permission !== 'denied') {
          Notification.requestPermission();
        }
      }
    },
    [projects, t],
  );

  const handleApprovalResolved = useCallback((event: WebSocketEvent) => {
    const e = event as ApprovalResolvedEvent;
    setPendingApprovals((prev) => {
      const count = (prev[e.project_id] ?? 1) - 1;
      const next = { ...prev };
      if (count <= 0) {
        delete next[e.project_id];
      } else {
        next[e.project_id] = count;
      }
      return next;
    });
  }, []);

  const handleDeviceStatus = useCallback((event: WebSocketEvent) => {
    const e = event as DeviceStatusEvent;
    const online = e.status === 'online';
    setDaemonOnline(online);
    // Re-fetch projects when daemon comes back online
    if (online) {
      listProjects();
    }
  }, [listProjects]);

  // Register/unregister WS listeners
  useEffect(() => {
    ws.on('agent.status', handleAgentStatus);
    ws.on('agent.status_summary', handleStatusSummary);
    ws.on('approval.request', handleApprovalRequest);
    ws.on('approval.resolved', handleApprovalResolved);
    ws.on('device.status', handleDeviceStatus);
    return () => {
      ws.off('agent.status', handleAgentStatus);
      ws.off('agent.status_summary', handleStatusSummary);
      ws.off('approval.request', handleApprovalRequest);
      ws.off('approval.resolved', handleApprovalResolved);
      ws.off('device.status', handleDeviceStatus);
    };
  }, [ws, handleAgentStatus, handleStatusSummary, handleApprovalRequest, handleApprovalResolved, handleDeviceStatus]);

  // Fix 2B: Listen for status poll overrides from ChatView
  useEffect(() => {
    function onStatusOverride(e: Event) {
      const { project_id, status } = (e as CustomEvent).detail;
      setAgentStatuses((prev) => ({ ...prev, [project_id]: status }));
    }
    window.addEventListener('agent-status-override', onStatusOverride);
    return () => window.removeEventListener('agent-status-override', onStatusOverride);
  }, []);

  // Credential-error surfacing: AgentErrorNotice's "Open Settings" action
  // (deep in ChatView) opens Global Settings — where the API key lives —
  // via the same window-event pattern as agent-status-override, avoiding a
  // callback threaded through ProjectDetail/ChatTab.
  useEffect(() => {
    function onOpenGlobalSettings() {
      setRoute({ name: 'settings' });
    }
    window.addEventListener('open-global-settings', onOpenGlobalSettings);
    return () => window.removeEventListener('open-global-settings', onOpenGlobalSettings);
  }, []);

  // Service worker registration + push subscription (remote mode only)
  useEffect(() => {
    if (!isRelayMode) return;
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) return;

    async function setupPush() {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js');

        const existing = await registration.pushManager.getSubscription();
        if (existing) return;

        if (Notification.permission === 'denied') return;
        if (Notification.permission === 'default') {
          const result = await Notification.requestPermission();
          if (result !== 'granted') return;
        }

        // Get VAPID key from relay
        const vapidResp = await fetch(window.location.origin + '/api/v1/push/vapid-key');
        if (!vapidResp.ok) return;
        const { key: vapidKey } = await vapidResp.json();
        if (!vapidKey) return;

        const subscription = await registration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: vapidKey,
        });

        const subJson = subscription.toJSON();
        await fetch(window.location.origin + '/api/v1/push/subscribe', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('relay_jwt') || ''}`,
          },
          body: JSON.stringify({ type: 'web_push', ...subJson }),
        });
      } catch (err) {
        console.warn('Push subscription failed:', err);
      }
    }

    setupPush();
  }, []);

  // Listen for service worker navigation messages
  useEffect(() => {
    if (!('serviceWorker' in navigator)) return;

    function handleMessage(event: MessageEvent) {
      if (event.data?.type === 'navigate' && event.data.projectId) {
        setRoute({ name: 'project', projectId: event.data.projectId, tab: 'chat', sessionId: undefined });
        setMobileView('content');
      }
    }

    navigator.serviceWorker.addEventListener('message', handleMessage);
    return () => navigator.serviceWorker.removeEventListener('message', handleMessage);
  }, []);

  // Handlers
  async function handleSetupComplete() {
    setSetupComplete(true);
    listProjects();
    await platform.getStatus();
  }

  function handleSelectProject(id: string) {
    setRoute({ name: 'project', projectId: id, tab: 'chat', sessionId: undefined });
    setMobileView('content');
  }

  // Workspace-zone Calendar item — a top-level surface. Mobile behavior mirrors
  // a project tap (swap to the content pane).
  function handleSelectCalendar() {
    setRoute({ name: 'calendar' });
    setMobileView('content');
  }

  // Workbench — a top-level Workspace-zone surface, sibling of Calendar
  // (spec 2026-07-23 §6). Same mobile behavior as a project/calendar tap.
  function handleSelectWorkbench() {
    setRoute({ name: 'workbench' });
    setMobileView('content');
  }

  function handleNewProject() {
    setRoute({ name: 'create' });
    setMobileView('content');
  }

  async function handleCreateProject(data: ProjectCreateRequest) {
    const created = await createProject(data);
    setRoute({ name: 'project', projectId: created.project_id, tab: 'chat', sessionId: undefined });
  }

  async function handleUpdateProject(data: ProjectUpdateRequest) {
    if (!selectedProjectId) return;
    await updateProject(selectedProjectId, data);
  }

  async function handleDeleteProject() {
    if (!selectedProjectId) return;
    await deleteProject(selectedProjectId);
    setRoute({ name: 'list' });
    setMobileView('sidebar');
  }

  // First-run wizard (before setup gate)
  if (needsWizard === true) {
    return (
      <SetupWizard
        onComplete={async () => {
          setNeedsWizard(false);
          setSetupComplete(true);
          listProjects();
          await platform.getStatus();
        }}
      />
    );
  }

  // Loading state
  if (setupComplete === null || needsWizard === null) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <span className="text-sm text-secondary">{t('app.loading')}</span>
      </div>
    );
  }

  // Setup gate
  if (!setupComplete) {
    return (
      <SetupGate
        setupIssues={platform.status?.setup_issues ?? []}
        onComplete={handleSetupComplete}
      />
    );
  }

  const selectedProject = route.name === 'project'
    ? projects.find((p) => p.project_id === route.projectId)
    : undefined;

  function handleMobileBack() {
    setMobileView('sidebar');
  }

  const sidebarHidden = isMobile && mobileView !== 'sidebar';
  const contentHidden = isMobile && mobileView !== 'content';

  return (
    <div className="flex h-dvh overflow-hidden">
      <div className={sidebarHidden ? 'hidden' : 'contents'}>
        <Sidebar
          projects={projects}
          agentStatuses={agentStatuses}
          statusSummaries={statusSummaries}
          pendingApprovals={pendingApprovals}
          route={route}
          connectionState={mapConnectionState(ws.connectionState, daemonOnline)}
          onSelectProject={handleSelectProject}
          onSelectCalendar={handleSelectCalendar}
          onSelectWorkbench={handleSelectWorkbench}
          onNewProject={handleNewProject}
          onSettings={() => {
            setRoute({ name: 'settings' });
            setMobileView('content');
          }}
        />
      </div>

      {/* NOTE: keep a space before `${` — Tailwind's scanner will not extract a
          class that runs straight into a template-literal interpolation, so
          `shadow-panel${...}` silently emits no CSS. */}
      <main className={`flex-1 overflow-hidden flex flex-col shadow-panel ${contentHidden ? 'hidden' : ''}`}>
        {/* Mobile back button */}
        {isMobile && mobileView === 'content' && (
          <button
            onClick={handleMobileBack}
            className="flex items-center gap-1.5 px-4 pt-3 pb-1 min-h-[44px] text-sm text-secondary hover:text-primary transition-colors"
          >
            <ArrowLeft size={16} />
            {t('app.mobileBack')}
          </button>
        )}

        {platform.status && platform.status.platform !== 'null' && !platform.status.setup_complete && (
          <div className="bg-warning/10 border border-warning/20 rounded-lg px-4 py-3 mx-4 mt-4 text-sm">
            <span className="text-warning font-medium">{t('app.sandboxWarning.title')}</span>{' '}
            <span className="text-secondary">
              {t('app.sandboxWarning.body')}
            </span>{' '}
            <button
              type="button"
              disabled={sandboxRetrying}
              onClick={() => {
                setSandboxRetrying(true);
                platform.triggerSetup()
                  .catch(() => {})
                  .finally(() => setSandboxRetrying(false));
              }}
              className="underline text-warning hover:opacity-80 disabled:opacity-50"
            >
              {sandboxRetrying ? t('app.sandboxWarning.retrying') : t('app.sandboxWarning.retry')}
            </button>
          </div>
        )}

        {route.name === 'settings' && (
          <GlobalSettings
            onBack={() => { setRoute({ name: 'list' }); setMobileView('sidebar'); }}
          />
        )}

        {/* Global calendar surface (spec 011 §0.4). Placeholder body until
            Wave 3; the component is kept so that swap is body-only. */}
        {route.name === 'calendar' && <CalendarPage />}

        {/* Global Workbench surface (spec 2026-07-23 §6) — sibling of
            Calendar in the Workspace zone. Same mount pattern: one component,
            no projectId here vs. the per-project lens below. */}
        {route.name === 'workbench' && <WorkbenchPage setRoute={setRoute} />}

        {route.name === 'project' && selectedProject && (
          // Scoped to the project view: a render crash here shows a recoverable
          // fallback instead of unmounting the whole app (the app shell/sidebar
          // stays alive). resetKey clears a prior crash when the user navigates.
          <ErrorBoundary
            resetKey={`${selectedProject.project_id}:${route.tab}:${route.sessionId ?? ''}:${route.pricing ? 'pricing' : route.settings ? 'settings' : ''}`}
          >
            {route.pricing ? (
              // Pricing-table editor overlay (P3-I). Back returns to the
              // settings surface; onSaved bumps the cost-refresh nonce so the
              // Budget meter recomputes against the new rates on return.
              <PricingEditorPage
                onBack={() =>
                  setRoute({ ...route, pricing: false, settings: true, settingsAnchor: 'budget' })
                }
                onSaved={() => setCostRefreshKey((k) => k + 1)}
              />
            ) : route.settings ? (
              <SettingsModalPage
                project={selectedProject}
                route={route}
                setRoute={setRoute}
                onSave={handleUpdateProject}
                onDelete={handleDeleteProject}
                onEditPricing={() =>
                  setRoute({ ...route, settings: false, pricing: true })
                }
                costRefreshKey={costRefreshKey}
              />
            ) : (
              <ProjectDetail
                project={selectedProject}
                agentStatus={agentStatuses[selectedProject.project_id] ?? 'idle'}
                statusSummary={statusSummaries[selectedProject.project_id]}
                route={route}
                setRoute={setRoute}
                triggers={triggers}
                onTriggerToggle={toggleTrigger}
                onTriggerDelete={deleteTrigger}
                globalDefaultModel={defaultModel}
              >
                {route.tab === 'queue' && (
                  <QueueTab
                    key={`queue-${selectedProject.project_id}`}
                    projectId={selectedProject.project_id}
                    project={selectedProject}
                  />
                )}
                {route.tab === 'chat' && (
                  <ChatTab
                    key={selectedProject.project_id}
                    project={selectedProject}
                    projects={projects}
                    agentStatus={agentStatuses[selectedProject.project_id] ?? 'idle'}
                    statusTick={statusTicks[selectedProject.project_id] ?? 0}
                    mentionAgents={agentsAvailable ?? []}
                    route={route}
                    setRoute={setRoute}
                    onRefreshProject={refreshProject}
                  />
                )}
                {route.tab === 'files' && (
                  <FileExplorer projectId={selectedProject.project_id} />
                )}
              </ProjectDetail>
            )}
          </ErrorBoundary>
        )}

        {route.name === 'list' && (
          <div className="flex flex-col items-center justify-center flex-1 min-h-0 gap-4">
            <p className="text-secondary text-sm">
              {projects.length === 0
                ? t('app.list.empty')
                : t('app.list.selectPrompt')}
            </p>
            {projects.length === 0 && (
              <button
                onClick={handleNewProject}
                className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150"
              >
                {t('app.newProject')}
              </button>
            )}
          </div>
        )}

        {/* 'blocked' route UI (BlockedBadge view) is a later task — minimal stub for now. */}
        {route.name === 'blocked' && (
          <div data-testid="blocked-route-stub" />
        )}
      </main>

      {/* New Project is a modal overlay (backlog #25), not page content —
          kept as a sibling of <main> rather than one of its route branches.
          'create' has no matching branch inside <main> above, so navigating
          here leaves <main> empty (none of the 'settings'/'calendar'/
          'project'/'list'/'blocked' conditions match 'create'); the modal's
          backdrop sits over that empty <main>, not over the previous route's
          content. Cancel routes back to 'list'. */}
      {route.name === 'create' && (
        <CreateProject
          onSubmit={handleCreateProject}
          onCancel={() => {
            setRoute({ name: 'list' });
            setMobileView('sidebar');
          }}
        />
      )}
    </div>
  );
}
