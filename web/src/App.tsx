// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { usePlatform } from './hooks/usePlatform';
import { useProjects } from './hooks/useProjects';
import { useAgent } from './hooks/useAgent';
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
import SetupGate from './components/SetupGate';
import SetupWizard from './components/SetupWizard';
import Sidebar from './components/Sidebar';
import CreateProject from './components/CreateProject';
import ProjectDetail, { type DetailTab } from './components/ProjectDetail';
import SettingsView from './components/SettingsView';
import ChatView from './components/ChatView';
import FileExplorer from './components/FileExplorer';
import GlobalSettings from './components/GlobalSettings';
import { api, isRelayMode } from './config';

type View = 'list' | 'create' | 'detail' | 'settings';

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
  const platform = usePlatform();
  const {
    projects,
    listProjects,
    createProject,
    updateProject,
    deleteProject,
  } = useProjects();
  const { cancelMessage } = useAgent();
  const ws = useWebSocket();

  const [setupComplete, setSetupComplete] = useState<boolean | null>(null);
  const [needsWizard, setNeedsWizard] = useState<boolean | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [view, setView] = useState<View>('list');
  const [tab, setTab] = useState<DetailTab>('chat');

  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentRunStatus>>({});
  const [statusTicks, setStatusTicks] = useState<Record<string, number>>({});
  const [statusSummaries, setStatusSummaries] = useState<Record<string, string>>({});
  const [pendingApprovals, setPendingApprovals] = useState<Record<string, number>>({});
  const [daemonOnline, setDaemonOnline] = useState(true);

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
        const name = project?.name ?? 'A project';
        if (Notification.permission === 'granted') {
          new Notification(`Orbital: ${name} needs your approval`);
        } else if (Notification.permission !== 'denied') {
          Notification.requestPermission();
        }
      }
    },
    [projects],
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
        handleSelectProject(event.data.projectId);
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
    setSelectedProjectId(id);
    setView('detail');
    setTab('chat');
    setMobileView('content');
  }

  function handleNewProject() {
    setView('create');
    setSelectedProjectId(null);
    setMobileView('content');
  }

  async function handleCreateProject(data: ProjectCreateRequest) {
    const created = await createProject(data);
    setSelectedProjectId(created.project_id);
    setView('detail');
    setTab('chat');
  }

  async function handleUpdateProject(data: ProjectUpdateRequest) {
    if (!selectedProjectId) return;
    await updateProject(selectedProjectId, data);
  }

  async function handleDeleteProject() {
    if (!selectedProjectId) return;
    await deleteProject(selectedProjectId);
    setSelectedProjectId(null);
    setView('list');
    setMobileView('sidebar');
  }

  async function handleCancelMessage() {
    if (!selectedProjectId) return;
    await cancelMessage(selectedProjectId);
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
        <span className="text-sm text-secondary">Loading...</span>
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

  const selectedProject = projects.find(
    (p) => p.project_id === selectedProjectId,
  );

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
          selectedProjectId={selectedProjectId}
          connectionState={mapConnectionState(ws.connectionState, daemonOnline)}
          onSelectProject={handleSelectProject}
          onNewProject={handleNewProject}
          onSettings={() => { setView('settings'); setSelectedProjectId(null); setMobileView('content'); }}
        />
      </div>

      <main className={`flex-1 overflow-hidden flex flex-col${contentHidden ? ' hidden' : ''}`}>
        {/* Mobile back button */}
        {isMobile && mobileView === 'content' && (
          <button
            onClick={handleMobileBack}
            className="flex items-center gap-1.5 px-4 pt-3 pb-1 min-h-[44px] text-sm text-secondary hover:text-primary transition-colors"
          >
            <ArrowLeft size={16} />
            Back
          </button>
        )}

        {platform.status && platform.status.platform !== 'null' && !platform.status.setup_complete && (
          <div className="bg-warning/10 border border-warning/20 rounded-lg px-4 py-3 mx-4 mt-4 text-sm">
            <span className="text-warning font-medium">{'\u26A0'} Agent sandbox is not configured.</span>{' '}
            <span className="text-secondary">
              Agents will run with your full system permissions.
              Reinstall Orbital to set up the sandbox, or run Orbital.exe --setup-sandbox as administrator.
            </span>
          </div>
        )}

        {view === 'settings' && (
          <GlobalSettings
            onBack={() => { setView(selectedProjectId ? 'detail' : 'list'); setMobileView('sidebar'); }}
          />
        )}

        {view === 'create' && (
          <CreateProject
            onSubmit={handleCreateProject}
            onCancel={() => { setView(selectedProjectId ? 'detail' : 'list'); setMobileView('sidebar'); }}
          />
        )}

        {view === 'detail' && selectedProject && (
          <ProjectDetail
            project={selectedProject}
            agentStatus={agentStatuses[selectedProject.project_id] ?? 'idle'}
            statusSummary={statusSummaries[selectedProject.project_id]}
            tab={tab}
            onTabChange={setTab}
            onStopAgent={handleCancelMessage}
            triggers={triggers}
            onTriggerToggle={toggleTrigger}
            onTriggerDelete={deleteTrigger}
          >
            {tab === 'settings' && (
              <SettingsView
                project={selectedProject}
                onSave={handleUpdateProject}
                onDelete={handleDeleteProject}
              />
            )}
            {tab === 'chat' && (
              <ChatView
                key={selectedProject.project_id}
                projectId={selectedProject.project_id}
                project={selectedProject}
                agentStatus={agentStatuses[selectedProject.project_id] ?? 'idle'}
                statusTick={statusTicks[selectedProject.project_id] ?? 0}
              />
            )}
            {tab === 'files' && (
              <FileExplorer projectId={selectedProject.project_id} />
            )}
          </ProjectDetail>
        )}

        {view === 'list' && !selectedProjectId && (
          <div className="flex flex-col items-center justify-center flex-1 min-h-0 gap-4">
            <p className="text-secondary text-sm">
              {projects.length === 0
                ? 'No projects yet. Create your first one.'
                : 'Select a project from the sidebar.'}
            </p>
            {projects.length === 0 && (
              <button
                onClick={handleNewProject}
                className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150"
              >
                + New Project
              </button>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
