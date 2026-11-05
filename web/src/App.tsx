// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { PanelLeft } from 'lucide-react';
import { usePlatform } from './hooks/usePlatform';
import { useProjects } from './hooks/useProjects';
import { useTriggers } from './hooks/useTriggers';
import {
  applyQuotaUpdate,
  fetchAgentAvailability,
  settleInstalledAgents,
  useInstalledAgents,
} from './hooks/useAgentAvailability';
import { useWebSocket, type ConnectionState } from './hooks/useWebSocket';
import { bumpPricingVersion } from './budget/pricingVersion';
import UpdatePill from './components/UpdatePill';
import type {
  AgentRunStatus,
  ApprovalRequestEvent,
  ApprovalResolvedEvent,
  AgentStatusEvent,
  DeviceStatusEvent,
  StatusSummaryEvent,
  ProjectCreateRequest,
  WebSocketEvent,
  SetupGateSettings,
} from './types';
import { routeKey, type Route } from './route';
import SetupGate from './components/SetupGate';
import SetupWizard from './components/SetupWizard';
import Sidebar from './components/Sidebar';
import EdgeStrip from './components/EdgeStrip';
import { effectivePinned, useWindowCovered } from './hooks/useWindowCover';
import CreateProject from './components/CreateProject';
import { requestOnboardingKickoff } from './utils/onboardingKickoff';
import FirstRunHome from './components/FirstRunHome';
import ImportProjectsDialog from './components/ImportProjectsDialog';
import { markFreshSession } from './utils/freshSession';
import { isFirstRun, needsSetupWizard, wizardLandingProjectId } from './utils/firstRun';
import FirstJourneyTour from './tour/FirstJourneyTour';
import {
  decideTourState, readTourState, shouldStartTour, writeTourState, type TourState,
} from './tour/tourLogic';
import ProjectDetail from './components/ProjectDetail';
import QueueTab from './components/QueueTab';
import SettingsModalPage from './components/SettingsModalPage';
import PricingEditorPage from './components/PricingEditorPage';
import ChatTab from './components/ChatTab';
import ErrorBoundary from './components/ErrorBoundary';
import GlobalSettings, { requestedSettingsSection } from './components/GlobalSettings';
import WorkbenchPage from './components/WorkbenchPage';
import { useWorkbenchFeed, useWorkbenchFeedSync } from './hooks/useWorkbenchFeed';
import { useCheckedMarks } from './hooks/useSeenReplies';
import { useHomeShortcut } from './hooks/useHomeShortcut';
import { noteAgentStatus } from './hooks/useFileHistory';
import { buildFeedGroups, unreadCounts } from './components/workbench/feedModel';
import { projectDisplayName } from './utils/projectDisplayName';
import { useT } from './i18n/useT';
import { api, isRelayMode } from './config';
import {
  SUB_AGENTS_RUNNING_EVENTS,
  createSubAgentsRunningSync,
} from './utils/projectStatus';

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
    reorderProjects,
    deleteProject,
  } = useProjects();
  const ws = useWebSocket();

  const [setupComplete, setSetupComplete] = useState<boolean | null>(null);
  const [needsWizard, setNeedsWizard] = useState<boolean | null>(null);
  const [route, setRoute] = useState<Route>({ name: 'list' });
  // One-shot deep link into Global Settings (the `open-global-settings`
  // event's detail.section). Kept beside the route rather than in it: only
  // the window event carries one, and the settings route stays a bare name.
  const [globalSettingsSection, setGlobalSettingsSection] = useState<string | undefined>(undefined);
  // Spec 078 amendment (2026-09-03): the edge strip's click pins the project
  // list as a docked column instead of navigating "home" — home is an empty
  // route now that the project is the work interface. In-memory only, never
  // persisted: hiding the list on entering a project is route-driven (D1),
  // so the pin resets whenever a project route is entered from any other
  // route (home, the Workbench, Settings — where the full list is on
  // screen anyway). It survives switching projects from inside the docked
  // list: the user pinned it to use it as a switcher.
  const [projectListPinned, setProjectListPinned] = useState(false);
  const prevRouteNameRef = useRef<Route['name']>(route.name);
  useLayoutEffect(() => {
    const prev = prevRouteNameRef.current;
    prevRouteNameRef.current = route.name;
    if (route.name === 'project' && prev !== 'project') setProjectListPinned(false);
  }, [route.name]);
  function toggleProjectListPinned() {
    setProjectListPinned((pinned) => !pinned);
  }
  // 2026-11-04: a maximized document covers the window except the 20px strip,
  // so a pinned list shows as the strip meanwhile and comes back after.
  const windowCovered = useWindowCovered();

  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentRunStatus>>({});
  const [statusTicks, setStatusTicks] = useState<Record<string, number>>({});
  const [statusSummaries, setStatusSummaries] = useState<Record<string, string>>({});
  const [pendingApprovals, setPendingApprovals] = useState<Record<string, number>>({});
  // Spec 095: a worker has an open turn in the project (run-status
  // `sub_agents_running`). A pinned dispatch runs no management turn, so
  // `agentStatuses` stays idle for the whole run; the project dot reads this.
  const [subAgentsRunning, setSubAgentsRunning] = useState<Record<string, boolean>>({});
  const [subAgentsSync] = useState(() =>
    createSubAgentsRunningSync(
      (projectId) =>
        api<{ sub_agents_running?: boolean }>(
          `/api/v2/agents/${encodeURIComponent(projectId)}/run-status`,
        ).then((result) => result.sub_agents_running),
      (projectId, running) =>
        setSubAgentsRunning((prev) =>
          (prev[projectId] ?? false) === running ? prev : { ...prev, [projectId]: running },
        ),
    ),
  );
  const [daemonOnline, setDaemonOnline] = useState(true);
  // Project-level agent availability. Lifted out of ChatView so a Files->Chat
  // tab switch doesn't block on the daemon's /agents/available cache miss
  // (cold call can take up to 8s). null = not yet loaded.
  const agentsAvailable = useInstalledAgents();
  // Global/default LLM model (from app settings) — header fallback when a
  // project has no per-project model pinned. Fetched once on mount.
  const [defaultModel, setDefaultModel] = useState<string>('');
  useEffect(() => {
    api<{ llm?: { model?: string } }>('/api/v2/settings')
      .then((d) => setDefaultModel(d?.llm?.model || ''))
      .catch(() => {});
  }, []);

  // Sandbox-repair affordance: the wizard's old sandbox step carried the only
  // Retry Setup button; with that step removed (spec 011 §0.7) the warning
  // banner hosts it instead. triggerSetup re-fetches status on success, which
  // hides the banner when setup completes.
  const [sandboxRetrying, setSandboxRetrying] = useState(false);
  // Derive selected project ID from route for convenience
  const selectedProjectId = route.name === 'project' ? route.projectId : null;

  // ── First-journey tour ────────────────────────────────────────────────
  // `tourState` is this device's one-time decision (see decideTourState: a new
  // install is `pending`, an install that already had a project is opted out).
  // It is decided the first time a LOADED project list is seen and never
  // revisited.
  const [tourState, setTourState] = useState<TourState | null>(() => {
    const stored = readTourState();
    return stored === 'pending' || stored === 'done' ? stored : null;
  });
  useEffect(() => {
    if (tourState !== null) return;
    const decided = decideTourState(readTourState(), projects);
    if (decided === null) return;
    writeTourState(decided);
    setTourState(decided);
  }, [projects, tourState]);
  // A replay from Settings → About shows the tour regardless of `tourState`.
  const [tourReplay, setTourReplay] = useState(false);
  const closeTour = useCallback(() => {
    writeTourState('done');
    setTourState('done');
    setTourReplay(false);
  }, []);

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

  // Fetch agent availability ONCE at app mount. Pre-filtered to the shape the
  // composer pin mark consumes: installed
  // sub-agents, no 'built-in'. The list is project-independent (the endpoint
  // takes no project and probes installed CLI binaries on disk — slow), so
  // fetching per project select and clearing meanwhile made the pin control
  // unmount and pop in late on every session open. Stale-while-revalidate:
  // the list is never cleared once loaded.
  // A sweep or an install that changed an agent says so over the WS, so the pin
  // menu picks it up without a reload. Otherwise it stays stale for the whole
  // page lifetime.
  const [agentsGeneration, setAgentsGeneration] = useState(0);
  useEffect(() => {
    const bump = () => setAgentsGeneration((g) => g + 1);
    ws.on('sub_agent_install_done', bump);
    ws.on('sub_agents.updated', bump);
    return () => {
      ws.off('sub_agent_install_done', bump);
      ws.off('sub_agents.updated', bump);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Spec 099: the same fetch fills the availability store (readiness +
  // subscription quota) the pin menu reads; `agent.quota_updated` keeps the
  // quota live between fetches.
  useEffect(() => {
    const onQuota = (e: WebSocketEvent) => {
      if (e.type === 'agent.quota_updated') applyQuotaUpdate(e.agent, e.snapshot);
    };
    ws.on('agent.quota_updated', onQuota);
    return () => ws.off('agent.quota_updated', onQuota);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    // First load failed → settle on empty (the @-dropdown shows no matches).
    fetchAgentAvailability().catch(settleInstalledAgents);
  }, [agentsGeneration]);

  // Mobile panel swap navigation. Spec 154 §5.10: the phone opens on the
  // content pane (the Workbench) unless this is the first run, which keeps
  // the project list (decided once, when the first project list loads).
  const [isMobile, setIsMobile] = useState(false);
  const [mobileView, setMobileView] = useState<'sidebar' | 'content'>('content');
  const mobileViewDecidedRef = useRef(false);
  useEffect(() => {
    if (mobileViewDecidedRef.current || projects.length === 0) return;
    mobileViewDecidedRef.current = true;
    if (isFirstRun(projects)) setMobileView('sidebar');
  }, [projects]);

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
      // First-run wizard: no key AND the key step was never skipped
      // (keyless onboarding). An unreachable daemon still shows it.
      const settingsData = await api<SetupGateSettings>('/api/v2/settings').catch(() => null);
      if (!cancelled) setNeedsWizard(needsSetupWizard(settingsData));
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
      const settleWorkers = subAgentsSync.track(p.project_id);
      api<{ project_id: string; status: AgentRunStatus; sub_agents_running?: boolean }>(
        `/api/v2/agents/${encodeURIComponent(p.project_id)}/run-status`,
      )
        .then((result) => {
          if (cancelled) return;
          setAgentStatuses((prev) => {
            if (prev[result.project_id] === result.status) return prev;
            return { ...prev, [result.project_id]: result.status };
          });
          settleWorkers(result.sub_agents_running);
        })
        .catch(() => {});
    }
    return () => { cancelled = true; };
  }, [setupComplete, projects, ws.connectionState, subAgentsSync]);

  // Spec 095: refetch the worker flag whenever it may have changed. Only the
  // flag is taken from these refetches — a response that lands after a newer
  // agent.status event would otherwise roll the manager status back.
  useEffect(() => {
    const onWorkerEvent = (event: WebSocketEvent) => {
      const projectId = (event as { project_id?: string }).project_id;
      if (projectId) subAgentsSync.refresh(projectId);
    };
    for (const type of SUB_AGENTS_RUNNING_EVENTS) ws.on(type, onWorkerEvent);
    return () => {
      for (const type of SUB_AGENTS_RUNNING_EVENTS) ws.off(type, onWorkerEvent);
    };
  }, [ws, subAgentsSync]);

  // Spec 154 §5.2: one App-level feed behind the Workbench, the strip
  // glyph's dot and the project-list counts (one predicate, one feed).
  useWorkbenchFeedSync(setupComplete === true);
  const feed = useWorkbenchFeed();
  const checkedMarks = useCheckedMarks();
  const feedGroups = useMemo(() => buildFeedGroups(feed, checkedMarks), [feed, checkedMarks]);
  const projectUnreadCounts = useMemo(() => unreadCounts(feedGroups), [feedGroups]);
  const anythingWaiting = feedGroups.waiting.length > 0;

  // Spec 154 §5.9: ⌘0 / Ctrl+0 goes home from anywhere (not while focus is
  // in the live browser view, which forwards keys to the agent's page).
  useHomeShortcut(() => {
    setRoute({ name: 'list' });
    setMobileView('content');
  });

  // Spec 146: asks changed in a project. The sidebar badge, an open Workbench
  // and the composer's tray all listen for this window event already.
  useEffect(() => {
    const onChanged = () => window.dispatchEvent(new Event('orbital:workbench-changed'));
    ws.on('workbench.changed', onChanged);
    return () => ws.off('workbench.changed', onChanged);
  }, [ws]);

  // WebSocket event handlers
  const handleAgentStatus = useCallback((event: WebSocketEvent) => {
    const e = event as AgentStatusEvent;
    noteAgentStatus(e.project_id, e.status);
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
  //
  // `detail.section` (optional) names a `data-settings-section` block to
  // scroll to — the composer pin's "API key required" row sends 'llm'.
  useEffect(() => {
    function onOpenGlobalSettings(e: Event) {
      setGlobalSettingsSection(requestedSettingsSection(e));
      setRoute({ name: 'settings' });
      setMobileView('content');
    }
    window.addEventListener('open-global-settings', onOpenGlobalSettings);
    return () => window.removeEventListener('open-global-settings', onOpenGlobalSettings);
  }, []);

  // Leaving Global Settings re-reads the project rows: a key saved there flips
  // `manager_ready` (the composer's "API key required" pill) and can change
  // each project's `effective_pinned_target`, and nothing else would refetch
  // them until the next reconnect.
  const prevRouteForProjectsRef = useRef<Route['name']>(route.name);
  useEffect(() => {
    const prev = prevRouteForProjectsRef.current;
    prevRouteForProjectsRef.current = route.name;
    if (prev === 'settings' && route.name !== 'settings') void listProjects();
  }, [route.name, listProjects]);

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
        setRoute({ name: 'project', projectId: event.data.projectId, sessionId: undefined });
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
    setRoute({ name: 'project', projectId: id, sessionId: undefined });
    setMobileView('content');
  }

  // Spec 154: the Workbench is home. The 工作台 row, the strip glyph, ⌘0 and
  // the phone header's mark all land here.
  function handleGoHome() {
    setRoute({ name: 'list' });
    setMobileView('content');
  }

  function handleNewProject() {
    setRoute({ name: 'create' });
    setMobileView('content');
  }

  async function handleCreateProject(data: ProjectCreateRequest) {
    const created = await createProject(data);
    // A project on an empty folder opens with the agent speaking first. A
    // folder with files gets the scan card instead, whose Skip does the same.
    if (created.is_empty_workspace) requestOnboardingKickoff(created.project_id);
    setRoute({ name: 'project', projectId: created.project_id, sessionId: undefined });
  }

  async function handleDeleteProject() {
    if (!selectedProjectId) return;
    await deleteProject(selectedProjectId);
    setRoute({ name: 'list' });
    setMobileView('sidebar');
  }

  // Spec 138: open an imported project on a NEW session, so its empty chat
  // shows the left-off cards. The minted id is marked fresh before the route
  // flips (spec 107: no history fetch for a session that cannot exist yet).
  async function openProjectOnNewSession(projectId: string) {
    let sessionId: string | undefined;
    try {
      const minted = await api<{ session_id?: string }>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/new-session`,
        { method: 'POST', body: JSON.stringify({}) },
      );
      sessionId = minted?.session_id || undefined;
      if (sessionId) markFreshSession(projectId, sessionId);
    } catch {
      // Still open the project; ChatTab picks its most recent session.
    }
    setRoute({ name: 'project', projectId, sessionId });
    setMobileView('content');
  }

  // "Import projects from your agents" (spec 138) — the dialog for users past
  // the wizard. Opens the most recently active imported project.
  const [importOpen, setImportOpen] = useState(false);
  async function handleImported(landingProjectId: string | null) {
    setImportOpen(false);
    const loaded = await listProjects();
    if (landingProjectId && loaded.some((p) => p.project_id === landingProjectId)) {
      await openProjectOnNewSession(landingProjectId);
    }
  }

  // First-run wizard (before setup gate)
  if (needsWizard === true) {
    return (
      <SetupWizard
        onComplete={async (outcome) => {
          setNeedsWizard(false);
          setSetupComplete(true);
          // Hand off straight into a project instead of the bare list: the
          // most recently active one the wizard just imported, on a new
          // session; otherwise the Create Project dialog. A new user was
          // never told that creating a project is the next step — so make it
          // the next step. Closing the dialog falls back to the first-run home.
          const loaded = await listProjects();
          const imported = outcome?.landingProjectId ?? null;
          const landing = wizardLandingProjectId(loaded);
          if (imported && loaded.some((p) => p.project_id === imported)) {
            await openProjectOnNewSession(imported);
          } else if (landing) {
            setRoute({ name: 'project', projectId: landing, sessionId: undefined });
          } else {
            setRoute({ name: 'create' });
          }
          setMobileView('content');
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

  // The tour's trigger is DERIVED, not an event: it shows while the user is on
  // the chat tab of a project they created and the tour is still pending (or
  // was asked for). Finishing or skipping flips `tourState` to done, which
  // turns this off — no "start" effect to keep in sync with the route.
  const onProjectChat = route.name === 'project' && route.page === undefined;
  // Held until the installed-agents probe has answered (`null` = still
  // loading; it shells out to find CLI binaries, so it is slow). The tour
  // freezes its list of stops when it starts, and the pin mark — the agents
  // stop's anchor — only renders once that list is in: starting early silently
  // dropped the stop.
  const tourShowing =
    onProjectChat && !!selectedProject && agentsAvailable !== null &&
    (tourReplay || shouldStartTour(tourState, selectedProject));

  function handleTakeTour() {
    // The tour walks through a project, so open the user's most recent one.
    const projectId = wizardLandingProjectId(projects);
    if (!projectId) return;
    setRoute({ name: 'project', projectId, sessionId: undefined });
    setMobileView('content');
    setTourReplay(true);
  }

  function handleMobileProjects() {
    setMobileView('sidebar');
  }

  const onHome = route.name === 'list' || route.name === 'create';
  const sidebarHidden = isMobile && mobileView !== 'sidebar';
  const contentHidden = isMobile && mobileView !== 'content';

  // Shared across both the home-route Sidebar and the EdgeStrip flyout's
  // Sidebar (spec 078 D3 — the flyout renders the real Sidebar, not a copy).
  const sidebarProps = {
    projects,
    agentStatuses,
    statusSummaries,
    pendingApprovals,
    subAgentsRunning,
    route,
    connectionState: mapConnectionState(ws.connectionState, daemonOnline),
    onSelectProject: handleSelectProject,
    onSelectHome: handleGoHome,
    waiting: anythingWaiting,
    unreadCounts: projectUnreadCounts,
    onNewProject: handleNewProject,
    onImportProjects: () => setImportOpen(true),
    onReorderProjects: reorderProjects,
    onSettings: () => {
      setGlobalSettingsSection(undefined);
      setRoute({ name: 'settings' });
      setMobileView('content');
    },
  };

  return (
    <div className="flex h-dvh overflow-hidden">
      <UpdatePill />
      {/* Desktop, a project open (any tab): the 260px Sidebar hides behind a
          20px EdgeStrip that floats it over the session column on hover and
          docks it on click (spec 078 D1-D3 + the pin amendment). Every other
          route, and mobile always, render exactly what rendered before this
          spec. */}
      {/* Home (the Workbench) docks the full project list, as every other
          non-project route does (2026-11-05; it used to sit in the collapsed
          edge strip). */}
      {!isMobile && route.name === 'project' ? (
        <EdgeStrip
          projects={projects}
          currentProjectId={route.name === 'project' ? route.projectId : undefined}
          waiting={anythingWaiting}
          onHome={handleGoHome}
          homeActive={onHome}
          agentStatuses={agentStatuses}
          pendingApprovals={pendingApprovals}
          subAgentsRunning={subAgentsRunning}
          pinned={effectivePinned(projectListPinned, windowCovered)}
          onTogglePin={toggleProjectListPinned}
        >
          <Sidebar {...sidebarProps} narrow />
        </EdgeStrip>
      ) : (
        <div className={sidebarHidden ? 'hidden' : 'contents'}>
          <Sidebar {...sidebarProps} />
        </div>
      )}

      {/* NOTE: keep a space before `${` — Tailwind's scanner will not extract a
          class that runs straight into a template-literal interpolation, so
          `shadow-panel${...}` silently emits no CSS. */}
      {/* pt-titlebar clears the macOS transparent-titlebar band (0 elsewhere).
          The band's fill is route-dependent so it matches the pane beneath it:
          the chat pane is white "paper" while the card-list surfaces keep the
          grey canvas so their cards still lift off it. Global settings opens
          with the card-coloured PageToolbar, so it takes the card band
          too. Both classes are inert off macOS, so the non-mac render is
          unchanged. */}
      <main className={`flex-1 overflow-hidden flex flex-col shadow-panel pt-titlebar ${route.name === 'project' || route.name === 'settings' ? 'titlebar-band-card' : 'titlebar-band-canvas'} ${onHome && !isFirstRun(projects) ? 'wb-band' : ''} ${contentHidden ? 'hidden' : ''}`}>
        {/* Phone header (spec 154 §5.9): the Orbital mark goes home, the
            right button opens the project list. Replaces the old Back row. */}
        {isMobile && mobileView === 'content' && (
          <div
            data-testid="mobile-header"
            className={`flex items-center gap-2 pl-4 pr-3 pt-4 pb-2.5 shrink-0 ${
              onHome && !isFirstRun(projects) ? 'bg-[#FAFBFC] dark:bg-background' : ''
            }`}
          >
            <button
              type="button"
              onClick={handleGoHome}
              aria-label={t('workbench.title')}
              data-testid="mobile-home"
              className="flex min-w-0 flex-1 items-center gap-2 min-h-[44px] text-left"
            >
              <img src="/icon-192.png" alt="" aria-hidden="true" className="h-6 w-6 shrink-0 rounded-md" />
              {selectedProject ? (
                <span className="truncate text-[15px] font-medium text-secondary">
                  {projectDisplayName(selectedProject, t)}
                </span>
              ) : (
                <span className="text-[17px] font-semibold text-primary">{t('sidebar.wordmark')}</span>
              )}
            </button>
            <button
              type="button"
              onClick={handleMobileProjects}
              aria-label={t('sidebar.projects')}
              title={t('sidebar.projects')}
              data-testid="mobile-projects"
              className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-lg text-secondary hover:text-primary"
            >
              <PanelLeft size={17} strokeWidth={1.8} aria-hidden="true" />
            </button>
          </div>
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
            scrollToSection={globalSettingsSection}
            onBack={() => { setRoute({ name: 'list' }); setMobileView('sidebar'); }}
            onTakeTour={handleTakeTour}
            canTakeTour={wizardLandingProjectId(projects) !== null}
          />
        )}

        {route.name === 'project' && selectedProject && (
          // Scoped to the project view: a render crash here shows a recoverable
          // fallback instead of unmounting the whole app (the app shell/sidebar
          // stays alive). resetKey clears a prior crash when the user navigates.
          <ErrorBoundary resetKey={routeKey(route)}>
            {/* Spec 137: one ProjectDetail for every page. Chat (no page) is
                the home and its body is the children; Tasks, Settings and
                Pricing are pages whose bodies are handed over in
                `pageContent` — ProjectDetail draws the shared toolbar and the
                page registry decides the rest. Only the open page's element
                is ever mounted. */}
            <ProjectDetail
              project={selectedProject}
              agentStatus={agentStatuses[selectedProject.project_id] ?? 'idle'}
              subAgentsRunning={subAgentsRunning[selectedProject.project_id] ?? false}
              statusSummary={statusSummaries[selectedProject.project_id]}
              route={route}
              setRoute={setRoute}
              triggers={triggers}
              onTriggerToggle={toggleTrigger}
              onTriggerDelete={deleteTrigger}
              globalDefaultModel={defaultModel}
              agentsAvailable={agentsAvailable ?? []}
              pageContent={{
                tasks: (
                  <QueueTab
                    key={`queue-${selectedProject.project_id}`}
                    projectId={selectedProject.project_id}
                    project={selectedProject}
                    agents={agentsAvailable ?? []}
                  />
                ),
                settings: (
                  <SettingsModalPage
                    project={selectedProject}
                    // Bound to THIS project, not "whatever is selected when the
                    // request fires": settings autosave, and a pause-then-save can
                    // outlive a switch to another project.
                    onSave={(data) => updateProject(selectedProject.project_id, data)}
                    onDelete={handleDeleteProject}
                    onEditPricing={() => setRoute({ ...route, page: 'pricing' })}
                    scrollToSection={route.settingsAnchor}
                  />
                ),
                // Pricing-table editor (P3-I), a page under Settings. onSaved
                // bumps the module-level pricing version so EVERY mounted
                // useCost (settings meter, header budget corner, queue banner)
                // recomputes against the new rates.
                pricing: <PricingEditorPage onSaved={bumpPricingVersion} />,
              }}
            >
              {route.page === undefined && (
                <ChatTab
                  key={selectedProject.project_id}
                  project={selectedProject}
                  projects={projects}
                  agentStatus={agentStatuses[selectedProject.project_id] ?? 'idle'}
                  statusTick={statusTicks[selectedProject.project_id] ?? 0}
                  agents={agentsAvailable ?? []}
                  route={route}
                  setRoute={setRoute}
                  onRefreshProject={refreshProject}
                />
              )}
            </ProjectDetail>
          </ErrorBoundary>
        )}

        {/* No project of their own yet → the first-run home. The old
            `projects.length === 0` "create your first one" branch that stood
            here never rendered: the daemon always creates Quick Tasks, so a
            loaded list is never empty. */}
        {/* One branch for 'list' AND 'create' so the home stays mounted behind
            the Create Project dialog's backdrop: the screen is not blank
            under it, and cancelling does not replay the entrance. */}
        {(route.name === 'list' || route.name === 'create') && isFirstRun(projects) && (
          <FirstRunHome onNewProject={handleNewProject} onImportProjects={() => setImportOpen(true)} />
        )}
        {/* Spec 154: home is the Workbench (not first run). 'create' keeps
            it mounted behind the Create Project dialog's backdrop. */}
        {onHome && !isFirstRun(projects) && (
          <WorkbenchPage
            setRoute={setRoute}
            projects={projects}
            agents={agentsAvailable ?? []}
            phone={isMobile}
            managerModel={defaultModel}
            agentStatuses={agentStatuses}
            statusTicks={statusTicks}
          />
        )}

        {/* 'blocked' route UI (BlockedBadge view) is a later task — minimal stub for now. */}
        {route.name === 'blocked' && (
          <div data-testid="blocked-route-stub" />
        )}
      </main>

      {/* New Project is a modal overlay (backlog #25), not page content —
          kept as a sibling of <main> rather than one of its route branches.
          Inside <main> above, 'create' matches only the first-run home; for
          everyone else it leaves <main> empty, so the modal's backdrop sits
          over an empty pane rather than the previous route's content. Cancel
          routes back to 'list'. */}
      {tourShowing && selectedProject && (
        <FirstJourneyTour
          folderName={
            (selectedProject.workspace || '').replace(/[\\/]+$/, '').split(/[\\/]/).filter(Boolean).pop()
            || selectedProject.name
          }
          agentNames={(agentsAvailable ?? []).map((a) => a.name)}
          onClose={closeTour}
        />
      )}

      {importOpen && (
        <ImportProjectsDialog
          onClose={() => setImportOpen(false)}
          onImported={(_result, landing) => { void handleImported(landing); }}
        />
      )}

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
