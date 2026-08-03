// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useRef, useEffect, useCallback } from 'react';
import type {
  Autonomy,
  FallbackModelEntry,
  NotificationPrefs,
  Project,
  ProjectUpdateRequest,
  ProviderRegistry,
} from '../types';
import { api, BASE_URL, isRelayMode, ApiError } from '../config';
import LLMProviderSettings, { type LLMValues } from './LLMProviderSettings';
import FallbackModelsEditor from './FallbackModelsEditor';
import { type SubAgentMemoryEntry } from './SubAgentMemoryCard';
import { type InstalledSubAgent } from './SubAgentToggleList';
import SubAgentCard from './SubAgentCard';
import BudgetSection from './BudgetSection';
import ProjectConnectorToggles from './ProjectConnectorToggles';
import { NetworkAccessSection } from './NetworkAccessSection';
import type { PendingDomainRequest } from '../types';
import SettingsRail, {
  scrollToSettingsSection,
  type SettingsRailSection,
} from './SettingsRail';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

interface SkillMeta {
  name: string;
  description: string;
  path: string;
  dir_name: string;
}

interface SettingsViewProps {
  project: Project;
  onSave: (data: ProjectUpdateRequest) => void;
  onDelete: () => void;
  /** Navigate to the pricing-table editor. */
  onEditPricing?: () => void;
  /**
   * Bumped after a pricing-table save so the Budget section's useCost
   * re-fetches and the meter/breakdown recompute against the new rates.
   */
  costRefreshKey?: number;
  /**
   * Optional section to scroll into view on mount — any `data-settings-section`
   * id (P3-G: the header budget corner deep-links here with 'budget').
   * Consumed once via the shared section-scroll mechanism.
   */
  scrollToSection?: string;
}

/**
 * Index-rail entries for the project settings document (spec 011 §0.8).
 * Conditional groups (scratch/relay-gated) and the reserved 'connectors' id
 * (section lands in a later wave) only get a rail entry once a matching
 * data-settings-section element actually exists in the DOM.
 */
export const PROJECT_SETTINGS_SECTIONS: SettingsRailSection[] = [
  { id: 'agent-name', labelKey: 'createProject.agentName.label' },
  { id: 'project-goals', labelKey: 'settings.projectGoals.label' },
  { id: 'project-instructions', labelKey: 'settings.projectInstructions.label' },
  { id: 'sub-agents', labelKey: 'settings.subAgents.label' },
  { id: 'skills', labelKey: 'settings.skills.label' },
  { id: 'notifications', labelKey: 'settings.notifications.label' },
  { id: 'llm', labelKey: 'llm.provider.heading' },
  { id: 'fallback-models', labelKey: 'fallback.heading' },
  { id: 'autonomy', labelKey: 'autonomy.level.label' },
  { id: 'budget', labelKey: 'settings.budget.label' },
  { id: 'network', labelKey: 'settingsRail.network' },
  { id: 'connectors', labelKey: 'settingsRail.connectors' },
  { id: 'workbench', labelKey: 'settings.workbench.label' },
  { id: 'danger', labelKey: 'settings.danger.title' },
];

const AUTONOMY_OPTIONS: {
  value: Autonomy;
  titleKey: StringKey;
  descriptionKey: StringKey;
}[] = [
  {
    value: 'hands_off',
    titleKey: 'autonomy.handsOff.title',
    descriptionKey: 'autonomy.handsOff.desc',
  },
  {
    value: 'check_in',
    titleKey: 'autonomy.checkIn.title',
    descriptionKey: 'autonomy.checkIn.desc',
  },
  {
    value: 'supervised',
    titleKey: 'autonomy.supervised.title',
    descriptionKey: 'autonomy.supervised.desc',
  },
];

export default function SettingsView({
  project,
  onSave,
  onDelete,
  onEditPricing = () => {},
  costRefreshKey,
  scrollToSection,
}: SettingsViewProps) {
  const t = useT();
  // Scroll container ref — SettingsRail scopes section discovery, scrollspy,
  // and jump-scrolls to this element; the anchor deep-link shares it.
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Deep-link: when opened with a section anchor (e.g. 'budget' from the
  // header corner click), scroll it into view once via the shared section
  // scroll mechanism. Best-effort — no-op in jsdom or for unknown ids.
  useEffect(() => {
    if (scrollToSection && scrollContainerRef.current) {
      scrollToSettingsSection(scrollContainerRef.current, scrollToSection);
    }
  }, [scrollToSection]);
  const [agentName, setAgentName] = useState(project.agent_name || project.name);
  const [projectGoals, setProjectGoals] = useState(project.project_goals_content || '');
  const [standingRules, setStandingRules] = useState(project.user_directives_content || '');
  const [autonomy, setAutonomy] = useState<Autonomy>(project.autonomy);
  const [saved, setSaved] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [notifPrefs, setNotifPrefs] = useState<NotificationPrefs>(
    project.notification_prefs || {},
  );

  // Workbench privacy toggle (spec §6 "Aggregation & privacy") — excludes this
  // project's flagged entries from the GLOBAL Workbench view. The
  // per-project lens is unaffected.
  const [workbenchExcludeGlobal, setWorkbenchExcludeGlobal] = useState(
    project.workbench_exclude_global ?? false,
  );

  // Skills state
  const [skills, setSkills] = useState<SkillMeta[]>([]);
  const [skillError, setSkillError] = useState('');
  const [skillSuccess, setSkillSuccess] = useState('');
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sub-agent MEMORY.md state
  const [memoryEntries, setMemoryEntries] = useState<SubAgentMemoryEntry[]>([]);
  const [memoryError, setMemoryError] = useState('');

  // Sub-agent enable/disable state. The list of installed sub-agents is
  // global (one daemon-level binary check); the denylist is per-project.
  const [installedSubAgents, setInstalledSubAgents] = useState<InstalledSubAgent[]>([]);
  // The binary probe behind GET /settings/sub-agents runs multi-second on a cold
  // daemon; without this flag the section's empty state doubles as its loading
  // state and tells the user they have no sub-agents installed (bug #36).
  const [loadingSubAgents, setLoadingSubAgents] = useState(!project.is_scratch);
  const [disabledSubAgents, setDisabledSubAgents] = useState<string[]>(
    project.disabled_sub_agents ?? [],
  );
  const [subAgentDeploymentInstructions, setSubAgentDeploymentInstructions] = useState(
    project.sub_agent_deployment_instructions ?? '',
  );

  // Per-project connector enablement (spec 011 §0.2 — authenticate globally,
  // enable per project). Saved through THIS form's payload, like every other
  // project field; the toggles component only reads/writes this state.
  const [enabledConnectors, setEnabledConnectors] = useState<string[]>(
    project.enabled_connectors ?? [],
  );

  // TOFU network grants (Plan 2). Like enabled_connectors, saved through
  // THIS form's payload — Task 2's PUT route applies the resulting grants to
  // a running agent's proxy immediately. pendingDomainRequests is display +
  // approve/dismiss state; approving moves an entry into approvedDomains and
  // drops it from pendingDomainRequests locally (persisted together on Save).
  const [approvedDomains, setApprovedDomains] = useState<string[]>(
    project.approved_domains ?? [],
  );
  const [pendingDomainRequests, setPendingDomainRequests] = useState<PendingDomainRequest[]>(
    project.pending_domain_requests ?? [],
  );

  // Budget state. Spend is NOT tracked here anymore — the Budget section reads
  // it exclusively from GET /cost via useCost (P3-F coupled removal of the
  // legacy budget_spent_usd accumulator).
  const [budgetLimit, setBudgetLimit] = useState<string>(
    project.budget_limit_usd != null ? String(project.budget_limit_usd) : '',
  );
  const [budgetCurrency, setBudgetCurrency] = useState<string>(
    project.budget_currency || 'USD',
  );
  const [budgetPeriod, setBudgetPeriod] = useState<
    'daily' | 'weekly' | 'monthly' | 'total'
  >(project.budget_period || 'daily');
  const [budgetAction, setBudgetAction] = useState<'pause' | 'stop'>(
    project.budget_action === 'stop' ? 'stop' : 'pause',
  );

  // Fallback models state
  const [fallbackModels, setFallbackModels] = useState<FallbackModelEntry[]>(
    project.llm_fallback_models || [],
  );
  const fallbackModelsRef = useRef<FallbackModelEntry[]>(fallbackModels);
  const [providers, setProviders] = useState<ProviderRegistry>({});

  useEffect(() => {
    let cancelled = false;
    api<ProviderRegistry>('/api/v2/providers')
      .then((data) => { if (!cancelled) setProviders(data); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const pid = project.project_id;

  // Fetch project detail (with disk content) on mount / project change
  useEffect(() => {
    let cancelled = false;
    setLoadingDetail(true);
    api<Project>(`/api/v2/projects/${encodeURIComponent(pid)}`)
      .then((detail) => {
        if (cancelled) return;
        setProjectGoals(detail.project_goals_content || '');
        setStandingRules(detail.user_directives_content || '');
        setBudgetLimit(detail.budget_limit_usd != null ? String(detail.budget_limit_usd) : '');
        if (detail.budget_currency) setBudgetCurrency(detail.budget_currency);
        if (detail.budget_period) setBudgetPeriod(detail.budget_period);
        setBudgetAction(detail.budget_action === 'stop' ? 'stop' : 'pause');
        if (Array.isArray(detail.enabled_connectors)) {
          setEnabledConnectors(detail.enabled_connectors);
        }
        if (Array.isArray(detail.approved_domains)) {
          setApprovedDomains(detail.approved_domains);
        }
        if (Array.isArray(detail.pending_domain_requests)) {
          setPendingDomainRequests(detail.pending_domain_requests);
        }
        setSubAgentDeploymentInstructions(detail.sub_agent_deployment_instructions ?? '');
      })
      .catch(() => {
        // On error, leave textareas with current (likely empty) values
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });
    return () => { cancelled = true; };
  }, [pid]);

  const fetchSkills = useCallback(async () => {
    if (project.is_scratch) return;
    try {
      const data = await api<SkillMeta[]>(
        `/api/v2/projects/${encodeURIComponent(pid)}/skills`,
      );
      setSkills(data);
    } catch {
      // silently ignore — skills section just shows empty
    }
  }, [pid, project.is_scratch]);

  useEffect(() => {
    fetchSkills();
  }, [fetchSkills]);

  // Load sub-agent MEMORY.md list. Re-runs whenever the project changes
  // (which includes when disabled_sub_agents has been edited and saved
  // via the parent — the parent re-fetches and passes a fresh project).
  const fetchMemories = useCallback(async () => {
    if (project.is_scratch) return;
    setMemoryError('');
    try {
      const data = await api<SubAgentMemoryEntry[]>(
        `/api/v2/projects/${encodeURIComponent(pid)}/sub-agent-memory`,
      );
      setMemoryEntries(data);
    } catch (e) {
      setMemoryError(e instanceof ApiError ? e.detail : t('settings.subAgentMemories.loadError'));
    }
  }, [pid, project.is_scratch]);

  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  // Load the installed sub-agents (global, daemon-level binary check) so the
  // user can enable/disable each for THIS project. Scratch projects don't
  // delegate to sub-agents, so skip.
  useEffect(() => {
    if (project.is_scratch) {
      setLoadingSubAgents(false);
      return;
    }
    let cancelled = false;
    setLoadingSubAgents(true);
    api<Array<{ slug: string; name: string; installed: boolean; ready: boolean }>>(
      '/api/v2/settings/sub-agents',
    )
      .then((data) => {
        if (cancelled) return;
        setInstalledSubAgents(
          data
            .filter((s) => s.installed && s.slug !== 'built-in')
            .map((s) => ({ slug: s.slug, name: s.name, ready: s.ready })),
        );
      })
      .catch(() => {
        // Non-fatal — the section falls back to its empty state.
      })
      // Must clear on BOTH outcomes or a failed probe wedges the section in
      // "checking…" forever.
      .finally(() => {
        if (!cancelled) setLoadingSubAgents(false);
      });
    return () => { cancelled = true; };
  }, [project.is_scratch]);

  const handleToggleSubAgent = useCallback((slug: string, enabled: boolean) => {
    setDisabledSubAgents((prev) =>
      enabled
        ? prev.filter((s) => s !== slug)
        : prev.includes(slug) ? prev : [...prev, slug],
    );
  }, []);

  async function handleDeleteSkill(dirName: string) {
    setSkillError('');
    try {
      await api(`/api/v2/projects/${encodeURIComponent(pid)}/skills/${encodeURIComponent(dirName)}`, {
        method: 'DELETE',
      });
      await fetchSkills();
    } catch (e) {
      setSkillError(e instanceof ApiError ? e.detail : t('settings.skills.deleteError'));
    }
  }

  async function handleSkillUpload(file: File) {
    setSkillError('');
    setSkillSuccess('');
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const base = isRelayMode ? window.location.origin : BASE_URL;
      const url = `${base}/api/v2/projects/${encodeURIComponent(pid)}/skills`;
      const headers: Record<string, string> = {};
      if (isRelayMode) {
        const token = localStorage.getItem('relay_jwt');
        if (token) headers['Authorization'] = `Bearer ${token}`;
      }
      const resp = await fetch(url, { method: 'POST', body: form, headers });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ detail: t('settings.skills.uploadError') }));
        throw new Error(body.detail || `HTTP ${resp.status}`);
      }
      setSkillSuccess(t('settings.skills.added'));
      setTimeout(() => setSkillSuccess(''), 2000);
      await fetchSkills();
    } catch (e) {
      setSkillError(e instanceof Error ? e.message : t('settings.skills.uploadError'));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  // Track LLM values from the shared component
  const llmValuesRef = useRef<LLMValues>({
    provider: project.provider,
    model: project.model,
    base_url: project.base_url || undefined,
    sdk: (project.sdk as 'openai' | 'anthropic') || 'openai',
  });

  function handleLLMChange(values: LLMValues) {
    llmValuesRef.current = values;
  }

  function handleFallbackChange(models: FallbackModelEntry[]) {
    setFallbackModels(models);
    fallbackModelsRef.current = models;
  }

  function handleSave(ev: React.FormEvent) {
    ev.preventDefault();
    const llm = llmValuesRef.current;
    const data: ProjectUpdateRequest = {
      agent_name: agentName,
      project_goals_content: projectGoals,
      user_directives_content: standingRules,
      model: llm.model,
      autonomy,
      base_url: llm.base_url || undefined,
      provider: llm.provider || undefined,
      sdk: llm.sdk,
      llm_fallback_models: fallbackModelsRef.current,
      budget_limit_usd: budgetLimit ? parseFloat(budgetLimit) : null,
      budget_currency: budgetCurrency,
      budget_period: budgetPeriod,
      budget_action: budgetAction,
      disabled_sub_agents: disabledSubAgents,
      sub_agent_deployment_instructions: subAgentDeploymentInstructions,
      enabled_connectors: enabledConnectors,
      approved_domains: approvedDomains,
      pending_domain_requests: pendingDomainRequests,
      workbench_exclude_global: workbenchExcludeGlobal,
    };
    if (llm.api_key) {
      data.api_key = llm.api_key;
    }
    onSave(data);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  function handleDelete() {
    onDelete();
  }

  // Index the per-project memory entries by slug so the merged sub-agent
  // cards can look up each agent's MEMORY.md state in O(1).
  const memoryBySlug = new Map(
    memoryEntries.map((e) => [e.agent_slug, e]),
  );

  return (
    <div ref={scrollContainerRef} className="h-full overflow-y-auto">
    {/* Index rail beside the single scrolling document (desktop) / jump menu
        above it (mobile). One document, one Save — no pagination. */}
    <div className="flex justify-center max-md:block">
    <SettingsRail
      sections={PROJECT_SETTINGS_SECTIONS}
      containerRef={scrollContainerRef}
    />
    <div className="max-w-[720px] w-full min-w-0 py-8 px-6 max-md:px-4">
      <form onSubmit={handleSave} className="space-y-6">
        {/* Agent Name */}
        <div data-settings-section="agent-name" className="scroll-mt-4">
          <label className="block text-sm font-medium text-primary mb-1.5">
            {t('createProject.agentName.label')}
          </label>
          <input
            type="text"
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
            placeholder={t('settings.agentName.placeholder')}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
          />
        </div>

        {/* Project Goals */}
        {!project.is_scratch && (
          <div data-settings-section="project-goals" className="scroll-mt-4">
            <label className="block text-sm font-medium text-primary mb-1.5">
              {t('settings.projectGoals.label')}
            </label>
            <textarea
              rows={6}
              value={projectGoals}
              onChange={(e) => setProjectGoals(e.target.value)}
              disabled={loadingDetail}
              placeholder={loadingDetail ? t('settings.loading') : t('settings.projectGoals.placeholder')}
              className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
            />
          </div>
        )}

        {/* Project Instructions */}
        <div data-settings-section="project-instructions" className="scroll-mt-4">
          <label className="block text-sm font-medium text-primary mb-1.5">
            {t('settings.projectInstructions.label')}
          </label>
          <p className="text-xs text-secondary mb-2">
            {t('settings.projectInstructions.hint')}
          </p>
          <textarea
            rows={4}
            value={standingRules}
            onChange={(e) => setStandingRules(e.target.value)}
            disabled={loadingDetail}
            placeholder={loadingDetail ? t('settings.loading') : t('settings.projectInstructions.placeholder')}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
          />
        </div>

        {/* Sub-Agents — merged per-agent card: enable toggle + auth badge +
            memory summary in the header, MEMORY.md editor as the expandable
            body. One card per installed sub-agent (a disabled one is dimmed
            but still expandable). */}
        {!project.is_scratch && (
          <div data-settings-section="sub-agents" className="scroll-mt-4">
            <label className="block text-sm font-medium text-primary mb-1.5">
              {t('settings.subAgents.label')}
              <span className="text-secondary font-normal"> {t('settings.subAgents.labelSuffix')}</span>
            </label>
            <p className="text-xs text-secondary mb-2">
              {t('settings.subAgents.hint')}
            </p>

            <label
              htmlFor="sub-agent-deployment-instructions"
              className="block text-sm font-medium text-primary mb-1.5"
            >
              {t('settings.subAgents.deploymentInstructions.label')}
            </label>
            <p className="text-xs text-secondary mb-2">
              {t('settings.subAgents.deploymentInstructions.hint')}
            </p>
            <textarea
              id="sub-agent-deployment-instructions"
              rows={4}
              maxLength={4000}
              value={subAgentDeploymentInstructions}
              onChange={(e) => setSubAgentDeploymentInstructions(e.target.value)}
              disabled={loadingDetail}
              placeholder={loadingDetail
                ? t('settings.loading')
                : t('settings.subAgents.deploymentInstructions.placeholder')}
              className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 mb-3 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
            />

            {memoryError && (
              <p className="text-xs text-error mb-2">{memoryError}</p>
            )}

            {loadingSubAgents ? (
              <p className="text-xs text-secondary/60 italic">
                {t('settings.subAgents.loading')}
              </p>
            ) : installedSubAgents.length === 0 ? (
              <p className="text-xs text-secondary/60 italic">
                {t('settings.subAgents.installHint')}
              </p>
            ) : (
              <>
                <div className="flex flex-col gap-2">
                  {installedSubAgents.map((agent) => (
                    <SubAgentCard
                      key={agent.slug}
                      projectId={pid}
                      agentName={agent.name}
                      slug={agent.slug}
                      enabled={!disabledSubAgents.includes(agent.slug)}
                      ready={agent.ready ?? true}
                      memory={memoryBySlug.get(agent.slug)}
                      onToggle={handleToggleSubAgent}
                      onMemorySaved={fetchMemories}
                    />
                  ))}
                </div>
                <p className="text-[11px] text-secondary/60 mt-1.5 italic">
                  {t('settings.subAgents.saveReminder')}
                </p>
                <p className="text-[11px] text-secondary/60 mt-1 italic">
                  {t('settings.subAgents.installHint')}
                </p>
              </>
            )}
          </div>
        )}

        {/* Skills */}
        {!project.is_scratch && (
          <div data-settings-section="skills" className="scroll-mt-4">
            <label className="block text-sm font-medium text-primary mb-1.5">
              {t('settings.skills.label')}
            </label>
            <p className="text-xs text-secondary mb-2">
              {t('settings.skills.hint')}
            </p>

            {skillError && (
              <p className="text-xs text-error mb-2">{skillError}</p>
            )}
            {skillSuccess && (
              <p className="text-xs text-success mb-2">{skillSuccess}</p>
            )}

            {skills.length > 0 ? (
              <div className="space-y-2 mb-3">
                {skills.map((s) => (
                  <div
                    key={s.name}
                    className="flex items-center justify-between bg-sidebar border border-border rounded-lg px-3 py-2"
                  >
                    <div className="min-w-0 mr-2">
                      <span className="text-sm font-medium text-primary block truncate">
                        {s.name}
                      </span>
                      <span className="text-xs text-secondary block truncate">
                        {s.description}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleDeleteSkill(s.dir_name || s.name)}
                      className="shrink-0 text-secondary hover:text-error transition-colors p-1 max-md:min-w-[44px] max-md:min-h-[44px] flex items-center justify-center"
                      title={t('settings.skills.deleteTitle', { name: s.name })}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-secondary/60 italic mb-3">
                {t('settings.skills.empty')}
              </p>
            )}

            <label className="flex items-center justify-center border border-dashed border-border rounded-lg px-3 py-3 cursor-pointer hover:border-accent/50 hover:bg-accent/5 transition-all duration-150 max-md:min-h-[44px]">
              <span className="text-sm text-secondary">
                {uploading ? t('settings.skills.uploading') : t('settings.skills.uploadCta')}
              </span>
              <input
                ref={fileInputRef}
                type="file"
                accept=".md,.zip"
                className="hidden"
                disabled={uploading}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleSkillUpload(f);
                }}
              />
            </label>
          </div>
        )}

        {/* Notification Preferences (remote mode only) */}
        {!project.is_scratch && isRelayMode && (
          <div data-settings-section="notifications" className="scroll-mt-4">
            <label className="block text-sm font-medium text-primary mb-1.5">
              {t('settings.notifications.label')}
            </label>
            <p className="text-xs text-secondary mb-2">
              {t('settings.notifications.hint')}
            </p>
            <div className="space-y-2">
              {([
                { key: 'task_completed', labelKey: 'settings.notifications.taskCompleted' },
                { key: 'errors', labelKey: 'settings.notifications.errors' },
                { key: 'agent_messages', labelKey: 'settings.notifications.agentMessages' },
                { key: 'trigger_started', labelKey: 'settings.notifications.triggerStarted' },
              ] as const).map(({ key, labelKey }) => (
                <label key={key} className="flex items-center gap-2 cursor-pointer max-md:min-h-[44px]">
                  <input
                    type="checkbox"
                    checked={notifPrefs[key] ?? (key !== 'trigger_started')}
                    onChange={(e) => {
                      const updated = { ...notifPrefs, [key]: e.target.checked };
                      setNotifPrefs(updated);
                      // Save immediately via API
                      onSave({ notification_prefs: updated });
                    }}
                    className="rounded border-border accent-accent"
                  />
                  <span className="text-sm text-primary">{t(labelKey)}</span>
                </label>
              ))}
            </div>
            <p className="text-xs text-secondary/60 mt-2 italic">
              {t('settings.notifications.approvalNote')}
            </p>
          </div>
        )}

        {/* LLM Provider (collapsible, from shared component) */}
        <div data-settings-section="llm" className="scroll-mt-4">
          <LLMProviderSettings
            mode="project"
            projectValues={{
              provider: project.provider,
              model: project.model,
              api_key: project.api_key,
              base_url: project.base_url,
              sdk: project.sdk,
            }}
            onChange={handleLLMChange}
          />
        </div>

        {/* Fallback Models */}
        <div data-settings-section="fallback-models" className="scroll-mt-4">
          <FallbackModelsEditor
            models={fallbackModels}
            onChange={handleFallbackChange}
            providers={providers}
          />
        </div>

        {/* Autonomy Level */}
        <div data-settings-section="autonomy" className="scroll-mt-4">
          <label className="block text-sm font-medium text-primary mb-2">
            {t('autonomy.level.label')}
          </label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {AUTONOMY_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setAutonomy(opt.value)}
                className={`text-left border rounded-lg p-3 transition-all duration-150 max-md:min-h-[44px] ${
                  autonomy === opt.value
                    ? 'border-accent bg-accent/5'
                    : 'border-border hover:border-secondary/40'
                }`}
              >
                <span className="text-sm font-medium text-primary block">
                  {t(opt.titleKey)}
                </span>
                <span className="text-xs text-secondary mt-1 block">
                  {t(opt.descriptionKey)}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Budget — limit-as-sentence, behavior cards, spend meter, breakdown.
            Spend is read EXCLUSIVELY from GET /cost (useCost), refreshed on the
            budget.spend_updated WS event. The data-settings-section tag anchors
            the header corner's deep-link (scrollToSection === 'budget') and
            the index rail. */}
        <div data-settings-section="budget" className="scroll-mt-4">
          <BudgetSection
            project={project}
            limit={budgetLimit}
            onLimitChange={setBudgetLimit}
            currency={budgetCurrency}
            onCurrencyChange={setBudgetCurrency}
            period={budgetPeriod}
            onPeriodChange={setBudgetPeriod}
            action={budgetAction}
            onActionChange={setBudgetAction}
            onEditPricing={onEditPricing}
            costRefreshKey={costRefreshKey}
          />
        </div>

        {/* Network access — TOFU allowlist (Plan 2 Task 7). Approved domains +
            pending requests, writing approved_domains / pending_domain_requests
            through this form's save (Task 2's PUT route live-rebuilds the
            proxy rules server-side). DOM order: budget → network → connectors
            → danger, matching PROJECT_SETTINGS_SECTIONS. */}
        <div data-settings-section="network" className="scroll-mt-4">
          <NetworkAccessSection
            approvedDomains={approvedDomains}
            pendingRequests={pendingDomainRequests}
            onChange={({ approvedDomains: next, pendingRequests }) => {
              setApprovedDomains(next);
              setPendingDomainRequests(pendingRequests);
            }}
          />
        </div>

        {/* Connectors — per-project enablement (spec 011 §0.2/§0.6, Task E1).
            One switch per globally-connected connector, writing
            enabled_connectors through this form's save. Mounting this fills
            the reserved 'connectors' rail entry (DOM order: budget → network →
            connectors → danger, matching PROJECT_SETTINGS_SECTIONS). */}
        <div data-settings-section="connectors" className="scroll-mt-4">
          <ProjectConnectorToggles
            enabledConnectors={enabledConnectors}
            onChange={setEnabledConnectors}
          />
        </div>

        {/* Workbench privacy toggle (spec 2026-07-23 §6). Per-project — the
            per-project Workbench lens is unaffected; this only controls
            whether the project's flagged entries are aggregated into the
            GLOBAL Workbench view. */}
        <div data-settings-section="workbench" className="scroll-mt-4">
          <label className="block text-sm font-medium text-primary mb-1.5">
            {t('settings.workbench.label')}
          </label>
          <label className="flex items-center gap-2 cursor-pointer max-md:min-h-[44px]">
            <input
              type="checkbox"
              checked={workbenchExcludeGlobal}
              onChange={(e) => setWorkbenchExcludeGlobal(e.target.checked)}
              className="rounded border-border accent-accent"
            />
            <span className="text-sm text-primary">
              {t('settings.workbench.excludeGlobal.label')}
            </span>
          </label>
          <p className="text-xs text-secondary mt-1.5">
            {t('settings.workbench.excludeGlobal.hint')}
          </p>
        </div>

        {/* Save */}
        <div className="flex items-center gap-3 pt-2">
          <button
            type="submit"
            className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 max-md:w-full max-md:min-h-[44px]"
          >
            {t('settings.save')}
          </button>
          {saved && (
            <span className="text-sm text-success">{t('settings.saved')}</span>
          )}
        </div>
      </form>

      {/* Danger zone */}
      {!project.is_scratch && (
        <div data-settings-section="danger" className="mt-12 border border-error/30 rounded-lg p-6 scroll-mt-4">
          <h3 className="text-sm font-semibold text-error mb-2">{t('settings.danger.title')}</h3>
          <p className="text-sm text-secondary mb-4">
            {t('settings.danger.body')}
          </p>
          {!confirmingDelete ? (
            <button
              onClick={() => setConfirmingDelete(true)}
              className="text-sm font-medium text-error border border-error/40 rounded-lg px-4 py-2 hover:bg-error/5 transition-all duration-150 max-md:w-full max-md:min-h-[44px]"
            >
              {t('settings.danger.delete')}
            </button>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={handleDelete}
                className="text-sm font-medium text-white bg-error rounded-lg px-4 py-2 hover:bg-error/90 transition-all duration-150 max-md:min-h-[44px]"
              >
                {t('settings.danger.confirmDelete')}
              </button>
              <button
                onClick={() => setConfirmingDelete(false)}
                className="text-sm font-medium text-secondary border border-border rounded-lg px-4 py-2 hover:bg-sidebar transition-all duration-150 max-md:min-h-[44px]"
              >
                {t('settings.danger.cancel')}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
    </div>
    </div>
  );
}
