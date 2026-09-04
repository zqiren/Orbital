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
} from '../types';
import { api, BASE_URL, isRelayMode, ApiError } from '../config';
import CardList from './CardList';
import MigrationNoteBanner from './MigrationNoteBanner';
import FallbackModelsEditor from './FallbackModelsEditor';
import { useCredentialCards } from '../hooks/useCredentialCards';
import { type SubAgentMemoryEntry } from './SubAgentMemoryCard';
import { type InstalledSubAgent } from './SubAgentToggleList';
import SubAgentCard from './SubAgentCard';
import BudgetSection from './BudgetSection';
import ProjectConnectorToggles from './ProjectConnectorToggles';
import BetaBadge from './BetaBadge';
import { NetworkAccessSection } from './NetworkAccessSection';
import type { PendingDomainRequest } from '../types';
import SettingsRail, {
  scrollToSettingsSection,
  type SettingsRailSection,
} from './SettingsRail';
import SettingsSection, { SettingsGroup, LabelWithHint } from './SettingsSection';
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
 *
 * `groupKey` mirrors the `SettingsGroup` chapters in the document below, so
 * the rail reads as the same five chapters rather than fourteen flat peers.
 * The array order must stay in step with DOM order — the rail renders in DOM
 * order, and a group whose entries were interleaved would print twice.
 */
export const PROJECT_SETTINGS_SECTIONS: SettingsRailSection[] = [
  { id: 'agent-name', labelKey: 'createProject.agentName.label', groupKey: 'settings.group.project' },
  { id: 'project-goals', labelKey: 'settings.projectGoals.label', groupKey: 'settings.group.project' },
  { id: 'project-instructions', labelKey: 'settings.projectInstructions.label', groupKey: 'settings.group.project' },
  { id: 'sub-agents', labelKey: 'settings.subAgents.label', groupKey: 'settings.group.capabilities' },
  { id: 'skills', labelKey: 'settings.skills.label', groupKey: 'settings.group.capabilities' },
  { id: 'connectors', labelKey: 'settingsRail.connectors', groupKey: 'settings.group.capabilities' },
  { id: 'llm', labelKey: 'cards.project.heading', groupKey: 'settings.group.model' },
  { id: 'fallback-models', labelKey: 'fallback.heading', groupKey: 'settings.group.model' },
  { id: 'autonomy', labelKey: 'autonomy.level.label', groupKey: 'settings.group.limits' },
  { id: 'budget', labelKey: 'settings.budget.label', groupKey: 'settings.group.limits' },
  { id: 'network', labelKey: 'settings.network.label', groupKey: 'settings.group.limits' },
  { id: 'notifications', labelKey: 'settings.notifications.label', groupKey: 'settings.group.preferences' },
  { id: 'workbench', labelKey: 'settings.workbench.label', groupKey: 'settings.group.preferences' },
  // No groupKey: a one-entry chapter whose heading repeats the entry is noise.
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
  // The credential card this project runs on (spec 082). `null` follows the
  // global default and is a value in its own right, not "unset". The provider
  // REGISTRY is no longer fetched here: a card already resolves provider,
  // endpoint and sdk, so this screen has no provider list to render.
  const [cardId, setCardId] = useState<string | null>(project.card_id ?? null);
  // The project surface now renders the same CardList as Global Settings,
  // so it needs the same two mutation hooks (a delete or a test here
  // updates the one shared list).
  const { cards, defaultCardId, loading: cardsLoading, refresh: refreshCards,
          applyCard } = useCredentialCards();

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

  function handleFallbackChange(models: FallbackModelEntry[]) {
    setFallbackModels(models);
    fallbackModelsRef.current = models;
  }

  function handleSave(ev: React.FormEvent) {
    ev.preventDefault();
    const data: ProjectUpdateRequest = {
      agent_name: agentName,
      project_goals_content: projectGoals,
      user_directives_content: standingRules,
      // Always PRESENT, null included: null is the explicit "follow the
      // global default card", and the daemon applies the field whenever it is
      // in the body. An `|| undefined` here would make "back to default"
      // unsavable — the wart spec 072 left behind on the old key field.
      card_id: cardId,
      autonomy,
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
        above it (mobile). One document, one Save — no pagination.

        Anchored LEFT, not centred. The rail + 720px column is a fixed 896px
        unit; centring it put half the slack on each side, so at a 1733px
        window (1473px pane) there were 288px of dead space to the LEFT of the
        rail — a navigation column floating in the middle of nowhere, detached
        from the nav column it conceptually continues. Anchoring turns that
        slack into one right-hand page margin, which is what a margin is.
        `pl-6` + the rail's own `pl-6` puts the first rail label 48px off the
        pane edge. At the 1200x800 default window (`desktop/main.py:1002`) the
        pane is 940px and this is visually identical to the old centring —
        which is why a flat max-w bump was the wrong fix: it would have
        regressed the default size to fix the maximised one.

        `max-md:pl-0` because below md this container is `block`, where the
        padding would stack on the content's own `max-md:px-4` and make the
        mobile page asymmetric (40px left, 16px right). */}
    <div className="flex justify-start pl-6 max-md:pl-0 max-md:block">
    <SettingsRail
      sections={PROJECT_SETTINGS_SECTIONS}
      containerRef={scrollContainerRef}
    />
    <div className="max-w-[720px] w-full min-w-0 py-8 px-6 max-md:px-4">
      <form onSubmit={handleSave}>
        <SettingsGroup title={t('settings.group.project')}>
        {/* Agent Name */}
        <SettingsSection id="agent-name" title={t('createProject.agentName.label')}>
          <input
            type="text"
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
            placeholder={t('settings.agentName.placeholder')}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
          />
        </SettingsSection>

        {/* Project Goals */}
        {!project.is_scratch && (
          <SettingsSection id="project-goals" title={t('settings.projectGoals.label')}>
            <textarea
              rows={6}
              value={projectGoals}
              onChange={(e) => setProjectGoals(e.target.value)}
              disabled={loadingDetail}
              placeholder={loadingDetail ? t('settings.loading') : t('settings.projectGoals.placeholder')}
              className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
            />
          </SettingsSection>
        )}

        {/* Project Instructions */}
        <SettingsSection
          id="project-instructions"
          title={t('settings.projectInstructions.label')}
          description={t('settings.projectInstructions.hint')}
        >
          <textarea
            rows={4}
            value={standingRules}
            onChange={(e) => setStandingRules(e.target.value)}
            disabled={loadingDetail}
            placeholder={loadingDetail ? t('settings.loading') : t('settings.projectInstructions.placeholder')}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
          />
        </SettingsSection>
        </SettingsGroup>

        <SettingsGroup title={t('settings.group.capabilities')}>
        {/* Sub-Agents — merged per-agent card: enable toggle + auth badge +
            memory summary in the header, MEMORY.md editor as the expandable
            body. One card per installed sub-agent (a disabled one is dimmed
            but still expandable). */}
        {!project.is_scratch && (
          <SettingsSection
            id="sub-agents"
            title={t('settings.subAgents.label')}
            description={t('settings.subAgents.hint')}
          >
            <div className="mt-4">
              <LabelWithHint
                htmlFor="sub-agent-deployment-instructions"
                hint={t('settings.subAgents.deploymentInstructions.hint')}
              >
                {t('settings.subAgents.deploymentInstructions.label')}
              </LabelWithHint>
            </div>
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
          </SettingsSection>
        )}

        {/* Skills */}
        {!project.is_scratch && (
          <SettingsSection
            id="skills"
            title={t('settings.skills.label')}
            description={t('settings.skills.hint')}
          >
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
          </SettingsSection>
        )}

        {/* Connectors — per-project enablement (spec 011 §0.2/§0.6, Task E1).
            One switch per globally-connected connector, writing
            enabled_connectors through this form's save. Mounting this fills
            the reserved 'connectors' rail entry. */}
        <SettingsSection
          id="connectors"
          title={t('connectors.heading')}
          suffix={<BetaBadge />}
          description={t('connectors.project.hint')}
        >
          <ProjectConnectorToggles
            enabledConnectors={enabledConnectors}
            onChange={setEnabledConnectors}
          />
        </SettingsSection>
        </SettingsGroup>

        <SettingsGroup title={t('settings.group.model')}>
        {/* Spec 082: one card picker replaces the provider/model/endpoint/key
            override. A project stores a card id and nothing else, so there is
            no longer a subset of fields for the daemon to pair wrongly. */}
        <SettingsSection
          id="llm"
          title={t('cards.project.heading')}
          description={t('cards.project.hint')}
        >
          <MigrationNoteBanner note={project.migration_note} cards={cards} />
          {/* The same list Global Settings shows: same rows, same highlight,
              same three actions. Here the highlight means "this project runs
              on it"; there it means "this is the default". */}
          <CardList
            mode="project"
            cards={cards}
            defaultCardId={defaultCardId}
            value={cardId}
            onChange={setCardId}
            loading={cardsLoading}
            onRefresh={refreshCards}
            onCardUpdated={applyCard}
            showAdd
            data-testid="project-card-picker"
          />
        </SettingsSection>

        {/* Fallback Models (collapsible — heading lives in the component) */}
        <SettingsSection id="fallback-models">
          <FallbackModelsEditor
            models={fallbackModels}
            onChange={handleFallbackChange}
            cards={cards}
            defaultCardId={defaultCardId}
          />
        </SettingsSection>
        </SettingsGroup>

        <SettingsGroup title={t('settings.group.limits')}>
        {/* Autonomy Level */}
        <SettingsSection id="autonomy" title={t('autonomy.level.label')}>
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
        </SettingsSection>

        {/* Budget — limit-as-sentence, behavior cards, spend meter, breakdown.
            Spend is read EXCLUSIVELY from GET /cost (useCost), refreshed on the
            budget.spend_updated WS event. The data-settings-section tag anchors
            the header corner's deep-link (scrollToSection === 'budget') and
            the index rail. */}
        <SettingsSection id="budget" title={t('settings.budget.label')}>
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
          />
        </SettingsSection>

        {/* Network access — TOFU allowlist (Plan 2 Task 7). Approved domains +
            pending requests, writing approved_domains / pending_domain_requests
            through this form's save (Task 2's PUT route live-rebuilds the
            proxy rules server-side). */}
        <SettingsSection
          id="network"
          title={t('settings.network.label')}
          description={t('settings.network.hint')}
        >
          <NetworkAccessSection
            approvedDomains={approvedDomains}
            pendingRequests={pendingDomainRequests}
            onChange={({ approvedDomains: next, pendingRequests }) => {
              setApprovedDomains(next);
              setPendingDomainRequests(pendingRequests);
            }}
          />
        </SettingsSection>
        </SettingsGroup>

        <SettingsGroup title={t('settings.group.preferences')}>
        {/* Notification Preferences (remote mode only). Unlike every other
            section these save on change, not on Save — hence the standalone
            note under the list. */}
        {!project.is_scratch && isRelayMode && (
          <SettingsSection
            id="notifications"
            title={t('settings.notifications.label')}
            description={t('settings.notifications.hint')}
          >
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
          </SettingsSection>
        )}

        {/* Workbench privacy toggle (spec 2026-07-23 §6). Per-project — the
            per-project Workbench lens is unaffected; this only controls
            whether the project's flagged entries are aggregated into the
            GLOBAL Workbench view. */}
        <SettingsSection
          id="workbench"
          title={t('settings.workbench.label')}
          description={t('settings.workbench.excludeGlobal.hint')}
        >
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
        </SettingsSection>
        </SettingsGroup>

        {/* Save */}
        {/* No rule above Save: a rule means "a chapter starts here" now. */}
        <div className="flex items-center gap-3 mt-12">
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
          <h3 className="text-[15px] font-semibold leading-6 text-error mb-2">{t('settings.danger.title')}</h3>
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
