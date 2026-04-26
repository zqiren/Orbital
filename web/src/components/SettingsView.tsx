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
}

const AUTONOMY_OPTIONS: {
  value: Autonomy;
  title: string;
  description: string;
}[] = [
  {
    value: 'hands_off',
    title: 'Hands-off',
    description: 'Agent works freely. Asks only for new access.',
  },
  {
    value: 'check_in',
    title: 'Check-in',
    description: 'Pauses for shell commands and file writes.',
  },
  {
    value: 'supervised',
    title: 'Supervised',
    description: 'Pauses for most actions. Safe mode.',
  },
];

export default function SettingsView({
  project,
  onSave,
  onDelete,
}: SettingsViewProps) {
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

  // Skills state
  const [skills, setSkills] = useState<SkillMeta[]>([]);
  const [skillError, setSkillError] = useState('');
  const [skillSuccess, setSkillSuccess] = useState('');
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Budget state
  const [budgetLimit, setBudgetLimit] = useState<string>(
    project.budget_limit_usd != null ? String(project.budget_limit_usd) : '',
  );
  const [budgetSpent, setBudgetSpent] = useState<number>(project.budget_spent_usd ?? 0);

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
        setBudgetSpent(detail.budget_spent_usd ?? 0);
        setBudgetLimit(detail.budget_limit_usd != null ? String(detail.budget_limit_usd) : '');
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

  async function handleDeleteSkill(dirName: string) {
    setSkillError('');
    try {
      await api(`/api/v2/projects/${encodeURIComponent(pid)}/skills/${encodeURIComponent(dirName)}`, {
        method: 'DELETE',
      });
      await fetchSkills();
    } catch (e) {
      setSkillError(e instanceof ApiError ? e.detail : 'Failed to delete skill');
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
        const body = await resp.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(body.detail || `HTTP ${resp.status}`);
      }
      setSkillSuccess('Skill added');
      setTimeout(() => setSkillSuccess(''), 2000);
      await fetchSkills();
    } catch (e) {
      setSkillError(e instanceof Error ? e.message : 'Upload failed');
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
    };
    if (llm.api_key) {
      data.api_key = llm.api_key;
    }
    onSave(data);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  async function handleResetSpend() {
    if (!confirm('Reset accumulated spend to $0?')) return;
    try {
      await api(`/api/v2/projects/${encodeURIComponent(pid)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ budget_spent_usd: 0 }),
      });
      setBudgetSpent(0);
    } catch {
      // silently ignore
    }
  }

  function handleDelete() {
    onDelete();
  }

  return (
    <div className="h-full overflow-y-auto">
    <div className="max-w-[720px] mx-auto py-8 px-6 max-md:px-4">
      <form onSubmit={handleSave} className="space-y-6">
        {/* Agent Name */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            Agent Name
          </label>
          <input
            type="text"
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
            placeholder="Display name for this agent"
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
          />
        </div>

        {/* Project Goals */}
        {!project.is_scratch && (
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              Project Goals
            </label>
            <textarea
              rows={6}
              value={projectGoals}
              onChange={(e) => setProjectGoals(e.target.value)}
              disabled={loadingDetail}
              placeholder={loadingDetail ? 'Loading…' : 'Define the agent\'s mission, scope, and rules...'}
              className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
            />
          </div>
        )}

        {/* Project Instructions */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            Project Instructions
          </label>
          <p className="text-xs text-secondary mb-2">
            Persistent instructions the agent always follows. E.g., &quot;always write tests&quot;, &quot;use PostgreSQL&quot;
          </p>
          <textarea
            rows={4}
            value={standingRules}
            onChange={(e) => setStandingRules(e.target.value)}
            disabled={loadingDetail}
            placeholder={loadingDetail ? 'Loading…' : 'One rule per line...'}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
          />
        </div>

        {/* Skills */}
        {!project.is_scratch && (
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              Skills
            </label>
            <p className="text-xs text-secondary mb-2">
              Operational patterns the agent follows. Skills are read before each task.
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
                      title={`Delete ${s.name}`}
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
                No skills installed. Add a SKILL.md file or zip below.
              </p>
            )}

            <label className="flex items-center justify-center border border-dashed border-border rounded-lg px-3 py-3 cursor-pointer hover:border-accent/50 hover:bg-accent/5 transition-all duration-150 max-md:min-h-[44px]">
              <span className="text-sm text-secondary">
                {uploading ? 'Uploading...' : 'Add Skill (.md or .zip)'}
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
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              Notifications
            </label>
            <p className="text-xs text-secondary mb-2">
              Push notifications for this project when the app is backgrounded.
            </p>
            <div className="space-y-2">
              {([
                { key: 'task_completed', label: 'Task completed' },
                { key: 'errors', label: 'Errors & crashes' },
                { key: 'agent_messages', label: 'Agent messages' },
                { key: 'trigger_started', label: 'Scheduled run started' },
              ] as const).map(({ key, label }) => (
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
                  <span className="text-sm text-primary">{label}</span>
                </label>
              ))}
            </div>
            <p className="text-xs text-secondary/60 mt-2 italic">
              Approvals always notify — agents are waiting on you.
            </p>
          </div>
        )}

        {/* LLM Provider (collapsible, from shared component) */}
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

        {/* Fallback Models */}
        <FallbackModelsEditor
          models={fallbackModels}
          onChange={handleFallbackChange}
          providers={providers}
        />

        {/* Autonomy Level */}
        <div>
          <label className="block text-sm font-medium text-primary mb-2">
            Autonomy Level
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
                  {opt.title}
                </span>
                <span className="text-xs text-secondary mt-1 block">
                  {opt.description}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Budget */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            Budget
          </label>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-secondary mb-1">
                Budget Limit (USD)
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                value={budgetLimit}
                onChange={(e) => setBudgetLimit(e.target.value)}
                placeholder="e.g. 5.00"
                className="w-48 text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
              />
              <p className="text-xs text-secondary mt-1">Leave empty for no limit</p>
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              {(() => {
                const limit = budgetLimit ? parseFloat(budgetLimit) : null;
                const overBudget = limit != null && budgetSpent >= limit;
                const nearBudget = limit != null && !overBudget && budgetSpent >= limit * 0.8;
                const colorClass = overBudget ? 'text-error' : nearBudget ? 'text-warning' : 'text-primary';
                return (
                  <>
                    <span className="text-xs text-secondary">
                      Spent: <span className={`font-medium ${colorClass}`}>${budgetSpent.toFixed(2)}</span>
                      {limit != null && (
                        <span className="text-secondary/60"> / ${limit.toFixed(2)}</span>
                      )}
                    </span>
                    {overBudget && (
                      <span className="text-xs text-error font-medium">Over budget</span>
                    )}
                    {nearBudget && (
                      <span className="text-xs text-warning font-medium">Nearing limit</span>
                    )}
                  </>
                );
              })()}
              {budgetSpent > 0 && (
                <button
                  type="button"
                  onClick={handleResetSpend}
                  className="text-xs text-secondary hover:text-primary underline transition-colors duration-150"
                >
                  Reset
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Save */}
        <div className="flex items-center gap-3 pt-2">
          <button
            type="submit"
            className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 max-md:w-full max-md:min-h-[44px]"
          >
            Save
          </button>
          {saved && (
            <span className="text-sm text-success">Saved</span>
          )}
        </div>
      </form>

      {/* Danger zone */}
      {!project.is_scratch && (
        <div className="mt-12 border border-error/30 rounded-lg p-6">
          <h3 className="text-sm font-semibold text-error mb-2">Danger Zone</h3>
          <p className="text-sm text-secondary mb-4">
            Delete this project? Orbital data (sessions, screenshots, logs)
            will be removed. Files you saved in your workspace folder will be
            kept.
          </p>
          {!confirmingDelete ? (
            <button
              onClick={() => setConfirmingDelete(true)}
              className="text-sm font-medium text-error border border-error/40 rounded-lg px-4 py-2 hover:bg-error/5 transition-all duration-150 max-md:w-full max-md:min-h-[44px]"
            >
              Delete Project
            </button>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={handleDelete}
                className="text-sm font-medium text-white bg-error rounded-lg px-4 py-2 hover:bg-error/90 transition-all duration-150 max-md:min-h-[44px]"
              >
                Confirm Delete
              </button>
              <button
                onClick={() => setConfirmingDelete(false)}
                className="text-sm font-medium text-secondary border border-border rounded-lg px-4 py-2 hover:bg-sidebar transition-all duration-150 max-md:min-h-[44px]"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      )}
    </div>
    </div>
  );
}
