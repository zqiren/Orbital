// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import { ChevronDown, ChevronRight, Loader2, X } from 'lucide-react';
import type {
  Autonomy,
  ProjectCreateRequest,
} from '../types';
import LLMProviderSettings from './LLMProviderSettings';
import FolderBrowserPanel from './FolderBrowserPanel';
import { ApiError } from '../config';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

interface CreateProjectProps {
  onSubmit: (data: ProjectCreateRequest) => Promise<void>;
  onCancel: () => void;
}

interface FormErrors {
  name?: string;
  workspace?: string;
}

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

function isAbsolutePath(path: string): boolean {
  return /^(?:[A-Za-z]:\\|\/|~\/)/.test(path.trim());
}

/** Last path segment, cross-platform (mirrors the server-side sanitizer
 * loosely — the server's sanitized value always wins on submit). */
function basename(path: string): string {
  const parts = path.trim().replace(/[\\/]+$/, '').split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || '';
}

export default function CreateProject({
  onSubmit,
  onCancel,
}: CreateProjectProps) {
  const t = useT();
  const [name, setName] = useState('');
  const [nameTouched, setNameTouched] = useState(false);
  const [agentName, setAgentName] = useState('');
  const [workspace, setWorkspace] = useState('');
  const [instructions, setInstructions] = useState('');
  const [autonomy, setAutonomy] = useState<Autonomy>('hands_off');
  const [budgetLimit, setBudgetLimit] = useState('');
  const [errors, setErrors] = useState<FormErrors>({});
  const [pickerExpanded, setPickerExpanded] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  /** Shared by both workspace-change paths (typing the path directly, or
   * picking/creating a folder in the embedded browser): re-derive the name
   * from the folder basename only while the user hasn't manually edited it. */
  function applyWorkspace(path: string) {
    setWorkspace(path);
    // Clear the name error too, not just workspace's: a stale 409
    // agent-name-collision message must not survive a folder re-selection
    // while the name is re-deriving to a (probably different) value.
    setErrors((prev) => ({ ...prev, workspace: undefined, name: undefined }));
    if (!nameTouched) {
      setName(basename(path));
    }
  }

  function handleWorkspaceInputChange(value: string) {
    applyWorkspace(value);
  }

  function handleWorkspaceSelect(path: string) {
    applyWorkspace(path);
    setPickerExpanded(false);
  }

  function handleNameChange(value: string) {
    setName(value);
    setNameTouched(true);
    setErrors((prev) => ({ ...prev, name: undefined }));
  }

  function validate(): FormErrors {
    const e: FormErrors = {};
    if (!name.trim()) e.name = t('createProject.name.required');
    if (!workspace.trim()) e.workspace = t('createProject.workspace.required');
    else if (!isAbsolutePath(workspace))
      e.workspace = t('createProject.workspace.absolute');
    return e;
  }

  async function handleSubmit(ev: React.FormEvent) {
    ev.preventDefault();
    if (submitting) return;
    const validationErrors = validate();
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) return;

    setSubmitError(null);
    setSubmitting(true);
    try {
      await onSubmit({
        name: name.trim(),
        workspace: workspace.trim(),
        instructions: instructions.trim() || undefined,
        autonomy,
        agent_name: agentName.trim() || undefined,
        budget_limit_usd: budgetLimit ? parseFloat(budgetLimit) : undefined,
      });
    } catch (err) {
      // agent_name collisions (likelier now that the name is often
      // auto-derived from a folder basename) surface inline on the name
      // field — never auto-suffixed, since the name is user-visible identity.
      if (err instanceof ApiError && err.status === 409) {
        setErrors((prev) => ({ ...prev, name: err.detail }));
      } else {
        setSubmitError(err instanceof ApiError ? err.detail : t('createProject.submitError'));
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
    >
      <div className="bg-background rounded-xl shadow-xl border border-border w-full max-w-[560px] max-h-[85vh] flex flex-col mx-4 animate-slide-up max-md:max-w-full max-md:max-h-full max-md:h-full max-md:mx-0 max-md:rounded-none">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border shrink-0">
          <h2 className="text-sm font-semibold text-primary">{t('createProject.title')}</h2>
          <button
            type="button"
            onClick={onCancel}
            className="text-secondary hover:text-primary transition-all duration-150 p-1 max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
          >
            <X size={16} />
          </button>
        </div>

        {/* Scrollable body — only this area scrolls when Advanced is expanded */}
        <form id="create-project-form" onSubmit={handleSubmit} className="flex-1 overflow-y-auto min-h-0 px-5 py-4 space-y-5">
          {/* Workspace (first — name derives from it) */}
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              {t('createProject.workspace.label')}
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={workspace}
                onChange={(e) => handleWorkspaceInputChange(e.target.value)}
                placeholder={t('createProject.workspace.placeholder')}
                className="flex-1 text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
              />
              <button
                type="button"
                onClick={() => setPickerExpanded((v) => !v)}
                className="text-sm font-medium text-accent border border-accent/30 rounded-lg px-4 py-2 hover:bg-accent/5 transition-all duration-150 shrink-0 max-md:min-h-[44px]"
              >
                {pickerExpanded ? t('createProject.workspace.hideBrowse') : t('createProject.browse')}
              </button>
            </div>
            {errors.workspace && (
              <p className="text-xs text-error mt-1">{errors.workspace}</p>
            )}
            {pickerExpanded && (
              <div className="mt-2 border border-border rounded-lg overflow-hidden">
                <FolderBrowserPanel compact onSelect={handleWorkspaceSelect} />
              </div>
            )}
          </div>

          {/* Project Name (pre-filled from workspace, still editable) */}
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              {t('createProject.name.label')}
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => handleNameChange(e.target.value)}
              placeholder={t('createProject.name.placeholder')}
              className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
            />
            {errors.name && (
              <p className="text-xs text-error mt-1">{errors.name}</p>
            )}
          </div>

          {/* LLM info: renders only the no-api-key warning; nothing otherwise */}
          <LLMProviderSettings mode="wizard" />

          {/* Advanced options (collapsed by default): Agent Name, Instructions, Autonomy, Budget */}
          <div>
            <button
              type="button"
              onClick={() => setAdvancedOpen((v) => !v)}
              className="flex items-center gap-1.5 text-sm font-medium text-secondary hover:text-primary transition-all duration-150"
            >
              {advancedOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              {t('createProject.advanced.label')}
            </button>

            {advancedOpen && (
              <div className="space-y-5 pt-4">
                {/* Agent Name */}
                <div>
                  <label className="block text-sm font-medium text-primary mb-1.5">
                    {t('createProject.agentName.label')} <span className="text-secondary font-normal">{t('createProject.agentName.optional')}</span>
                  </label>
                  <input
                    type="text"
                    value={agentName}
                    onChange={(e) => setAgentName(e.target.value)}
                    placeholder={t('createProject.agentName.placeholder')}
                    className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
                  />
                  <p className="text-xs text-secondary mt-1">
                    {t('createProject.agentName.hint')}
                  </p>
                </div>

                {/* Instructions */}
                <div>
                  <label className="block text-sm font-medium text-primary mb-1.5">
                    {t('createProject.instructions.label')}
                  </label>
                  <textarea
                    rows={5}
                    value={instructions}
                    onChange={(e) => setInstructions(e.target.value)}
                    placeholder={t('createProject.instructions.placeholder')}
                    className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y"
                  />
                </div>

                {/* Autonomy Level */}
                <div>
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

                {/* Budget Limit */}
                <div>
                  <label className="block text-sm font-medium text-primary mb-1.5">
                    {t('createProject.budget.label')} <span className="text-secondary font-normal">{t('createProject.agentName.optional')}</span>
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    value={budgetLimit}
                    onChange={(e) => setBudgetLimit(e.target.value)}
                    placeholder={t('createProject.budget.placeholder')}
                    className="w-48 text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
                  />
                  <p className="text-xs text-secondary mt-1">
                    {t('createProject.budget.hint')}
                  </p>
                </div>
              </div>
            )}
          </div>

          {submitError && (
            <p className="text-sm text-error">{submitError}</p>
          )}
        </form>

        {/* Footer */}
        <div className="border-t border-border px-5 py-3 shrink-0 flex items-center justify-end gap-2 max-md:flex-col-reverse">
          <button
            type="button"
            onClick={onCancel}
            className="text-sm text-secondary hover:text-primary transition-all duration-150 px-4 py-2 max-md:w-full max-md:min-h-[44px]"
          >
            {t('createProject.cancel')}
          </button>
          <button
            type="submit"
            form="create-project-form"
            disabled={submitting}
            className="inline-flex items-center justify-center gap-2 bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 max-md:w-full max-md:min-h-[44px]"
          >
            {submitting && <Loader2 size={14} className="animate-spin" />}
            {t('createProject.deploy')}
          </button>
        </div>
      </div>
    </div>
  );
}
