// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import type {
  Autonomy,
  ProjectCreateRequest,
} from '../types';
import LLMProviderSettings from './LLMProviderSettings';
import FolderPickerModal from './FolderPickerModal';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

interface CreateProjectProps {
  onSubmit: (data: ProjectCreateRequest) => void;
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

export default function CreateProject({
  onSubmit,
  onCancel,
}: CreateProjectProps) {
  const t = useT();
  const [name, setName] = useState('');
  const [agentName, setAgentName] = useState('');
  const [workspace, setWorkspace] = useState('');
  const [instructions, setInstructions] = useState('');
  const [autonomy, setAutonomy] = useState<Autonomy>('hands_off');
  const [budgetLimit, setBudgetLimit] = useState('');
  const [errors, setErrors] = useState<FormErrors>({});
  const [pickerOpen, setPickerOpen] = useState(false);

  function validate(): FormErrors {
    const e: FormErrors = {};
    if (!name.trim()) e.name = t('createProject.name.required');
    if (!workspace.trim()) e.workspace = t('createProject.workspace.required');
    else if (!isAbsolutePath(workspace))
      e.workspace = t('createProject.workspace.absolute');
    return e;
  }

  function handleSubmit(ev: React.FormEvent) {
    ev.preventDefault();
    const validationErrors = validate();
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) return;

    onSubmit({
      name: name.trim(),
      workspace: workspace.trim(),
      instructions: instructions.trim() || undefined,
      model: '',
      api_key: '',
      autonomy,
      agent_name: agentName.trim() || undefined,
      budget_limit_usd: budgetLimit ? parseFloat(budgetLimit) : undefined,
    });
  }

  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
    <div className="max-w-[720px] mx-auto py-10 px-6 max-md:px-4">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-xl font-semibold text-primary">{t('createProject.title')}</h1>
        <button
          onClick={onCancel}
          className="text-sm text-secondary hover:text-primary transition-all duration-150"
        >
          {t('createProject.cancel')}
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Project Name */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            {t('createProject.name.label')}
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setErrors((prev) => ({ ...prev, name: undefined }));
            }}
            placeholder={t('createProject.name.placeholder')}
            autoFocus
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
          />
          {errors.name && (
            <p className="text-xs text-error mt-1">{errors.name}</p>
          )}
          <p className="text-xs text-secondary mt-1">
            {t('createProject.name.hint')}
          </p>
        </div>

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

        {/* Workspace */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            {t('createProject.workspace.label')}
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={workspace}
              onChange={(e) => {
                setWorkspace(e.target.value);
                setErrors((prev) => ({ ...prev, workspace: undefined }));
              }}
              placeholder={t('createProject.workspace.placeholder')}
              className="flex-1 text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
            />
            <button
              type="button"
              onClick={() => setPickerOpen(true)}
              className="text-sm font-medium text-accent border border-accent/30 rounded-lg px-4 py-2 hover:bg-accent/5 transition-all duration-150 shrink-0 max-md:min-h-[44px]"
            >
              {t('createProject.browse')}
            </button>
          </div>
          {errors.workspace && (
            <p className="text-xs text-error mt-1">{errors.workspace}</p>
          )}
          <FolderPickerModal
            open={pickerOpen}
            onSelect={(path) => {
              setWorkspace(path);
              setErrors((prev) => ({ ...prev, workspace: undefined }));
              setPickerOpen(false);
            }}
            onClose={() => setPickerOpen(false)}
          />
        </div>

        {/* Instructions */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            {t('createProject.instructions.label')}
          </label>
          <textarea
            rows={6}
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            placeholder={t('createProject.instructions.placeholder')}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y"
          />
        </div>

        {/* LLM Provider (wizard mode - shows global config status) */}
        <LLMProviderSettings mode="wizard" />

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

        {/* Submit */}
        <div className="pt-2">
          <button
            type="submit"
            className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 max-md:w-full max-md:min-h-[44px]"
          >
            {t('createProject.deploy')}
          </button>
        </div>
      </form>
    </div>
    </div>
  );
}
