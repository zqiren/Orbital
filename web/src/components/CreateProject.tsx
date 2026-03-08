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

function isAbsolutePath(path: string): boolean {
  return /^(?:[A-Za-z]:\\|\/|~\/)/.test(path.trim());
}

export default function CreateProject({
  onSubmit,
  onCancel,
}: CreateProjectProps) {
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
    if (!name.trim()) e.name = 'Project name is required.';
    if (!workspace.trim()) e.workspace = 'Workspace path is required.';
    else if (!isAbsolutePath(workspace))
      e.workspace = 'Workspace must be an absolute path.';
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
        <h1 className="text-xl font-semibold text-primary">New Project</h1>
        <button
          onClick={onCancel}
          className="text-sm text-secondary hover:text-primary transition-all duration-150"
        >
          Cancel
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Project Name */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            Project Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g., Refactor Auth Module"
            autoFocus
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
          />
          {errors.name && (
            <p className="text-xs text-error mt-1">{errors.name}</p>
          )}
          <p className="text-xs text-secondary mt-1">
            Use letters, numbers, hyphens, and underscores only.
          </p>
        </div>

        {/* Agent Name */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            Agent Name <span className="text-secondary font-normal">(optional)</span>
          </label>
          <input
            type="text"
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
            placeholder="e.g., CodeBot"
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
          />
          <p className="text-xs text-secondary mt-1">
            A friendly name for your agent. Defaults to project name.
          </p>
        </div>

        {/* Workspace */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            Workspace
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={workspace}
              onChange={(e) => setWorkspace(e.target.value)}
              placeholder="Select a folder or type a path..."
              className="flex-1 text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
            />
            <button
              type="button"
              onClick={() => setPickerOpen(true)}
              className="text-sm font-medium text-accent border border-accent/30 rounded-lg px-4 py-2 hover:bg-accent/5 transition-all duration-150 shrink-0 max-md:min-h-[44px]"
            >
              Browse
            </button>
          </div>
          {errors.workspace && (
            <p className="text-xs text-error mt-1">{errors.workspace}</p>
          )}
          <FolderPickerModal
            open={pickerOpen}
            onSelect={(path) => { setWorkspace(path); setPickerOpen(false); }}
            onClose={() => setPickerOpen(false)}
          />
        </div>

        {/* Instructions */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            Instructions
          </label>
          <textarea
            rows={6}
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            placeholder="Describe what the agent should do..."
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y"
          />
        </div>

        {/* LLM Provider (wizard mode - shows global config status) */}
        <LLMProviderSettings mode="wizard" />

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

        {/* Budget Limit */}
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">
            Budget Limit (USD) <span className="text-secondary font-normal">(optional)</span>
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
          <p className="text-xs text-secondary mt-1">
            Leave empty for no limit. Agent pauses when spend reaches the limit.
          </p>
        </div>

        {/* Submit */}
        <div className="pt-2">
          <button
            type="submit"
            className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 max-md:w-full max-md:min-h-[44px]"
          >
            Deploy Agent
          </button>
        </div>
      </form>
    </div>
    </div>
  );
}
