// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect } from 'react';
import type { FallbackModelEntry, ProviderRegistry } from '../types';
import LLMProviderSettings from './LLMProviderSettings';
import FallbackModelsEditor from './FallbackModelsEditor';
import CredentialStore from './CredentialStore';
import PairPhone from './PairPhone';

interface GlobalSettingsProps {
  onBack: () => void;
}

const API_BASE = import.meta.env.VITE_API_BASE || '';

export default function GlobalSettings({ onBack }: GlobalSettingsProps) {
  const [userPreferences, setUserPreferences] = useState('');
  const [scratchWorkspace, setScratchWorkspace] = useState('');
  const [fallbackModels, setFallbackModels] = useState<FallbackModelEntry[]>([]);
  const [providers, setProviders] = useState<ProviderRegistry>({});
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/api/v2/settings`)
      .then(r => r.json())
      .then(data => {
        setUserPreferences(data.user_preferences_content || '');
        setScratchWorkspace(data.scratch_workspace || '');
        setFallbackModels(data.llm?.fallback_models || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/v2/providers`)
      .then(r => r.json())
      .then(data => setProviders(data))
      .catch(() => {});
  }, []);

  async function handleSave() {
    await fetch(`${API_BASE}/api/v2/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_preferences_content: userPreferences,
        scratch_workspace: scratchWorkspace || undefined,
        llm_fallback_models: fallbackModels,
      }),
    });
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      <div className="max-w-[720px] mx-auto py-10 px-6 max-md:px-4">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-xl font-semibold text-primary">Global Settings</h1>
          <button
            onClick={onBack}
            className="text-sm text-secondary hover:text-primary transition-all duration-150"
          >
            Back
          </button>
        </div>

        <p className="text-sm text-secondary mb-6">
          Configure default LLM settings. These are used as fallback when a project
          does not specify its own API key, model, or base URL.
        </p>

        <LLMProviderSettings mode="global" />

        {/* Fallback Models */}
        <div className="mt-6">
          <FallbackModelsEditor
            models={fallbackModels}
            onChange={setFallbackModels}
            providers={providers}
          />
        </div>

        {/* About You */}
        <div className="mt-8 pt-6 border-t border-border space-y-4">
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              About You
            </label>
            <p className="text-xs text-secondary mb-2">
              Preferences that apply across all projects. E.g., &quot;I&apos;m a senior Python developer&quot;, &quot;I prefer concise responses&quot;
            </p>
            <textarea
              rows={4}
              value={userPreferences}
              onChange={(e) => setUserPreferences(e.target.value)}
              placeholder="Tell your agents about yourself..."
              disabled={loading}
              className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
            />
          </div>

          {/* Quick Tasks Workspace */}
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              Quick Tasks Workspace
            </label>
            <p className="text-xs text-secondary mb-2">
              Where Quick Tasks stores files
            </p>
            <input
              type="text"
              value={scratchWorkspace}
              onChange={(e) => setScratchWorkspace(e.target.value)}
              placeholder="Default: ~/.agent-os/scratch/"
              disabled={loading}
              className="w-full text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 disabled:opacity-50"
            />
          </div>

          {/* Save button */}
          <div className="flex items-center gap-3 pt-2">
            <button
              onClick={handleSave}
              disabled={loading}
              className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
            >
              Save
            </button>
            {saved && (
              <span className="text-sm text-success">Saved</span>
            )}
          </div>
        </div>

        {/* Credentials */}
        <div className="mt-8 pt-6 border-t border-border space-y-3">
          <div>
            <label className="block text-sm font-medium text-primary mb-1">
              Saved Credentials
            </label>
            <p className="text-xs text-secondary mb-3">
              Website passwords stored in your system keychain. Agents always ask permission before using them.
            </p>
          </div>
          <CredentialStore />
        </div>

        {/* Phone Pairing section */}
        <div className="mt-10 pt-8 border-t border-border">
          <PairPhone />
        </div>
      </div>
    </div>
  );
}
