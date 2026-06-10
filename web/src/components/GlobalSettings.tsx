// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect } from 'react';
import type { FallbackModelEntry, ProviderRegistry } from '../types';
import LLMProviderSettings from './LLMProviderSettings';
import FallbackModelsEditor from './FallbackModelsEditor';
import CredentialStore from './CredentialStore';
import PairPhone from './PairPhone';
import SubAgentSettings from './SubAgentSettings';
import { useLocale } from '../i18n/LocaleContext';
import { LOCALES } from '../i18n/locales';
import { useT } from '../i18n/useT';

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
  const { locale, setLocale } = useLocale();
  const t = useT();

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
          <h1 className="text-xl font-semibold text-primary">{t('global.title')}</h1>
          <button
            onClick={onBack}
            className="text-sm text-secondary hover:text-primary transition-all duration-150"
          >
            {t('global.back')}
          </button>
        </div>

        {/* Language */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-primary mb-1.5">
            {t('global.language')}
          </label>
          <select
            value={locale}
            onChange={(e) => setLocale(e.target.value as typeof locale)}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
          >
            {LOCALES.map((l) => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
        </div>

        <p className="text-sm text-secondary mb-6">
          {t('global.intro')}
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
              {t('global.aboutYou.label')}
            </label>
            <p className="text-xs text-secondary mb-2">
              {t('global.aboutYou.hint')}
            </p>
            <textarea
              rows={4}
              value={userPreferences}
              onChange={(e) => setUserPreferences(e.target.value)}
              placeholder={t('global.aboutYou.placeholder')}
              disabled={loading}
              className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
            />
          </div>

          {/* Quick Tasks Workspace */}
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              {t('global.scratch.label')}
            </label>
            <p className="text-xs text-secondary mb-2">
              {t('global.scratch.hint')}
            </p>
            <input
              type="text"
              value={scratchWorkspace}
              onChange={(e) => setScratchWorkspace(e.target.value)}
              placeholder={t('global.scratch.placeholder')}
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
              {t('settings.save')}
            </button>
            {saved && (
              <span className="text-sm text-success">{t('settings.saved')}</span>
            )}
          </div>
        </div>

        {/* Credentials */}
        <div className="mt-8 pt-6 border-t border-border space-y-3">
          <div>
            <label className="block text-sm font-medium text-primary mb-1">
              {t('global.credentials.label')}
            </label>
            <p className="text-xs text-secondary mb-3">
              {t('global.credentials.hint')}
            </p>
          </div>
          <CredentialStore />
        </div>

        {/* Sub-agents */}
        <div className="mt-10 pt-8 border-t border-border">
          <h2 className="text-base font-semibold text-primary mb-3">{t('global.subAgents.heading')}</h2>
          <SubAgentSettings />
        </div>

        {/* Phone Pairing section */}
        <div className="mt-10 pt-8 border-t border-border">
          <PairPhone />
        </div>
      </div>
    </div>
  );
}
