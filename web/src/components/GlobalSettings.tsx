// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useRef } from 'react';
import type { FallbackModelEntry, ProviderRegistry } from '../types';
import LLMProviderSettings from './LLMProviderSettings';
import FallbackModelsEditor from './FallbackModelsEditor';
import CredentialStore from './CredentialStore';
import BrowserSignInCard from './BrowserSignInCard';
import PairPhone from './PairPhone';
import SubAgentSettings from './SubAgentSettings';
import ConnectorSettings from './ConnectorSettings';
import SettingsRail, { type SettingsRailSection } from './SettingsRail';
import { useLocale } from '../i18n/LocaleContext';
import { LOCALES } from '../i18n/locales';
import { useT } from '../i18n/useT';

interface GlobalSettingsProps {
  onBack: () => void;
}

const API_BASE = import.meta.env.VITE_API_BASE || '';

/**
 * Index-rail entries for the global settings document (spec 011 §0.8).
 * 'connectors' is reserved — its section lands in a later wave; the rail
 * renders an entry only once a matching data-settings-section element exists.
 */
export const GLOBAL_SETTINGS_SECTIONS: SettingsRailSection[] = [
  { id: 'language', labelKey: 'global.language' },
  { id: 'llm', labelKey: 'llm.global.heading' },
  { id: 'fallback-models', labelKey: 'fallback.heading' },
  { id: 'about-you', labelKey: 'global.aboutYou.label' },
  { id: 'quick-tasks-workspace', labelKey: 'global.scratch.label' },
  { id: 'credentials', labelKey: 'global.credentials.label' },
  { id: 'browser-sign-in', labelKey: 'global.browserSignIn.title' },
  { id: 'connectors', labelKey: 'settingsRail.connectors' },
  { id: 'sub-agents', labelKey: 'global.subAgents.heading' },
  { id: 'phone-pairing', labelKey: 'settingsRail.phone' },
];

export default function GlobalSettings({ onBack }: GlobalSettingsProps) {
  const [userPreferences, setUserPreferences] = useState('');
  const [scratchWorkspace, setScratchWorkspace] = useState('');
  const [fallbackModels, setFallbackModels] = useState<FallbackModelEntry[]>([]);
  const [providers, setProviders] = useState<ProviderRegistry>({});
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);
  const { locale, setLocale } = useLocale();
  const t = useT();
  // Scroll container ref — the SettingsRail scopes section discovery,
  // scrollspy, and jump-scrolls to this element.
  const scrollContainerRef = useRef<HTMLDivElement>(null);

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
    <div ref={scrollContainerRef} className="flex-1 min-h-0 overflow-y-auto">
      {/* Index rail beside the single scrolling document (desktop) / jump
          menu above it (mobile). The forms themselves are untouched — one
          document, one Save. */}
      <div className="flex justify-center max-md:block">
        <SettingsRail
          sections={GLOBAL_SETTINGS_SECTIONS}
          containerRef={scrollContainerRef}
        />
      <div className="max-w-[720px] w-full min-w-0 py-10 px-6 max-md:px-4">
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
        <div data-settings-section="language" className="mb-6 scroll-mt-4">
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

        <div data-settings-section="llm" className="scroll-mt-4">
          <LLMProviderSettings mode="global" />
        </div>

        {/* Fallback Models */}
        <div data-settings-section="fallback-models" className="mt-6 scroll-mt-4">
          <FallbackModelsEditor
            models={fallbackModels}
            onChange={setFallbackModels}
            providers={providers}
          />
        </div>

        {/* About You */}
        <div className="mt-8 pt-6 border-t border-border space-y-4">
          <div data-settings-section="about-you" className="scroll-mt-4">
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
          <div data-settings-section="quick-tasks-workspace" className="scroll-mt-4">
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
        <div data-settings-section="credentials" className="mt-8 pt-6 border-t border-border space-y-3 scroll-mt-4">
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

        {/* Browser Sign-In */}
        <div data-settings-section="browser-sign-in" className="mt-8 pt-6 border-t border-border scroll-mt-4">
          <BrowserSignInCard />
        </div>

        {/* Connectors — global catalog + auth (spec 011 §0.2/§0.6, Task E1).
            Mounting this fills the reserved 'connectors' rail entry. */}
        <div data-settings-section="connectors" className="mt-8 pt-6 border-t border-border scroll-mt-4">
          <ConnectorSettings />
        </div>

        {/* Sub-agents */}
        <div data-settings-section="sub-agents" className="mt-10 pt-8 border-t border-border scroll-mt-4">
          <h2 className="text-base font-semibold text-primary mb-3">{t('global.subAgents.heading')}</h2>
          <SubAgentSettings />
        </div>

        {/* Phone Pairing section */}
        <div data-settings-section="phone-pairing" className="mt-10 pt-8 border-t border-border scroll-mt-4">
          <PairPhone />
        </div>
      </div>
      </div>
    </div>
  );
}
