// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useRef } from 'react';
import { ArrowLeft } from 'lucide-react';
import type { FallbackModelEntry, ProviderRegistry } from '../types';
import BetaBadge from './BetaBadge';
import SettingsSection, { SettingsGroup } from './SettingsSection';
import LLMProviderSettings from './LLMProviderSettings';
import FallbackModelsEditor from './FallbackModelsEditor';
import CredentialStore from './CredentialStore';
import BrowserSignInCard from './BrowserSignInCard';
import PairPhone from './PairPhone';
import SubAgentSettings from './SubAgentSettings';
import ConnectorSettings from './ConnectorSettings';
import TelemetrySettings from './TelemetrySettings';
import AboutSection from './AboutSection';
import SettingsRail, { type SettingsRailSection } from './SettingsRail';
import { useLocale } from '../i18n/LocaleContext';
import { LOCALES } from '../i18n/locales';
import { useT } from '../i18n/useT';
import Select from './Select';

interface GlobalSettingsProps {
  onBack: () => void;
}

const API_BASE = import.meta.env.VITE_API_BASE || '';

/**
 * Index-rail entries for the global settings document (spec 011 §0.8).
 *
 * `groupKey` mirrors the `SettingsGroup` chapters in the document below, the
 * same way PROJECT_SETTINGS_SECTIONS does — the rail reads as four chapters
 * plus a trailing entry rather than twelve flat peers. The array order must
 * stay in step with DOM order: the rail renders in DOM order, and a chapter
 * whose entries were interleaved would print twice.
 *
 * Order changed with the chapter migration — About You / Quick Tasks Workspace
 * moved ABOVE the model sections so General is contiguous. They were
 * previously stranded between Fallback Models and Credentials, which is why
 * no chapter could be drawn around anything.
 */
export const GLOBAL_SETTINGS_SECTIONS: SettingsRailSection[] = [
  { id: 'language', labelKey: 'global.language', groupKey: 'settings.group.general' },
  { id: 'about-you', labelKey: 'global.aboutYou.label', groupKey: 'settings.group.general' },
  { id: 'quick-tasks-workspace', labelKey: 'global.scratch.label', groupKey: 'settings.group.general' },
  { id: 'llm', labelKey: 'llm.global.heading', groupKey: 'settings.group.model' },
  { id: 'fallback-models', labelKey: 'fallback.heading', groupKey: 'settings.group.model' },
  { id: 'credentials', labelKey: 'global.credentials.label', groupKey: 'settings.group.capabilities' },
  { id: 'browser-sign-in', labelKey: 'global.browserSignIn.title', groupKey: 'settings.group.capabilities' },
  { id: 'connectors', labelKey: 'settingsRail.connectors', groupKey: 'settings.group.capabilities' },
  { id: 'sub-agents', labelKey: 'global.subAgents.heading', groupKey: 'settings.group.capabilities' },
  { id: 'phone-pairing', labelKey: 'settingsRail.phone', groupKey: 'settings.group.device' },
  { id: 'privacy', labelKey: 'telemetry.heading', groupKey: 'settings.group.device' },
  // No groupKey: a one-entry chapter whose heading repeats the entry is noise
  // (same call as project settings' Danger Zone).
  { id: 'about', labelKey: 'update.about.heading' },
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
    <div className="flex flex-col flex-1 min-h-0 bg-background">
      {/* Header band — mirrors SettingsModalPage's exactly, because the two
          settings surfaces are one design and used to disagree about it. This
          page previously opened with an h1 + a text "Back" link floating
          INSIDE the scrolling column, while project settings had a ruled band
          with a back arrow, a title and a subtitle. Same track (rail-width
          spacer + the same 720px column, same inner padding), same
          justification, so the title lands on the left edge of the fields
          under it. */}
      <div className="pt-5 pb-4 border-b border-border">
        <div className="flex justify-start pl-6 max-md:pl-0 max-md:block">
          <div className="w-44 shrink-0 max-lg:hidden" aria-hidden="true" />
          <div className="flex flex-col gap-1 max-w-[720px] w-full min-w-0 px-6 max-md:px-4">
            <button
              onClick={onBack}
              data-testid="global-settings-back-button"
              className="flex items-center gap-1.5 text-sm text-secondary hover:text-primary transition-colors w-fit"
            >
              <ArrowLeft size={14} />
              {t('global.back')}
            </button>
            <h1 className="text-lg font-semibold text-primary mt-1" data-testid="global-settings-title">
              {t('global.title')}
            </h1>
            <p className="text-sm text-secondary">{t('global.subtitle')}</p>
          </div>
        </div>
      </div>

      {/* Scrollable body */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto min-h-0">
        {/* Index rail beside the single scrolling document (desktop) / jump
            menu above it (mobile).

            Left-anchored rather than centred, identically to SettingsView —
            see the full reasoning there. The two surfaces must move together:
            they are the same layout and the report named both. */}
        <div className="flex justify-start pl-6 max-md:pl-0 max-md:block">
          <SettingsRail
            sections={GLOBAL_SETTINGS_SECTIONS}
            containerRef={scrollContainerRef}
          />
          <div className="max-w-[720px] w-full min-w-0 py-8 px-6 max-md:px-4">

            <SettingsGroup title={t('settings.group.general')}>
              <SettingsSection id="language" title={t('global.language')}>
                <Select
                  value={locale}
                  onChange={(e) => setLocale(e.target.value as typeof locale)}
                  className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
                >
                  {LOCALES.map((l) => (
                    <option key={l.code} value={l.code}>{l.label}</option>
                  ))}
                </Select>
              </SettingsSection>

              <SettingsSection
                id="about-you"
                title={t('global.aboutYou.label')}
                description={t('global.aboutYou.hint')}
              >
                <textarea
                  rows={4}
                  value={userPreferences}
                  onChange={(e) => setUserPreferences(e.target.value)}
                  placeholder={t('global.aboutYou.placeholder')}
                  disabled={loading}
                  className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
                />
              </SettingsSection>

              <SettingsSection
                id="quick-tasks-workspace"
                title={t('global.scratch.label')}
                description={t('global.scratch.hint')}
              >
                <input
                  type="text"
                  value={scratchWorkspace}
                  onChange={(e) => setScratchWorkspace(e.target.value)}
                  placeholder={t('global.scratch.placeholder')}
                  disabled={loading}
                  className="w-full text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 disabled:opacity-50"
                />
              </SettingsSection>
            </SettingsGroup>

            <SettingsGroup title={t('settings.group.model')}>
              {/* Untitled, like project settings' LLM section: the child's own
                  disclosure button is the heading. `global.intro` rides INSIDE
                  the section rather than sitting between the chapter heading
                  and it — a bare <p> as SettingsGroup's first child would take
                  the `first:mt-0` reset that belongs to the first section. */}
              <SettingsSection id="llm">
                <p className="text-[13px] leading-relaxed text-secondary mb-4">
                  {t('global.intro')}
                </p>
                <LLMProviderSettings mode="global" />
              </SettingsSection>

              <SettingsSection id="fallback-models">
                <FallbackModelsEditor
                  models={fallbackModels}
                  onChange={setFallbackModels}
                  providers={providers}
                />
              </SettingsSection>
            </SettingsGroup>

            <SettingsGroup title={t('settings.group.capabilities')}>
              <SettingsSection
                id="credentials"
                title={t('global.credentials.label')}
                description={t('global.credentials.hint')}
              >
                <CredentialStore />
              </SettingsSection>

              <SettingsSection
                id="browser-sign-in"
                title={t('global.browserSignIn.title')}
              >
                <BrowserSignInCard />
              </SettingsSection>

              {/* Connectors — global catalog + auth (spec 011 §0.2/§0.6, Task
                  E1). Title/badge/hint hoisted out of ConnectorSettings onto
                  the section, exactly as project settings does it for its own
                  connectors section. */}
              <SettingsSection
                id="connectors"
                title={t('connectors.heading')}
                suffix={<BetaBadge />}
                description={t('connectors.global.hint')}
              >
                <ConnectorSettings />
              </SettingsSection>

              <SettingsSection
                id="sub-agents"
                title={t('global.subAgents.heading')}
                description={t('subAgentSettings.intro')}
              >
                <SubAgentSettings />
              </SettingsSection>
            </SettingsGroup>

            <SettingsGroup title={t('settings.group.device')}>
              <SettingsSection id="phone-pairing" title={t('settingsRail.phone')}>
                <PairPhone />
              </SettingsSection>

              {/* Telemetry toggle + verbatim payload viewer (spec 046 §6). */}
              <SettingsSection id="privacy" title={t('telemetry.heading')}>
                <TelemetrySettings />
              </SettingsSection>
            </SettingsGroup>

            {/* Ungrouped trailing section, mirroring project settings' Danger
                Zone: a one-entry chapter whose heading repeats the entry is
                noise. */}
            <SettingsSection id="about" title={t('update.about.heading')}>
              <AboutSection />
            </SettingsSection>

            {/* One document, one Save — at the BOTTOM, like project settings.
                It used to sit mid-document, directly under Quick Tasks
                Workspace, while ALSO persisting `llm_fallback_models` from the
                Fallback Models editor rendered above it. The file's own header
                comment already claimed "one document, one Save"; this makes
                that true. */}
            <div className="flex items-center gap-3 mt-12">
              <button
                onClick={handleSave}
                disabled={loading}
                data-testid="global-settings-save"
                className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 max-md:w-full max-md:min-h-[44px]"
              >
                {t('settings.save')}
              </button>
              {saved && (
                <span className="text-sm text-success">{t('settings.saved')}</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
