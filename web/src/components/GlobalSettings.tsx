// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useRef, useCallback } from 'react';
import type { FallbackModelEntry, ProviderRegistry } from '../types';
import { useCredentialCards } from '../hooks/useCredentialCards';
import { useAutosave } from '../hooks/useAutosave';
import { api } from '../config';
import BetaBadge from './BetaBadge';
import SettingsSection, { SettingsGroup, LabelWithHint } from './SettingsSection';
import PageToolbar from './PageToolbar';
import AppearanceSection from './AppearanceSection';
import AutosaveStatusPill from './AutosaveStatusPill';
import CredentialCards from './CredentialCards';
import FallbackModelsEditor from './FallbackModelsEditor';
import CredentialStore from './CredentialStore';
import BrowserSignInCard from './BrowserSignInCard';
import PairPhone from './PairPhone';
import SubAgentSettings from './SubAgentSettings';
import ConnectorSettings from './ConnectorSettings';
import TelemetrySettings from './TelemetrySettings';
import SendFeedback from './SendFeedback';
import AboutSection from './AboutSection';
import SettingsRail, { scrollToSettingsSection, type SettingsRailSection } from './SettingsRail';
import { useLocale } from '../i18n/LocaleContext';
import { LOCALES } from '../i18n/locales';
import { useT } from '../i18n/useT';
import Select from './Select';

interface GlobalSettingsProps {
  onBack: () => void;
  /** First-journey tour replay, surfaced in About (see AboutSection). */
  onTakeTour?: () => void;
  canTakeTour?: boolean;
  /**
   * A `data-settings-section` id to scroll into view on open (the composer
   * pin's "API key required" row sends 'llm'). Applied once the settings have
   * loaded, so the fields above it have their final height.
   */
  scrollToSection?: string;
}

const API_BASE = import.meta.env.VITE_API_BASE || '';

/** The fields this page edits directly; every other section saves itself. */
interface GlobalSettingsPatch {
  user_preferences_content: string;
  user_memory_content: string;
  user_memory_enabled: boolean;
  scratch_workspace: string;
  llm_fallback_models: FallbackModelEntry[];
}

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
  { id: 'appearance', labelKey: 'appearance.title', groupKey: 'settings.group.general' },
  { id: 'about-you', labelKey: 'global.aboutYou.label', groupKey: 'settings.group.general' },
  { id: 'user-memory', labelKey: 'global.userMemory.label', groupKey: 'settings.group.general' },
  { id: 'quick-tasks-workspace', labelKey: 'global.scratch.label', groupKey: 'settings.group.general' },
  { id: 'llm', labelKey: 'cards.heading', groupKey: 'settings.group.model' },
  { id: 'fallback-models', labelKey: 'fallback.heading', groupKey: 'settings.group.model' },
  { id: 'credentials', labelKey: 'global.credentials.label', groupKey: 'settings.group.capabilities' },
  { id: 'browser-sign-in', labelKey: 'global.browserSignIn.title', groupKey: 'settings.group.capabilities' },
  { id: 'connectors', labelKey: 'settingsRail.connectors', groupKey: 'settings.group.capabilities' },
  { id: 'sub-agents', labelKey: 'global.subAgents.heading', groupKey: 'settings.group.capabilities' },
  { id: 'phone-pairing', labelKey: 'settingsRail.phone', groupKey: 'settings.group.device' },
  { id: 'privacy', labelKey: 'telemetry.heading', groupKey: 'settings.group.device' },
  { id: 'feedback', labelKey: 'feedback.button', groupKey: 'settings.group.device' },
  // No groupKey: a one-entry chapter whose heading repeats the entry is noise
  // (same call as project settings' Danger Zone).
  { id: 'about', labelKey: 'update.about.heading' },
];

/**
 * The section an `open-global-settings` window event asks for
 * (`detail.section`), or undefined for a bare "open settings". App's listener
 * feeds it to `scrollToSection`. Anything that is not a non-empty string is
 * ignored, so an older dispatcher with no detail still just opens the page.
 */
export function requestedSettingsSection(event: Event): string | undefined {
  const detail = (event as CustomEvent<unknown>).detail;
  if (!detail || typeof detail !== 'object') return undefined;
  const section = (detail as { section?: unknown }).section;
  return typeof section === 'string' && section ? section : undefined;
}

export default function GlobalSettings({ onBack, onTakeTour, canTakeTour, scrollToSection }: GlobalSettingsProps) {
  const [userPreferences, setUserPreferences] = useState('');
  const [userMemory, setUserMemory] = useState('');
  const [userMemoryEnabled, setUserMemoryEnabled] = useState(true);
  const [scratchWorkspace, setScratchWorkspace] = useState('');
  const [fallbackModels, setFallbackModels] = useState<FallbackModelEntry[]>([]);
  const [providers, setProviders] = useState<ProviderRegistry>({});
  // The card list is shared by the Credentials section and the fallback chain
  // below it: both offer the same cards, so they must read the same list.
  const { cards, defaultCardId } = useCredentialCards();
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
        setUserMemory(data.user_memory_content || '');
        setUserMemoryEnabled(data.user_memory_enabled !== false);
        setScratchWorkspace(data.scratch_workspace || '');
        setFallbackModels(data.llm?.fallback_models || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Deep link: same shared section-scroll as project settings' anchor.
  useEffect(() => {
    if (loading || !scrollToSection || !scrollContainerRef.current) return;
    scrollToSettingsSection(scrollContainerRef.current, scrollToSection);
  }, [loading, scrollToSection]);

  useEffect(() => {
    fetch(`${API_BASE}/api/v2/providers`)
      .then(r => r.json())
      .then(data => setProviders(data))
      .catch(() => {});
  }, []);

  // Save-as-you-go, exactly like project settings. This page used to end in
  // one Save button below thirteen sections — a user who edited About You had
  // to scroll the whole document to keep the change, past sections that had
  // all been saving themselves the entire time. PUT /settings ignores fields
  // it is not sent, so each edit travels as its own partial patch.
  const savePatch = useCallback(async (patch: Partial<GlobalSettingsPatch>) => {
    await api('/api/v2/settings', { method: 'PUT', body: JSON.stringify(patch) });
  }, []);
  const autosave = useAutosave<GlobalSettingsPatch>(savePatch);
  const { saveNow, saveSoon } = autosave;

  return (
    <div className="flex flex-col flex-1 min-h-0 bg-background">
      {/* Same toolbar as project settings — the two settings surfaces are one
          design. No parent crumb: global settings belongs to no project. */}
      <PageToolbar
        title={t('global.title')}
        onBack={onBack}
        backTestId="global-settings-back-button"
        titleTestId="global-settings-title"
      />

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

            {/* Leaving a field sends what was typed without waiting out the
                pause — same wrapper as SettingsView. */}
            <div onBlur={() => void autosave.flush()}>
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

              <SettingsSection id="appearance" title={t('appearance.title')}>
                <AppearanceSection />
              </SettingsSection>

              <SettingsSection
                id="about-you"
                title={t('global.aboutYou.label')}
                description={t('global.aboutYou.hint')}
              >
                <textarea
                  rows={4}
                  value={userPreferences}
                  onChange={(e) => {
                    setUserPreferences(e.target.value);
                    saveSoon({ user_preferences_content: e.target.value });
                  }}
                  placeholder={t('global.aboutYou.placeholder')}
                  disabled={loading}
                  className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
                />
              </SettingsSection>

              {/* Spec 073 — user-level memory: agent-filed facts about the
                  user, injected into every project's prompt. The textarea is
                  the edit/prune surface (full overwrite on each save, like
                  About You — the two are separate files so neither save
                  clobbers the other); the toggle saves on the flip. */}
              <SettingsSection
                id="user-memory"
                title={t('global.userMemory.label')}
                description={t('global.userMemory.hint')}
              >
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div className="min-w-0">
                    <LabelWithHint hint={t('global.userMemory.toggle.hint')}>
                      {t('global.userMemory.toggle.label')}
                    </LabelWithHint>
                  </div>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={userMemoryEnabled}
                    aria-label={t('global.userMemory.toggle.label')}
                    disabled={loading}
                    onClick={() => {
                      setUserMemoryEnabled(!userMemoryEnabled);
                      saveNow({ user_memory_enabled: !userMemoryEnabled });
                    }}
                    data-testid="user-memory-toggle"
                    className={`shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-150 disabled:opacity-50 ${
                      userMemoryEnabled ? 'bg-accent' : 'bg-border'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-150 ${
                        userMemoryEnabled ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
                <textarea
                  rows={4}
                  value={userMemory}
                  onChange={(e) => {
                    setUserMemory(e.target.value);
                    saveSoon({ user_memory_content: e.target.value });
                  }}
                  placeholder={t('global.userMemory.placeholder')}
                  disabled={loading || !userMemoryEnabled}
                  data-testid="user-memory-textarea"
                  className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y disabled:opacity-50"
                />
                {/* Every manager reads this when it delegates; an empty
                    memory shows what a who-does-what line looks like. */}
                {!loading && !userMemory.trim() && (
                  <p className="mt-1.5 text-xs text-secondary" data-testid="user-memory-routing-hint">
                    {t('global.userMemory.routingHint')}
                  </p>
                )}
              </SettingsSection>

              <SettingsSection
                id="quick-tasks-workspace"
                title={t('global.scratch.label')}
                description={t('global.scratch.hint')}
              >
                <input
                  type="text"
                  value={scratchWorkspace}
                  onChange={(e) => {
                    setScratchWorkspace(e.target.value);
                    // An emptied field is not sent, as before: the daemon
                    // keeps its current workspace rather than taking ''.
                    if (e.target.value) saveSoon({ scratch_workspace: e.target.value });
                  }}
                  placeholder={t('global.scratch.placeholder')}
                  disabled={loading}
                  className="w-full text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 disabled:opacity-50"
                />
              </SettingsSection>
            </SettingsGroup>

            <SettingsGroup title={t('settings.group.model')}>
              {/* Spec 082: the single global provider form is gone — this is
                  the flat credential-card list. Unlike the old mount it takes
                  a real `title`, because a list of cards has no disclosure
                  button of its own to act as the heading. */}
              <SettingsSection
                id="llm"
                title={t('cards.heading')}
                description={t('cards.intro')}
              >
                <CredentialCards providers={providers} />
              </SettingsSection>

              <SettingsSection id="fallback-models">
                <FallbackModelsEditor
                  models={fallbackModels}
                  onChange={(next) => {
                    setFallbackModels(next);
                    saveNow({ llm_fallback_models: next });
                  }}
                  cards={cards}
                  defaultCardId={defaultCardId}
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
                description={t('global.browserSignIn.body')}
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
                description={
                  <>
                    {t('connectors.global.hint')}
                    <span className="block mt-1.5">{t('connectors.betaNote')}</span>
                  </>
                }
              >
                <ConnectorSettings />
              </SettingsSection>

              <SettingsSection
                id="sub-agents"
                title={t('global.subAgents.heading')}
                description={
                  <>
                    {t('subAgentSettings.intro')}
                    <span className="block mt-1.5">{t('subAgentSettings.installHint')}</span>
                    <span className="block mt-1.5">{t('subAgentSettings.credNote')}</span>
                    <span className="block mt-1.5">{t('subAgentSettings.loginNote')}</span>
                  </>
                }
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

              {/* Send feedback with diagnostics (spec 113 D): preview-then-upload
                  of a sanitised bundle; explicit action, independent of the
                  telemetry toggle above. */}
              <SettingsSection id="feedback" title={t('feedback.button')}>
                <SendFeedback />
              </SettingsSection>
            </SettingsGroup>

            {/* Ungrouped trailing section, mirroring project settings' Danger
                Zone: a one-entry chapter whose heading repeats the entry is
                noise. */}
            <SettingsSection id="about" title={t('update.about.heading')}>
              <AboutSection onTakeTour={onTakeTour} canTakeTour={canTakeTour} />
            </SettingsSection>

            </div>

            <p className="text-xs text-secondary/70 mt-12" data-testid="global-settings-autosave-hint">
              {t('settings.autosave.hint')}
            </p>
          </div>
        </div>
        {/* Sticky, so it must sit inside the scroll container. */}
        <AutosaveStatusPill
          status={autosave.status}
          error={autosave.error}
          onRetry={() => void autosave.retry()}
        />
      </div>
    </div>
  );
}
