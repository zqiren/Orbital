// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SetupWizard — first-run onboarding (Spec 011 §0.7, Task I1; spec 138).
 *
 * Four steps: `agents` → `projects` → `api_key` → `accounts`.
 *
 *  1. Agents — the coding agents found on this computer, all checked. An
 *     unchecked one goes into the global `sub_agents_hidden` denylist, which
 *     seeds new projects' disabled agents. Agents not found are chips that open
 *     the existing locate card. Continuing with no agent at all makes step 3
 *     compulsory (the Orbital agent is then the only thing that can work).
 *  2. Projects — the folders and groups found in the user's agent history
 *     (ImportProjectsStep). Skipped when the scan finds nothing to import.
 *     The scan starts the moment the wizard mounts (useOnboardingScan).
 *  3. The Orbital agent — the key form, skippable unless compulsory; skipped
 *     entirely when a key already exists.
 *  4. Connect Your Accounts — connectors and the agent browser sign-in.
 *
 * The `accounts` step merges the former `browser_warmup` step with featured
 * connector cards into one "Connect Your Accounts" screen with two labeled
 * groups (API connectors / agent browser sign-in) — two back-to-back Google
 * prompts on separate screens would otherwise confuse users.
 *
 * Keyless onboarding (2026-10-21): the key step is skippable. The key is
 * Orbital's own (its management agent); coding agents keep their own logins,
 * so a user without a key can still work through a pinned agent. Skip
 * persists `setup_api_key_skipped` so the first-run gate stops showing the
 * wizard (see `needsSetupWizard` in utils/firstRun).
 *
 * Finish hands App the project to land in (the most recently active imported
 * one) and, on a fresh install, marks the first-journey tour pending — the
 * import would otherwise make a new user look like an upgrade to the tour's
 * once-only decision.
 */

import { useState, useEffect, useCallback, useRef, type ReactNode } from 'react';
import { Loader2, Globe, Plug } from 'lucide-react';
import { api, ApiError } from '../config';
import { useCredentialCards } from '../hooks/useCredentialCards';
import type { Connector, ConnectorListResponse, Project } from '../types';
import LLMProviderSettings from './LLMProviderSettings';
import TokenCostSign from './TokenCostSign';
import ImportProjectsStep from './ImportProjectsStep';
import WizardAgentsStep from './WizardAgentsStep';
import { useSubAgentList } from '../hooks/useSubAgentList';
import { useOnboardingScan } from '../hooks/useOnboardingScan';
import { hasImportables, importIntro } from '../utils/onboarding';
import { sendTelemetry } from '../utils/telemetry';
import { isFirstRun } from '../utils/firstRun';
import { readTourState, writeTourState } from '../tour/tourLogic';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';
import BetaBadge from './BetaBadge';
import { useLocale } from '../i18n/LocaleContext';
import { LOCALES } from '../i18n/locales';
import { warmupUrlForLocale } from '../utils/warmupUrl';
import Select from './Select';

type WizardStep = 'agents' | 'projects' | 'api_key' | 'accounts';

const STEP_ORDER: WizardStep[] = ['agents', 'projects', 'api_key', 'accounts'];
const STEP_LABEL: Record<WizardStep, StringKey> = {
  agents: 'wizard.steps.agents',
  projects: 'wizard.steps.projects',
  api_key: 'wizard.steps.key',
  accounts: 'wizard.steps.accounts',
};
/** The `wizard_step_reached` enum (telemetry allowlist). */
const STEP_TELEMETRY: Record<WizardStep, string> = {
  agents: 'agents',
  projects: 'projects',
  api_key: 'key',
  accounts: 'accounts',
};

export interface WizardOutcome {
  /** The imported project to land in, on a new session. null = none. */
  landingProjectId: string | null;
}

interface SetupWizardProps {
  onComplete: (outcome?: WizardOutcome) => void;
}

/** Ambient language toggle (Spec 008 A2), reachable on every wizard step.
 *  Owns its own hooks rather than taking props so WizardCard can place it
 *  without either step having to thread locale state through. */
function LocaleToggle() {
  const t = useT();
  const { locale, setLocale } = useLocale();
  return (
    <div className="inline-flex shrink-0 items-center gap-1.5">
      <Globe className="w-3.5 h-3.5 text-secondary" aria-hidden="true" />
      <Select
        aria-label={t('global.language')}
        data-testid="wizard-locale-select"
        value={locale}
        onChange={(e) => setLocale(e.target.value as typeof locale)}
        className="text-xs bg-card border border-border rounded-lg px-2 py-1 text-primary focus:outline-none focus:border-accent transition-all duration-150"
      >
        {LOCALES.map((l) => (
          <option key={l.code} value={l.code}>{l.label}</option>
        ))}
      </Select>
    </div>
  );
}

/** "1 Agents · 2 Projects · 3 Orbital agent · 4 Accounts" */
function StepIndicator({ current }: { current: WizardStep }) {
  const t = useT();
  return (
    <ol className="mb-4 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-secondary" data-testid="wizard-steps">
      {STEP_ORDER.map((s, i) => (
        <li key={s} className="inline-flex items-center gap-2">
          {i > 0 && <span aria-hidden="true" className="text-secondary/50">·</span>}
          <span
            aria-current={s === current ? 'step' : undefined}
            className={s === current ? 'font-medium text-primary' : undefined}
          >
            <span className="font-mono">{i + 1}</span> {t(STEP_LABEL[s])}
          </span>
        </li>
      ))}
    </ol>
  );
}

/** Shared card chrome for every wizard step.
 *
 *  The language toggle lives here rather than above the card because the
 *  steps render separate cards — hoisting it out of them was what left it
 *  floating as loose chrome over the dialog. Declaring it once in the shared
 *  header keeps it inside the card and identical across steps.
 *
 *  bg-card, not bg-background: this is content that must lift off the canvas
 *  (see the surface ladder in index.css). Both onboarding screens were setting
 *  the same token as the page behind them, leaving a 1px border to do all the
 *  separation work — the exact flatness that ladder was introduced to fix. */
function WizardCard({
  step,
  title,
  intro,
  children,
}: {
  step: WizardStep;
  title: string;
  intro: string;
  children: ReactNode;
}) {
  return (
    <div className="surface-float-static p-8 max-md:p-5">
      <StepIndicator current={step} />
      <div className="flex items-start justify-between gap-4 mb-3">
        <h1 className="text-2xl font-semibold text-primary">{title}</h1>
        <LocaleToggle />
      </div>
      <p className="text-secondary text-sm leading-relaxed mb-6">{intro}</p>
      {children}
    </div>
  );
}

export default function SetupWizard({ onComplete }: SetupWizardProps) {
  const t = useT();
  const { locale } = useLocale();
  const [step, setStep] = useState<WizardStep>('agents');
  const [nextError, setNextError] = useState<string | null>(null);
  const [skippingKey, setSkippingKey] = useState(false);
  const [skipError, setSkipError] = useState<string | null>(null);
  // The provider the form must EDIT. undefined while loading and when there is
  // genuinely none, which is the true first run — there the form's create
  // branch is right, and the new provider becomes the default by being first.
  const { cards, defaultCardId, refresh: refreshCards } = useCredentialCards();
  const defaultCard = cards.find((c) => c.id === defaultCardId) ?? undefined;
  const [checkingKey, setCheckingKey] = useState(false);
  const saveRef = useRef<(() => Promise<boolean>) | null>(null);
  // Spec 160 §5.3: imported projects this key would read once (from the
  // key form's own settings read, made after the import).
  const [unscannedImports, setUnscannedImports] = useState(0);
  // Spec 167: the worker a keyless read would spend (product name), for the
  // sign beside Skip. null: no worker ready, so Skip reads nothing.
  const [scanAgent, setScanAgent] = useState<string | null>(null);
  // Whether the key form's settings read has reported yet. Until it has,
  // Skip cannot tell from state whether anything waits, so it reads the
  // settings itself (a Skip clicked at once must not turn the read off).
  const [scanKnown, setScanKnown] = useState(false);
  // Round 2 opt-out (per batch, never lasting), shared by the signs beside
  // Next and Skip: one batch, one switch.
  const [scanOn, setScanOn] = useState(true);
  const skipReads = unscannedImports >= 1 && scanAgent !== null;

  // Accounts step — API connector cards. null = fetch not resolved yet;
  // fetch failure resolves to [] so the browser group renders alone.
  const [connectors, setConnectors] = useState<Connector[] | null>(null);
  const [connectingId, setConnectingId] = useState<string | null>(null);
  const [connectError, setConnectError] = useState<{ id: string; message: string } | null>(null);

  // Accounts step — agent browser sign-in (former browser_warmup content)
  const [browserLoading, setBrowserLoading] = useState(false);
  const [browserDone, setBrowserDone] = useState(false);
  const [browserError, setBrowserError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (pollRef.current !== null) clearInterval(pollRef.current);
  }, []);

  // Settings at mount: whether a key exists (step 3 is then skipped) and the
  // agents already hidden (they start unchecked).
  const [keySet, setKeySet] = useState(false);
  const [hiddenBefore, setHiddenBefore] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    api<{ llm: { api_key_set: boolean }; sub_agents_hidden?: string[] }>('/api/v2/settings')
      .then((data) => {
        if (cancelled || !data) return;
        setKeySet(data.llm?.api_key_set === true);
        if (Array.isArray(data.sub_agents_hidden)) setHiddenBefore(data.sub_agents_hidden);
      })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, []);

  // A fresh install: no project of its own yet and no tour decision made. The
  // tour is marked pending at Finish so an import does not opt them out.
  const freshRef = useRef<boolean | null>(null);
  useEffect(() => {
    if (readTourState() === 'done') { freshRef.current = false; return; }
    api<Project[]>('/api/v2/projects')
      .then((list) => { freshRef.current = !Array.isArray(list) || list.length === 0 || isFirstRun(list); })
      .catch(() => { freshRef.current = true; });
  }, []);

  // ── Step 1: agents. The history scan starts here (it feeds the counts). ──
  const scan = useOnboardingScan();
  const { entries: agentEntries, reload: reloadAgents } = useSubAgentList();
  const [agentTicks, setAgentTicks] = useState<Record<string, boolean>>({});
  const installedAgents = (agentEntries ?? []).filter((e) => e.installed);
  const agentChecked = (slug: string) => agentTicks[slug] ?? !hiddenBefore.includes(slug);
  const checkedAgents = installedAgents.filter((e) => agentChecked(e.slug)).length;
  const noAgents = agentEntries !== null && installedAgents.length === 0;
  const [savingAgents, setSavingAgents] = useState(false);
  const [agentsError, setAgentsError] = useState<string | null>(null);
  // No coding agent to continue with: the key step loses its Skip.
  const [keyCompulsory, setKeyCompulsory] = useState(false);
  // The project Finish lands in, from the import.
  const landingRef = useRef<string | null>(null);

  const reportedRef = useRef(new Set<WizardStep>());
  useEffect(() => {
    if (reportedRef.current.has(step)) return;
    reportedRef.current.add(step);
    sendTelemetry('wizard_step_reached', { step: STEP_TELEMETRY[step] });
  }, [step]);

  // A key that already exists answers step 3, compulsory or not.
  const afterProjects = useCallback(() => {
    setStep(keySet ? 'accounts' : 'api_key');
  }, [keySet]);

  const nothingToImport = scan.settled && (scan.error || !scan.importable || !hasImportables(scan.importable));

  const handleAgentsContinue = useCallback(async () => {
    setSavingAgents(true);
    setAgentsError(null);
    const installedSlugs = new Set(installedAgents.map((e) => e.slug));
    const hidden = [
      // Hidden agents that are not on this list keep their state.
      ...hiddenBefore.filter((slug) => !installedSlugs.has(slug)),
      ...installedAgents.filter((e) => !agentChecked(e.slug)).map((e) => e.slug),
    ];
    try {
      await api('/api/v2/settings', {
        method: 'PUT',
        body: JSON.stringify({ sub_agents_hidden: hidden }),
      });
    } catch {
      setAgentsError(t('wizard.skipKey.failed'));
      setSavingAgents(false);
      return;
    }
    setSavingAgents(false);
    setKeyCompulsory(checkedAgents === 0);
    if (nothingToImport) afterProjects();
    else setStep('projects');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [installedAgents, hiddenBefore, agentTicks, checkedAgents, nothingToImport, afterProjects, t]);

  // The scan can finish empty after the user reached the Projects step.
  useEffect(() => {
    if (step === 'projects' && nothingToImport && !scan.error) afterProjects();
  }, [step, nothingToImport, scan.error, afterProjects]);

  // Featured connector catalog — fetched on entering the accounts step and
  // re-fetched after every successful connect (connectors sharing an
  // auth_provider come online together, so the whole list must refresh).
  const refreshConnectors = useCallback(async () => {
    try {
      const data = await api<ConnectorListResponse>('/api/v2/connectors');
      setConnectors(data.connectors);
    } catch {
      // Catalog unreachable — never a broken screen: the browser sign-in
      // group stands alone.
      setConnectors([]);
    }
  }, []);

  useEffect(() => {
    if (step !== 'accounts') return;
    refreshConnectors();
  }, [step, refreshConnectors]);

  const featured = (connectors ?? []).filter(
    (c) => c.featured && c.status === 'available',
  );

  // Save settings then advance to the accounts step.
  //
  // The form saves the DEFAULT provider (see the `card` prop below), so the
  // key it just wrote is the key `llm.api_key_set` reports. Before that it
  // minted a NEW provider on every press: a new one only becomes the default
  // when there is none, so with a default already present (any migrated
  // install) `api_key_set` stayed false, Next did nothing at all, and each
  // press leaked another provider. Five accumulated on the reporting user's
  // machine before they gave up.
  const handleNext = useCallback(async () => {
    setCheckingKey(true);
    setNextError(null);
    try {
      if (saveRef.current) {
        const ok = await saveRef.current();
        if (!ok) {
          // The form renders its own reason; do not also advance.
          return;
        }
      }
      const data = await api<{ llm: { api_key_set: boolean } }>('/api/v2/settings');
      if (data.llm.api_key_set) {
        setStep('accounts');
      } else {
        // Saved, yet still no usable key. Silence here is what made this look
        // like a dead button rather than a failure.
        setNextError(t('wizard.apiKey.notSaved'));
      }
    } catch (err: unknown) {
      setNextError(err instanceof Error && err.message ? err.message : t('wizard.apiKey.notSaved'));
    } finally {
      setCheckingKey(false);
    }
  }, [t]);

  // Skip the key: persist the choice so the first-run gate stops showing the
  // wizard, then continue to the accounts step exactly as Next does. Nothing
  // is saved from the form — a half-filled provider stays unsaved. Spec 167:
  // Skip skips Orbital's key, not the read — with the sign shown and its
  // switch on, the user's own agent reads the imported projects once.
  const handleSkipKey = useCallback(async () => {
    setSkippingKey(true);
    setSkipError(null);
    setNextError(null);
    try {
      let reads = skipReads;
      if (!scanKnown) {
        // The form's own read has not landed: ask the daemon directly rather
        // than guess "nothing waits" (import_scan:false is final for the
        // wizard's imports).
        const data = await api<{
          unscanned_imports?: number;
          import_scan_agent?: { slug: string; name: string } | null;
        }>('/api/v2/settings');
        reads = (data.unscanned_imports ?? 0) >= 1 && !!data.import_scan_agent;
      }
      await api('/api/v2/settings', {
        method: 'PUT',
        body: JSON.stringify({ setup_api_key_skipped: true, import_scan: reads && scanOn }),
      });
      setStep('accounts');
    } catch {
      setSkipError(t('wizard.skipKey.failed'));
    } finally {
      setSkippingKey(false);
    }
  }, [t, skipReads, scanKnown, scanOn]);

  // Connector connect — the backend runs a BLOCKING loopback OAuth flow (up
  // to ~300 s while the user completes browser consent), so the card shows a
  // "waiting for browser consent…" busy state for the whole await.
  const handleConnect = useCallback(async (c: Connector) => {
    setConnectError(null);
    setConnectingId(c.id);
    try {
      await api(`/api/v2/connectors/${encodeURIComponent(c.id)}/connect`, {
        method: 'POST',
      });
      await refreshConnectors();
    } catch (e) {
      if (e instanceof ApiError && e.status === 400) {
        // OAuth client not configured — short hint, finishable later in Settings.
        setConnectError({ id: c.id, message: t('wizard.accounts.clientHint') });
      } else if (e instanceof ApiError) {
        setConnectError({ id: c.id, message: e.detail });
      } else {
        setConnectError({ id: c.id, message: t('wizard.accounts.connectFailed') });
      }
    } finally {
      setConnectingId(null);
    }
  }, [refreshConnectors, t]);

  // Browser warmup handler — fire-and-forget launch, then poll status
  const handleOpenBrowser = useCallback(async () => {
    setBrowserLoading(true);
    setBrowserError(null);
    try {
      await api('/api/v2/platform/browser/warmup', {
        method: 'POST',
        body: JSON.stringify({ url: warmupUrlForLocale(locale) }),
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : t('wizard.browser.openFailed');
      setBrowserError(msg);
      setBrowserLoading(false);
      return;
    }
    // Poll until warmup browser is closed
    pollRef.current = window.setInterval(async () => {
      try {
        const status = await api<{ active: boolean }>('/api/v2/platform/browser/warmup/status');
        if (!status.active) {
          if (pollRef.current !== null) clearInterval(pollRef.current);
          pollRef.current = null;
          setBrowserDone(true);
          setBrowserLoading(false);
        }
      } catch {
        // Status endpoint failed — keep polling
      }
    }, 2000);
  }, [t, locale]);

  const finish = useCallback(() => {
    sendTelemetry('wizard_step_reached', { step: 'done' });
    // null = the projects check has not answered: the wizard showing at all
    // is the strong signal of a fresh install.
    if (freshRef.current !== false) writeTourState('pending');
    onComplete({ landingProjectId: landingRef.current });
  }, [onComplete]);

  return (
    // pt-titlebar: this is a full-screen surface rendered outside the app
    // shell, so it clears the macOS titlebar band itself — the native titlebar
    // view sits above the webview and eats clicks in that strip. Safe against
    // min-h-screen because box-sizing is border-box globally, so the padding
    // comes out of the 100vh rather than adding to it.
    <div className="min-h-screen flex items-center justify-center bg-background px-4 py-8 pt-titlebar">
      <div className="w-full max-w-xl">
        {step === 'agents' && (
          <WizardCard
            step="agents"
            title={t('wizard.welcome')}
            intro={noAgents ? t('wizard.agents.introNone') : t('wizard.agents.intro')}
          >
            {agentEntries === null ? (
              <div className="inline-flex items-center gap-2 text-sm text-secondary" data-testid="wizard-agents-loading">
                <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
                {t('wizard.agents.loading')}
              </div>
            ) : (
              <WizardAgentsStep
                entries={agentEntries}
                history={scan.history}
                ticks={Object.fromEntries(installedAgents.map((e) => [e.slug, agentChecked(e.slug)]))}
                onToggle={(slug, checked) => setAgentTicks((prev) => ({ ...prev, [slug]: checked }))}
                reload={reloadAgents}
              />
            )}
            <div className="mt-6 flex flex-col items-end">
              <button
                type="button"
                onClick={handleAgentsContinue}
                disabled={agentEntries === null || savingAgents}
                data-testid="wizard-agents-continue"
                className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 inline-flex items-center gap-2"
              >
                {savingAgents && <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />}
                {checkedAgents === 0
                  ? t('wizard.agents.continueNone')
                  : t(checkedAgents === 1 ? 'wizard.agents.continue.one' : 'wizard.agents.continue.other', { n: checkedAgents })}
              </button>
              {agentsError && (
                <p className="text-xs text-error mt-2" role="alert" data-testid="wizard-agents-error">
                  {agentsError}
                </p>
              )}
            </div>
          </WizardCard>
        )}

        {step === 'projects' && (
          <WizardCard
            step="projects"
            title={t('importProjects.title')}
            intro={importIntro(t, scan.history?.sessions_total, scan.importable)}
          >
            <ImportProjectsStep
              scan={scan}
              // Spec 160: with no key yet, the read waits for the key step's
              // Save or Skip (each with its sign); with one, the key step is skipped and the import
              // itself starts it.
              deferScan={!keySet}
              onBack={() => setStep('agents')}
              onSkip={afterProjects}
              onImported={(_result, landing) => {
                landingRef.current = landing;
                afterProjects();
              }}
            />
          </WizardCard>
        )}

        {step === 'api_key' && (
          <WizardCard step="api_key" title={t('wizard.apiKey.title')} intro={t('wizard.intro')}>
            {/* `card` = the default provider, so Save PUTs it. Without this the
                form takes its "create a new one" branch every press. */}
            <LLMProviderSettings
              mode="global"
              card={defaultCard}
              hideSaveButton
              saveRef={saveRef}
              onUnscannedImports={(n, agent) => { setUnscannedImports(n); setScanAgent(agent); setScanKnown(true); }}
              importScan={scanOn}
              providerPicker="cards"
              // One-click sign-in creates (or re-keys) a card on the daemon;
              // re-read so "Adjust settings" edits THAT card, not the one
              // that happened to be default at mount.
              onCardSaved={() => { void refreshCards(); }}
            />

            {/* First-run telemetry disclosure (spec 046 §6) — one sentence,
                non-blocking; the full inspectable payload lives in Global
                Settings → Data & privacy. */}
            <p
              className="mt-4 text-xs text-secondary/80"
              data-testid="wizard-telemetry-disclosure"
            >
              {t('telemetry.wizard.disclosure')}
            </p>

            <div className="mt-6 flex flex-col items-end">
              <div className="flex items-center gap-2.5">
                {/* Spec 160 §5.3: saving the key reads the imported projects
                    once; the count says so, the sentence waits for a hover. */}
                <TokenCostSign n={unscannedImports} scan={scanOn} onScanChange={setScanOn} />
                <button
                  onClick={handleNext}
                  disabled={checkingKey || skippingKey}
                  className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 inline-flex items-center gap-2"
                >
                  {checkingKey ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      {t('wizard.checking')}
                    </>
                  ) : (
                    t('wizard.next')
                  )}
                </button>
              </div>
              {nextError && (
                <p className="text-xs text-error mt-2" data-testid="wizard-next-error">
                  {nextError}
                </p>
              )}
            </div>

            {/* Keyless path: a quiet secondary action under Next, with the
                consequence spelled out right beside it. Not offered when the
                user continued without any coding agent — the key is then the
                only way anything gets done. */}
            {keyCompulsory ? (
              <div
                className="mt-5 rounded-lg border border-warning/30 bg-warning-surface px-3.5 py-3 text-xs leading-relaxed text-warning-text"
                data-testid="wizard-key-compulsory"
              >
                {t('wizard.apiKey.compulsory')}{' '}
                <button
                  type="button"
                  onClick={() => setStep('agents')}
                  data-testid="wizard-key-back-to-agents"
                  className="font-medium underline underline-offset-2 hover:no-underline"
                >
                  {t('wizard.apiKey.compulsoryBack')}
                </button>
              </div>
            ) : (
              <div className="mt-5 pt-4 border-t border-border flex flex-col items-end text-right">
                <div className="flex items-center gap-2.5">
                  {/* Spec 167: Skip reads with the user's own agent; the sign
                      says which, and how many, and holds the switch. */}
                  <TokenCostSign n={unscannedImports} kind="importAgent" agent={scanAgent}
                    scan={scanOn} onScanChange={setScanOn} />
                  <button
                    type="button"
                    onClick={handleSkipKey}
                    disabled={checkingKey || skippingKey}
                    data-testid="wizard-skip-key"
                    className="text-sm font-medium text-secondary hover:text-primary underline-offset-2 hover:underline transition-colors duration-150 disabled:opacity-50 max-md:min-h-[44px]"
                  >
                    {t('wizard.skipKey')}
                  </button>
                </div>
                <p
                  className="mt-1 text-xs text-secondary/80 leading-relaxed max-w-[380px]"
                  data-testid="wizard-skip-key-hint"
                >
                  {t('wizard.skipKey.hint')}
                </p>
                {skipError && (
                  <p className="text-xs text-error mt-2" role="alert" data-testid="wizard-skip-error">
                    {skipError}
                  </p>
                )}
              </div>
            )}
          </WizardCard>
        )}

        {step === 'accounts' && (
          <WizardCard step="accounts" title={t('wizard.accounts.title')} intro={t('wizard.accounts.intro')}>
            {/* Group (a): API connectors — featured, available catalog
                entries only. Hidden entirely when the fetch failed or
                nothing qualifies. */}
            {featured.length > 0 && (
              <div className="mb-6" data-testid="wizard-connectors-group">
                <div className="flex items-center gap-2 mb-1">
                  <Plug className="w-4 h-4 text-accent" />
                  <h2 className="text-sm font-medium text-primary">
                    {t('wizard.accounts.connectorsHeading')}
                  </h2>
                  <BetaBadge />
                </div>
                <p className="text-xs text-secondary mb-1">
                  {t('wizard.accounts.connectorsHint')}
                </p>
                <p className="text-xs text-secondary/80 mb-3">
                  {t('connectors.betaNote')}
                </p>
                <ul className="space-y-2">
                  {featured.map((c) => {
                    const connecting = connectingId === c.id;
                    return (
                      <li
                        key={c.id}
                        data-testid={`wizard-connector-${c.id}`}
                        className="surface-float-static surface-float-nested px-3 py-2.5"
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-xl shrink-0" aria-hidden="true">
                            {c.icon}
                          </span>
                          <div className="min-w-0 flex-1">
                            <span className="block text-sm font-medium text-primary truncate">
                              {c.name}
                            </span>
                            {c.connected && (
                              <p
                                className="text-xs text-secondary truncate"
                                data-testid={`wizard-connector-status-${c.id}`}
                              >
                                {t('wizard.accounts.connectedAs', { account: c.account ?? '' })}
                              </p>
                            )}
                          </div>
                          {!c.connected && (
                            <button
                              type="button"
                              data-testid={`wizard-connector-connect-${c.id}`}
                              disabled={connectingId !== null}
                              onClick={() => handleConnect(c)}
                              className="shrink-0 text-xs font-medium text-white bg-accent rounded-lg px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-60 inline-flex items-center gap-1.5 max-md:min-h-[44px]"
                            >
                              {connecting ? (
                                <>
                                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                  {t('wizard.accounts.waitingConsent')}
                                </>
                              ) : (
                                t('wizard.accounts.connect')
                              )}
                            </button>
                          )}
                        </div>
                        {connectError?.id === c.id && (
                          <p
                            className="mt-2 text-xs text-error"
                            role="alert"
                            data-testid={`wizard-connector-error-${c.id}`}
                          >
                            {connectError.message}
                          </p>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {/* Group (b): agent browser sign-in — the former browser_warmup
                card content, visually parallel to group (a). */}
            <div data-testid="wizard-browser-group">
              <div className="flex items-center gap-2 mb-1">
                <Globe className="w-4 h-4 text-accent" />
                <h2 className="text-sm font-medium text-primary">
                  {t('wizard.accounts.browserHeading')}
                </h2>
              </div>
              <p className="text-secondary text-xs leading-relaxed mb-1.5">
                {t('wizard.browser.body1')}
              </p>
              <p className="text-secondary text-xs leading-relaxed mb-3">
                {t('wizard.browser.body2')}
              </p>

              {browserError && (
                <div className="bg-error/10 border border-error/20 rounded-lg px-4 py-3 mb-3 text-sm text-error">
                  {browserError}
                </div>
              )}

              {browserDone ? (
                <div
                  className="inline-flex items-center gap-2 text-sm text-success"
                  data-testid="wizard-browser-done"
                >
                  <span aria-hidden="true">{'✓'}</span>
                  {t('wizard.accounts.browserDone')}
                </div>
              ) : browserLoading ? (
                <div className="inline-flex items-center gap-2 text-sm text-secondary">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {t('wizard.browser.opening')}
                </div>
              ) : (
                <button
                  onClick={handleOpenBrowser}
                  data-testid="wizard-open-browser"
                  className="border border-border text-primary text-sm font-medium rounded-lg px-4 py-2 hover:bg-surface transition-all duration-150"
                >
                  {t('wizard.openBrowser')}
                </button>
              )}
            </div>

            <div className="mt-8 flex justify-end gap-3">
              <button
                onClick={finish}
                data-testid="wizard-skip"
                className="border border-border text-primary text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-surface transition-all duration-150"
              >
                {t('wizard.skip')}
              </button>
              <button
                onClick={finish}
                data-testid="wizard-get-started"
                className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150"
              >
                {t('wizard.getStarted')}
              </button>
            </div>
          </WizardCard>
        )}
      </div>
    </div>
  );
}
