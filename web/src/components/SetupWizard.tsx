// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SetupWizard — first-run onboarding (Spec 011 §0.7, Task I1).
 *
 * Two steps: `api_key` → `accounts`. The old `sandbox_status` step is gone on
 * all platforms — macOS needs nothing (sandbox-exec ships with the OS) and
 * Windows sandbox setup moved into the installer. The standalone sandbox
 * warning banner in App.tsx covers the installer-skipped/failed case.
 *
 * The `accounts` step merges the former `browser_warmup` step with featured
 * connector cards into one "Connect Your Accounts" screen with two labeled
 * groups (API connectors / agent browser sign-in) — two back-to-back Google
 * prompts on separate screens would otherwise confuse users. Everything is
 * skippable; wizard completion stays derived from `llm.api_key_set` upstream.
 */

import { useState, useEffect, useCallback, useRef, type ReactNode } from 'react';
import { Loader2, Globe, Plug } from 'lucide-react';
import { api, ApiError } from '../config';
import { useCredentialCards } from '../hooks/useCredentialCards';
import type { Connector, ConnectorListResponse } from '../types';
import LLMProviderSettings from './LLMProviderSettings';
import ImportProjectsStep from './ImportProjectsStep';
import { useT } from '../i18n/useT';
import BetaBadge from './BetaBadge';
import { useLocale } from '../i18n/LocaleContext';
import { LOCALES } from '../i18n/locales';
import { warmupUrlForLocale } from '../utils/warmupUrl';
import Select from './Select';

type WizardStep = 'api_key' | 'accounts';

interface SetupWizardProps {
  onComplete: () => void;
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

/** Shared card chrome for both wizard steps.
 *
 *  The language toggle lives here rather than above the card because the two
 *  steps render separate cards — hoisting it out of them was what left it
 *  floating as loose chrome over the dialog. Declaring it once in the shared
 *  header keeps it inside the card and identical across steps.
 *
 *  bg-card, not bg-background: this is content that must lift off the canvas
 *  (see the surface ladder in index.css). Both onboarding screens were setting
 *  the same token as the page behind them, leaving a 1px border to do all the
 *  separation work — the exact flatness that ladder was introduced to fix. */
function WizardCard({
  title,
  intro,
  children,
}: {
  title: string;
  intro: string;
  children: ReactNode;
}) {
  return (
    <div className="bg-card border border-border rounded-lg p-8">
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
  const [step, setStep] = useState<WizardStep>('api_key');
  const [nextError, setNextError] = useState<string | null>(null);
  // The provider the form must EDIT. undefined while loading and when there is
  // genuinely none, which is the true first run — there the form's create
  // branch is right, and the new provider becomes the default by being first.
  const { cards, defaultCardId } = useCredentialCards();
  const defaultCard = cards.find((c) => c.id === defaultCardId) ?? undefined;
  const [checkingKey, setCheckingKey] = useState(false);
  const saveRef = useRef<(() => Promise<boolean>) | null>(null);

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

  // Auto-advance if API key is already configured
  useEffect(() => {
    let cancelled = false;
    async function checkKey() {
      try {
        const data = await api<{ llm: { api_key_set: boolean } }>('/api/v2/settings');
        if (!cancelled && data.llm.api_key_set) {
          setStep('accounts');
        }
      } catch {
        // ignore
      }
    }
    checkKey();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  return (
    // pt-titlebar: this is a full-screen surface rendered outside the app
    // shell, so it clears the macOS titlebar band itself — the native titlebar
    // view sits above the webview and eats clicks in that strip. Safe against
    // min-h-screen because box-sizing is border-box globally, so the padding
    // comes out of the 100vh rather than adding to it.
    <div className="min-h-screen flex items-center justify-center bg-background px-4 pt-titlebar">
      <div className="w-full max-w-lg">
        {step === 'api_key' && (
          <WizardCard title={t('wizard.welcome')} intro={t('wizard.intro')}>
            {/* `card` = the default provider, so Save PUTs it. Without this the
                form takes its "create a new one" branch every press. */}
            <LLMProviderSettings
              mode="global"
              card={defaultCard}
              hideSaveButton
              saveRef={saveRef}
              providerPicker="cards"
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

            <div className="mt-6 flex justify-end">
              <button
                onClick={handleNext}
                disabled={checkingKey}
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
              {nextError && (
                <p className="text-xs text-error mt-2" data-testid="wizard-next-error">
                  {nextError}
                </p>
              )}
            </div>
          </WizardCard>
        )}

        {step === 'accounts' && (
          <WizardCard title={t('wizard.accounts.title')} intro={t('wizard.accounts.intro')}>
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
                        className="border border-border rounded-lg px-3 py-2.5"
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

            {/* Group (c): import existing projects — a disclose-then-scan,
                confirm-each list of projects discovered on disk from other CLI
                agents and Obsidian (backlog #34). Link-only; each confirmed row
                calls the standard project-creation flow. */}
            <div className="mt-6 pt-6 border-t border-border">
              <ImportProjectsStep />
            </div>

            <div className="mt-8 flex justify-end gap-3">
              <button
                onClick={onComplete}
                data-testid="wizard-skip"
                className="border border-border text-primary text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-surface transition-all duration-150"
              >
                {t('wizard.skip')}
              </button>
              <button
                onClick={onComplete}
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
