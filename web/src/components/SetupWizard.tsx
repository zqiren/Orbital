// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useCallback, useRef } from 'react';
import { Loader2, Shield, AlertTriangle, Globe } from 'lucide-react';
import { api } from '../config';
import type { PlatformStatus } from '../types';
import LLMProviderSettings from './LLMProviderSettings';
import { useT } from '../i18n/useT';

type WizardStep = 'api_key' | 'sandbox_status' | 'browser_warmup';

interface SetupWizardProps {
  onComplete: () => void;
}

export default function SetupWizard({ onComplete }: SetupWizardProps) {
  const t = useT();
  const [step, setStep] = useState<WizardStep>('api_key');
  const [checkingKey, setCheckingKey] = useState(false);
  const saveRef = useRef<(() => Promise<boolean>) | null>(null);

  // Sandbox step state
  const [sandboxLoading, setSandboxLoading] = useState(false);
  const [sandboxStatus, setSandboxStatus] = useState<PlatformStatus | null>(null);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);

  // Browser warmup step state
  const [browserLoading, setBrowserLoading] = useState(false);
  const [browserDone, setBrowserDone] = useState(false);
  const [browserError, setBrowserError] = useState<string | null>(null);

  // Auto-advance if API key is already configured
  useEffect(() => {
    let cancelled = false;
    async function checkKey() {
      try {
        const data = await api<{ llm: { api_key_set: boolean } }>('/api/v2/settings');
        if (!cancelled && data.llm.api_key_set) {
          setStep('sandbox_status');
        }
      } catch {
        // ignore
      }
    }
    checkKey();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch platform status when entering sandbox_status step
  useEffect(() => {
    if (step !== 'sandbox_status') return;
    let cancelled = false;
    async function fetchPlatformStatus() {
      setSandboxLoading(true);
      try {
        const data = await api<PlatformStatus>('/api/v2/platform/status');
        if (cancelled) return;
        // Non-Windows users skip sandbox screen entirely
        if (data.platform === 'null') {
          setStep('browser_warmup');
          return;
        }
        setSandboxStatus(data);
      } catch {
        // If we can't reach the platform endpoint, skip to browser warmup
        if (!cancelled) setStep('browser_warmup');
      } finally {
        if (!cancelled) setSandboxLoading(false);
      }
    }
    fetchPlatformStatus();
    return () => { cancelled = true; };
  }, [step, onComplete]);

  // Save settings then advance to sandbox step
  const handleNext = useCallback(async () => {
    setCheckingKey(true);
    try {
      // Auto-save settings before advancing
      if (saveRef.current) {
        await saveRef.current();
      }
      const data = await api<{ llm: { api_key_set: boolean } }>('/api/v2/settings');
      if (data.llm.api_key_set) {
        setStep('sandbox_status');
      }
    } catch {
      // ignore — user can retry
    } finally {
      setCheckingKey(false);
    }
  }, []);

  // Sandbox step handlers
  const handleSkipSandbox = useCallback(async () => {
    try {
      await api('/api/v2/platform/skip', { method: 'POST' });
    } catch {
      // Continue even if skip call fails
    }
    setStep('browser_warmup');
  }, []);

  const handleRetrySandbox = useCallback(async () => {
    setRetrying(true);
    setSetupError(null);
    try {
      await api('/api/v2/platform/setup', { method: 'POST' });
      // Re-fetch status to confirm
      const data = await api<PlatformStatus>('/api/v2/platform/status');
      setSandboxStatus(data);
    } catch (e) {
      const msg = e instanceof Error ? e.message : t('wizard.setupFailed');
      // Clean up common error messages
      if (msg.includes('administrator') || msg.includes('elevation') || msg.includes('privilege')) {
        setSetupError(t('wizard.sandboxIncomplete.adminError'));
      } else {
        setSetupError(msg);
      }
    } finally {
      setRetrying(false);
    }
  }, []);

  // Browser warmup handler — fire-and-forget launch, then poll status
  const handleOpenBrowser = useCallback(async () => {
    setBrowserLoading(true);
    setBrowserError(null);
    try {
      await api('/api/v2/platform/browser/warmup', { method: 'POST' });
    } catch (e) {
      const msg = e instanceof Error ? e.message : t('wizard.browser.openFailed');
      setBrowserError(msg);
      setBrowserLoading(false);
      return;
    }
    // Poll until warmup browser is closed
    const poll = setInterval(async () => {
      try {
        const status = await api<{ active: boolean }>('/api/v2/platform/browser/warmup/status');
        if (!status.active) {
          clearInterval(poll);
          setBrowserDone(true);
          setBrowserLoading(false);
        }
      } catch {
        // Status endpoint failed — keep polling
      }
    }, 2000);
  }, []);

  // Render sandbox confirmation (green — setup complete)
  function renderSandboxComplete() {
    return (
      <div className="bg-background border border-border rounded-lg p-8">
        <div className="flex items-center gap-3 mb-4">
          <Shield className="w-6 h-6 text-success" />
          <h1 className="text-2xl font-semibold text-primary">
            {t('wizard.sandboxComplete.title')}
          </h1>
        </div>

        <div className="space-y-3 mb-6">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-success font-medium">{'\u2713'}</span>
            <span className="text-primary">{t('wizard.sandboxComplete.item1')}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-success font-medium">{'\u2713'}</span>
            <span className="text-primary">{t('wizard.sandboxComplete.item2')}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-success font-medium">{'\u2713'}</span>
            <span className="text-primary">{t('wizard.sandboxComplete.item3')}</span>
          </div>
        </div>

        <p className="text-secondary text-sm leading-relaxed mb-6">
          {t('wizard.sandboxComplete.body')}
        </p>

        <div className="flex justify-end">
          <button
            onClick={() => setStep('browser_warmup')}
            className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150"
          >
            {t('wizard.next')}
          </button>
        </div>
      </div>
    );
  }

  // Render sandbox warning (setup incomplete)
  function renderSandboxIncomplete() {
    return (
      <div className="bg-background border border-border rounded-lg p-8">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle className="w-6 h-6 text-warning" />
          <h1 className="text-2xl font-semibold text-primary">
            {t('wizard.sandboxIncomplete.title')}
          </h1>
        </div>

        <p className="text-secondary text-sm leading-relaxed mb-4">
          {t('wizard.sandboxIncomplete.body')}
        </p>

        <div className="bg-surface border border-border rounded-lg p-4 mb-6">
          <p className="text-secondary text-sm mb-2">{t('wizard.sandboxIncomplete.howto')}</p>
          <ol className="text-secondary text-sm space-y-1 list-decimal list-inside">
            <li>{t('wizard.sandboxIncomplete.step1')}</li>
            <li>
              {t('wizard.sandboxIncomplete.step2')}
            </li>
          </ol>
        </div>

        {setupError && (
          <div className="bg-error/10 border border-error/20 rounded-lg px-4 py-3 mb-4 text-sm text-error">
            {setupError}
          </div>
        )}

        <div className="flex justify-end gap-3">
          <button
            onClick={handleSkipSandbox}
            className="border border-border text-primary text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-surface transition-all duration-150"
          >
            {t('wizard.continueNoSandbox')}
          </button>
          <button
            onClick={handleRetrySandbox}
            disabled={retrying}
            className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 inline-flex items-center gap-2"
          >
            {retrying ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                {t('wizard.retrying')}
              </>
            ) : (
              'Retry Setup'
            )}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4">
      <div className="w-full max-w-lg">
        <div className="text-center mb-8">
          <span className="font-mono text-lg text-primary tracking-tight">
            {t('setupGate.wordmark')}
          </span>
        </div>

        {step === 'api_key' && (
          <div className="bg-background border border-border rounded-lg p-8">
            <h1 className="text-2xl font-semibold text-primary mb-3">
              {t('wizard.welcome')}
            </h1>
            <p className="text-secondary text-sm leading-relaxed mb-6">
              {t('wizard.intro')}
            </p>

            <LLMProviderSettings mode="global" hideSaveButton saveRef={saveRef} />

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
            </div>
          </div>
        )}

        {step === 'sandbox_status' && sandboxLoading && (
          <div className="bg-background border border-border rounded-lg p-8 flex items-center justify-center">
            <Loader2 className="w-5 h-5 animate-spin text-secondary" />
          </div>
        )}

        {step === 'sandbox_status' && !sandboxLoading && sandboxStatus && (
          sandboxStatus.setup_complete
            ? renderSandboxComplete()
            : renderSandboxIncomplete()
        )}

        {step === 'browser_warmup' && (
          <div className="bg-background border border-border rounded-lg p-8">
            <div className="flex items-center gap-3 mb-4">
              <Globe className="w-6 h-6 text-accent" />
              <h1 className="text-2xl font-semibold text-primary">
                {t('wizard.browser.title')}
              </h1>
            </div>

            <p className="text-secondary text-sm leading-relaxed mb-4">
              {t('wizard.browser.body1')}
            </p>
            <p className="text-secondary text-sm leading-relaxed mb-6">
              {t('wizard.browser.body2')}
            </p>

            {browserError && (
              <div className="bg-error/10 border border-error/20 rounded-lg px-4 py-3 mb-4 text-sm text-error">
                {browserError}
              </div>
            )}

            <div className="flex justify-end gap-3">
              <button
                onClick={onComplete}
                className="border border-border text-primary text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-surface transition-all duration-150"
              >
                {t('wizard.skip')}
              </button>

              {browserDone ? (
                <button
                  onClick={onComplete}
                  className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150"
                >
                  {t('wizard.getStarted')}
                </button>
              ) : (
                <button
                  onClick={handleOpenBrowser}
                  disabled={browserLoading}
                  className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 inline-flex items-center gap-2"
                >
                  {browserLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      {t('wizard.browser.opening')}
                    </>
                  ) : (
                    t('wizard.openBrowser')
                  )}
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
