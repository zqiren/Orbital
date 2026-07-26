// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import { api } from '../config';
import type { PlatformStatus } from '../types';
import { useT } from '../i18n/useT';

interface SetupGateProps {
  setupIssues: string[];
  onComplete: () => void;
}

export default function SetupGate({ setupIssues, onComplete }: SetupGateProps) {
  const t = useT();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSetup() {
    setLoading(true);
    setError(null);
    try {
      await api<PlatformStatus>('/api/v2/platform/setup', {
        method: 'POST',
      });
      onComplete();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : t('setupGate.setupError'),
      );
      setLoading(false);
    }
  }

  return (
    // pt-titlebar: full-screen surface outside the app shell — clears the
    // macOS titlebar band on its own. See SetupWizard for the reasoning.
    <div className="min-h-screen flex items-center justify-center bg-background px-4 pt-titlebar">
      <div className="w-full max-w-md">
        {/* No standalone wordmark above the card: the heading below already
            names the app, and stacking both put "Orbital" on screen twice in
            the same eyeful. bg-card (not bg-background) so the dialog lifts
            off the canvas instead of relying on a 1px border — see the surface
            ladder in index.css. Kept in step with SetupWizard, which users can
            see back-to-back with this screen. */}
        <div className="bg-card border border-border rounded-lg p-8">
          <h1 className="text-2xl font-semibold text-primary mb-3">
            {t('setupGate.welcome')}
          </h1>
          <p className="text-secondary text-sm leading-relaxed mb-6">
            {t('setupGate.intro')}
          </p>

          {setupIssues.length > 0 && (
            <div className="mb-6 space-y-2">
              {setupIssues.map((issue, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-sm text-warning bg-warning/5 border border-warning/20 rounded-lg px-3 py-2"
                >
                  <span className="shrink-0 mt-0.5">!</span>
                  <span>{issue}</span>
                </div>
              ))}
            </div>
          )}

          {error && (
            <div className="mb-6 text-sm text-error bg-error/5 border border-error/20 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          {loading ? (
            <div className="text-center py-4">
              <p className="text-sm text-secondary">
                {t('setupGate.settingUp')}
              </p>
            </div>
          ) : (
            <button
              onClick={handleSetup}
              className="w-full bg-accent text-white text-sm font-medium rounded-lg px-4 py-2.5 hover:bg-accent/90 transition-all duration-150"
            >
              {error ? t('setupGate.retry') : t('setupGate.setup')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
