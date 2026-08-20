// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import { useT } from '../i18n/useT';

/**
 * Data & privacy section (spec 046 §6). Two pieces:
 *
 *  - The telemetry toggle — saved immediately on flip (self-contained section
 *    like CredentialStore/ConnectorSettings, not part of the big form Save).
 *  - The payload viewer — the exact JSON of the last-sent and next-pending
 *    ping, verbatim. This transparency is the load-bearing trust feature
 *    behind the default-on consent model; never summarize or prettify beyond
 *    JSON indentation.
 *
 * Follows GlobalSettings' local raw-fetch convention.
 */

const API_BASE = import.meta.env.VITE_API_BASE || '';

interface PayloadResponse {
  last_sent: Record<string, unknown> | null;
  next_pending: Record<string, unknown> | null;
}

export default function TelemetrySettings() {
  const t = useT();
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [payload, setPayload] = useState<PayloadResponse | null>(null);
  const [payloadError, setPayloadError] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/v2/settings`)
      .then((r) => r.json())
      .then((data) => setEnabled(data.telemetry_enabled !== false))
      .catch(() => setEnabled(true));
    fetch(`${API_BASE}/api/v2/settings/telemetry-payload`)
      .then((r) => r.json())
      .then((data: PayloadResponse) => setPayload(data))
      .catch(() => setPayloadError(true));
  }, []);

  async function handleToggle() {
    if (enabled === null) return;
    const next = !enabled;
    setEnabled(next);
    try {
      await fetch(`${API_BASE}/api/v2/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ telemetry_enabled: next }),
      });
    } catch {
      setEnabled(!next); // revert on failure — never show an unsaved state
    }
  }

  return (
    <div>
      {/* Section heading is the enclosing SettingsSection's `title`
          (GlobalSettings) — see ConnectorSettings for the same note. */}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <label className="block text-sm font-medium text-primary mb-1">
            {t('telemetry.toggle.label')}
          </label>
          <p className="text-xs text-secondary">{t('telemetry.toggle.hint')}</p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={enabled === true}
          aria-label={t('telemetry.toggle.label')}
          disabled={enabled === null}
          onClick={handleToggle}
          data-testid="telemetry-toggle"
          className={`shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-150 disabled:opacity-50 ${
            enabled ? 'bg-accent' : 'bg-border'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-150 ${
              enabled ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      <div className="mt-4">
        <h3 className="text-sm font-medium text-primary mb-2">
          {t('telemetry.viewer.heading')}
        </h3>
        {payloadError ? (
          <p className="text-xs text-secondary">{t('telemetry.viewer.error')}</p>
        ) : (
          <div className="space-y-3">
            <div>
              <p className="text-xs text-secondary mb-1">
                {t('telemetry.viewer.lastSent')}
              </p>
              {payload?.last_sent ? (
                <pre
                  data-testid="telemetry-last-sent"
                  className="text-[11px] font-mono bg-sidebar border border-border rounded-lg p-3 overflow-x-auto whitespace-pre"
                >
                  {JSON.stringify(payload.last_sent, null, 2)}
                </pre>
              ) : (
                <p className="text-xs text-secondary/70">
                  {t('telemetry.viewer.never')}
                </p>
              )}
            </div>
            {payload?.next_pending && (
              <div>
                <p className="text-xs text-secondary mb-1">
                  {t('telemetry.viewer.nextPending')}
                </p>
                <pre
                  data-testid="telemetry-next-pending"
                  className="text-[11px] font-mono bg-sidebar border border-border rounded-lg p-3 overflow-x-auto whitespace-pre"
                >
                  {JSON.stringify(payload.next_pending, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
