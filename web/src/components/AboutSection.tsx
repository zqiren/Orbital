// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import { useT } from '../i18n/useT';

/**
 * About row in Global Settings: the running version (the app finally knows
 * it — spec 046 §7 version injection) + a manual update check against
 * GET /api/v2/update-status. Companion to the passive UpdatePill.
 */

const API_BASE = import.meta.env.VITE_API_BASE || '';

interface UpdateStatus {
  current: string;
  update_available: boolean;
  latest: string | null;
  url: string | null;
}

interface AboutSectionProps {
  /** Replay the first-journey tour. Omitted where there is no app shell to
   * tour (the component is also rendered standalone in tests). */
  onTakeTour?: () => void;
  /** False while the user has no project of their own — the tour walks
   * through a project, so there is nothing to show yet. */
  canTakeTour?: boolean;
}

export default function AboutSection({ onTakeTour, canTakeTour = false }: AboutSectionProps = {}) {
  const t = useT();
  const [status, setStatus] = useState<UpdateStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(false);
  const [checked, setChecked] = useState(false);

  async function fetchStatus() {
    const r = await fetch(`${API_BASE}/api/v2/update-status`);
    return (await r.json()) as UpdateStatus;
  }

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => {});
  }, []);

  async function handleCheck() {
    setChecking(true);
    setError(false);
    try {
      setStatus(await fetchStatus());
      setChecked(true);
    } catch {
      setError(true);
    } finally {
      setChecking(false);
    }
  }

  return (
    <div>
      {/* Section heading is the enclosing SettingsSection's `title`
          (GlobalSettings) — see ConnectorSettings for the same note. */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-sm text-secondary font-mono" data-testid="about-version">
          {status ? t('update.about.version', { version: status.current }) : '…'}
        </span>
        <button
          type="button"
          onClick={handleCheck}
          disabled={checking}
          data-testid="about-check-updates"
          className="text-xs text-accent hover:underline disabled:opacity-50"
        >
          {checking ? t('update.about.checking') : t('update.about.check')}
        </button>
      </div>
      <p className="text-xs text-secondary mt-1.5" data-testid="about-check-result">
        {error
          ? t('update.about.error')
          : status?.update_available && status.latest
            ? (
                <a
                  href={status.url ?? '#'}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent hover:underline"
                >
                  {t('update.about.available', { version: status.latest })}
                </a>
              )
            : checked
              ? t('update.about.upToDate')
              : null}
      </p>
      {onTakeTour && (
        <div className="mt-4 pt-4 border-t border-border">
          <button
            type="button"
            onClick={onTakeTour}
            disabled={!canTakeTour}
            data-testid="about-take-tour"
            className="text-sm font-medium text-accent border border-accent/30 rounded-lg px-4 py-2 hover:bg-accent/5 transition-all duration-150 disabled:opacity-50 disabled:hover:bg-transparent max-md:min-h-[44px]"
          >
            {t('tour.takeTour')}
          </button>
          <p className="text-xs text-secondary mt-1.5">
            {canTakeTour ? t('tour.takeTour.hint') : t('tour.takeTour.needsProject')}
          </p>
        </div>
      )}
    </div>
  );
}
