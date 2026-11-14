// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import { X } from 'lucide-react';
import type { UpdateInstallerLinks, WebSocketEvent } from '../types';
import { useWebSocket } from '../hooks/useWebSocket';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import { regionForLocale, type Locale } from '../i18n/locales';

/**
 * Bottom-left "new version available" pill (notify-only update tier).
 *
 * Sources: the daemon announces `update.available` once per version over the
 * global WS; a mount fetch of /api/v2/update-status covers page loads that
 * happen after the announce. Clicking downloads this platform's installer in
 * the system browser (GitHub, or the China mirror for a zh UI); a platform
 * without one (Linux) gets the website. Opening the installer stays manual
 * (no auto-update until signing lands).
 *
 * Dismissal is remembered PER VERSION in localStorage: never nags twice for
 * the same release, reappears for the next one.
 */

const API_BASE = import.meta.env.VITE_API_BASE || '';
const DISMISS_KEY = 'orbital.updateDismissed';

interface UpdateInfo {
  version: string;
  url: string;
  installer?: UpdateInstallerLinks | null;
}

/** Where a click on the pill goes: the installer from the mirror the UI
 * language picks, else the website the daemon forwarded. */
function updateHref(
  info: { url?: string | null; installer?: UpdateInstallerLinks | null },
  locale: Locale,
): string {
  return info.installer?.[regionForLocale(locale)] || info.url || '';
}

export default function UpdatePill() {
  const t = useT();
  const { locale } = useLocale();
  const ws = useWebSocket();
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [dismissed, setDismissed] = useState<string | null>(
    () => localStorage.getItem(DISMISS_KEY),
  );

  useEffect(() => {
    fetch(`${API_BASE}/api/v2/update-status`)
      .then((r) => r.json())
      .then((data) => {
        if (data.update_available && data.latest) {
          setUpdate({ version: data.latest, url: data.url || '', installer: data.installer });
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const handler = (event: WebSocketEvent) => {
      if (event.type !== 'update.available') return;
      setUpdate({ version: event.version, url: event.url, installer: event.installer });
    };
    ws.on('update.available', handler);
    return () => {
      ws.off('update.available', handler);
    };
  }, [ws]);

  if (!update || dismissed === update.version) return null;

  function handleDismiss() {
    if (!update) return;
    localStorage.setItem(DISMISS_KEY, update.version);
    setDismissed(update.version);
  }

  return (
    <div
      data-testid="update-pill"
      className="fixed bottom-4 left-4 z-50 flex items-center gap-1 rounded-full bg-card border border-border shadow-panel pl-3 pr-1.5 py-1.5"
    >
      <a
        href={updateHref(update, locale)}
        target="_blank"
        rel="noopener noreferrer"
        className="text-xs font-medium text-accent hover:underline"
      >
        {t('update.pill.label', { version: update.version })}
      </a>
      <button
        type="button"
        onClick={handleDismiss}
        aria-label={t('update.pill.dismiss')}
        data-testid="update-pill-dismiss"
        className="p-1 rounded-full text-secondary hover:text-primary hover:bg-border/50 transition-colors duration-150"
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  );
}
