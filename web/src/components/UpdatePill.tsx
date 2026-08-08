// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import { X } from 'lucide-react';
import type { WebSocketEvent } from '../types';
import { useWebSocket } from '../hooks/useWebSocket';
import { useT } from '../i18n/useT';

/**
 * Bottom-left "new version available" pill (notify-only update tier).
 *
 * Sources: the daemon announces `update.available` once per version over the
 * global WS; a mount fetch of /api/v2/update-status covers page loads that
 * happen after the announce. Clicking opens the GitHub release page in the
 * system browser — download/install stays manual (no auto-update until
 * signing lands).
 *
 * Dismissal is remembered PER VERSION in localStorage: never nags twice for
 * the same release, reappears for the next one.
 */

const API_BASE = import.meta.env.VITE_API_BASE || '';
const DISMISS_KEY = 'orbital.updateDismissed';

interface UpdateInfo {
  version: string;
  url: string;
}

export default function UpdatePill() {
  const t = useT();
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
          setUpdate({ version: data.latest, url: data.url || '' });
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const handler = (event: WebSocketEvent) => {
      if (event.type !== 'update.available') return;
      setUpdate({ version: event.version, url: event.url });
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
        href={update.url}
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
