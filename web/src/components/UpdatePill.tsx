// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useLayoutEffect, useState, useSyncExternalStore, type ReactNode } from 'react';
import { X } from 'lucide-react';
import type { UpdateInstallerLinks, WebSocketEvent } from '../types';
import { useWebSocket } from '../hooks/useWebSocket';
import { useUpdateInstall, type InstallState } from '../hooks/useUpdateInstall';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import { regionForLocale, type Locale } from '../i18n/locales';
import { isRelayMode } from '../config';

/**
 * Bottom-left "new version" pill.
 *
 * Sources: the daemon announces `update.available` once per version over the
 * global WS; a mount fetch of /api/v2/update-status covers page loads that
 * happen after the announce (and carries the in-app upgrade's `install`
 * state).
 *
 * One click upgrades in place when this app can update itself
 * (`install.can_install`: the packaged desktop app, signed on macOS, an
 * installed copy on Windows): download with a quiet percentage, verify,
 * restart on the new version (agent_os/updater.py). Otherwise, and as the
 * fallback after a failure, the click downloads this platform's installer in
 * the system browser (GitHub, or the China mirror for a zh UI), as before.
 * A phone on the relay never restarts the desktop app; it keeps the link.
 *
 * Dismissal is remembered PER VERSION in localStorage: never nags twice for
 * the same release, reappears for the next one.
 *
 * Where it sits: fixed over the bottom-left corner, which on every route that
 * shows the project list is the sidebar footer's connection line. The pill
 * takes that line's place while it shows: the Sidebar reads
 * `useUpdatePillShown()` and hides the line (keeping its box, so the footer
 * never changes height). A line that is NOT "connected" outranks the pill —
 * the pill steps aside so the warning stays visible — except while the
 * upgrade restarts the app, when the outage is the restart the user asked for
 * and "Restarting…" is the truthful label.
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

// Whether the pill is on screen, for the sidebar footer (see the header).
let pillShown = false;
const shownListeners = new Set<() => void>();

function setPillShown(next: boolean) {
  if (pillShown === next) return;
  pillShown = next;
  shownListeners.forEach((l) => l());
}

function subscribeShown(l: () => void) {
  shownListeners.add(l);
  return () => {
    shownListeners.delete(l);
  };
}

// eslint-disable-next-line react-refresh/only-export-components
export function useUpdatePillShown(): boolean {
  return useSyncExternalStore(subscribeShown, () => pillShown, () => false);
}

export type PillConnectionState = 'connected' | 'reconnecting' | 'disconnected' | 'daemon_offline';

function readDismissed(): string | null {
  try {
    return localStorage.getItem(DISMISS_KEY);
  } catch {
    return null;
  }
}

export default function UpdatePill({
  connectionState = 'connected',
}: {
  /** The sidebar's connection line state (App's mapConnectionState). */
  connectionState?: PillConnectionState;
} = {}) {
  const t = useT();
  const { locale } = useLocale();
  const ws = useWebSocket();
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [seed, setSeed] = useState<InstallState | null>(null);
  const [dismissed, setDismissed] = useState<string | null>(readDismissed);
  const { install, refused, start, reset } = useUpdateInstall(seed);

  useEffect(() => {
    fetch(`${API_BASE}/api/v2/update-status`)
      .then((r) => r.json())
      .then((data) => {
        if (data.update_available && data.latest) {
          setUpdate({ version: data.latest, url: data.url || '', installer: data.installer });
        }
        if (data.install) setSeed(data.install as InstallState);
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

  const phase = install?.state ?? 'idle';
  // A run (or its failure) from before a reload still shows, even when the
  // announce has not arrived yet.
  const version = update?.version || install?.version || '';
  const shown =
    !!version &&
    !(phase === 'idle' && (!update || dismissed === update.version)) &&
    (connectionState === 'connected' || phase === 'installing');
  // Layout effects, so the line and the pill swap in the same paint.
  useLayoutEffect(() => setPillShown(shown), [shown]);
  useLayoutEffect(() => () => setPillShown(false), []);
  if (!shown) return null;

  const href = updateHref(update ?? { url: '' }, locale);
  const inApp = !!install?.can_install && !isRelayMode && !refused;

  function handleDismiss() {
    try {
      localStorage.setItem(DISMISS_KEY, version);
    } catch {
      /* per-device convenience only */
    }
    setDismissed(version);
    setSeed(null);
    reset();
  }

  function handleStart() {
    void start(regionForLocale(locale));
  }

  const textBtn = 'text-xs font-medium text-accent hover:underline';
  const quiet = 'text-xs font-medium text-secondary tabular-nums';
  let body: ReactNode;
  let canDismiss = true;

  if (refused || phase === 'failed') {
    body = (
      <span className="flex items-center gap-2" data-testid="update-pill-failed">
        <span className="text-xs font-medium text-primary">{t('update.pill.failed')}</span>
        {inApp && (
          <button type="button" onClick={handleStart} className={textBtn}
            data-testid="update-pill-retry">
            {t('update.pill.retry')}
          </button>
        )}
        {href && (
          <a href={href} target="_blank" rel="noopener noreferrer" className={textBtn}
            data-testid="update-pill-manual">
            {t('update.pill.manual')}
          </a>
        )}
      </span>
    );
  } else if (phase === 'downloading') {
    canDismiss = false;
    body = (
      <span className={quiet} data-testid="update-pill-progress">
        {t('update.pill.downloading', { n: String(install?.progress ?? 0) })}
      </span>
    );
  } else if (phase === 'verifying') {
    canDismiss = false;
    body = <span className={quiet} data-testid="update-pill-status">{t('update.pill.verifying')}</span>;
  } else if (phase === 'installing') {
    canDismiss = false;
    body = <span className={quiet} data-testid="update-pill-status">{t('update.pill.restarting')}</span>;
  } else if (phase === 'ready' && inApp) {
    body = (
      <button type="button" onClick={handleStart} className={textBtn}
        data-testid="update-pill-action">
        {t('update.pill.ready')}
      </button>
    );
  } else if (inApp) {
    body = (
      <button type="button" onClick={handleStart} className={textBtn}
        data-testid="update-pill-action">
        {t('update.pill.install', { version })}
      </button>
    );
  } else {
    body = (
      <a href={href} target="_blank" rel="noopener noreferrer" className={textBtn}>
        {t('update.pill.label', { version })}
      </a>
    );
  }

  const progress = phase === 'downloading' ? Math.max(0, Math.min(100, install?.progress ?? 0)) : null;

  return (
    <div
      data-testid="update-pill"
      data-phase={refused ? 'failed' : phase}
      className={`fixed bottom-4 left-4 z-50 flex items-center gap-1 rounded-full bg-card border border-border shadow-panel overflow-hidden py-1.5 ${canDismiss ? 'pl-3 pr-1.5' : 'px-3'}`}
    >
      {body}
      {canDismiss && (
        <button
          type="button"
          onClick={handleDismiss}
          aria-label={t('update.pill.dismiss')}
          data-testid="update-pill-dismiss"
          className="p-1 rounded-full text-secondary hover:text-primary hover:bg-border/50 transition-colors duration-150"
        >
          <X className="w-3 h-3" />
        </button>
      )}
      {progress !== null && (
        <span
          aria-hidden="true"
          data-testid="update-pill-bar"
          className="absolute left-0 bottom-0 h-0.5 bg-accent transition-[width] duration-500"
          style={{ width: `${progress}%` }}
        />
      )}
    </div>
  );
}
