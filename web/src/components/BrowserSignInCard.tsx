// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';
import { Globe, Loader2 } from 'lucide-react';
import { api } from '../config';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import { warmupUrlForLocale } from '../utils/warmupUrl';

/**
 * Global Settings card: open the headed sign-in browser at any time when no
 * session is running. Same open/poll pattern as the setup wizard's warmup
 * step; the backend refuses with 409 while a session runs, and that detail
 * is shown inline.
 */
export default function BrowserSignInCard() {
  const t = useT();
  const { locale } = useLocale();
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (pollRef.current !== null) clearInterval(pollRef.current);
  }, []);

  const handleOpenBrowser = useCallback(async () => {
    setOpening(true);
    setError(null);
    try {
      await api('/api/v2/platform/browser/warmup', {
        method: 'POST',
        body: JSON.stringify({ url: warmupUrlForLocale(locale) }),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : t('global.browserSignIn.openFailed'));
      setOpening(false);
      return;
    }
    // Poll until the user closes the sign-in browser
    pollRef.current = window.setInterval(async () => {
      try {
        const status = await api<{ active: boolean }>('/api/v2/platform/browser/warmup/status');
        if (!status.active) {
          if (pollRef.current !== null) clearInterval(pollRef.current);
          pollRef.current = null;
          setOpening(false);
        }
      } catch {
        // Status endpoint failed — keep polling
      }
    }, 2000);
  }, [t, locale]);

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <Globe className="w-4 h-4 text-accent" />
        <label className="block text-sm font-medium text-primary">
          {t('global.browserSignIn.title')}
        </label>
      </div>
      <p className="text-xs text-secondary mb-3">
        {t('global.browserSignIn.body')}
      </p>

      {error && (
        <div className="bg-error/10 border border-error/20 rounded-lg px-4 py-3 mb-3 text-sm text-error">
          {error}
        </div>
      )}

      {opening ? (
        <div className="inline-flex items-center gap-2 text-sm text-secondary">
          <Loader2 className="w-4 h-4 animate-spin" />
          {t('global.browserSignIn.opening')}
        </div>
      ) : (
        <button
          onClick={handleOpenBrowser}
          className="border border-border text-primary text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-surface transition-all duration-150"
        >
          {t('global.browserSignIn.open')}
        </button>
      )}
    </div>
  );
}
