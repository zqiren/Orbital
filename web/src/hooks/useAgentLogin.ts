// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The in-app sign-in for a coding agent: the same login job Sub-agent
 * settings starts (POST /settings/sub-agents/{slug}/login), with its progress
 * over the WebSocket — the browser link it printed, then complete or failed.
 * Lifted from the setup wizard's agent row so the Welcome page and the chat's
 * "signed out" notice start it the same way.
 */

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import { useT } from '../i18n/useT';
import { useWebSocket } from './useWebSocket';
import type { WebSocketEvent } from '../types';
import { pickLoginUrl } from '../utils/loginUrl';

export interface AgentLogin {
  /** The job is running (the browser is open, or opening). */
  busy: boolean;
  /** The sign-in link the CLI printed, once it did. */
  url: string | null;
  /** Why the last attempt failed, already translated. */
  failed: string | null;
  start: () => Promise<void>;
}

/** `onSignedIn` runs when the job completes (refetch the agent list). */
export function useAgentLogin(slug: string, onSignedIn: () => void): AgentLogin {
  const t = useT();
  const ws = useWebSocket();
  const [busy, setBusy] = useState(false);
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    const onProgress = (event: WebSocketEvent) => {
      if (event.type !== 'login.progress' || event.slug !== slug || event.kind === 'update') return;
      const found = pickLoginUrl(event.line);
      if (found) setUrl((prev) => prev ?? found);
    };
    const onComplete = (event: WebSocketEvent) => {
      if (event.type !== 'login.complete' || event.slug !== slug || event.kind === 'update') return;
      setBusy(false);
      setUrl(null);
      onSignedIn();
    };
    const onFailed = (event: WebSocketEvent) => {
      if (event.type !== 'login.failed' || event.slug !== slug || event.kind === 'update') return;
      setBusy(false);
      setUrl(null);
      setFailed(t('subAgentCard.loginFailed'));
    };
    ws.on('login.progress', onProgress);
    ws.on('login.complete', onComplete);
    ws.on('login.failed', onFailed);
    return () => {
      ws.off('login.progress', onProgress);
      ws.off('login.complete', onComplete);
      ws.off('login.failed', onFailed);
    };
  }, [ws, slug, onSignedIn, t]);

  const start = useCallback(async () => {
    setBusy(true);
    setFailed(null);
    setUrl(null);
    try {
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(slug)}/login`, { method: 'POST' });
    } catch (e) {
      setBusy(false);
      setFailed(e instanceof Error && e.message ? e.message : t('subAgentCard.loginFailed'));
    }
  }, [slug, t]);

  return { busy, url, failed, start };
}
