// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';

export interface SessionAsk {
  id: string;
  text: string;
  due: string | null;
  /** Choices the agent offered (2-4), tapped as the answer; [] when none. */
  options: string[];
  /** Markdown context under the question ("" when none). */
  detail: string;
  /** The goals card's "needs changes" option (batch 1110 item 2); absent otherwise. */
  change_option?: string | null;
}

/** The open asks this conversation opened. Refetched on `workbench.changed`
 *  (App re-dispatches it as a window event) and when `refreshKey` changes, so a
 *  dropped socket still catches up at the end of a turn. */
export function useSessionAsks(projectId: string, sessionId: string | undefined, refreshKey: unknown) {
  const [asks, setAsks] = useState<SessionAsk[]>([]);

  const refresh = useCallback(() => {
    if (sessionId === undefined) {
      setAsks([]);
      return;
    }
    api<{ asks?: SessionAsk[] }>(
      `/api/v2/workbench/${encodeURIComponent(projectId)}/session-asks?session_id=${encodeURIComponent(sessionId)}`,
    )
      .then((body) => setAsks(body?.asks ?? []))
      .catch(() => setAsks([]));
  }, [projectId, sessionId]);

  useEffect(() => {
    refresh();
    window.addEventListener('orbital:workbench-changed', refresh);
    return () => window.removeEventListener('orbital:workbench-changed', refresh);
  }, [refresh, refreshKey]);

  const close = useCallback(async (askId: string, kind: 'fulfilled' | 'irrelevant') => {
    setAsks((prev) => prev.filter((a) => a.id !== askId));
    await api(`/api/v2/workbench/${encodeURIComponent(projectId)}/entries/${encodeURIComponent(askId)}/exit`, {
      method: 'POST',
      body: JSON.stringify({ kind }),
    }).catch(() => refresh());
    window.dispatchEvent(new Event('orbital:workbench-changed'));
  }, [projectId, refresh]);

  /** The user just answered this ask from the composer: drop it on screen
   *  now; the inject route closes it and the next feed agrees. */
  const dismiss = useCallback((askId: string) => {
    setAsks((prev) => prev.filter((a) => a.id !== askId));
  }, []);

  return { asks, close, dismiss };
}
