// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import type { SessionListEntry } from '../types';

// useSessions
//
// Fetches the per-project session list (GET /api/v2/projects/{pid}/sessions)
// and exposes each entry — including the persistent `last_terminal_event`
// field — to the UI. Consumers render durable per-session indicators
// (warning glyph for errors, etc.) that survive WS disconnects and page
// reloads, since the backend stores the field outside of WS event flow.
//
// Phase 1B's SessionSidebar / SessionListItem are the primary consumers;
// this hook also stands alone for ad-hoc rendering (e.g. existing project
// view headers).
export function useSessions(projectId: string | null) {
  const [sessions, setSessions] = useState<SessionListEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!projectId) {
      setSessions([]);
      return [];
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api<SessionListEntry[]>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/sessions`,
      );
      setSessions(data);
      return data;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to list sessions';
      setError(msg);
      return [];
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { sessions, loading, error, refresh };
}
