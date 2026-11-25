// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import type { NotReady } from '../utils/agentIssue';

/** The fields of GET /api/v2/settings/sub-agents the wizard's Agents step
 *  reads (Sub-agent settings keeps its own, fuller type). */
export interface SubAgentListEntry {
  slug: string;
  name: string;
  installed: boolean;
  credentials_configured: boolean;
  supports_login?: boolean;
  auth_expired?: boolean;
  override_path?: string | null;
  override_missing?: boolean;
  /** Installed, dependencies met, signed in, above its version floor, its
   *  handshake not failing, its sign-in not expired, no turn refused by its
   *  vendor. Absent on an older daemon. */
  ready?: boolean;
  /** Why not, when not ready: `{reason, detail}` (utils/agentIssue). Absent
   *  on an older daemon. */
  not_ready?: NotReady | null;
  /** The manifest declares an updater (the Update action). */
  supports_update?: boolean;
  /** The per-OS install command rides here as an `install_agent` action. */
  setup_actions?: Array<{ action: string; command: string | null }>;
}

export const isSignedIn = (e: SubAgentListEntry) => e.credentials_configured && e.auth_expired !== true;

/** The supported sub-agents, installed or not. `reload(true)` re-runs
 *  detection (POST …/refresh) instead of reading the cached list. */
export function useSubAgentList() {
  const [entries, setEntries] = useState<SubAgentListEntry[] | null>(null);
  const [error, setError] = useState(false);
  const reload = useCallback(async (refresh = false) => {
    try {
      const data = await api<SubAgentListEntry[]>(
        refresh ? '/api/v2/settings/sub-agents/refresh' : '/api/v2/settings/sub-agents',
        refresh ? { method: 'POST' } : undefined,
      );
      setEntries(Array.isArray(data) ? data : []);
      setError(false);
    } catch {
      setError(true);
      setEntries((prev) => prev ?? []);
    }
  }, []);
  useEffect(() => {
    let cancelled = false;
    api<SubAgentListEntry[]>('/api/v2/settings/sub-agents')
      .then((data) => { if (!cancelled) setEntries(Array.isArray(data) ? data : []); })
      .catch(() => {
        if (cancelled) return;
        setError(true);
        setEntries((prev) => prev ?? []);
      });
    return () => { cancelled = true; };
  }, []);
  return { entries, error, reload };
}
