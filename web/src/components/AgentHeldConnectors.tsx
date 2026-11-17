// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Agents' own connectors — below Orbital's connectors in Global Settings.
 *
 * The MCP servers an agent holds itself (Claude Code's claude.ai account
 * connectors, its Figma or Playwright servers, Codex's local ones), as the
 * agent last reported them while it ran (GET /api/v2/connectors/agent-held).
 * Only that agent can use them; the others hand it the task. So there is
 * nothing to set: each agent is its logo and the names, read-only, and one
 * hover sign carries the meaning. An agent never seen running is absent; with
 * none (or an older daemon) nothing renders at all.
 */

import { useEffect, useState } from 'react';
import { api } from '../config';
import type { AgentHeldConnectors as Row, AgentHeldConnectorsResponse } from '../types';
import { AgentLogo } from './history/AgentLogo';
import HintSign from './HintSign';
import { useT } from '../i18n/useT';

/** Statuses that mean the agent can call it now (or is connecting). */
const READY = new Set(['connected', 'pending']);

export default function AgentHeldConnectors() {
  const t = useT();
  const [rows, setRows] = useState<Row[]>([]);

  useEffect(() => {
    let cancelled = false;
    api<AgentHeldConnectorsResponse>('/api/v2/connectors/agent-held')
      .then((data) => {
        if (cancelled || !Array.isArray(data?.agents)) return;
        setRows(data.agents.filter((a) => Array.isArray(a.connectors) && a.connectors.length > 0));
      })
      .catch(() => {
        // An older daemon (404) or a failed load: show nothing.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (rows.length === 0) return null;

  return (
    <div
      data-testid="agent-held-connectors"
      className="surface-float-static relative mt-2 px-3 py-2.5"
    >
      <span className="absolute right-3 top-3">
        <HintSign label={t('connectors.agentHeld.hintAria')} align="right">
          {t('connectors.agentHeld.hint')}
        </HintSign>
      </span>
      <div className="space-y-2.5 pr-6">
        {rows.map((a) => (
          <div key={a.id} className="flex items-start gap-3" data-testid={`agent-held-${a.id}`}>
            <span className="mt-[5px] inline-flex shrink-0">
              <AgentLogo slug={a.id} title={a.name} />
            </span>
            <ul
              aria-label={t('connectors.agentHeld.listAria', { agent: a.name })}
              className="flex min-w-0 flex-1 flex-wrap gap-1.5"
            >
              {a.connectors.map((c) => {
                const ready = READY.has(c.status);
                return (
                  <li
                    key={c.raw_name}
                    data-ready={ready ? 'true' : 'false'}
                    className={`max-w-full truncate rounded-md bg-sidebar px-2 py-0.5 text-xs ${
                      ready ? 'text-primary' : 'text-secondary/70'
                    }`}
                  >
                    {c.name}
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
