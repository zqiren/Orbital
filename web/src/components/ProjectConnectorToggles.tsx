// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ProjectConnectorToggles — the Project Settings → Connectors section
 * (Spec 011 §0.2/§0.6, Task E1).
 *
 * "Authenticate globally, enable per project": OAuth happens once in Global
 * Settings; each project then opts in per connector — the project is the
 * trust boundary. This component renders one switch per connector, writing
 * `enabled_connectors` THROUGH THE PARENT FORM (SettingsView holds the state
 * and includes it in the existing project-update save payload — per-project
 * enablement is deliberately not a connector route).
 *
 * Connectors that exist in the catalog but aren't connected globally render
 * disabled with a "connect it in Global Settings first" hint.
 */

import { useEffect, useState } from 'react';
import { api } from '../config';
import type { Connector, ConnectorListResponse } from '../types';
import { useT } from '../i18n/useT';
import BetaBadge from './BetaBadge';

export interface ProjectConnectorTogglesProps {
  /** Connector ids currently enabled for this project (form state, owned by
   *  SettingsView, saved with the rest of the project form). */
  enabledConnectors: string[];
  /** Called with the full next id list on every toggle. */
  onChange: (ids: string[]) => void;
}

export default function ProjectConnectorToggles({
  enabledConnectors,
  onChange,
}: ProjectConnectorTogglesProps) {
  const t = useT();
  // null = load in flight.
  const [connectors, setConnectors] = useState<Connector[] | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api<ConnectorListResponse>('/api/v2/connectors')
      .then((data) => {
        if (!cancelled) setConnectors(data.connectors);
      })
      .catch(() => {
        if (!cancelled) {
          setLoadError(true);
          setConnectors([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function handleToggle(id: string, enabled: boolean) {
    onChange(
      enabled
        ? enabledConnectors.includes(id)
          ? enabledConnectors
          : [...enabledConnectors, id]
        : enabledConnectors.filter((x) => x !== id),
    );
  }

  return (
    <div data-testid="project-connector-toggles">
      <label className="flex items-center gap-2 text-sm font-medium text-primary mb-1.5">
        {t('connectors.heading')}
        <BetaBadge />
      </label>
      <p className="text-xs text-secondary mb-2">{t('connectors.project.hint')}</p>

      {loadError && (
        <p className="text-xs text-error mb-2" role="alert">
          {t('connectors.loadError')}
        </p>
      )}

      {connectors === null ? (
        <p className="text-xs text-secondary/60 italic">{t('connectors.loading')}</p>
      ) : connectors.length === 0 ? (
        <p className="text-xs text-secondary/60 italic">{t('connectors.project.empty')}</p>
      ) : (
        <>
          <div className="flex flex-col gap-1">
            {connectors.map((c) => (
              <label
                key={c.id}
                className={`flex items-center gap-2 py-1 max-md:min-h-[44px] ${
                  c.connected ? 'cursor-pointer' : 'opacity-60 cursor-default'
                }`}
              >
                <input
                  type="checkbox"
                  checked={c.connected && enabledConnectors.includes(c.id)}
                  disabled={!c.connected}
                  onChange={(e) => handleToggle(c.id, e.target.checked)}
                  aria-label={t('connectors.project.toggleAria', { name: c.name })}
                  data-testid={`project-connector-toggle-${c.id}`}
                  className="rounded border-border accent-accent"
                />
                <span aria-hidden="true">{c.icon}</span>
                <span className="text-sm text-primary truncate">{c.name}</span>
                {!c.connected && (
                  <span
                    className="text-xs text-secondary italic truncate"
                    data-testid={`project-connector-hint-${c.id}`}
                  >
                    {t('connectors.project.connectFirst')}
                  </span>
                )}
              </label>
            ))}
          </div>
          <p className="text-[11px] text-secondary/60 mt-1.5 italic">
            {t('connectors.project.saveNote')}
          </p>
        </>
      )}
    </div>
  );
}
