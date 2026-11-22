// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ConnectorSettings — the Global Settings → Connectors section
 * (Spec 011 §0.2/§0.6, Task E1).
 *
 * Card list from GET /api/v2/connectors: icon, name, and a status line —
 * "Connected as {account}" with a Disconnect action, a Connect button, or a
 * "Pending verification" badge (visible but NOT connectable, e.g. Gmail while
 * Google's restricted-scope review is in flight).
 *
 * Connect runs the backend's BLOCKING loopback OAuth flow (up to ~300 s while
 * the user completes browser consent), so the card shows a "waiting for
 * browser consent…" busy state. A 400 means the OAuth client isn't configured
 * — surface the BYO-client hint (settings keys connector_google_client_id /
 * connector_google_client_secret) alongside the backend detail.
 *
 * Disconnect is PROVIDER-scoped: tokens are stored per `auth_provider`, so
 * disconnecting any Google connector takes ALL Google connectors offline. The
 * inline confirm therefore names the provider ("Disconnect Google account?"),
 * never the single service.
 *
 * A "Custom MCP server" form (name + URL + auth type) covers Tier-0 servers;
 * custom entries (`custom-*` ids) get a Remove action.
 *
 * Per-project enablement is NOT here — it rides the project-update flow via
 * `enabled_connectors` (see ProjectConnectorToggles).
 *
 * Each CONNECTED card also carries who may use it (ConnectorAccessControl):
 * all agents by default, or selected ones. A change saves at once through
 * PUT /api/v2/connectors/{id}/access — optimistic, and put back to the last
 * saved value, with the API's error on the card, when the save fails.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from '../config';
import type {
  Connector,
  ConnectorAccess,
  ConnectorAgent,
  ConnectorAgentsResponse,
  ConnectorListResponse,
} from '../types';
import ConnectorAccessControl, { DEFAULT_ACCESS } from './ConnectorAccessControl';
import AgentHeldConnectors from './AgentHeldConnectors';
import { AgentLogo } from './history/AgentLogo';
import { useT } from '../i18n/useT';
import { LabelWithHint } from './SettingsSection';
import Select from './Select';

/** "google" → "Google" for the provider-scoped disconnect copy. Provider ids
 *  are backend slugs, not UI chrome — capitalizing is display-only. */
function providerLabel(provider: string): string {
  return provider ? provider.charAt(0).toUpperCase() + provider.slice(1) : provider;
}

function isCustom(c: Connector): boolean {
  return c.id.startsWith('custom-');
}

/**
 * The slot for a small "comes from" mark beside a connector's name: the
 * logo of the agent the connector came from. Nothing supplies an origin yet
 * (no API for it), so every row passes undefined and this renders nothing;
 * it exists so the row's layout already has the place when one does.
 */
export function ConnectorOrigin({ agent }: { agent?: string }) {
  if (!agent) return null;
  return (
    <span className="inline-flex shrink-0" data-testid="connector-origin">
      <AgentLogo slug={agent} />
    </span>
  );
}

interface CardError {
  id: string;
  /** Dynamic backend detail (rendered untranslated) or a translated fallback. */
  message: string;
  /** True on the 400 client-not-configured path — adds the BYO-client hint. */
  clientHint: boolean;
}

export default function ConnectorSettings() {
  const t = useT();
  // null = initial load in flight.
  const [connectors, setConnectors] = useState<Connector[] | null>(null);
  const [loadError, setLoadError] = useState(false);
  /** Connector id whose (blocking) connect call is in flight. */
  const [connectingId, setConnectingId] = useState<string | null>(null);
  /** Connector id showing the provider-scoped disconnect confirm. */
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [cardError, setCardError] = useState<CardError | null>(null);

  // Custom MCP server add form.
  const [customName, setCustomName] = useState('');
  const [customUrl, setCustomUrl] = useState('');
  const [customAuth, setCustomAuth] = useState<'none' | 'oauth2'>('none');
  const [adding, setAdding] = useState(false);
  const [customError, setCustomError] = useState('');

  // Agents a connector can be shared with; null = loading.
  const [agents, setAgents] = useState<ConnectorAgent[] | null | 'error'>(null);
  /** Last access value the daemon confirmed, per connector — what a failed
   *  save goes back to. */
  const savedAccess = useRef<Record<string, ConnectorAccess>>({});
  /** Latest access save per connector, so a slow earlier reply cannot
   *  overwrite a newer choice. */
  const accessSeq = useRef<Record<string, number>>({});

  const refresh = useCallback(async () => {
    try {
      const data = await api<ConnectorListResponse>('/api/v2/connectors');
      for (const c of data.connectors) savedAccess.current[c.id] = c.access ?? DEFAULT_ACCESS;
      setConnectors(data.connectors);
      setLoadError(false);
    } catch {
      setLoadError(true);
      // Keep whatever we already had; an empty list beats a spinner forever.
      setConnectors((prev) => prev ?? []);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    let cancelled = false;
    api<ConnectorAgentsResponse>('/api/v2/connectors/agents')
      .then((data) => {
        if (!cancelled) setAgents(Array.isArray(data?.agents) ? data.agents : []);
      })
      .catch(() => {
        if (!cancelled) setAgents('error');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function setAccess(id: string, access: ConnectorAccess) {
    setConnectors((prev) => prev?.map((x) => (x.id === id ? { ...x, access } : x)) ?? prev);
  }

  async function handleAccessChange(c: Connector, next: ConnectorAccess) {
    setCardError(null);
    const seq = (accessSeq.current[c.id] ?? 0) + 1;
    accessSeq.current[c.id] = seq;
    setAccess(c.id, next);
    try {
      const res = await api<{ id: string; access?: ConnectorAccess }>(
        `/api/v2/connectors/${encodeURIComponent(c.id)}/access`,
        { method: 'PUT', body: JSON.stringify(next) },
      );
      const saved = res?.access ?? next;
      savedAccess.current[c.id] = saved;
      if (accessSeq.current[c.id] === seq) setAccess(c.id, saved);
    } catch (e) {
      if (accessSeq.current[c.id] !== seq) return;
      setAccess(c.id, savedAccess.current[c.id] ?? DEFAULT_ACCESS);
      setCardError({
        id: c.id,
        // 422 carries FastAPI's validation list, not a sentence.
        message:
          e instanceof ApiError && e.status !== 422 && e.detail
            ? e.detail
            : t('connectors.access.saveError'),
        clientHint: false,
      });
    }
  }

  async function handleConnect(c: Connector) {
    setCardError(null);
    setConnectingId(c.id);
    try {
      // Blocking loopback flow — resolves only after browser consent (or
      // failure/timeout), so this await can legitimately take minutes.
      await api(`/api/v2/connectors/${encodeURIComponent(c.id)}/connect`, {
        method: 'POST',
      });
      await refresh();
    } catch (e) {
      if (e instanceof ApiError) {
        // 400 = OAuth client not configured (BYO-client hint); 409 = pending
        // verification; both carry a backend detail worth showing verbatim.
        setCardError({ id: c.id, message: e.detail, clientHint: e.status === 400 });
      } else {
        setCardError({ id: c.id, message: t('connectors.error.connectFailed'), clientHint: false });
      }
    } finally {
      setConnectingId(null);
    }
  }

  async function handleDisconnect(c: Connector) {
    setCardError(null);
    try {
      await api(`/api/v2/connectors/${encodeURIComponent(c.id)}/disconnect`, {
        method: 'POST',
      });
      setConfirmingId(null);
      await refresh();
    } catch (e) {
      setConfirmingId(null);
      setCardError({
        id: c.id,
        message: e instanceof ApiError ? e.detail : t('connectors.error.disconnectFailed'),
        clientHint: false,
      });
    }
  }

  async function handleRemoveCustom(c: Connector) {
    setCardError(null);
    try {
      await api(`/api/v2/connectors/custom/${encodeURIComponent(c.id)}`, {
        method: 'DELETE',
      });
      await refresh();
    } catch (e) {
      setCardError({
        id: c.id,
        message: e instanceof ApiError ? e.detail : t('connectors.error.disconnectFailed'),
        clientHint: false,
      });
    }
  }

  async function handleAddCustom(ev: React.FormEvent) {
    ev.preventDefault();
    if (!customName.trim() || !customUrl.trim()) return;
    setCustomError('');
    setAdding(true);
    try {
      await api('/api/v2/connectors/custom', {
        method: 'POST',
        body: JSON.stringify({
          name: customName.trim(),
          url: customUrl.trim(),
          auth_type: customAuth,
        }),
      });
      setCustomName('');
      setCustomUrl('');
      setCustomAuth('none');
      await refresh();
    } catch (e) {
      setCustomError(e instanceof ApiError ? e.detail : t('connectors.custom.error'));
    } finally {
      setAdding(false);
    }
  }

  return (
    <div data-testid="connector-settings">
      {/* Heading, Beta badge and the global hint are the enclosing
          SettingsSection's `title` / `suffix` / `description` (GlobalSettings),
          mirroring how project settings titles its own connectors section.
          Rendering them here too would print the section name twice, at two
          different weights. The beta caveat rides in that same ⓘ description:
          it is worth reading once, not on every visit. */}

      {loadError && (
        <p className="text-caption text-error mb-2" role="alert">
          {t('connectors.loadError')}
        </p>
      )}

      {connectors === null ? (
        <p className="text-caption text-secondary italic">{t('connectors.loading')}</p>
      ) : (
        <ul className="space-y-2">
          {connectors.map((c) => {
            const pending = c.status === 'pending_verification';
            const connecting = connectingId === c.id;
            return (
              <li
                key={c.id}
                data-testid={`connector-card-${c.id}`}
                className="surface-float-static relative px-3 py-2.5"
              >
                <div className="flex items-center gap-3">
                  <span className="text-heading shrink-0" aria-hidden="true">
                    {c.icon}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-body font-medium text-primary truncate">
                        {c.name}
                      </span>
                      {/* Reserved: where a connector came from (e.g. an
                          agent's own MCP server), as that agent's logo. No
                          source feeds it yet, so nothing renders. */}
                      <ConnectorOrigin agent={undefined} />
                      {pending && (
                        <span
                          data-testid={`connector-pending-${c.id}`}
                          className="shrink-0 type-label text-secondary border border-border rounded-full px-2 py-0.5"
                        >
                          {t('connectors.pendingBadge')}
                        </span>
                      )}
                    </div>
                    <p
                      className="text-caption text-secondary truncate"
                      data-testid={`connector-status-${c.id}`}
                    >
                      {c.connected
                        ? t('connectors.card.connectedAs', { account: c.account ?? '' })
                        : pending
                          ? t('connectors.pendingHint')
                          : t('connectors.card.notConnected')}
                    </p>
                    {/* Dynamic backend text (session failure etc.) — not UI chrome. */}
                    {c.enabled_error && (
                      <p className="text-caption text-error truncate">{c.enabled_error}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    {c.connected && (
                      <ConnectorAccessControl
                        connectorId={c.id}
                        connectorName={c.name}
                        access={c.access ?? DEFAULT_ACCESS}
                        agents={agents}
                        onChange={(next) => void handleAccessChange(c, next)}
                      />
                    )}
                    {c.connected ? (
                      <button
                        type="button"
                        data-testid={`connector-disconnect-${c.id}`}
                        onClick={() => setConfirmingId(c.id)}
                        className="text-caption font-medium text-secondary border border-border rounded-lg px-3 py-1.5 hover:text-error hover:border-error/40 transition-all duration-150 max-md:min-h-[44px]"
                      >
                        {t('connectors.disconnect')}
                      </button>
                    ) : (
                      !pending && (
                        <button
                          type="button"
                          data-testid={`connector-connect-${c.id}`}
                          disabled={connecting}
                          onClick={() => handleConnect(c)}
                          className="text-caption font-medium text-white bg-accent rounded-lg px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-60 max-md:min-h-[44px]"
                        >
                          {connecting ? t('connectors.connecting') : t('connectors.connect')}
                        </button>
                      )
                    )}
                    {isCustom(c) && (
                      <button
                        type="button"
                        data-testid={`connector-remove-${c.id}`}
                        aria-label={t('connectors.custom.removeAria', { name: c.name })}
                        onClick={() => handleRemoveCustom(c)}
                        className="text-caption font-medium text-secondary hover:text-error transition-colors px-1.5 py-1.5 max-md:min-h-[44px]"
                      >
                        {t('connectors.custom.remove')}
                      </button>
                    )}
                  </div>
                </div>

                {/* Provider-scoped disconnect confirm — names the account
                    provider, NOT this one service: one shared token, all of
                    the provider's connectors go offline together. */}
                {confirmingId === c.id && (
                  <div
                    data-testid={`connector-disconnect-confirm-${c.id}`}
                    className="mt-2 pt-2 border-t border-border"
                  >
                    <p className="text-body font-medium text-primary">
                      {t('connectors.disconnectConfirm.title', {
                        provider: providerLabel(c.auth_provider),
                      })}
                    </p>
                    <p className="text-caption text-secondary mt-0.5 mb-2">
                      {t('connectors.disconnectConfirm.body', {
                        provider: providerLabel(c.auth_provider),
                      })}
                    </p>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        data-testid={`connector-disconnect-confirm-btn-${c.id}`}
                        onClick={() => handleDisconnect(c)}
                        className="text-caption font-medium text-white bg-error rounded-lg px-3 py-1.5 hover:bg-error/90 transition-all duration-150 max-md:min-h-[44px]"
                      >
                        {t('connectors.disconnectConfirm.confirm')}
                      </button>
                      <button
                        type="button"
                        onClick={() => setConfirmingId(null)}
                        className="text-caption font-medium text-secondary border border-border rounded-lg px-3 py-1.5 hover:bg-card-hover transition-all duration-150 max-md:min-h-[44px]"
                      >
                        {t('connectors.disconnectConfirm.cancel')}
                      </button>
                    </div>
                  </div>
                )}

                {cardError?.id === c.id && (
                  <div className="mt-2" role="alert" data-testid={`connector-error-${c.id}`}>
                    <p className="text-caption text-error">{cardError.message}</p>
                    {cardError.clientHint && (
                      <p className="text-caption text-secondary mt-0.5">
                        {t('connectors.error.clientHint')}
                      </p>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {/* The connectors each agent holds itself: read-only, below Orbital's. */}
      <AgentHeldConnectors />

      {/* Custom MCP server (Tier-0, Spec 011 §3): a power-user escape hatch —
          paste a server URL, no catalog work required. */}
      <form onSubmit={handleAddCustom} className="mt-4 pt-3 border-t border-border">
        <LabelWithHint hint={t('connectors.custom.hint')}>
          {t('connectors.custom.heading')}
        </LabelWithHint>
        <div className="flex flex-col gap-2">
          <div className="flex gap-2 max-md:flex-col">
            <input
              type="text"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              placeholder={t('connectors.custom.name.placeholder')}
              aria-label={t('connectors.custom.name.label')}
              data-testid="connector-custom-name"
              className="w-40 max-md:w-full text-body bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary focus:outline-none focus:border-accent transition-all duration-150"
            />
            <input
              type="text"
              value={customUrl}
              onChange={(e) => setCustomUrl(e.target.value)}
              placeholder={t('connectors.custom.url.placeholder')}
              aria-label={t('connectors.custom.url.label')}
              data-testid="connector-custom-url"
              className="flex-1 text-body font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary focus:outline-none focus:border-accent transition-all duration-150"
            />
            <Select
              value={customAuth}
              onChange={(e) => setCustomAuth(e.target.value as 'none' | 'oauth2')}
              aria-label={t('connectors.custom.auth.label')}
              data-testid="connector-custom-auth"
              className="text-body bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
            >
              <option value="none">{t('connectors.custom.auth.none')}</option>
              <option value="oauth2">{t('connectors.custom.auth.oauth2')}</option>
            </Select>
          </div>
          {customError && (
            <p className="text-caption text-error" role="alert" data-testid="connector-custom-error">
              {customError}
            </p>
          )}
          <div>
            <button
              type="submit"
              data-testid="connector-custom-add"
              disabled={adding || !customName.trim() || !customUrl.trim()}
              className="text-caption font-medium text-white bg-accent rounded-lg px-4 py-2 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 max-md:min-h-[44px]"
            >
              {adding ? t('connectors.custom.adding') : t('connectors.custom.add')}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
