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
 */

import { useCallback, useEffect, useState } from 'react';
import { api, ApiError } from '../config';
import type { Connector, ConnectorListResponse } from '../types';
import { useT } from '../i18n/useT';
import BetaBadge from './BetaBadge';
import Select from './Select';

/** "google" → "Google" for the provider-scoped disconnect copy. Provider ids
 *  are backend slugs, not UI chrome — capitalizing is display-only. */
function providerLabel(provider: string): string {
  return provider ? provider.charAt(0).toUpperCase() + provider.slice(1) : provider;
}

function isCustom(c: Connector): boolean {
  return c.id.startsWith('custom-');
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

  const refresh = useCallback(async () => {
    try {
      const data = await api<ConnectorListResponse>('/api/v2/connectors');
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
      <h2 className="text-base font-semibold text-primary mb-1 flex items-center gap-2">
        {t('connectors.heading')}
        <BetaBadge />
      </h2>
      <p className="text-xs text-secondary mb-1">{t('connectors.global.hint')}</p>
      <p className="text-xs text-secondary/80 mb-3">{t('connectors.betaNote')}</p>

      {loadError && (
        <p className="text-xs text-error mb-2" role="alert">
          {t('connectors.loadError')}
        </p>
      )}

      {connectors === null ? (
        <p className="text-xs text-secondary/60 italic">{t('connectors.loading')}</p>
      ) : (
        <ul className="space-y-2">
          {connectors.map((c) => {
            const pending = c.status === 'pending_verification';
            const connecting = connectingId === c.id;
            return (
              <li
                key={c.id}
                data-testid={`connector-card-${c.id}`}
                className="border border-border rounded-lg px-3 py-2.5 bg-sidebar"
              >
                <div className="flex items-center gap-3">
                  <span className="text-xl shrink-0" aria-hidden="true">
                    {c.icon}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-primary truncate">
                        {c.name}
                      </span>
                      {pending && (
                        <span
                          data-testid={`connector-pending-${c.id}`}
                          className="shrink-0 text-2xs uppercase tracking-wide text-secondary border border-border rounded-full px-2 py-0.5"
                        >
                          {t('connectors.pendingBadge')}
                        </span>
                      )}
                    </div>
                    <p
                      className="text-xs text-secondary truncate"
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
                      <p className="text-xs text-error truncate">{c.enabled_error}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {c.connected ? (
                      <button
                        type="button"
                        data-testid={`connector-disconnect-${c.id}`}
                        onClick={() => setConfirmingId(c.id)}
                        className="text-xs font-medium text-secondary border border-border rounded-lg px-3 py-1.5 hover:text-error hover:border-error/40 transition-all duration-150 max-md:min-h-[44px]"
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
                          className="text-xs font-medium text-white bg-accent rounded-lg px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-60 max-md:min-h-[44px]"
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
                        className="text-xs font-medium text-secondary hover:text-error transition-colors px-1.5 py-1.5 max-md:min-h-[44px]"
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
                    <p className="text-sm font-medium text-primary">
                      {t('connectors.disconnectConfirm.title', {
                        provider: providerLabel(c.auth_provider),
                      })}
                    </p>
                    <p className="text-xs text-secondary mt-0.5 mb-2">
                      {t('connectors.disconnectConfirm.body', {
                        provider: providerLabel(c.auth_provider),
                      })}
                    </p>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        data-testid={`connector-disconnect-confirm-btn-${c.id}`}
                        onClick={() => handleDisconnect(c)}
                        className="text-xs font-medium text-white bg-error rounded-lg px-3 py-1.5 hover:bg-error/90 transition-all duration-150 max-md:min-h-[44px]"
                      >
                        {t('connectors.disconnectConfirm.confirm')}
                      </button>
                      <button
                        type="button"
                        onClick={() => setConfirmingId(null)}
                        className="text-xs font-medium text-secondary border border-border rounded-lg px-3 py-1.5 hover:bg-card-hover transition-all duration-150 max-md:min-h-[44px]"
                      >
                        {t('connectors.disconnectConfirm.cancel')}
                      </button>
                    </div>
                  </div>
                )}

                {cardError?.id === c.id && (
                  <div className="mt-2" role="alert" data-testid={`connector-error-${c.id}`}>
                    <p className="text-xs text-error">{cardError.message}</p>
                    {cardError.clientHint && (
                      <p className="text-xs text-secondary mt-0.5">
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

      {/* Custom MCP server (Tier-0, Spec 011 §3): a power-user escape hatch —
          paste a server URL, no catalog work required. */}
      <form onSubmit={handleAddCustom} className="mt-4 pt-3 border-t border-border">
        <label className="block text-sm font-medium text-primary mb-1">
          {t('connectors.custom.heading')}
        </label>
        <p className="text-xs text-secondary mb-2">{t('connectors.custom.hint')}</p>
        <div className="flex flex-col gap-2">
          <div className="flex gap-2 max-md:flex-col">
            <input
              type="text"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              placeholder={t('connectors.custom.name.placeholder')}
              aria-label={t('connectors.custom.name.label')}
              data-testid="connector-custom-name"
              className="w-40 max-md:w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
            />
            <input
              type="text"
              value={customUrl}
              onChange={(e) => setCustomUrl(e.target.value)}
              placeholder={t('connectors.custom.url.placeholder')}
              aria-label={t('connectors.custom.url.label')}
              data-testid="connector-custom-url"
              className="flex-1 text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
            />
            <Select
              value={customAuth}
              onChange={(e) => setCustomAuth(e.target.value as 'none' | 'oauth2')}
              aria-label={t('connectors.custom.auth.label')}
              data-testid="connector-custom-auth"
              className="text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
            >
              <option value="none">{t('connectors.custom.auth.none')}</option>
              <option value="oauth2">{t('connectors.custom.auth.oauth2')}</option>
            </Select>
          </div>
          {customError && (
            <p className="text-xs text-error" role="alert" data-testid="connector-custom-error">
              {customError}
            </p>
          )}
          <div>
            <button
              type="submit"
              data-testid="connector-custom-add"
              disabled={adding || !customName.trim() || !customUrl.trim()}
              className="text-xs font-medium text-white bg-accent rounded-lg px-4 py-2 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 max-md:min-h-[44px]"
            >
              {adding ? t('connectors.custom.adding') : t('connectors.custom.add')}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
