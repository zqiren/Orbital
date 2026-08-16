// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../config';
import { useWebSocket } from '../hooks/useWebSocket';
import MessageAvatar from './MessageAvatar';
import type { StringKey } from '../i18n/strings';
import { useT } from '../i18n/useT';
import type { WebSocketEvent } from '../types';
import Select from './Select';

// ---------------------------------------------------------------------------
// Types — wire shape of /api/v2/settings/sub-agents
// ---------------------------------------------------------------------------

interface ParamSchemaEntry {
  /** Allowed values, or null for free-text. */
  allowed: string[] | null;
  /** Effective runtime value when no override has been persisted. */
  default?: string | null;
}

/** Orbital-managed install state for one agent. `supported` is false both for
 *  agents Orbital can't install at all and for a platform the install isn't
 *  wired up on yet. Optional for backward-compat with older daemons. */
interface InstallInfo {
  supported: boolean;
  state: 'installed' | 'installing' | 'failed' | 'not_installed';
  job_id?: string | null;
  /** Platforms the manifest declares an Orbital install for. Present only on
   *  daemons that echo it; it is what lets an unsupported platform explain
   *  itself instead of the agent silently vanishing from the list. */
  platforms?: string[];
}

/** One manifest-declared credential. Emitted per entry by newer daemons; when
 *  absent we fall back to `missing_credentials` (required-and-missing keys
 *  only), which is enough to offer the field that unblocks the agent. */
interface CredentialSpec {
  key: string;
  label?: string | null;
  type?: string | null;
  required?: boolean;
  configured?: boolean;
  /** True when the agent's own CLI ingests this credential (`codex login`),
   *  which is a different mechanism from a key Orbital holds. */
  has_setup_command?: boolean;
}

interface SubAgentEntry {
  slug: string;
  name: string;
  installed: boolean;
  binary_path: string | null;
  version: string | null;
  ready: boolean;
  dependencies_met: boolean;
  missing_dependencies: string[];
  credentials_configured: boolean;
  missing_credentials: string[];
  /** True when the agent exposes an interactive login (OAuth) flow. When
   *  false, the agent only accepts an API key — hide the Login button so it
   *  can't 400. Optional for backward-compat with older daemon payloads. */
  supports_login?: boolean;
  setup_actions: Array<{ action: string; label: string; command: string | null; blocking: boolean }>;
  config: Record<string, string>;
  param_schema: Record<string, ParamSchemaEntry>;
  /** Orbital-managed install; absent on daemons without the install route. */
  install?: InstallInfo;
  /** False for agents whose transport reports no tool rows at all — the card
   *  says so instead of letting the silence read as a hang. */
  emits_tool_activity?: boolean;
  /** Manifest-declared credentials, when the daemon reports them. */
  credentials?: CredentialSpec[];
}

// Map daemon param names (kebab-case) to camelCase request body fields.
const PARAM_REQUEST_KEY: Record<string, string> = {
  'model': 'model',
  'effort': 'effort',
  'permission-mode': 'permission_mode',
  'approval-mode': 'approval_mode',
};

// Friendly labels for known params (catalog keys, resolved via t() at render).
const PARAM_LABEL_KEY: Record<string, StringKey> = {
  'model': 'subAgentCard.param.model',
  'effort': 'subAgentCard.param.effort',
  'permission-mode': 'subAgentCard.param.permissionMode',
  'approval-mode': 'subAgentCard.param.approvalMode',
};

function paramOptionLabel(
  slug: string,
  paramKey: string,
  value: string,
  t: (key: StringKey) => string,
): string {
  if (slug === 'cursor' && paramKey === 'permission-mode') {
    if (value === 'auto') return t('subAgentCard.permission.autoDefault');
    if (value === 'ask') return t('subAgentCard.permission.ask');
  }
  // Provider model IDs and unknown provider-native option IDs are dynamic and
  // intentionally remain untranslated.
  return value;
}

// Sub-agents that accept an API key via stdin (vs. an interactive OAuth flow).
const API_KEY_LOGIN_SLUGS = new Set<string>(['codex']);

// The credential key that can be copied from the global LLM provider settings
// server-side. Keyed on the credential, not on a slug: any agent declaring it
// gets the checkbox, and the daemon 409s if the provider isn't DeepSeek.
const PROVIDER_KEY_CREDENTIAL = 'DEEPSEEK_API_KEY';

// Extra guidance for specific credential keys (not slugs — a key means the
// same thing wherever it is declared).
const CREDENTIAL_HINT_KEY: Record<string, StringKey> = {
  'DEEPSEEK_BASE_URL': 'subAgentCard.cred.baseUrlHint',
};

/** Whether this agent gets a row on the daemon-level list.
 *
 *  Installed agents always show. A not-installed agent shows only when Orbital
 *  can install it itself (or is mid-install / just failed one) — otherwise the
 *  row would offer nothing but a status pill, which is why the list was
 *  installed-only before the install route existed. */
function isVisibleEntry(entry: SubAgentEntry): boolean {
  if (entry.installed === true) return true;
  // A non-empty `platforms` list — echoed from the manifest — is the daemon's
  // "Orbital installs this agent" signal; `[]` means bring-your-own, which
  // stays hidden exactly as it was before the install route existed.
  // `supported` then narrows that to this host, and an excluded host gets an
  // explanation on the card rather than a vanished row.
  return (entry.install?.platforms?.length ?? 0) > 0;
}

/** The credentials this card can take a value for: manifest-declared secrets
 *  the agent's own CLI does not ingest. Falls back to the required-and-missing
 *  keys on daemons that don't declare credentials — enough to offer the field
 *  that unblocks the agent, though optional ones stay invisible there. */
function credentialFieldsFor(entry: SubAgentEntry): CredentialSpec[] {
  if (entry.credentials && entry.credentials.length > 0) {
    return entry.credentials.filter(
      c => (c.type ?? 'secret') === 'secret' && c.has_setup_command !== true,
    );
  }
  return (entry.missing_credentials ?? []).map(key => ({
    key,
    required: true,
    configured: false,
  }));
}

interface Props {
  /** When true, render as a standalone full-height page with header/back button.
   *  When false (default), render as an embeddable section without page chrome. */
  standalone?: boolean;
  onBack?: () => void;
}

export default function SubAgentSettings({ standalone = false, onBack }: Props) {
  const t = useT();
  const [entries, setEntries] = useState<SubAgentEntry[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api<SubAgentEntry[]>('/api/v2/settings/sub-agents');
      setEntries(data);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const data = await api<SubAgentEntry[]>(
        '/api/v2/settings/sub-agents/refresh',
        { method: 'POST' },
      );
      setEntries(data);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setRefreshing(false);
    }
  }, []);

  // This daemon-level screen lists installed sub-agents plus the ones Orbital
  // can install for the user. Agents whose CLI isn't on the machine and can't
  // be installed from here are hidden; the helper line tells the user how to
  // surface one. (The /settings/sub-agents payload still includes them — other
  // screens consume the full list — so we filter here in the frontend.)
  const visibleEntries = entries === null
    ? null
    : entries.filter(isVisibleEntry);

  const inner = (
    <>
      <div className="flex items-center justify-between mb-3">
        {standalone ? (
          <h1 className="text-xl font-semibold text-primary">{t('subAgentSettings.title')}</h1>
        ) : (
          <p className="text-sm text-secondary">
            {t('subAgentSettings.intro')}
          </p>
        )}
        <div className="flex items-center gap-2 ml-3 shrink-0">
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="text-sm text-secondary hover:text-primary transition-all duration-150 disabled:opacity-50"
          >
            {refreshing ? t('subAgentSettings.refreshing') : t('subAgentSettings.refresh')}
          </button>
          {standalone && onBack && (
            <button
              onClick={onBack}
              className="text-sm text-secondary hover:text-primary transition-all duration-150"
            >
              {t('global.back')}
            </button>
          )}
        </div>
      </div>

      <p className="text-xs text-secondary mb-1">
        {t('subAgentSettings.installHint')}
      </p>
      <p className="text-xs text-secondary mb-4 italic">
        {t('subAgentSettings.credNote')}
      </p>

      {error && (
        <div className="bg-warning/10 border border-warning/20 rounded-lg px-4 py-3 mb-4 text-sm text-warning">
          {error}
        </div>
      )}

      {loading && entries === null && (
        <div className="text-sm text-secondary">{t('subAgentSettings.loading')}</div>
      )}

      {visibleEntries !== null && visibleEntries.length === 0 && (
        <div className="text-sm text-secondary">
          {t('subAgentSettings.noInstalled')}
        </div>
      )}

      {visibleEntries !== null && visibleEntries.length > 0 && (
        <div className="flex flex-col gap-4">
          {visibleEntries.map(entry => (
            <SubAgentCard
              key={entry.slug}
              entry={entry}
              onChanged={load}
            />
          ))}
        </div>
      )}
    </>
  );

  if (standalone) {
    return (
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-[720px] mx-auto py-10 px-6 max-md:px-4">
          {inner}
        </div>
      </div>
    );
  }
  return <div>{inner}</div>;
}

// ---------------------------------------------------------------------------
// Single sub-agent card
// ---------------------------------------------------------------------------

interface CardProps {
  entry: SubAgentEntry;
  onChanged: () => void | Promise<void>;
}

function SubAgentCard({ entry, onChanged }: CardProps) {
  const t = useT();
  const ws = useWebSocket();
  // Local copy of the config so the form is editable before save.
  const [draft, setDraft] = useState<Record<string, string>>({ ...entry.config });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [loginBusy, setLoginBusy] = useState(false);
  const [logoutBusy, setLogoutBusy] = useState(false);
  const [apiKeyDraft, setApiKeyDraft] = useState('');
  const [showApiKeyForm, setShowApiKeyForm] = useState(false);
  // Install: `installing` starts from the entry so a page refresh mid-install
  // resumes the progress display without waiting for the next WS line.
  const [installing, setInstalling] = useState(entry.install?.state === 'installing');
  const [installBusy, setInstallBusy] = useState(false);
  const [installLine, setInstallLine] = useState<string | null>(null);
  const [installError, setInstallError] = useState<string | null>(null);

  const paramKeys = Object.keys(entry.param_schema);
  const isApiKeyFlow = API_KEY_LOGIN_SLUGS.has(entry.slug);
  // Hide the Login button when the daemon reports the agent has no interactive
  // login flow (API-key-only agents). Treat undefined as "supported" so older
  // daemon payloads keep the previous behavior. The API-key flow has its own
  // "Set API Key" button below, so it's never affected by this gate.
  const supportsLogin = entry.supports_login !== false;

  const credentialFields = useMemo(() => credentialFieldsFor(entry), [entry]);
  // Orbital holds the key itself for agents with no login flow and no CLI
  // credential store of their own. Driven by the entry's declared credentials,
  // never by a slug list — the codex stdin path above is a different mechanism
  // and keeps its own branch.
  const usesManagedCredentials =
    !supportsLogin && !isApiKeyFlow && credentialFields.length > 0;

  const canInstall = entry.install?.supported === true;
  const installUnsupported = entry.install !== undefined && !canInstall;
  const installFailed = installError !== null || entry.install?.state === 'failed';

  // Keep the local install view in step with a refetched entry (resume after
  // refresh, and the settled state once a job ends).
  const entryInstallState = entry.install?.state;
  useEffect(() => {
    if (entryInstallState === 'installing') setInstalling(true);
    else if (entryInstallState === 'installed') setInstalling(false);
  }, [entryInstallState]);

  // Install job events are daemon-global and carry the slug, so one handler per
  // card filtering on it is enough — the route 409s a second job for the same
  // slug, so there is never more than one in flight to confuse.
  useEffect(() => {
    const onProgress = (event: WebSocketEvent) => {
      if (event.type !== 'sub_agent_install_progress') return;
      if (event.slug !== entry.slug) return;
      setInstalling(true);
      setInstallError(null);
      setInstallLine(event.line);
    };
    const onDone = (event: WebSocketEvent) => {
      if (event.type !== 'sub_agent_install_done') return;
      if (event.slug !== entry.slug) return;
      setInstalling(false);
      setInstallLine(null);
      setInstallError(null);
      void onChanged();
    };
    const onFailed = (event: WebSocketEvent) => {
      if (event.type !== 'sub_agent_install_failed') return;
      if (event.slug !== entry.slug) return;
      setInstalling(false);
      setInstallError(event.error);
    };
    ws.on('sub_agent_install_progress', onProgress);
    ws.on('sub_agent_install_done', onDone);
    ws.on('sub_agent_install_failed', onFailed);
    return () => {
      ws.off('sub_agent_install_progress', onProgress);
      ws.off('sub_agent_install_done', onDone);
      ws.off('sub_agent_install_failed', onFailed);
    };
  }, [ws, entry.slug, onChanged]);

  const handleInstall = async () => {
    setInstallBusy(true);
    setInstallError(null);
    setInstallLine(null);
    try {
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/install`, {
        method: 'POST',
      });
      // 202 — the job streams over the WS from here.
      setInstalling(true);
    } catch (e) {
      setInstallError(e instanceof Error ? e.message : String(e));
    } finally {
      setInstallBusy(false);
    }
  };

  const handleChange = (paramKey: string, value: string) => {
    setSaved(false);
    setDraft(d => ({ ...d, [paramKey]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    setActionError(null);
    try {
      // Translate kebab-case keys into the body's snake_case field names.
      const body: Record<string, string> = {};
      for (const [k, v] of Object.entries(draft)) {
        const reqKey = PARAM_REQUEST_KEY[k] ?? k.replace(/-/g, '_');
        body[reqKey] = v;
      }
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/config`, {
        method: 'PUT',
        body: JSON.stringify(body),
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      await onChanged();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const handleLogin = async () => {
    setLoginBusy(true);
    setActionError(null);
    try {
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/login`, {
        method: 'POST',
      });
      // Login progress streams via WebSocket; we just kick the job off.
      // Refresh after a short delay to pick up the new auth state.
      setTimeout(() => { void onChanged(); }, 1500);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoginBusy(false);
    }
  };

  const handleLogout = async () => {
    setLogoutBusy(true);
    setActionError(null);
    try {
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/logout`, {
        method: 'POST',
      });
      await onChanged();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setLogoutBusy(false);
    }
  };

  const handleApiKeySubmit = async () => {
    if (!apiKeyDraft.trim()) return;
    setLoginBusy(true);
    setActionError(null);
    try {
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/api-key`, {
        method: 'POST',
        body: JSON.stringify({ api_key: apiKeyDraft }),
      });
      setApiKeyDraft('');
      setShowApiKeyForm(false);
      await onChanged();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoginBusy(false);
    }
  };

  return (
    <div className="bg-sidebar/30 border border-border rounded-lg p-4">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div>
          <div className="flex items-center gap-2">
            <MessageAvatar variant="agent" agentHandle={entry.slug} />
            <h2 className="text-base font-medium text-primary">{entry.name}</h2>
            <code className="text-xs text-secondary/80 font-mono">{entry.slug}</code>
          </div>
          <div className="mt-1 flex items-center gap-3 text-xs">
            <StatusPill
              label={entry.installed ? t('subAgentCard.installed') : t('subAgentCard.notInstalled')}
              variant={entry.installed ? 'success' : 'warning'}
            />
            <StatusPill
              label={usesManagedCredentials
                ? (entry.credentials_configured ? t('subAgentCard.keySaved') : t('subAgentCard.keyNeeded'))
                : (entry.credentials_configured ? t('subAgentCard.loggedIn') : t('subAgentCard.notLoggedIn'))}
              variant={entry.credentials_configured ? 'success' : 'warning'}
            />
            {entry.version && (
              <span className="text-secondary/80 font-mono">v{entry.version}</span>
            )}
          </div>
        </div>
      </div>

      {entry.emits_tool_activity === false && (
        <p
          data-testid={`sub-agent-silent-note-${entry.slug}`}
          className="text-xs text-secondary mt-2"
        >
          {t('subAgentCard.silentUntilDone')}
        </p>
      )}

      {!entry.installed && entry.missing_dependencies.length > 0 && (
        <p className="text-xs text-secondary mt-1 mb-2">
          {t('subAgentCard.missingDeps', { list: entry.missing_dependencies.join(', ') })}
        </p>
      )}

      {/* Orbital-managed install */}
      {!entry.installed && canInstall && (
        <div
          data-testid={`sub-agent-install-${entry.slug}`}
          className="flex flex-col gap-2 mt-3 mb-3"
        >
          {installing ? (
            <div className="flex items-center gap-2 text-xs text-secondary">
              <span
                data-testid={`sub-agent-install-spinner-${entry.slug}`}
                className="inline-block h-3 w-3 rounded-full border-2 border-accent border-t-transparent animate-spin"
                aria-hidden="true"
              />
              <span>{t('subAgentCard.installing')}</span>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={handleInstall}
                disabled={installBusy}
                className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
              >
                {installBusy
                  ? t('subAgentCard.installStarting')
                  : installFailed
                    ? t('subAgentCard.installRetry')
                    : t('subAgentCard.install')}
              </button>
              <span className="text-xs text-secondary">{t('subAgentCard.installHint')}</span>
            </div>
          )}
          {installLine && (
            // Raw installer (npm) output — dynamic, never translated.
            <p
              data-testid={`sub-agent-install-progress-${entry.slug}`}
              className="text-[11px] font-mono text-secondary/80 truncate"
            >
              {installLine}
            </p>
          )}
          {installError ? (
            <p
              data-testid={`sub-agent-install-error-${entry.slug}`}
              className="text-xs text-warning"
            >
              {t('subAgentCard.installFailed', { error: installError })}
            </p>
          ) : entryInstallState === 'failed' ? (
            // A job that failed before this page loaded: the entry keeps the
            // outcome but not the message, so say that much rather than
            // offering a bare Retry with no explanation.
            <p
              data-testid={`sub-agent-install-error-${entry.slug}`}
              className="text-xs text-warning"
            >
              {t('subAgentCard.installFailedEarlier')}
            </p>
          ) : null}
        </div>
      )}

      {!entry.installed && installUnsupported && (
        <p
          data-testid={`sub-agent-install-unsupported-${entry.slug}`}
          className="text-xs text-secondary mt-3 mb-3"
        >
          {t('subAgentCard.installUnsupported')}
        </p>
      )}

      {/* Login / Logout */}
      <div className="flex flex-wrap items-center gap-2 mt-3 mb-3">
        {!entry.credentials_configured && !isApiKeyFlow && supportsLogin && (
          <button
            onClick={handleLogin}
            disabled={loginBusy || !entry.installed}
            className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
          >
            {loginBusy ? t('subAgentCard.startingLogin') : t('subAgentCard.login')}
          </button>
        )}
        {!entry.credentials_configured && isApiKeyFlow && (
          <button
            onClick={() => setShowApiKeyForm(s => !s)}
            disabled={!entry.installed}
            className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
          >
            {t('subAgentCard.setApiKey')}
          </button>
        )}
        {/* Logout drives the agent's own CLI logout command; agents whose key
            Orbital holds have a per-credential Remove instead. */}
        {entry.credentials_configured && !usesManagedCredentials && (
          <button
            onClick={handleLogout}
            disabled={logoutBusy}
            className="text-xs bg-sidebar border border-border text-primary rounded px-3 py-1.5 hover:bg-sidebar/80 transition-all duration-150 disabled:opacity-50"
          >
            {logoutBusy ? t('subAgentCard.loggingOut') : t('subAgentCard.logout')}
          </button>
        )}
      </div>

      {showApiKeyForm && isApiKeyFlow && (
        <div className="bg-sidebar/40 border border-border rounded p-3 mb-3 flex flex-col gap-2">
          <label className="text-xs text-secondary">
            {t('subAgentCard.apiKeyHint', { slug: entry.slug })}
          </label>
          <input
            type="password"
            value={apiKeyDraft}
            onChange={e => setApiKeyDraft(e.target.value)}
            placeholder={t('llm.apiKey.placeholder')}
            className="w-full text-sm font-mono bg-sidebar border border-border rounded px-2 py-1.5 text-primary focus:outline-none focus:border-accent"
          />
          <div className="flex gap-2">
            <button
              onClick={handleApiKeySubmit}
              disabled={loginBusy || !apiKeyDraft.trim()}
              className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 disabled:opacity-50"
            >
              {loginBusy ? t('subAgentCard.sending') : t('subAgentCard.submit')}
            </button>
            <button
              onClick={() => { setShowApiKeyForm(false); setApiKeyDraft(''); }}
              className="text-xs text-secondary hover:text-primary"
            >
              {t('subAgentCard.cancel')}
            </button>
          </div>
        </div>
      )}

      {usesManagedCredentials && (
        <ManagedCredentials
          slug={entry.slug}
          fields={credentialFields}
          onChanged={onChanged}
        />
      )}

      {/* Configuration */}
      {paramKeys.length > 0 ? (
        <div className="border-t border-border pt-3 mt-1 flex flex-col gap-3">
          {paramKeys.map(paramKey => {
            const schema = entry.param_schema[paramKey];
            const value = draft[paramKey] ?? schema.default ?? '';
            return (
              <div key={paramKey} className="flex flex-col gap-1">
                <label className="text-xs font-medium text-primary">
                  {PARAM_LABEL_KEY[paramKey] ? t(PARAM_LABEL_KEY[paramKey]) : paramKey}
                </label>
                {schema.allowed ? (
                  <Select
                    value={value}
                    onChange={e => handleChange(paramKey, e.target.value)}
                    disabled={!entry.installed}
                    className="text-sm bg-sidebar border border-border rounded px-2 py-1.5 text-primary focus:outline-none focus:border-accent disabled:opacity-50"
                  >
                    {!schema.default && (
                      <option value="">{t('subAgentCard.param.default')}</option>
                    )}
                    {/* A saved value absent from the (possibly live-populated)
                        list would render the select BLANK — keep it visible so
                        the user can see the stale override they're replacing.
                        Saving it again is rejected server-side. */}
                    {value !== '' && !schema.allowed.includes(value) && (
                      <option value={value}>{value}</option>
                    )}
                    {schema.allowed.map(opt => (
                      <option key={opt} value={opt}>
                        {paramOptionLabel(
                          entry.slug,
                          paramKey,
                          opt,
                          t,
                        )}
                      </option>
                    ))}
                  </Select>
                ) : (
                  <input
                    type="text"
                    value={value}
                    onChange={e => handleChange(paramKey, e.target.value)}
                    disabled={!entry.installed}
                    placeholder={t('subAgentCard.param.cliDefault')}
                    className="text-sm font-mono bg-sidebar border border-border rounded px-2 py-1.5 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent disabled:opacity-50"
                  />
                )}
                {entry.slug === 'cursor' && paramKey === 'permission-mode' && (
                  <p className="text-xs text-secondary">
                    {t('subAgentCard.permission.cursorHint')}
                  </p>
                )}
              </div>
            );
          })}
          <div className="flex items-center gap-2 pt-1">
            <button
              onClick={handleSave}
              disabled={saving || !entry.installed}
              className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
            >
              {saving ? t('subAgentCard.saving') : t('subAgentCard.save')}
            </button>
            {saved && <span className="text-xs text-success">{t('subAgentCard.saved')}</span>}
          </div>
        </div>
      ) : (
        <p className="text-xs text-secondary border-t border-border pt-3 mt-1">
          {t('subAgentCard.noParams')}
        </p>
      )}

      {actionError && (
        <p className="text-xs text-warning mt-2">{actionError}</p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Managed credentials — write-only key fields for agents Orbital holds the
// credential for (no interactive login, no CLI credential store of their own).
// Values go out over POST /credential and never come back: the card shows
// "stored" and a Remove button, never key text.
// ---------------------------------------------------------------------------

interface ManagedCredentialsProps {
  slug: string;
  fields: CredentialSpec[];
  onChanged: () => void | Promise<void>;
}

function ManagedCredentials({ slug, fields, onChanged }: ManagedCredentialsProps) {
  const t = useT();
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [useProviderKey, setUseProviderKey] = useState(false);
  // The server-side copy path only means anything when the global LLM
  // provider is DeepSeek with a key saved — otherwise the daemon 409s. Hide
  // the checkbox entirely when it can't work; on fetch failure it stays
  // hidden and manual paste remains.
  const [providerKeyAvailable, setProviderKeyAvailable] = useState(false);
  useEffect(() => {
    let cancelled = false;
    // Promise.resolve: harness-proof against api() doubles that return a bare
    // value (single-shot test mocks fall through to undefined) — an unhandled
    // .then here is a teardown-timing flake, not a render failure.
    Promise.resolve(api<{ llm?: { provider?: string; api_key_set?: boolean } }>('/api/v2/settings'))
      .then(d => {
        if (cancelled) return;
        setProviderKeyAvailable(d?.llm?.provider === 'deepseek' && d?.llm?.api_key_set === true);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const setError = (key: string, message: string) =>
    setErrors(prev => ({ ...prev, [key]: message }));

  const handleSave = async (key: string) => {
    const viaProvider = key === PROVIDER_KEY_CREDENTIAL && useProviderKey;
    const value = (drafts[key] ?? '').trim();
    if (!viaProvider && !value) return;
    setBusyKey(key);
    setError(key, '');
    try {
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(slug)}/credential`, {
        method: 'POST',
        // The provider-key path copies the global LLM key server-side, so the
        // client never has to hold text it can't read back. The daemon 409s
        // when that provider isn't DeepSeek or has no key.
        body: JSON.stringify(viaProvider ? { key, use_llm_provider_key: true } : { key, value }),
      });
      setDrafts(d => ({ ...d, [key]: '' }));
      setUseProviderKey(false);
      await onChanged();
    } catch (e) {
      setError(key, e instanceof Error ? e.message : String(e));
    } finally {
      setBusyKey(null);
    }
  };

  const handleRemove = async (key: string) => {
    setBusyKey(key);
    setError(key, '');
    try {
      await api(
        `/api/v2/settings/sub-agents/${encodeURIComponent(slug)}/credential/${encodeURIComponent(key)}`,
        { method: 'DELETE' },
      );
      await onChanged();
    } catch (e) {
      setError(key, e instanceof Error ? e.message : String(e));
    } finally {
      setBusyKey(null);
    }
  };

  return (
    <div
      data-testid={`sub-agent-credentials-${slug}`}
      className="border-t border-border pt-3 mt-1 flex flex-col gap-3"
    >
      <div>
        <h3 className="text-xs font-medium text-primary">{t('subAgentCard.cred.heading')}</h3>
        <p className="text-xs text-secondary">{t('subAgentCard.cred.note')}</p>
      </div>

      {fields.map(field => {
        const isProviderKey = field.key === PROVIDER_KEY_CREDENTIAL;
        const viaProvider = isProviderKey && useProviderKey;
        const busy = busyKey === field.key;
        const hintKey = CREDENTIAL_HINT_KEY[field.key];
        const draft = drafts[field.key] ?? '';
        return (
          <div key={field.key} className="flex flex-col gap-1">
            <label htmlFor={`cred-${slug}-${field.key}`} className="text-xs font-medium text-primary">
              {/* Manifest-supplied label — backend copy, left untranslated. */}
              {field.label || field.key}{' '}
              <span className="font-normal text-secondary">
                {field.required === false
                  ? t('subAgentCard.cred.optional')
                  : t('subAgentCard.cred.required')}
              </span>
            </label>

            {field.configured && (
              <div className="flex items-center gap-2">
                <StatusPill label={t('subAgentCard.cred.stored')} variant="success" />
                <button
                  data-testid={`sub-agent-cred-remove-${field.key}`}
                  onClick={() => handleRemove(field.key)}
                  disabled={busy}
                  className="text-xs text-secondary hover:text-primary disabled:opacity-50"
                >
                  {t('subAgentCard.cred.remove')}
                </button>
              </div>
            )}

            {isProviderKey && providerKeyAvailable && (
              <label className="flex items-center gap-2 text-xs text-secondary">
                <input
                  type="checkbox"
                  data-testid={`sub-agent-cred-provider-${slug}`}
                  checked={useProviderKey}
                  onChange={e => setUseProviderKey(e.target.checked)}
                />
                {t('subAgentCard.cred.useProviderKey')}
              </label>
            )}

            <input
              id={`cred-${slug}-${field.key}`}
              data-testid={`sub-agent-cred-input-${field.key}`}
              type="password"
              value={draft}
              disabled={viaProvider}
              onChange={e => setDrafts(d => ({ ...d, [field.key]: e.target.value }))}
              placeholder={t('subAgentCard.cred.placeholder')}
              className="text-sm font-mono bg-sidebar border border-border rounded px-2 py-1.5 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent disabled:opacity-50"
            />

            {hintKey && <p className="text-xs text-secondary">{t(hintKey)}</p>}

            <div className="flex items-center gap-2 pt-1">
              <button
                data-testid={`sub-agent-cred-save-${field.key}`}
                onClick={() => handleSave(field.key)}
                disabled={busy || (!viaProvider && !draft.trim())}
                className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
              >
                {busy ? t('subAgentCard.sending') : t('subAgentCard.save')}
              </button>
            </div>

            {errors[field.key] && (
              <p
                data-testid={`sub-agent-cred-error-${field.key}`}
                className="text-xs text-warning"
              >
                {errors[field.key]}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function StatusPill({ label, variant }: { label: string; variant: 'success' | 'warning' }) {
  const cls = variant === 'success'
    ? 'bg-success/10 text-success border-success/20'
    : 'bg-warning/10 text-warning border-warning/20';
  return (
    <span className={`px-2 py-0.5 rounded-full border text-[11px] ${cls}`}>
      {label}
    </span>
  );
}
