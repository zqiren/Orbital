// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { api } from '../config';
import type { StringKey } from '../i18n/strings';
import { useT } from '../i18n/useT';

// ---------------------------------------------------------------------------
// Types — wire shape of /api/v2/settings/sub-agents
// ---------------------------------------------------------------------------

interface ParamSchemaEntry {
  /** Allowed values, or null for free-text. */
  allowed: string[] | null;
  /** Effective runtime value when no override has been persisted. */
  default?: string | null;
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

  // This daemon-level screen lists ONLY installed sub-agents. Agents whose
  // CLI isn't on the machine are hidden; the helper line tells the user how to
  // surface one. (The /settings/sub-agents payload still includes them — other
  // screens consume the full list — so we filter here in the frontend.)
  const installedEntries = entries === null
    ? null
    : entries.filter(e => e.installed === true);

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

      {installedEntries !== null && installedEntries.length === 0 && (
        <div className="text-sm text-secondary">
          {t('subAgentSettings.noInstalled')}
        </div>
      )}

      {installedEntries !== null && installedEntries.length > 0 && (
        <div className="flex flex-col gap-4">
          {installedEntries.map(entry => (
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
  // Local copy of the config so the form is editable before save.
  const [draft, setDraft] = useState<Record<string, string>>({ ...entry.config });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [loginBusy, setLoginBusy] = useState(false);
  const [logoutBusy, setLogoutBusy] = useState(false);
  const [apiKeyDraft, setApiKeyDraft] = useState('');
  const [showApiKeyForm, setShowApiKeyForm] = useState(false);

  const paramKeys = Object.keys(entry.param_schema);
  const isApiKeyFlow = API_KEY_LOGIN_SLUGS.has(entry.slug);
  // Hide the Login button when the daemon reports the agent has no interactive
  // login flow (API-key-only agents). Treat undefined as "supported" so older
  // daemon payloads keep the previous behavior. The API-key flow has its own
  // "Set API Key" button below, so it's never affected by this gate.
  const supportsLogin = entry.supports_login !== false;

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
            <h2 className="text-base font-medium text-primary">{entry.name}</h2>
            <code className="text-xs text-secondary/80 font-mono">{entry.slug}</code>
          </div>
          <div className="mt-1 flex items-center gap-3 text-xs">
            <StatusPill
              label={entry.installed ? t('subAgentCard.installed') : t('subAgentCard.notInstalled')}
              variant={entry.installed ? 'success' : 'warning'}
            />
            <StatusPill
              label={entry.credentials_configured ? t('subAgentCard.loggedIn') : t('subAgentCard.notLoggedIn')}
              variant={entry.credentials_configured ? 'success' : 'warning'}
            />
            {entry.version && (
              <span className="text-secondary/80 font-mono">v{entry.version}</span>
            )}
          </div>
        </div>
      </div>

      {!entry.installed && entry.missing_dependencies.length > 0 && (
        <p className="text-xs text-secondary mt-1 mb-2">
          {t('subAgentCard.missingDeps', { list: entry.missing_dependencies.join(', ') })}
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
        {entry.credentials_configured && (
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
                  <select
                    value={value}
                    onChange={e => handleChange(paramKey, e.target.value)}
                    disabled={!entry.installed}
                    className="text-sm bg-sidebar border border-border rounded px-2 py-1.5 text-primary focus:outline-none focus:border-accent disabled:opacity-50"
                  >
                    {!schema.default && (
                      <option value="">{t('subAgentCard.param.default')}</option>
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
                  </select>
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
