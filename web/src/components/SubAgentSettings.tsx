// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../config';
import { useWebSocket } from '../hooks/useWebSocket';
import MessageAvatar from './MessageAvatar';
import NotDetectedRow from './NotDetectedRow';
import type { StringKey } from '../i18n/strings';
import { useT } from '../i18n/useT';
import { LabelWithHint } from './SettingsSection';
import type { WebSocketEvent } from '../types';
import Select from './Select';
import { pickLoginUrl } from '../utils/loginUrl';
import { SendDiagnosticsLink } from './SendFeedback';

// ---------------------------------------------------------------------------
// Types — wire shape of /api/v2/settings/sub-agents
// ---------------------------------------------------------------------------

interface ParamSchemaEntry {
  /** Allowed values, or null for free-text. */
  allowed: string[] | null;
  /** Effective runtime value when no override has been persisted. */
  default?: string | null;
  /** Display names for ``allowed`` values, when the agent publishes them
   *  (claude-code: the CLI's own model list, e.g. opus → "Opus 5.5").
   *  Absent on older daemons; unlabeled values render as-is. */
  labels?: Record<string, string> | null;
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
  /** The agent's own app holds this key (DSH Desktop) and uses it over
   *  anything Orbital injects: the card names the owner instead of offering
   *  a field whose value would be ignored. */
  stored_by?: string | null;
}

/** Which copy of the agent Orbital drives. `desktop_app`: the CLI inside a
 *  vendor app (DeepSeek Harness, ChatGPT); `override`: the user's pick. */
interface SourceInfo {
  kind: 'desktop_app' | 'override' | 'path' | 'auto_detect' | 'check_command';
  label?: string | null;
  app_path?: string | null;
  app_version?: string | null;
}

/** ACP agents: did the binary answer `initialize`? Only `failed` blocks. */
interface HandshakeInfo {
  state: 'ok' | 'failed' | 'unknown';
  detail?: string | null;
}

/** The vendor app that ships this agent, and where to download it. */
interface DesktopAppInfo {
  label: string;
  download_url?: string | null;
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
  /** Spec 116 D4: the installed CLI is older than `min_version`, the oldest
   *  version Orbital can drive. The card then shows the Update button
   *  (`supports_update`: the manifest declares an updater) instead of Login. */
  update_needed?: boolean;
  min_version?: string | null;
  supports_update?: boolean;
  /** The locate card: the launcher the user pasted (null when detection
   *  found the agent on its own), and whether that file has since gone. */
  override_path?: string | null;
  override_missing?: boolean;
  /** A dispatch failed with an authentication error while the CLI's status
   *  probe still says signed in (it reads the local token file and cannot see
   *  expiry). The card says "Sign-in expired" and offers Login. */
  auth_expired?: boolean;
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
  /** For an agent whose credential follows its configured model's provider
   *  (Pi): a credential-card key or the CLI's own login. `unknown` is still
   *  dispatchable, but only the CLI can confirm it. Absent for agents whose
   *  credential is a login or a key Orbital holds. */
  credential_state?: CredentialState | null;
  source?: SourceInfo | null;
  handshake?: HandshakeInfo | null;
  desktop_app?: DesktopAppInfo | null;
}

type CredentialState = 'configured' | 'missing' | 'unknown';

const CREDENTIAL_STATE_KEY: Record<CredentialState, StringKey> = {
  configured: 'subAgentCard.credState.configured',
  missing: 'subAgentCard.credState.missing',
  unknown: 'subAgentCard.credState.unknown',
};

/** Per-agent note under the card header, for agents whose runtime differs
 *  from the sandboxed default in a way the user has to know about. */
const HOST_NOTE_KEY: Record<string, StringKey> = {
  dsh: 'subAgentCard.dsh.providersNote',
  pi: 'subAgentCard.pi.hostNote',
  'minimax-code': 'subAgentCard.minimaxCode.signInNote',
};

/** What to run by hand when the in-app updater leaves the CLI below the
 *  floor (spec 116 D3: a pre-2.1.288 npm updater can "succeed" with only the
 *  placeholder stub installed). Dynamic command text, never translated. */
const MANUAL_UPDATE_COMMAND: Record<string, string> = {
  'claude-code': 'npm install -g @anthropic-ai/claude-code@latest',
};

/** An Update terminal whose re-probe found no runnable CLI (1018 bench item
 *  5). `reason` / `manual_command` are additive on `login.failed`; older
 *  daemons send neither, so this is false for them. */
function isCliMissingEvent(event: WebSocketEvent): event is WebSocketEvent & {
  slug: string; reason: string; manual_command?: string;
} {
  if (event.type !== 'login.failed' || event.kind !== 'update') return false;
  return (event as { reason?: string }).reason === 'cli_missing_after_update';
}

/** What a missing-CLI notice is waiting for: the recovery command, and
 *  whether a refetch has shown the agent gone yet (only a gone → back
 *  transition clears it, so the stale row on screen when the event lands
 *  cannot). */
interface CliMissing {
  command: string | null;
  seenGone: boolean;
}

/** The dotted number out of a `--version` line ("2.1.100 (Claude Code)" →
 *  "2.1.100") for the floor notice; the raw line stays on the version pill. */
export function shortVersion(raw: string | null | undefined): string {
  const m = raw?.match(/\d+\.\d+\.\d+/);
  return m ? m[0] : (raw ?? '?');
}

/** Per-agent hint under the free-text model field, for CLIs whose model ids
 *  are not guessable from the vendor's model names. */
const MODEL_HINT_KEY: Record<string, StringKey> = {
  pi: 'subAgentCard.pi.modelHint',
  'kimi-code': 'subAgentCard.kimiCode.modelHint',
  'minimax-code': 'subAgentCard.minimaxCode.modelHint',
};

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

// Sub-agents that ALSO accept an API key via stdin. They keep the interactive
// login too: for codex that is the ChatGPT sign-in most people use.
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
  // used to stay hidden. Spec 115 D4: a bring-your-own agent that is not
  // detected now gets a compact "Not detected" row (isNotDetectedEntry)
  // instead of vanishing — the reporter of #57 read the silence as "codex
  // is not supported". `supported` then narrows the installable set to this
  // host, and an excluded host gets an explanation on the card.
  return true;
}

/** Spec 115 D4: bring-your-own (no Orbital install) and the CLI was not found. */
function isNotDetectedEntry(entry: SubAgentEntry): boolean {
  return entry.installed !== true && (entry.install?.platforms?.length ?? 0) === 0;
}

/** Spec 127 D3: an Orbital-installed agent that is not installed and whose
 *  Orbital install is not on offer — this host is outside its platform list,
 *  or the last run failed — gets the locate card *under* its card, never in
 *  place of it: the card keeps Install / Retry, the failure text and the
 *  credential form; the row is the way to a launcher the user installed by
 *  hand. A successful install, or one still on offer, shows the card alone. */
function showsLocateFallback(entry: SubAgentEntry): boolean {
  // Found, but it did not answer the handshake: the way out is another copy.
  if (entry.installed === true) return entry.handshake?.state === 'failed';
  if ((entry.install?.platforms?.length ?? 0) === 0) return false;
  return entry.install?.supported === false || entry.install?.state === 'failed';
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
  const ws = useWebSocket();
  const [cliMissing, setCliMissing] = useState<Record<string, CliMissing>>({});
  // "Change location" on a detected card opens the locate row under it.
  const [locating, setLocating] = useState<Record<string, boolean>>({});

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

  // Spec 136 A.2: the list is served from the daemon's last sweep, however
  // old; a background sweep (or a late model list) that changes it says so
  // over the WS. Refetch quietly — the cards stay up, no spinner.
  useEffect(() => {
    const onUpdated = () => {
      api<SubAgentEntry[]>('/api/v2/settings/sub-agents')
        .then(setEntries)
        .catch(() => { /* the next update or Refresh tries again */ });
    };
    ws.on('sub_agents.updated', onUpdated);
    return () => { ws.off('sub_agents.updated', onUpdated); };
  }, [ws]);

  // 1018 bench item 5: an update that took the CLI with it. Held here, not in
  // the card, because the refetched "not installed" entry may render as the
  // not-detected row instead of a card. The daemon already dropped its cached
  // detection, so refetch at once rather than waiting for Refresh.
  useEffect(() => {
    const onFailed = (event: WebSocketEvent) => {
      if (!isCliMissingEvent(event)) return;
      setCliMissing(m => ({
        ...m,
        [event.slug]: { command: event.manual_command ?? null, seenGone: false },
      }));
      void load();
    };
    ws.on('login.failed', onFailed);
    return () => { ws.off('login.failed', onFailed); };
  }, [ws, load]);

  useEffect(() => {
    if (entries === null) return;
    setCliMissing(m => {
      let changed = false;
      const next: Record<string, CliMissing> = { ...m };
      for (const [slug, state] of Object.entries(m)) {
        const entry = entries.find(e => e.slug === slug);
        if (!entry) continue;
        if (!entry.installed && !state.seenGone) {
          next[slug] = { ...state, seenGone: true };
          changed = true;
        } else if (entry.installed && state.seenGone) {
          delete next[slug];
          changed = true;
        }
      }
      return changed ? next : m;
    });
  }, [entries]);

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
  // be installed from here get a one-line "Not detected" row (spec 115 D4);
  // project settings and the pin switcher keep hiding them. (The filter is
  // kept as the seam for any future hide rule.)
  const visibleEntries = entries === null
    ? null
    : entries.filter(isVisibleEntry);

  const inner = (
    <>
      {/* Embedded in GlobalSettings, the heading AND the intro line are the
          enclosing SettingsSection's `title` / `description` — printing the
          intro here too would say the same sentence twice, once loose and
          once behind the ⓘ. With nothing in the left slot, `justify-between`
          would strand Refresh on the LEFT, so the row switches to
          `justify-end`. The standalone page still owns its own h1. */}
      <div className={`flex items-center mb-3 ${standalone ? 'justify-between' : 'justify-end'}`}>
        {standalone && (
          <h1 className="text-xl font-semibold text-primary">{t('subAgentSettings.title')}</h1>
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

      {/* Embedded in Global Settings these three notes live behind the
          section's ⓘ, with the rest of its description. The standalone page
          has no section heading to hang them on, so it still prints them. */}
      {standalone && (
        <>
          <p className="text-xs text-secondary mb-1">
            {t('subAgentSettings.installHint')}
          </p>
          <p className="text-xs text-secondary mb-1 italic">
            {t('subAgentSettings.credNote')}
          </p>
          <p className="text-xs text-secondary mb-4 italic">
            {t('subAgentSettings.loginNote')}
          </p>
        </>
      )}

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
            <div key={entry.slug} className="flex flex-col gap-2">
              {isNotDetectedEntry(entry)
                ? (
                  <NotDetectedRow
                    slug={entry.slug}
                    name={entry.name}
                    overridePath={entry.override_path}
                    overrideMissing={entry.override_missing}
                    onChanged={load}
                    desktopApp={entry.desktop_app}
                    fallbackHint={entry.desktop_app
                      ? t('subAgentCard.desktop.locateHint', { app: entry.desktop_app.label })
                      : undefined}
                  />
                )
                : (
                  <SubAgentCard
                    entry={entry}
                    onChanged={load}
                    onChangeLocation={() => setLocating(m => ({ ...m, [entry.slug]: !m[entry.slug] }))}
                  />
                )}
              {!isNotDetectedEntry(entry) && (showsLocateFallback(entry) || locating[entry.slug]) && (
                <NotDetectedRow
                  slug={entry.slug}
                  name={entry.name}
                  overridePath={entry.override_path}
                  overrideMissing={entry.override_missing}
                  onChanged={async () => {
                    setLocating(m => ({ ...m, [entry.slug]: false }));
                    await load();
                  }}
                  fallbackHint={entry.installed
                    ? t('subAgentCard.locate.changeHint', { name: entry.name })
                    : t('subAgentCard.locate.orbitalInstallFallback', { name: entry.name })}
                />
              )}
              {cliMissing[entry.slug] && (
                <p
                  data-testid={`sub-agent-update-missing-${entry.slug}`}
                  className="text-xs text-error px-1"
                >
                  {t('subAgentCard.updateCliMissing', {
                    name: entry.name,
                    command: cliMissing[entry.slug].command
                      ?? MANUAL_UPDATE_COMMAND[entry.slug]
                      ?? entry.setup_actions.find(a => a.action === 'install_agent')?.command
                      ?? '',
                  })}
                  {' '}
                  <SendDiagnosticsLink context={{ kind: 'dispatch_failed', agent: entry.slug, detail: 'update_failed' }} />
                </p>
              )}
            </div>
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
  /** Open the locate row under this card: point Orbital at another copy. */
  onChangeLocation?: () => void;
}

function SubAgentCard({ entry, onChanged, onChangeLocation }: CardProps) {
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
  // Jobs whose terminal event (done/failed) already arrived. A preflight
  // failure (no Node on PATH) lands over the WS before the POST's 202
  // resolves; without this the handler would re-enter `installing` after the
  // error and the spinner would sit beside "Install failed" forever
  // (spec 127 V4 on Windows, 2026-10-22).
  const settledInstallJobs = useRef<Set<string>>(new Set());
  const [installBusy, setInstallBusy] = useState(false);
  const [installLine, setInstallLine] = useState<string | null>(null);
  const [installError, setInstallError] = useState<string | null>(null);
  // Login: the CLI opens a browser and completes the OAuth handshake on its
  // own (the result travels server-side), so the ONLY thing the user may need
  // is the URL — printed once, in case the browser did not open. `loginNotice`
  // carries the terminal outcome: 'unverified' means the CLI exited 0 but the
  // credential re-check still reported unconfigured, which is NOT a failure (a
  // slow keychain write looks identical), so it reads as "couldn't confirm".
  const [loginUrl, setLoginUrl] = useState<string | null>(null);
  const [loginNotice, setLoginNotice] = useState<'unverified' | 'failed' | null>(null);
  // Update (spec 116 D2/D3): the same job runner as login, told apart by
  // `kind: 'update'` on every event. `updating` spans the job; the notice is
  // 'failed' when the re-probed version is still below the floor (or the
  // updater failed), 'busy' when the daemon refused (409) because a worker
  // of this agent is running (D8).
  const [updateBusy, setUpdateBusy] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [updateNotice, setUpdateNotice] = useState<'failed' | 'busy' | null>(null);

  const paramKeys = Object.keys(entry.param_schema);
  const updateNeeded = entry.update_needed === true;
  const authExpired = entry.auth_expired === true && entry.credentials_configured;
  const supportsUpdate = entry.supports_update !== false;
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
  // Every key optional (dsh: it runs on whatever provider the user set up
  // in its own app): no "key needed / saved" pill — whether a task can be
  // served is the agent's call, and the host note says where to set it up.
  const keyOptional = usesManagedCredentials
    && credentialFields.every(f => f.required === false);
  // Set only for agents whose credential follows the configured model's
  // provider; Orbital owns no login for those, so neither Login nor Logout.
  const credentialState = entry.credential_state ?? null;

  const canInstall = entry.install?.supported === true;
  const installUnsupported = entry.install !== undefined && !canInstall;
  const installFailed = installError !== null || entry.install?.state === 'failed';

  // Keep the local install view in step with a refetched entry (resume after
  // refresh, and the settled state once a job ends).
  const entryInstallState = entry.install?.state;
  useEffect(() => {
    if (entryInstallState === 'installing') setInstalling(true);
    else if (entryInstallState === 'installed' || entryInstallState === 'failed') setInstalling(false);
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
      if (typeof event.job_id === 'string') settledInstallJobs.current.add(event.job_id);
      setInstalling(false);
      setInstallLine(null);
      setInstallError(null);
      void onChanged();
    };
    const onFailed = (event: WebSocketEvent) => {
      if (event.type !== 'sub_agent_install_failed') return;
      if (event.slug !== entry.slug) return;
      if (typeof event.job_id === 'string') settledInstallJobs.current.add(event.job_id);
      setInstalling(false);
      setInstallError(event.error);
      // Spec 127 D3: the daemon's job registry now says `failed`; refetch
      // so the list can hang the locate card under this card. The live
      // error text above outranks the entry's generic "failed earlier".
      void onChanged();
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

  // Login job events. Daemon-global and slug-carrying, same shape as the
  // install trio above. The progress lines are already ANSI/OSC-stripped by the
  // daemon (the CLI prints its URL wrapped in an OSC-8 hyperlink), so the first
  // non-loopback URL we see is directly linkable (see pickLoginUrl).
  useEffect(() => {
    const onLoginProgress = (event: WebSocketEvent) => {
      if (event.type !== 'login.progress') return;
      if (event.slug !== entry.slug) return;
      if (event.kind === 'update') return; // updater output carries no sign-in URL
      const url = pickLoginUrl(event.line);
      if (url) setLoginUrl(prev => prev ?? url);
    };
    const onLoginComplete = (event: WebSocketEvent) => {
      if (event.type !== 'login.complete') return;
      if (event.slug !== entry.slug) return;
      if (event.kind === 'update') {
        setUpdating(false);
        // D3: `verified` is the re-probed version reaching the floor. An
        // exit-0 that left the stub behind is a failure here, unlike login.
        setUpdateNotice(event.verified === false ? 'failed' : null);
        void onChanged();
        return;
      }
      setLoginBusy(false);
      setLoginUrl(null);
      // `verified === undefined` means an older daemon that does not re-check;
      // treat only an explicit false as unconfirmed.
      setLoginNotice(event.verified === false ? 'unverified' : null);
      void onChanged();
    };
    const onLoginFailed = (event: WebSocketEvent) => {
      if (event.type !== 'login.failed') return;
      if (event.slug !== entry.slug) return;
      if (event.kind === 'update') {
        setUpdating(false);
        // 1018 bench item 5: an update that left no runnable CLI is reported
        // by the list (the refetched entry reads "not installed", which can
        // swap this card for the not-detected row), and it refetches itself.
        if (isCliMissingEvent(event)) {
          setUpdateNotice(null);
          return;
        }
        setUpdateNotice('failed');
        // The daemon re-probed and dropped its cached detection on every
        // update terminal; show what is on disk now, not the pre-update row.
        void onChanged();
        return;
      }
      setLoginBusy(false);
      setLoginUrl(null);
      setLoginNotice('failed');
    };
    ws.on('login.progress', onLoginProgress);
    ws.on('login.complete', onLoginComplete);
    ws.on('login.failed', onLoginFailed);
    return () => {
      ws.off('login.progress', onLoginProgress);
      ws.off('login.complete', onLoginComplete);
      ws.off('login.failed', onLoginFailed);
    };
  }, [ws, entry.slug, onChanged]);

  const handleInstall = async () => {
    setInstallBusy(true);
    setInstallError(null);
    setInstallLine(null);
    try {
      const res = (await api(
        `/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/install`,
        { method: 'POST' },
      )) as { job_id?: string } | undefined;
      // 202 — the job streams over the WS from here. Unless its terminal
      // event already arrived while this request was in flight: then the
      // handlers above have the final word and the spinner must stay off.
      const jobId = res?.job_id;
      if (typeof jobId !== 'string' || !settledInstallJobs.current.has(jobId)) {
        setInstalling(true);
      }
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
    // Clear the previous attempt's URL/outcome so a retry never shows a stale
    // link or a stale "couldn't confirm" from the run before it.
    setLoginUrl(null);
    setLoginNotice(null);
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

  const handleUpdate = async () => {
    setUpdateBusy(true);
    setActionError(null);
    setUpdateNotice(null);
    try {
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/update`, {
        method: 'POST',
      });
      // Progress streams over the WS; the terminal event ends `updating`.
      setUpdating(true);
    } catch (e) {
      const status = e && typeof e === 'object' ? (e as { status?: number }).status : undefined;
      if (status === 409) {
        setUpdateNotice('busy');
      } else {
        setActionError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setUpdateBusy(false);
    }
  };

  const handleLogout = async () => {
    setLogoutBusy(true);
    setActionError(null);
    try {
      const result = await api<{ return_code: number; stderr?: string }>(
        `/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/logout`,
        { method: 'POST' },
      );
      await onChanged();
      // The route answers 200 with the CLI's exit code, so a refused logout
      // only shows up here.
      if (result.return_code !== 0) {
        setActionError(result.stderr?.trim() || t('subAgentCard.logoutFailed'));
      }
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setLogoutBusy(false);
    }
  };

  // The locate card's Clear: forget the pasted location; detection runs on
  // its own again (the row may come back as "not detected").
  const [clearBusy, setClearBusy] = useState(false);
  const handleClearLocation = async () => {
    setClearBusy(true);
    setActionError(null);
    try {
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/locate`, {
        method: 'DELETE',
      });
      await onChanged();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setClearBusy(false);
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
            {credentialState ? (
              <StatusPill
                label={t(CREDENTIAL_STATE_KEY[credentialState], { name: entry.name })}
                variant={credentialState === 'configured' ? 'success' : 'warning'}
              />
            ) : keyOptional ? null : (
              <StatusPill
                label={usesManagedCredentials
                  ? (entry.credentials_configured ? t('subAgentCard.keySaved') : t('subAgentCard.keyNeeded'))
                  : authExpired
                    ? t('subAgentCard.authExpired')
                    : (entry.credentials_configured ? t('subAgentCard.loggedIn') : t('subAgentCard.notLoggedIn'))}
                variant={entry.credentials_configured && !authExpired ? 'success' : 'warning'}
              />
            )}
            {entry.version && (
              <span className="text-secondary/80 font-mono">v{entry.version}</span>
            )}
            {entry.installed && entry.source?.kind === 'desktop_app' && entry.source.label && (
              <span
                data-testid={`sub-agent-source-${entry.slug}`}
                title={[entry.source.app_path, entry.source.app_version].filter(Boolean).join(' · ')}
                className="px-2 py-0.5 rounded-full border text-[11px] bg-sidebar text-secondary border-border"
              >
                {t('subAgentCard.source.desktop', { app: entry.source.label })}
              </span>
            )}
            {entry.installed && !entry.override_path && onChangeLocation && (
              <button
                type="button"
                onClick={onChangeLocation}
                data-testid={`sub-agent-locate-change-${entry.slug}`}
                className="text-secondary hover:text-primary transition-all duration-150"
              >
                {t('subAgentCard.locate.change')}
              </button>
            )}
            {entry.override_path && (
              <>
                <span
                  data-testid={`sub-agent-locate-chip-${entry.slug}`}
                  title={entry.override_path}
                  className="px-2 py-0.5 rounded-full border text-[11px] bg-sidebar text-secondary border-border"
                >
                  {t('subAgentCard.locate.chip')}
                </span>
                <button
                  type="button"
                  onClick={handleClearLocation}
                  disabled={clearBusy}
                  data-testid={`sub-agent-locate-clear-${entry.slug}`}
                  className="text-secondary hover:text-primary transition-all duration-150 disabled:opacity-50"
                >
                  {t('subAgentCard.locate.clear')}
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      {HOST_NOTE_KEY[entry.slug] && (
        <p
          data-testid={`sub-agent-host-note-${entry.slug}`}
          className="text-xs text-secondary mt-2"
        >
          {t(HOST_NOTE_KEY[entry.slug])}
        </p>
      )}

      {entry.emits_tool_activity === false && (
        <p
          data-testid={`sub-agent-silent-note-${entry.slug}`}
          className="text-xs text-secondary mt-2"
        >
          {t('subAgentCard.silentUntilDone')}
        </p>
      )}

      {entry.installed && entry.handshake?.state === 'failed' && (
        <div
          data-testid={`sub-agent-handshake-failed-${entry.slug}`}
          className="flex flex-col gap-1 mt-2"
        >
          <p className="text-xs text-warning">
            {t('subAgentCard.handshakeFailed', { name: entry.name })}
          </p>
          {entry.handshake.detail && (
            // The agent's own words — dynamic, never translated.
            <pre className="text-[11px] font-mono text-secondary/80 whitespace-pre-wrap break-words">
              {entry.handshake.detail}
            </pre>
          )}
        </div>
      )}

      {!entry.installed && entry.missing_dependencies.length > 0 && (
        <p className="text-xs text-secondary mt-1 mb-2">
          {t('subAgentCard.missingDeps', { list: entry.missing_dependencies.join(', ') })}
        </p>
      )}

      {/* Spec 116 D4: too old to drive — say "update", offer the updater,
          and hide Login (a too-old CLI cannot be logged in through Orbital). */}
      {updateNeeded && (
        <div
          data-testid={`sub-agent-update-${entry.slug}`}
          className="flex flex-col gap-2 mt-3 mb-3"
        >
          <p className="text-xs text-warning">
            {t('subAgentCard.updateNeeded', {
              name: entry.name,
              version: shortVersion(entry.version),
              min: entry.min_version ?? '?',
            })}
          </p>
          {supportsUpdate && (
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={handleUpdate}
                disabled={updateBusy || updating || !entry.installed}
                className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
              >
                {updateBusy || updating ? t('subAgentCard.updating') : t('subAgentCard.update')}
              </button>
              {updating && (
                <span
                  data-testid={`sub-agent-update-spinner-${entry.slug}`}
                  className="inline-block h-3 w-3 rounded-full border-2 border-accent border-t-transparent animate-spin"
                  aria-hidden="true"
                />
              )}
            </div>
          )}
          {updateNotice && (
            <p
              data-testid={`sub-agent-update-notice-${entry.slug}`}
              className={`text-[11px] ${updateNotice === 'busy' ? 'text-warning' : 'text-error'}`}
            >
              {updateNotice === 'busy'
                ? t('subAgentCard.updateBusy', { name: entry.name })
                : t('subAgentCard.updateFailed', {
                    command: MANUAL_UPDATE_COMMAND[entry.slug] ?? entry.setup_actions.find(a => a.action === 'install_agent')?.command ?? '',
                  })}
              {updateNotice !== 'busy' && (
                <>
                  {' '}
                  <SendDiagnosticsLink context={{ kind: 'dispatch_failed', agent: entry.slug, detail: 'update_failed' }} />
                </>
              )}
            </p>
          )}
        </div>
      )}

      {authExpired && supportsLogin && (
        <p
          data-testid={`sub-agent-auth-expired-${entry.slug}`}
          className="text-xs text-warning mt-2"
        >
          {t('subAgentCard.authExpiredNote', { name: entry.name })}
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
        {(!entry.credentials_configured || authExpired) && supportsLogin && !updateNeeded && (
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
        {entry.credentials_configured && !authExpired && !usesManagedCredentials && !credentialState && (
          <button
            onClick={handleLogout}
            disabled={logoutBusy}
            className="text-xs bg-sidebar border border-border text-primary rounded px-3 py-1.5 hover:bg-sidebar/80 transition-all duration-150 disabled:opacity-50"
          >
            {logoutBusy ? t('subAgentCard.loggingOut') : t('subAgentCard.logout')}
          </button>
        )}
      </div>

      {/* The CLI opens the browser itself and finishes the handshake without
          any input from us; this link is the fallback for when it cannot. */}
      {loginUrl && (
        <p className="text-[11px] text-secondary mb-3">
          {t('subAgentCard.loginBrowserHint')}{' '}
          <a
            href={loginUrl}
            target="_blank"
            rel="noreferrer noopener"
            data-testid={`sub-agent-login-url-${entry.slug}`}
            className="text-accent underline break-all"
          >
            {t('subAgentCard.loginOpenLink')}
          </a>
        </p>
      )}
      {loginNotice && (
        <p
          data-testid={`sub-agent-login-notice-${entry.slug}`}
          className={`text-[11px] mb-3 ${
            loginNotice === 'unverified' ? 'text-warning' : 'text-error'
          }`}
        >
          {loginNotice === 'unverified'
            ? t('subAgentCard.loginUnverified')
            : t('subAgentCard.loginFailed')}
          {' '}
          <SendDiagnosticsLink context={{ kind: 'login_failed', agent: entry.slug, detail: loginNotice }} />
        </p>
      )}

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
                        {schema.labels?.[opt] ?? paramOptionLabel(
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
                {MODEL_HINT_KEY[entry.slug] && paramKey === 'model' && (
                  <p className="text-xs text-secondary" data-testid={`sub-agent-model-hint-${entry.slug}`}>
                    {t(MODEL_HINT_KEY[entry.slug])}
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
      {/* The note sat in full under this heading on EVERY API-key card — the
          same two lines three or four times down the page. Behind the ⓘ it is
          one click away on each, and printed on none. */}
      <div>
        <LabelWithHint className="!text-xs" hint={t('subAgentCard.cred.note')}>
          {t('subAgentCard.cred.heading')}
        </LabelWithHint>
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

            {field.stored_by ? (
              <div data-testid={`sub-agent-cred-stored-by-${field.key}`} className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <StatusPill label={t('subAgentCard.cred.storedBy', { owner: field.stored_by })} variant="success" />
                </div>
                <p className="text-xs text-secondary">
                  {t('subAgentCard.cred.storedByNote', { owner: field.stored_by })}
                </p>
              </div>
            ) : (<>
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
            </>)}
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
