// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The Welcome page's Agents band: every coding agent DETECTED on this
 * computer, as tiles. Ready (installed and signed in) is a quiet check; an
 * installed agent that is signed out offers Sign in (the in-app login job,
 * useAgentLogin); an agent that takes a key instead says where the key goes.
 * Nothing here is a choice: hiding an agent stays in Settings.
 *
 * Agents not detected sit behind "添加其他 Agent" (with none detected:
 * "安装了但没找到？"): their chips open the existing locate card, and the
 * install line carries "Check again". A failed list fetch is not "none
 * found": it says so, with Try again.
 */

import { useCallback, useState } from 'react';
import { Check, ChevronDown, ChevronUp, KeyRound, Loader2, Plus } from 'lucide-react';
import type { SubAgentListEntry } from '../../hooks/useSubAgentList';
import { useAgentLogin } from '../../hooks/useAgentLogin';
import { useT } from '../../i18n/useT';
import { sendTelemetry } from '../../utils/telemetry';
import LoginCodeField from '../LoginCodeField';
import NotDetectedRow from '../NotDetectedRow';
import AgentGlyph from './AgentGlyph';
import { agentStateOf, isReady, type AgentState } from './welcomeModel';

const tileClass =
  'surface-float-static surface-float-nested flex h-[46px] min-w-0 items-center gap-2.5 !rounded-[12px] pl-3 pr-2';

function AgentTile({ entry, state, emphasis, onChanged }: {
  entry: SubAgentListEntry;
  state: AgentState;
  /** Nothing is ready: Sign in is the main action. */
  emphasis: boolean;
  onChanged: () => void;
}) {
  const t = useT();
  const login = useAgentLogin(entry.slug, onChanged);
  return (
    <li data-testid={`welcome-agent-${entry.slug}`} data-state={state} className="min-w-0">
      <div className={tileClass}>
        <AgentGlyph slug={entry.slug} size={22} />
        <span className="min-w-0 flex-1 truncate text-body font-medium text-primary">{entry.name}</span>
        {state === 'ready' && (
          <Check
            aria-label={t('wizard.agents.signedIn')}
            className="h-4 w-4 shrink-0 text-success-text"
            strokeWidth={2.2}
            data-testid={`welcome-agent-ready-${entry.slug}`}
          />
        )}
        {state === 'key' && (
          <span
            className="inline-flex min-w-0 shrink items-center gap-1 text-caption text-secondary"
            title={t('wizard.agents.needsKey')}
            data-testid={`welcome-agent-key-${entry.slug}`}
          >
            <KeyRound aria-hidden="true" className="h-3.5 w-3.5 shrink-0" strokeWidth={1.6} />
            <span className="truncate">{t('welcome.agents.keyInSettings')}</span>
          </span>
        )}
        {state === 'out' && (
          <button
            type="button"
            onClick={() => void login.start()}
            disabled={login.busy}
            aria-busy={login.busy || undefined}
            title={login.busy ? t('subAgentCard.loginBrowserHint') : undefined}
            data-testid={`welcome-agent-login-${entry.slug}`}
            className={`inline-flex h-[30px] shrink-0 items-center gap-1.5 rounded-[8px] px-3 text-caption font-medium transition-colors duration-150 disabled:opacity-70 max-md:min-h-[44px] ${
              emphasis ? 'bg-accent text-white hover:bg-accent/90' : 'bg-card-hover text-primary hover:bg-nav-hover'
            }`}
          >
            {login.busy && <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />}
            {!login.busy && t('wizard.agents.signIn')}
          </button>
        )}
      </div>
      {login.busy && login.url && (
        <a
          href={login.url}
          target="_blank"
          rel="noreferrer"
          className="mt-1 block truncate px-1 text-caption text-accent-text underline"
        >
          {t('subAgentCard.loginOpenLink')}
        </a>
      )}
      {/* A login that asks for the code its browser page shows (Antigravity):
          renders nothing until the CLI asks, and nothing for other agents. */}
      {login.busy && <LoginCodeField slug={entry.slug} className="mt-1.5" />}
      {login.failed && (
        <p className="mt-1 px-1 text-caption text-error-text" role="alert">{login.failed}</p>
      )}
    </li>
  );
}

export interface AgentsBandProps {
  entries: SubAgentListEntry[] | null;
  /** The list could not be fetched. */
  error: boolean;
  reload: (refresh?: boolean) => Promise<void>;
}

/** The open band's body (the header is the page's). */
export function AgentsBandBody({ entries, error, reload }: AgentsBandProps) {
  const t = useT();
  const [addOpen, setAddOpen] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const onChanged = useCallback(() => { void reload(); }, [reload]);

  if (error && (entries === null || entries.length === 0)) {
    return (
      <div className="mt-2 flex items-center gap-2 text-body text-secondary" role="alert" data-testid="welcome-agents-error">
        <span>{t('welcome.agents.loadFailed')}</span>
        <button
          type="button"
          disabled={checking}
          onClick={async () => { setChecking(true); await reload(); setChecking(false); }}
          data-testid="welcome-agents-retry"
          className="inline-flex items-center gap-1.5 font-medium text-accent-text hover:underline disabled:opacity-60 max-md:min-h-[44px]"
        >
          {checking && <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />}
          {t('importProjects.retry')}
        </button>
      </div>
    );
  }
  if (entries === null) {
    return (
      <div className="mt-2 inline-flex items-center gap-2 text-body text-secondary" data-testid="welcome-agents-loading">
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
        {t('wizard.agents.loading')}
      </div>
    );
  }

  const installed = entries
    .filter((e) => e.installed)
    .map((e) => ({ entry: e, state: agentStateOf(e) }))
    .sort((a, b) => Number(a.state !== 'ready') - Number(b.state !== 'ready'));
  const missing = entries.filter((e) => !e.installed);
  const none = installed.length === 0;
  const emphasis = !installed.some((a) => a.state === 'ready');
  const pickedEntry = picked ? missing.find((e) => e.slug === picked) : undefined;

  return (
    <div data-testid="welcome-agents">
      {!none && (
        <ul className="mt-2 grid grid-cols-2 gap-2 max-[520px]:grid-cols-1" data-testid="welcome-agent-list">
          {installed.map(({ entry, state }) => (
            <AgentTile key={entry.slug} entry={entry} state={state} emphasis={emphasis} onChanged={onChanged} />
          ))}
        </ul>
      )}
      {missing.length > 0 && (
        <div className={none ? 'mt-1' : 'mt-2'}>
          <button
            type="button"
            onClick={() => { setAddOpen((v) => !v); if (addOpen) setPicked(null); }}
            aria-expanded={addOpen}
            data-testid="welcome-agents-add"
            className="inline-flex items-center gap-1 text-caption text-secondary transition-colors duration-150 hover:text-primary max-md:min-h-[44px]"
          >
            {!none && <Plus className="h-3.5 w-3.5" strokeWidth={2} aria-hidden="true" />}
            {none ? t('wizard.agents.notFound.heroHeading') : t('welcome.agents.add')}
            {addOpen
              ? <ChevronUp className="h-3 w-3" aria-hidden="true" />
              : <ChevronDown className="h-3 w-3" aria-hidden="true" />}
          </button>
          {addOpen && (
            <div className="mt-2" data-testid="welcome-agents-missing">
              <div className="flex flex-wrap gap-1.5" role="group" aria-label={t('welcome.agents.add')}>
                {missing.map((e) => (
                  <button
                    key={e.slug}
                    type="button"
                    onClick={() => setPicked((cur) => (cur === e.slug ? null : e.slug))}
                    aria-pressed={picked === e.slug}
                    data-testid={`welcome-locate-chip-${e.slug}`}
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-caption transition-colors duration-150 max-md:min-h-[44px] ${
                      picked === e.slug
                        ? 'border-accent bg-accent/10 text-primary'
                        : 'border-border bg-sidebar text-secondary hover:text-primary'
                    }`}
                  >
                    <AgentGlyph slug={e.slug} size={14} />
                    {e.name}
                  </button>
                ))}
              </div>
              {pickedEntry && (
                <div className="mt-2">
                  <NotDetectedRow
                    slug={pickedEntry.slug}
                    name={pickedEntry.name}
                    overridePath={pickedEntry.override_path}
                    overrideMissing={pickedEntry.override_missing}
                    onChanged={async () => {
                      sendTelemetry('agent_located');
                      setPicked(null);
                      await reload();
                    }}
                  />
                </div>
              )}
              {none && (
                <p className="mt-2 text-caption text-secondary" data-testid="welcome-install-line">
                  {t('wizard.agents.install')}{' '}
                  <span className="text-primary">{missing.map((e) => e.name).join(' · ')}</span>
                </p>
              )}
              <button
                type="button"
                disabled={checking}
                onClick={async () => { setChecking(true); await reload(true); setChecking(false); }}
                data-testid="welcome-agents-recheck"
                className="mt-2 inline-flex items-center gap-1.5 text-caption font-medium text-accent-text hover:underline disabled:opacity-60 max-md:min-h-[44px]"
              >
                {checking && <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />}
                {t('wizard.agents.checkAgain')}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** The folded line: ready marks + a check, then how many still need a sign-in. */
export function AgentsSummary({ entries }: { entries: SubAgentListEntry[] | null }) {
  const t = useT();
  const installed = (entries ?? []).filter((e) => e.installed);
  const ready = installed.filter(isReady);
  const out = installed.filter((e) => agentStateOf(e) === 'out').length;
  const shown = ready.length ? ready : installed;
  return (
    <span className="flex min-w-0 items-center gap-2" data-testid="welcome-agents-summary">
      <span className="flex shrink-0 items-center gap-1">
        {shown.slice(0, 8).map((e) => (
          <AgentGlyph key={e.slug} slug={e.slug} size={18} title={e.name} dim={!ready.length} />
        ))}
        {shown.length > 8 && <span className="text-caption text-secondary">+{shown.length - 8}</span>}
      </span>
      {ready.length > 0 && (
        <Check aria-label={t('wizard.agents.signedIn')} className="h-3.5 w-3.5 shrink-0 text-success-text" strokeWidth={2.2} />
      )}
      {out > 0 && (
        <span className="truncate text-caption text-warning-text">
          {ready.length > 0 ? '· ' : ''}
          {t('welcome.agents.toSignIn', { n: out })}
        </span>
      )}
    </span>
  );
}
