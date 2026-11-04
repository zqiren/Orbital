// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * WizardAgentsStep — wizard step 1 (spec 138): the coding agents Orbital found
 * on this computer, each checked; unchecking one hides it from the pin switcher
 * and delegation in projects created from now on (the global denylist the
 * wizard PUTs as `sub_agents_hidden`). Nothing is uninstalled.
 *
 * Each agent keeps its own login, so a not-signed-in agent offers the same
 * login job Sub-agent settings starts (POST /settings/sub-agents/{slug}/login,
 * progress over the WebSocket). Agents that were not detected are chips; a
 * chip expands the existing locate card (NotDetectedRow) as-is, and its Save
 * refetches the list so the agent moves up into the found list.
 *
 * Spec 153: an agent that keeps private notes (Claude Code's `memory/`) gets a
 * third status segment, "N memories only it can use"; clicking it expands the
 * per-folder breakdown under the row. Hidden at zero.
 */

import { useCallback, useEffect, useState } from 'react';
import { isSignedIn, type SubAgentListEntry } from '../hooks/useSubAgentList';
import { Loader2 } from 'lucide-react';
import { api } from '../config';
import { useT } from '../i18n/useT';
import { useWebSocket } from '../hooks/useWebSocket';
import type { WebSocketEvent } from '../types';
import { pickLoginUrl } from '../utils/loginUrl';
import {
  conversationCount,
  notesBreakdown,
  notesCount,
  type HistoryStatus,
  type NotesBreakdownRow,
} from '../utils/onboarding';
import { sendTelemetry } from '../utils/telemetry';
import MessageAvatar from './MessageAvatar';
import NotDetectedRow from './NotDetectedRow';

function AgentRow({
  entry,
  checked,
  conversations,
  notes = null,
  breakdown = [],
  onToggle,
  onChanged,
}: {
  entry: SubAgentListEntry;
  checked: boolean;
  /** null = this agent has no history reader (no count shown). */
  conversations: number | null;
  /** Spec 153: private notes only this agent can use; null hides the segment. */
  notes?: number | null;
  breakdown?: NotesBreakdownRow[];
  onToggle: (checked: boolean) => void;
  onChanged: () => void;
}) {
  const t = useT();
  const ws = useWebSocket();
  const [busy, setBusy] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const signedIn = isSignedIn(entry);

  useEffect(() => {
    const onProgress = (event: WebSocketEvent) => {
      if (event.type !== 'login.progress' || event.slug !== entry.slug || event.kind === 'update') return;
      const found = pickLoginUrl(event.line);
      if (found) setUrl((prev) => prev ?? found);
    };
    const onComplete = (event: WebSocketEvent) => {
      if (event.type !== 'login.complete' || event.slug !== entry.slug || event.kind === 'update') return;
      setBusy(false);
      setUrl(null);
      onChanged();
    };
    const onFailed = (event: WebSocketEvent) => {
      if (event.type !== 'login.failed' || event.slug !== entry.slug || event.kind === 'update') return;
      setBusy(false);
      setUrl(null);
      setFailed(t('subAgentCard.loginFailed'));
    };
    ws.on('login.progress', onProgress);
    ws.on('login.complete', onComplete);
    ws.on('login.failed', onFailed);
    return () => {
      ws.off('login.progress', onProgress);
      ws.off('login.complete', onComplete);
      ws.off('login.failed', onFailed);
    };
  }, [ws, entry.slug, onChanged, t]);

  const handleLogin = async () => {
    setBusy(true);
    setFailed(null);
    setUrl(null);
    try {
      await api(`/api/v2/settings/sub-agents/${encodeURIComponent(entry.slug)}/login`, { method: 'POST' });
    } catch (e) {
      setBusy(false);
      setFailed(e instanceof Error && e.message ? e.message : t('subAgentCard.loginFailed'));
    }
  };

  // An agent without a login flow takes an API key instead, which lives in
  // Settings → Sub-agents; say so rather than offer a Sign in that can't work.
  const keyOnly = entry.supports_login === false;
  let status: string;
  if (!signedIn) status = keyOnly ? t('wizard.agents.needsKey') : t('wizard.agents.notSignedIn');
  else if (conversations === null) status = t('wizard.agents.signedIn');
  else status = t(conversations === 1 ? 'wizard.agents.signedInCount.one' : 'wizard.agents.signedInCount.other', { n: conversations });

  return (
    <li data-testid={`wizard-agent-${entry.slug}`} className="rounded-lg border border-border px-3 py-2.5">
      <div className="flex items-center gap-3">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onToggle(e.target.checked)}
          aria-label={entry.name}
          data-testid={`wizard-agent-check-${entry.slug}`}
          className="h-4 w-4 shrink-0 cursor-pointer accent-accent"
        />
        <MessageAvatar variant="agent" agentHandle={entry.slug} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-primary">{entry.name}</p>
          <p
            className={`text-xs ${notes === null ? 'truncate' : 'leading-snug'} ${signedIn ? 'text-secondary' : 'text-warning-text'}`}
            data-testid={`wizard-agent-status-${entry.slug}`}
          >
            {status}
            {notes !== null && (
              <>
                {' · '}
                <button
                  type="button"
                  onClick={() => setNotesOpen((v) => !v)}
                  aria-expanded={notesOpen}
                  disabled={breakdown.length === 0}
                  data-testid={`wizard-agent-notes-${entry.slug}`}
                  className="whitespace-nowrap border-b border-dotted border-muted text-primary transition-colors duration-150 hover:border-accent hover:text-accent disabled:cursor-default disabled:hover:border-muted disabled:hover:text-primary"
                >
                  {t(notes === 1 ? 'wizard.agents.notes.one' : 'wizard.agents.notes.other', { n: notes })}
                </button>
              </>
            )}
          </p>
        </div>
        {!signedIn && !keyOnly && (
          <button
            type="button"
            onClick={handleLogin}
            disabled={busy}
            data-testid={`wizard-agent-login-${entry.slug}`}
            className="shrink-0 inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-primary transition-colors duration-150 hover:bg-surface disabled:opacity-60 max-md:min-h-[44px]"
          >
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />}
            {busy ? t('subAgentCard.startingLogin') : t('wizard.agents.signIn')}
          </button>
        )}
      </div>
      {notesOpen && notes !== null && breakdown.length > 0 && (
        <ul
          aria-label={t('wizard.agents.notes.byFolder')}
          data-testid={`wizard-agent-notes-breakdown-${entry.slug}`}
          className="mb-0.5 mt-2 flex flex-col gap-[3px] pl-[66px] text-xs"
        >
          {breakdown.map((row) => (
            <li key={`${row.notFound ? 'nf' : 'f'}:${row.label}`} className="flex justify-between gap-3 py-0.5 text-secondary">
              <span className={`min-w-0 truncate ${row.notFound ? 'italic text-muted' : 'text-primary'}`}>
                {row.notFound ? `${t('wizard.agents.notes.folderNotFound')} · ${row.label}` : row.label}
              </span>
              <span className="shrink-0 font-mono">{row.n}</span>
            </li>
          ))}
        </ul>
      )}
      {busy && (
        <p className="mt-1.5 pl-[66px] text-xs text-secondary">
          {t('subAgentCard.loginBrowserHint')}{' '}
          {url && (
            <a href={url} target="_blank" rel="noreferrer" className="text-accent underline">
              {t('subAgentCard.loginOpenLink')}
            </a>
          )}
        </p>
      )}
      {failed && (
        <p className="mt-1.5 pl-[66px] text-xs text-error" role="alert">{failed}</p>
      )}
    </li>
  );
}

interface WizardAgentsStepProps {
  entries: SubAgentListEntry[];
  history: HistoryStatus | null;
  /** Unchecked = hidden. Keyed by slug; absent = checked. */
  ticks: Record<string, boolean>;
  onToggle: (slug: string, checked: boolean) => void;
  reload: (refresh?: boolean) => Promise<void>;
}

export default function WizardAgentsStep({ entries, history, ticks, onToggle, reload }: WizardAgentsStepProps) {
  const t = useT();
  const installed = entries.filter((e) => e.installed);
  const missing = entries.filter((e) => !e.installed);
  const none = installed.length === 0;
  // With nothing found, the locate card is the hero: the first chip opens.
  const [open, setOpen] = useState<string | null>(() => (none ? missing[0]?.slug ?? null : null));
  const [checking, setChecking] = useState(false);

  const onLocated = useCallback(async () => {
    sendTelemetry('agent_located');
    setOpen(null);
    await reload();
  }, [reload]);

  const conversationsFor = (slug: string): number | null =>
    conversationCount(history?.by_agent, slug);

  const notFound = missing.length > 0 && (
    <section className={none ? '' : 'mt-6 border-t border-border pt-5'} data-testid="wizard-not-found">
      <h2 className="text-sm font-medium text-primary">
        {none ? t('wizard.agents.notFound.heroHeading') : t('wizard.agents.notFound.heading')}
      </h2>
      <p className="mb-3 mt-0.5 text-xs text-secondary">{t('wizard.agents.notFound.hint')}</p>
      <div className="flex flex-wrap gap-2" role="group" aria-label={t('wizard.agents.notFound.heading')}>
        {missing.map((e) => (
          <button
            key={e.slug}
            type="button"
            onClick={() => setOpen((cur) => (cur === e.slug ? null : e.slug))}
            aria-expanded={open === e.slug}
            data-testid={`wizard-locate-chip-${e.slug}`}
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors duration-150 max-md:min-h-[44px] ${
              open === e.slug
                ? 'border-accent bg-accent/10 text-primary'
                : 'border-border bg-sidebar text-secondary hover:text-primary'
            }`}
          >
            {e.name}
          </button>
        ))}
      </div>
      {open && missing.some((e) => e.slug === open) && (() => {
        const e = missing.find((m) => m.slug === open)!;
        return (
          <div className="mt-3">
            <NotDetectedRow
              slug={e.slug}
              name={e.name}
              overridePath={e.override_path}
              overrideMissing={e.override_missing}
              onChanged={onLocated}
            />
          </div>
        );
      })()}
      {none && (
        <div className="mt-4 text-xs text-secondary" data-testid="wizard-install-line">
          <p>{t('wizard.agents.install')}</p>
          <p className="mt-0.5 text-primary">{missing.map((e) => e.name).join(' · ')}</p>
          <button
            type="button"
            disabled={checking}
            onClick={async () => {
              setChecking(true);
              await reload(true);
              setChecking(false);
            }}
            data-testid="wizard-agents-recheck"
            className="mt-2 inline-flex items-center gap-1.5 text-xs font-medium text-accent hover:underline disabled:opacity-60 max-md:min-h-[44px]"
          >
            {checking && <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />}
            {t('wizard.agents.checkAgain')}
          </button>
        </div>
      )}
    </section>
  );

  return (
    <div data-testid="wizard-agents">
      {installed.length > 0 && (
        <ul className="space-y-2" data-testid="wizard-agent-list">
          {installed.map((e) => (
            <AgentRow
              key={e.slug}
              entry={e}
              checked={ticks[e.slug] ?? true}
              conversations={conversationsFor(e.slug)}
              notes={notesCount(history, e.slug)}
              breakdown={notesBreakdown(history, e.slug)}
              onToggle={(c) => onToggle(e.slug, c)}
              onChanged={() => { void reload(); }}
            />
          ))}
        </ul>
      )}
      {notFound}
    </div>
  );
}
