// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * LeftOffCards — "pick up where you left off" in a project's new-session empty
 * chat (spec 138). They replace the example prompts once the project has any
 * sessions, Orbital's own or imported from another agent's history:
 *
 *   - Last time: your last message verbatim, which agent it went to and when,
 *     plus the branch / uncommitted / commits-since chips that are non-zero.
 *   - Didn't finish: up to three abnormal stops, each with its cause pill.
 *
 * Each card is one button that opens that session through the app's normal
 * session-select navigation (`onOpen`); imported `imp_…` ids materialise on
 * open. There is no agent-switch logic here — the cause pill says why it
 * stopped and the user picks the agent in the pin switcher.
 *
 * GET /projects/{pid}/left-off answers fast (the scan is cached), so nothing
 * renders while it loads — showing the examples and then swapping them for
 * cards would flash. A failed or empty answer renders `fallback` (today's
 * example prompts). The `data-left-off-state` markers let the tour wait for
 * the answer before it fixes its stops.
 */

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ArrowRight } from 'lucide-react';
import { api } from '../config';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import type { StringKey } from '../i18n/strings';
import { formatRelativeTime } from '../utils/relativeTime';
import { displayAnswerText } from '../utils/askAnswer';
import {
  agentDisplayName,
  hasLeftOff,
  parseLeftOff,
  type LeftOffResponse,
  type LeftOffUnfinished,
  type StopCause,
} from '../utils/onboarding';
import { sendTelemetry } from '../utils/telemetry';
import MessageAvatar from './MessageAvatar';

interface LeftOffCardsProps {
  projectId: string;
  /** Installed sub-agents, for display names. */
  agents: ReadonlyArray<{ slug: string; name: string }>;
  onOpen: (sessionId: string) => void;
  /** What the empty chat shows when there is nothing to pick up. */
  fallback: ReactNode;
}

const CAUSE_KEY: Record<StopCause, StringKey> = {
  usage_limit: 'leftOff.cause.usageLimit',
  model_unavailable: 'leftOff.cause.modelUnavailable',
  network_auth: 'leftOff.cause.networkAuth',
  interrupted: 'leftOff.cause.interrupted',
  no_reply: 'leftOff.cause.noReply',
};

const sectionLabel = 'block text-secondary font-semibold';
const sectionLabelStyle = { fontSize: '9.5px', letterSpacing: '0.8px', textTransform: 'uppercase' } as const;
const cardShell =
  'group flex w-full flex-col gap-1.5 surface-float px-3.5 py-3 text-left active:scale-[0.99] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 motion-reduce:transition-none motion-reduce:active:scale-100';
const chip = 'rounded-full border border-border bg-sidebar px-2 py-0.5 text-[11px] text-secondary';

export default function LeftOffCards({ projectId, agents, onOpen, fallback }: LeftOffCardsProps) {
  const t = useT();
  const { locale } = useLocale();
  const [state, setState] = useState<{ projectId: string; data: LeftOffResponse | null } | null>(null);
  const shownRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api<unknown>(`/api/v2/projects/${encodeURIComponent(projectId)}/left-off`)
      .then((raw) => { if (!cancelled) setState({ projectId, data: parseLeftOff(raw) }); })
      .catch(() => { if (!cancelled) setState({ projectId, data: null }); });
    return () => { cancelled = true; };
  }, [projectId]);

  const loaded = state !== null && state.projectId === projectId;
  const data = loaded ? state.data : null;
  const showCards = loaded && hasLeftOff(data);

  useEffect(() => {
    if (!showCards || shownRef.current === projectId) return;
    shownRef.current = projectId;
    sendTelemetry('leftoff_shown');
  }, [showCards, projectId]);

  if (!loaded) return <div data-left-off-state="loading" aria-hidden="true" />;
  // The marker tells the tour the answer is in and there are no cards.
  if (!showCards || !data) return <>{fallback}<span hidden data-left-off-state="none" /></>;

  const name = (slug: string | null) => agentDisplayName(slug, agents);
  const timeOf = (iso: string) =>
    new Date(iso).toLocaleTimeString(locale === 'zh' ? 'zh-CN' : 'en-US', { hour: 'numeric', minute: '2-digit' });
  const causeText = (r: LeftOffUnfinished) =>
    r.cause === 'usage_limit' && r.resets_at
      ? t('leftOff.cause.usageLimitAt', { time: timeOf(r.resets_at) })
      : t(CAUSE_KEY[r.cause]);
  const open = (
    <span className="ml-auto inline-flex shrink-0 items-center gap-1 text-xs font-medium text-secondary transition-colors duration-150 group-hover:text-accent">
      {t('leftOff.open')}
      <ArrowRight size={12} aria-hidden="true" />
    </span>
  );
  const quote = (text: string) => (
    <span className="line-clamp-3 text-[13px] leading-snug text-primary [overflow-wrap:anywhere]">{`“${text}”`}</span>
  );

  const last = data.last_time;
  const git = last?.git ?? null;

  return (
    <div data-tour="left-off" data-testid="left-off" className="mt-6 flex w-full flex-col gap-4 text-left">
      {last && (
        <div className="flex flex-col gap-2">
          <span aria-hidden="true" className={sectionLabel} style={sectionLabelStyle}>{t('leftOff.lastTime')}</span>
          <button
            type="button"
            data-testid="left-off-last"
            className={`${cardShell} max-md:min-h-[44px]`}
            onClick={() => {
              sendTelemetry('leftoff_opened', { kind: 'last_time' });
              onOpen(last.session_id);
            }}
          >
            <span className="flex w-full items-center gap-2">
              <MessageAvatar variant="agent" agentHandle={last.agent ?? undefined} />
              <span className="min-w-0 truncate text-xs text-secondary">
                {t('leftOff.youTo', { agent: name(last.agent), when: formatRelativeTime(last.at, t) })}
              </span>
              {open}
            </span>
            {quote(displayAnswerText(last.user_text))}
            {git && (git.branch || git.uncommitted > 0 || git.commits_since > 0) && (
              <span className="flex flex-wrap gap-1.5">
                {git.branch && (
                  <span data-testid="left-off-chip-branch" className={`${chip} max-w-full truncate font-mono`}>{git.branch}</span>
                )}
                {git.uncommitted > 0 && (
                  <span data-testid="left-off-chip-uncommitted" className={chip}>
                    {t(git.uncommitted === 1 ? 'leftOff.uncommitted.one' : 'leftOff.uncommitted.other', { n: git.uncommitted })}
                  </span>
                )}
                {git.commits_since > 0 && (
                  <span data-testid="left-off-chip-commits" className={chip}>
                    {t(git.commits_since === 1 ? 'leftOff.commitsSince.one' : 'leftOff.commitsSince.other', { n: git.commits_since })}
                  </span>
                )}
              </span>
            )}
          </button>
        </div>
      )}
      {data.unfinished.length > 0 && (
        <div className="flex flex-col gap-2" data-testid="left-off-unfinished">
          <span aria-hidden="true" className={sectionLabel} style={sectionLabelStyle}>{t('leftOff.unfinished')}</span>
          {data.unfinished.map((r) => (
            <button
              key={r.session_id}
              type="button"
              data-testid={`left-off-unfinished-row-${r.session_id}`}
              className={`${cardShell} max-md:min-h-[44px]`}
              onClick={() => {
                sendTelemetry('leftoff_opened', { kind: 'unfinished', cause: r.cause });
                onOpen(r.session_id);
              }}
            >
              <span className="flex w-full flex-wrap items-center gap-x-2 gap-y-1">
                <MessageAvatar variant="agent" agentHandle={r.agent ?? undefined} />
                <span className="min-w-0 truncate text-xs text-secondary">
                  {t('leftOff.agentWhen', { agent: name(r.agent), when: formatRelativeTime(r.at, t) })}
                </span>
                <span
                  data-testid={`left-off-cause-${r.session_id}`}
                  className={`shrink-0 rounded-full border px-2 py-0.5 text-[11px] ${
                    r.cause === 'usage_limit'
                      ? 'border-warning/30 bg-warning-surface text-warning-text'
                      : 'border-border bg-sidebar text-secondary'
                  }`}
                >
                  {causeText(r)}
                </span>
                {open}
              </span>
              {quote(displayAnswerText(r.user_text))}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
