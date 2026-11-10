// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Piece 3 Part D: honest sub-agent status badge + user stop control.
//
// Status vocabulary (SDK claude-code; other transports never report
// background-running):
//   running            — turn open, agent producing output ("Working")
//     + awaiting_approval — the open turn waits on the user's permission
//                          answer ("Needs you"; still dot, no ring)
//   background-running — turn done BUT tracked background work is alive
//                        ("Background"). Looks done, isn't done.
//   idle               — turn done, nothing real underneath.
//
// The stop button cancels the turn + kills the agent + its TRACKED children,
// with an honest warning: work detached via raw shell '&'/'nohup' escapes the
// process tree and cannot be guaranteed stopped (accepted limitation, see
// REPORT-piece3-child-classification.md).

import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { api } from '../config';
import { useT } from '../i18n/useT';
import {
  useSubAgentStatus,
  type SubAgentInfo,
  type SubAgentStatusState,
} from '../hooks/useSubAgentStatus';

interface StopResult {
  handle: string;
  status: string;
  background_terminated: string[];
  warning: string;
}

interface Props {
  projectId: string;
  sessionId?: string;
  /** Spec 099: the session's status from a parent-owned useSubAgentStatus
   *  (ChatView shares it with the pin control's running dot). Omitted, the
   *  bar runs its own. */
  live?: SubAgentStatusState;
}

/** What the chip shows: the reported status, or `needs-you` while the open
 *  turn waits on the user's approval. A status this build does not know
 *  (a newer daemon) shows as idle. */
type ChipState = 'running' | 'needs-you' | 'background-running' | 'idle';

function chipState(a: SubAgentInfo): ChipState {
  if (a.status === 'running') return a.awaiting_approval === true ? 'needs-you' : 'running';
  if (a.status === 'background-running') return 'background-running';
  return 'idle';
}

const BADGE_LABEL_KEY: Record<ChipState, 'subAgentBar.working' | 'subAgentBar.needsYou' | 'subAgentBar.background' | 'subAgentBar.idle'> = {
  running: 'subAgentBar.working',
  'needs-you': 'subAgentBar.needsYou',
  'background-running': 'subAgentBar.background',
  idle: 'subAgentBar.idle',
};

/** The dot before the status word (spec 145): green while its turn runs,
 *  with the pin switcher's moving ring; amber and still when it waits on
 *  the user (the word in amber too, as the queue's "needs you"); amber for
 *  work alive after the turn; grey when idle. The ring means "producing",
 *  so only `running` has it. Stable state colours at readable contrast in
 *  both themes; the chip's name carries the identity. */
const DOT_CLASS: Record<ChipState, string> = {
  running: 'bg-success-text',
  'needs-you': 'bg-warning-text',
  'background-running': 'bg-warning-text',
  idle: 'bg-idle',
};

export default function SubAgentStatusBar({ projectId, sessionId, live }: Props) {
  const t = useT();
  const own = useSubAgentStatus(projectId, sessionId, { enabled: !live });
  const { agents, refresh } = live ?? own;
  const [confirming, setConfirming] = useState<SubAgentInfo | null>(null);
  const [stopping, setStopping] = useState<string | null>(null);
  const [lastStop, setLastStop] = useState<StopResult | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  const doStop = useCallback(async (agent: SubAgentInfo) => {
    setStopping(agent.handle);
    setConfirming(null);
    try {
      const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      const result = await api<StopResult>(
        `/api/v2/agents/${projectId}/sub-agents/${agent.handle}/stop${qs}`,
        { method: 'POST' },
      );
      setLastStop(result);
      setTimeout(() => { if (alive.current) setLastStop(null); }, 8000);
    } catch {
      /* surfaced via lifecycle message in chat regardless */
    } finally {
      if (alive.current) {
        setStopping(null);
        refresh();
      }
    }
  }, [projectId, sessionId, refresh]);

  // Spec 169 C: nothing left to stop — the confirmed agent left the line or
  // went idle — closes the dialog (adjusting state during render, not in an
  // effect, so it never paints a frame asking to stop an idle worker).
  if (confirming) {
    const current = agents.find((a) => a.handle === confirming.handle);
    if (!current || chipState(current) === 'idle') setConfirming(null);
  }

  if (agents.length === 0 && !lastStop) return null;

  // A line of the session heading, not a bar of its own (spec 137 follow-up,
  // C4 of the heading options): while a worker runs, the agents sit under
  // the session title as flat tiles, and the messages scroll under a fade
  // below this line instead of the heading's (`.sub-agent-line` in
  // index.css). Same horizontal inset as the title and the messages.
  return (
    <div
      data-testid="sub-agent-status-bar"
      className="sub-agent-line relative z-10 shrink-0 flex items-center gap-2 overflow-x-auto px-4 pt-2 pb-2.5 md:px-7 md:pt-0.5 bg-card text-xs"
    >
      {agents.map((a) => {
        const state = chipState(a);
        return (
        <div
          key={a.handle}
          data-testid={`sub-agent-chip-${a.handle}`}
          className="flex items-center gap-1.5 shrink-0 rounded-md bg-background dark:bg-elevated pl-2 pr-1 py-0.5 text-primary"
        >
          <span className="max-w-[96px] truncate">{a.display_name}</span>
          <span
            data-testid={`sub-agent-badge-${a.handle}`}
            data-status={a.status}
            data-state={state}
            className={`inline-flex items-center gap-1.5 text-2xs leading-4 ${
              state === 'needs-you' ? 'text-warning' : 'text-secondary'}`}
          >
            {/* The dot sits in a 12px box so the ring grows inside room of
                its own, clear of the name and the word; the box is under
                the badge's 16px line, so the chip keeps its height. */}
            <span aria-hidden="true" className="relative grid place-items-center w-3 h-3 shrink-0">
              {state === 'running' && (
                <span
                  data-testid={`sub-agent-ring-${a.handle}`}
                  className="absolute inset-[2.5px] rounded-full border-[1.5px] border-success-text
                             motion-safe:animate-ping [animation-duration:1.6s] motion-reduce:opacity-40"
                />
              )}
              <span
                data-testid={`sub-agent-dot-${a.handle}`}
                aria-hidden="true"
                className={`relative w-[7px] h-[7px] rounded-full ${DOT_CLASS[state]}`}
              />
            </span>
            {t(BADGE_LABEL_KEY[state])}
          </span>
          {state !== 'idle' && (
            <button
              data-testid={`sub-agent-stop-${a.handle}`}
              title={t('subAgentBar.stopTitle', { name: a.display_name })}
              aria-label={t('subAgentBar.stopTitle', { name: a.display_name })}
              disabled={stopping === a.handle}
              onClick={() => setConfirming(a)}
              className="grid place-items-center h-5 w-5 rounded text-[9px] text-secondary hover:text-error-text hover:bg-error-surface disabled:opacity-40"
            >
              ■
            </button>
          )}
        </div>
        );
      })}

      {lastStop && (
        <span data-testid="sub-agent-stop-result" className="text-secondary shrink-0">
          {t('subAgentBar.stopped', { handle: lastStop.handle })}
          {lastStop.background_terminated.length > 0 &&
            t('subAgentBar.bgTerminated', { n: lastStop.background_terminated.length })}
        </span>
      )}

      {/* Spec 169 C: portalled to <body>, as CardList's dialogs are. Inside
          the chip line it sat under the line's overflow box and z-10
          stacking context: the packaged app's WKWebView clipped the
          viewport backdrop to the line (a grey band that ate clicks, the
          dialog itself off-strip), and elsewhere the heading stayed undimmed
          above it. React context (i18n) carries through the portal; theme
          tokens live on :root. */}
      {confirming && createPortal(
        <div
          data-testid="sub-agent-stop-confirm"
          className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4"
          onClick={() => setConfirming(null)}
        >
          <div
            className="w-full max-w-sm surface-float-pop p-4 text-sm text-primary"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="font-medium mb-2">{t('subAgentBar.confirm.title', { name: confirming.display_name })}</div>
            <p className="text-secondary mb-2">
              {confirming.status === 'running'
                ? t('subAgentBar.confirm.running')
                : t('subAgentBar.confirm.background')}
            </p>
            {confirming.status === 'background-running' &&
              confirming.background_commands.length > 0 && (
              <ul className="mb-2 max-h-24 overflow-y-auto rounded bg-background p-2 font-mono text-[11px] text-warning-text">
                {confirming.background_commands.map((c, i) => (
                  <li key={i} className="truncate">{c}</li>
                ))}
              </ul>
            )}
            <p data-testid="sub-agent-stop-warning" className="text-[11px] text-secondary mb-3">
              {t('subAgentBar.confirm.warning')}
            </p>
            <div className="flex justify-end gap-2">
              <button
                data-testid="sub-agent-stop-cancel"
                onClick={() => setConfirming(null)}
                className="rounded px-3 py-1.5 text-secondary hover:bg-card-hover"
              >
                {t('subAgentBar.confirm.cancel')}
              </button>
              <button
                data-testid="sub-agent-stop-confirm-button"
                onClick={() => doStop(confirming)}
                className="rounded bg-error px-3 py-1.5 text-white hover:opacity-90"
              >
                {t('subAgentBar.confirm.stop')}
              </button>
            </div>
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}
