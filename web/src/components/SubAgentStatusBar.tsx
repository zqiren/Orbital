// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Piece 3 Part D: honest sub-agent status badge + user stop control.
//
// Status vocabulary (SDK claude-code; other transports never report
// background-running):
//   running            — turn open, agent producing output ("Working")
//   background-running — turn done BUT tracked background work is alive
//                        ("Background"). Looks done, isn't done.
//   idle               — turn done, nothing real underneath.
//
// The stop button cancels the turn + kills the agent + its TRACKED children,
// with an honest warning: work detached via raw shell '&'/'nohup' escapes the
// process tree and cannot be guaranteed stopped (accepted limitation, see
// REPORT-piece3-child-classification.md).

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../config';
import { useWebSocket } from '../hooks/useWebSocket';
import { useT } from '../i18n/useT';
import { isWorkerHandle } from '../utils/subAgentHandle';
import type { SubAgentRunStatus, WebSocketEvent } from '../types';

interface SubAgentInfo {
  handle: string;
  display_name: string;
  status: SubAgentRunStatus;
  background_commands: string[];
}

interface StopResult {
  handle: string;
  status: string;
  background_terminated: string[];
  warning: string;
}

interface Props {
  projectId: string;
  sessionId?: string;
}

const BADGE_LABEL_KEY: Record<SubAgentInfo['status'], 'subAgentBar.working' | 'subAgentBar.background' | 'subAgentBar.idle'> = {
  running: 'subAgentBar.working',
  'background-running': 'subAgentBar.background',
  idle: 'subAgentBar.idle',
};

const BADGE_CLASS: Record<SubAgentInfo['status'], string> = {
  running: 'bg-blue-500/15 text-blue-400 border-blue-500/40 animate-pulse',
  'background-running': 'bg-amber-500/15 text-amber-400 border-amber-500/40',
  idle: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
};

export default function SubAgentStatusBar({ projectId, sessionId }: Props) {
  const t = useT();
  const [agents, setAgents] = useState<SubAgentInfo[]>([]);
  const [confirming, setConfirming] = useState<SubAgentInfo | null>(null);
  const [stopping, setStopping] = useState<string | null>(null);
  const [lastStop, setLastStop] = useState<StopResult | null>(null);
  const { on, off } = useWebSocket();
  const alive = useRef(true);

  const abortRef = useRef<AbortController | null>(null);
  const refresh = useCallback(async () => {
    // Bug #48 (fix C): a session switch re-fires this; abort the superseded
    // request instead of letting discarded responses pile up.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      const data = await api<{ session_id: string | null; agents: SubAgentInfo[] }>(
        `/api/v2/agents/${projectId}/sub-agents/status${qs}`,
        { signal: controller.signal },
      );
      if (alive.current) {
        // Fanout workers (spec 009 §0.5) get their own live surface —
        // FanoutCard — and must NOT also show up as chips here.
        const next = (data?.agents ?? []).filter((a) => !isWorkerHandle(a.handle));
        // Identity-stable update: unchanged payloads keep the previous array
        // reference so effects keyed on state don't re-fire (and unstable
        // hook identities — e.g. test mocks recreating on/off per render —
        // cannot produce a render loop through this setState).
        setAgents((prev) =>
          JSON.stringify(prev) === JSON.stringify(next) ? prev : next,
        );
      }
    } catch {
      /* daemon may be restarting — badge just goes quiet */
    }
  }, [projectId, sessionId]);

  // Refresh on every sub-agent lifecycle signal + light poll while non-idle.
  useEffect(() => {
    alive.current = true;
    refresh();
    const handler = (e: WebSocketEvent) => {
      if ('project_id' in e && e.project_id && e.project_id !== projectId) return;
      refresh();
    };
    const events = [
      'sub_agent.started', 'sub_agent.completed', 'sub_agent.error',
      'sub_agent.failed', 'sub_agent.stopped', 'sub_agent.turn_interrupted',
      'chat.sub_agent_message', 'agent.status',
    ] as const;
    events.forEach((ev) => on(ev, handler));
    return () => {
      alive.current = false;
      abortRef.current?.abort();
      events.forEach((ev) => off(ev, handler));
    };
  }, [projectId, on, off, refresh]);

  useEffect(() => {
    const anyActive = agents.some((a) => a.status !== 'idle');
    if (!anyActive) return;
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [agents, refresh]);

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

  if (agents.length === 0 && !lastStop) return null;

  return (
    <div
      data-testid="sub-agent-status-bar"
      className="flex items-center gap-2 overflow-x-auto px-3 py-1.5 border-b border-zinc-800 bg-zinc-900/60 text-xs"
    >
      {agents.map((a) => (
        <div
          key={a.handle}
          data-testid={`sub-agent-chip-${a.handle}`}
          className="flex items-center gap-1.5 shrink-0 rounded-full border border-zinc-700 bg-zinc-800/80 pl-2 pr-1 py-0.5"
        >
          <span className="text-zinc-300 max-w-[96px] truncate">{a.display_name}</span>
          <span
            data-testid={`sub-agent-badge-${a.handle}`}
            data-status={a.status}
            className={`rounded-full border px-1.5 py-px text-2xs leading-4 ${BADGE_CLASS[a.status]}`}
          >
            {t(BADGE_LABEL_KEY[a.status])}
          </span>
          {a.status !== 'idle' && (
            <button
              data-testid={`sub-agent-stop-${a.handle}`}
              title={t('subAgentBar.stopTitle', { name: a.display_name })}
              aria-label={t('subAgentBar.stopTitle', { name: a.display_name })}
              disabled={stopping === a.handle}
              onClick={() => setConfirming(a)}
              className="grid place-items-center h-5 w-5 rounded-full text-red-400 hover:bg-red-500/20 disabled:opacity-40"
            >
              ■
            </button>
          )}
        </div>
      ))}

      {lastStop && (
        <span data-testid="sub-agent-stop-result" className="text-zinc-400 shrink-0">
          {t('subAgentBar.stopped', { handle: lastStop.handle })}
          {lastStop.background_terminated.length > 0 &&
            t('subAgentBar.bgTerminated', { n: lastStop.background_terminated.length })}
        </span>
      )}

      {confirming && (
        <div
          data-testid="sub-agent-stop-confirm"
          className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4"
          onClick={() => setConfirming(null)}
        >
          <div
            className="w-full max-w-sm rounded-lg border border-zinc-700 bg-zinc-900 p-4 text-sm text-zinc-200"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="font-medium mb-2">{t('subAgentBar.confirm.title', { name: confirming.display_name })}</div>
            <p className="text-zinc-400 mb-2">
              {confirming.status === 'running'
                ? t('subAgentBar.confirm.running')
                : t('subAgentBar.confirm.background')}
            </p>
            {confirming.status === 'background-running' &&
              confirming.background_commands.length > 0 && (
              <ul className="mb-2 max-h-24 overflow-y-auto rounded bg-zinc-800/80 p-2 font-mono text-[11px] text-amber-300">
                {confirming.background_commands.map((c, i) => (
                  <li key={i} className="truncate">{c}</li>
                ))}
              </ul>
            )}
            <p data-testid="sub-agent-stop-warning" className="text-[11px] text-zinc-500 mb-3">
              {t('subAgentBar.confirm.warning')}
            </p>
            <div className="flex justify-end gap-2">
              <button
                data-testid="sub-agent-stop-cancel"
                onClick={() => setConfirming(null)}
                className="rounded px-3 py-1.5 text-zinc-300 hover:bg-zinc-800"
              >
                {t('subAgentBar.confirm.cancel')}
              </button>
              <button
                data-testid="sub-agent-stop-confirm-button"
                onClick={() => doStop(confirming)}
                className="rounded bg-red-600 px-3 py-1.5 text-white hover:bg-red-500"
              >
                {t('subAgentBar.confirm.stop')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
