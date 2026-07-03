// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Task 5 (spec 009 §0.5): row-click destination from FanoutCard. Replaces the
// chat message area (never a modal — see ChatView's drillIn conditional) with
// a read-only transcript for one sub-agent handle, plus a composer gated on
// the transcript's `resumable` flag. No live stream in v1: polls the
// transcript every 3s while the handle's status is non-idle, reusing the same
// /sub-agents/status endpoint SubAgentStatusBar already polls (matches its
// pattern rather than inventing a second status source).
//
// WKWebView note (CLAUDE.md): no transform-based reveal animation for the
// swap — ChatView renders this as a plain conditional, and this component
// itself is a static layout with no enter/exit transition.

import { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronLeft, Send } from 'lucide-react';
import { api } from '../config';
import { useAgent } from '../hooks/useAgent';
import { useT } from '../i18n/useT';
import type { SubAgentTranscriptResult } from '../types';

interface SubAgentDrillInProps {
  projectId: string;
  sessionId?: string;
  handle: string;
  displayName: string;
  onBack: () => void;
}

const POLL_MS = 3000;

export default function SubAgentDrillIn({ projectId, sessionId, handle, displayName, onBack }: SubAgentDrillInProps) {
  const t = useT();
  const { getSubAgentTranscript, injectMessage } = useAgent();
  const [transcript, setTranscript] = useState<SubAgentTranscriptResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [composerText, setComposerText] = useState('');
  const [sending, setSending] = useState(false);
  const aliveRef = useRef(true);
  // Guards the 3s status poll: null when not currently polling. Cleared
  // (and the timer stopped) the moment a tick observes the handle is idle,
  // so the poll doesn't run forever once the task is done — `startStatusPoll`
  // is called again from `handleSend` to resume it, covering the case where
  // the user sends a follow-up that respawns work after it went idle.
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchTranscript = useCallback(async () => {
    try {
      const result = await getSubAgentTranscript(projectId, handle, sessionId);
      if (aliveRef.current) {
        setTranscript(result);
        setNotFound(false);
      }
    } catch {
      if (aliveRef.current) setNotFound(true);
    } finally {
      if (aliveRef.current) setLoading(false);
    }
  }, [getSubAgentTranscript, projectId, handle, sessionId]);

  useEffect(() => {
    aliveRef.current = true;
    setLoading(true);
    fetchTranscript();
    return () => {
      aliveRef.current = false;
    };
  }, [fetchTranscript]);

  // Poll every 3s while this handle's status is non-idle (matches
  // SubAgentStatusBar's polling pattern / status source). Stops itself the
  // moment a tick sees idle — otherwise this would poll the status endpoint
  // forever for as long as the drill-in view stays open, long after the
  // task finished.
  const startStatusPoll = useCallback(() => {
    if (pollTimerRef.current) return; // already polling
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    const tick = async () => {
      try {
        const data = await api<{ agents: Array<{ handle: string; status: string }> }>(
          `/api/v2/agents/${projectId}/sub-agents/status${qs}`,
        );
        const agent = data?.agents?.find((a) => a.handle === handle);
        const running = !!agent && agent.status !== 'idle';
        if (!running) {
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
          return;
        }
        if (aliveRef.current) fetchTranscript();
      } catch {
        /* daemon may be restarting — skip this tick, keep polling */
      }
    };
    pollTimerRef.current = setInterval(tick, POLL_MS);
  }, [projectId, handle, sessionId, fetchTranscript]);

  useEffect(() => {
    startStatusPoll();
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [startStatusPoll]);

  async function handleSend() {
    const text = composerText.trim();
    if (!text || sending) return;
    setSending(true);
    try {
      await injectMessage(projectId, text, handle, undefined, undefined, sessionId);
      setComposerText('');
      await fetchTranscript();
      // The poll may have stopped (handle had gone idle) — a follow-up send
      // likely respawned work, so resume it.
      startStatusPoll();
    } finally {
      if (aliveRef.current) setSending(false);
    }
  }

  const headerName = transcript?.display_name || displayName;
  const resumable = transcript?.resumable ?? false;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="shrink-0 flex items-center gap-2 px-4 py-2 border-b border-border">
        <button
          type="button"
          onClick={onBack}
          aria-label={t('fanout.drillin.back')}
          className="flex items-center gap-1 text-sm text-secondary hover:text-primary rounded px-1.5 py-1 -ml-1.5"
        >
          <ChevronLeft size={16} />
          {t('fanout.drillin.back')}
        </button>
        <span className="text-sm font-medium text-primary truncate">{headerName}</span>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4 flex flex-col gap-3">
        {loading && (
          <div className="text-sm text-secondary">{t('fanout.drillin.loading')}</div>
        )}
        {!loading && notFound && (
          <div className="text-sm text-secondary">{t('fanout.drillin.error')}</div>
        )}
        {!loading && !notFound && transcript && transcript.entries.length === 0 && (
          <div className="text-sm text-secondary">{t('fanout.drillin.empty')}</div>
        )}
        {!loading && !notFound && transcript?.entries.map((entry, idx) => (
          <div key={`${entry.timestamp}-${idx}`} data-testid="drillin-entry" className="text-sm">
            <div className="text-xs text-secondary font-mono mb-0.5">{entry.source}</div>
            <div className="whitespace-pre-wrap break-words text-primary">{entry.content}</div>
          </div>
        ))}
      </div>

      <div className="shrink-0 px-4 pb-4 pt-2">
        <div className="relative flex items-center gap-2 bg-background border border-border rounded-[10px] shadow-lg px-3 py-2">
          <input
            type="text"
            data-testid="drillin-composer-input"
            value={composerText}
            onChange={(e) => setComposerText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={!resumable}
            placeholder={resumable ? undefined : t('fanout.drillin.readonly')}
            className="flex-1 text-[13px] bg-transparent focus:outline-none disabled:opacity-50"
          />
          <button
            type="button"
            data-testid="drillin-composer-send"
            onClick={handleSend}
            disabled={!resumable || !composerText.trim() || sending}
            aria-label={t('chat.send')}
            className="shrink-0 p-1.5 rounded-lg text-accent hover:bg-accent/10 disabled:opacity-40 disabled:cursor-default"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
