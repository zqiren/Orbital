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
// Round 2 (2026-07-05, Task D, issues 2+3): a worker transcript whose
// `session_uuid` is discoverable (live via the fanout registry mid-batch,
// disk fallback afterward — see agents_v2.py's transcript endpoint) is its
// OWN recorded chat session. Rendering it through the flat `entries` list
// looked nothing like the main chat and showed "(no transcript yet)" for a
// still-running worker (issue 2) whose entries hadn't landed yet even though
// its session file already had content (issue 3). D2 fetches that session's
// chat history and renders it via the same `transformChatHistory` +
// `ChatMessage` the main conversation uses. Falls back to the original flat
// entries rendering when `session_uuid` is absent (old fanouts predating the
// field) or `kind !== 'worker'` (cli handles keep their existing behavior
// unconditionally, even if a session_uuid were ever present on one).
//
// WKWebView note (CLAUDE.md): no transform-based reveal animation for the
// swap — ChatView renders this as a plain conditional, and this component
// itself is a static layout with no enter/exit transition.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, Send } from 'lucide-react';
import { api } from '../config';
import { useAgent } from '../hooks/useAgent';
import { useT } from '../i18n/useT';
import type { ChatMessage as ChatMessageType, SubAgentTranscriptResult } from '../types';
import type { StringKey } from '../i18n/strings';
import { transformChatHistory, type DisplayItem } from '../utils/chatTransform';
import ChatMessage from './ChatMessage';
import { useWebSocket } from '../hooks/useWebSocket';
import type { FanoutTaskUpdateEvent, StreamDeltaEvent, WebSocketEvent } from '../types';

interface SubAgentDrillInProps {
  projectId: string;
  sessionId?: string;
  handle: string;
  displayName: string;
  onBack: () => void;
}

const POLL_MS = 3000;

type WorkerCapsuleItem = Extract<DisplayItem, { type: 'agent_run' }>;
type WorkerMessageItem = Extract<
  DisplayItem,
  { type: 'user_message' | 'agent_message' | 'sub_agent_message' }
>;
type Translator = (key: StringKey, vars?: Record<string, string | number>) => string;

function formatToolBreakdown(counts: Record<string, number>): string {
  const entries = Object.entries(counts);
  if (entries.length === 0) return '';
  entries.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return entries.map(([n, c]) => (c === 1 ? n : `${c} ${n}s`)).join(', ');
}

function formatCapsuleDuration(startedAt: number, endedAt: number | null, lessThan1s: string): string {
  if (!endedAt || endedAt <= startedAt) return lessThan1s;
  const ms = endedAt - startedAt;
  if (ms < 1000) return lessThan1s;
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  return rs ? `${m}m ${rs}s` : `${m}m`;
}

/**
 * Compact, read-only rendering of an `agent_run` capsule inside the
 * chat-shaped worker view. Duplicates ChatView's capsule class strings and
 * `capsuleSummaryText` format verbatim (ChatView.tsx ~2643-2731 /
 * capsuleSummaryText ~83-92) rather than importing — ChatView exports
 * neither as a reusable component, and this read-only view has no
 * expand/collapse or live-status-overlay concerns a worker's own transcript
 * is always a finished turn once it appears here. Flagged for a later
 * shared-renderer refactor (same note the plan calls out).
 */
function WorkerCapsule({ capsule, t }: { capsule: WorkerCapsuleItem; t: Translator }) {
  const breakdown = formatToolBreakdown(capsule.tool_call_count_by_name);
  const duration = formatCapsuleDuration(capsule.started_at, capsule.ended_at, t('duration.lessThan1s'));
  const head = breakdown || (capsule.has_thinking ? t('chat.capsule.thinking') : t('chat.capsule.agentStep'));
  let summary = `${head} · ${duration}`;
  if (capsule.status === 'error' || capsule.status === 'stopped') {
    summary += ` ${t('chat.capsule.stoppedAtError')}`;
  }
  return (
    <div data-testid="drillin-capsule" className="ml-9 flex gap-[10px]">
      <div className="w-0.5 shrink-0 rounded-sm bg-border" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 w-full text-left font-mono text-[11.5px] text-secondary">
          <span className="truncate text-primary font-medium">{summary}</span>
        </div>
      </div>
    </div>
  );
}

export default function SubAgentDrillIn({ projectId, sessionId, handle, displayName, onBack }: SubAgentDrillInProps) {
  const t = useT();
  const { getSubAgentTranscript, injectMessage } = useAgent();
  const [transcript, setTranscript] = useState<SubAgentTranscriptResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [composerText, setComposerText] = useState('');
  const [sending, setSending] = useState(false);
  // Round 2 (Task D2): the worker's own chat history, fetched when the
  // transcript names a `session_uuid` for a `kind === 'worker'` handle. null
  // means "not applicable" (no session_uuid, or a cli handle, or the fetch
  // hasn't resolved yet) — the render falls back to `transcript.entries` in
  // all of those cases.
  const [workerMessages, setWorkerMessages] = useState<ChatMessageType[] | null>(null);
  // Live streaming buffer: worker chat.stream_delta text accumulates here
  // while the worker's turn is in flight (the session JSONL only receives
  // COMPLETE messages, so without this the view sits still through a long
  // generation). Cleared when the final delta lands and the persisted
  // message supersedes it (refetch), or on a terminal fanout.task_update.
  const [liveStream, setLiveStream] = useState('');
  const { on, off } = useWebSocket();
  // The worker session id deltas are addressed to — set by fetchTranscript,
  // read synchronously inside WS handlers (which close over the ref).
  const sessionUuidRef = useRef<string | null>(null);
  const liveStreamStartedAtRef = useRef<string>('');
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
      if (!aliveRef.current) return;
      setTranscript(result);
      setNotFound(false);
      sessionUuidRef.current =
        result.kind === 'worker' && result.session_uuid ? result.session_uuid : null;
      if (result.kind === 'worker' && result.session_uuid) {
        // The worker's own recorded chat session — same endpoint ChatView
        // uses for the main conversation, scoped via ?session_id to the
        // worker's F2 stem (agents_v2.py's direct-file-match path resolves
        // it). Refetched on every poll tick (see startStatusPoll below) so a
        // still-running worker's transcript fills in live, fixing issue 3
        // ("(no transcript yet)" for a worker that's actively producing one).
        try {
          const messages = await api<ChatMessageType[]>(
            `/api/v2/agents/${encodeURIComponent(projectId)}/chat?session_id=${encodeURIComponent(result.session_uuid)}`,
          );
          if (aliveRef.current) setWorkerMessages(messages);
        } catch {
          if (aliveRef.current) setWorkerMessages(null);
        }
      } else {
        setWorkerMessages(null);
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

  // Live streaming: the backend addresses worker deltas to the WORKER's own
  // session_uuid (never the viewed chat session), so filtering on
  // sessionUuidRef makes this subscription see exactly this worker's stream.
  // On the final delta (or a terminal fanout.task_update for this handle,
  // which covers a missed final), the persisted message is about to appear in
  // the session JSONL — drop the buffer and refetch so the durable rendering
  // takes over without a duplicate.
  useEffect(() => {
    function handleDelta(event: WebSocketEvent) {
      const e = event as StreamDeltaEvent;
      if (!e.session_id || e.session_id !== sessionUuidRef.current) return;
      if (e.is_final) {
        setLiveStream('');
        liveStreamStartedAtRef.current = '';
        fetchTranscript();
        return;
      }
      if (e.text) {
        if (!liveStreamStartedAtRef.current) {
          liveStreamStartedAtRef.current = new Date().toISOString();
        }
        setLiveStream((prev) => prev + e.text);
      }
    }
    function handleTaskUpdate(event: WebSocketEvent) {
      const e = event as FanoutTaskUpdateEvent;
      if (e.handle !== handle) return;
      if (e.status !== 'running') {
        setLiveStream('');
        liveStreamStartedAtRef.current = '';
        fetchTranscript();
      }
    }
    on('chat.stream_delta', handleDelta);
    on('fanout.task_update', handleTaskUpdate);
    return () => {
      off('chat.stream_delta', handleDelta);
      off('fanout.task_update', handleTaskUpdate);
    };
  }, [on, off, handle, fetchTranscript]);

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

  // `transformChatHistory`'s translator param defaults to English
  // (ship-English-first, per CLAUDE.md's i18n rules) — acceptable here since
  // the worker's own chat history is read-only debug-adjacent content, not
  // core chat chrome. null (rather than []) distinguishes "no worker chat to
  // show" from "worker chat is empty", so the flat-entries fallback below can
  // tell the two apart.
  const workerItems = useMemo<DisplayItem[] | null>(() => {
    if (!workerMessages) return null;
    return transformChatHistory(workerMessages, undefined);
  }, [workerMessages]);

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
        {!loading && !notFound && workerItems && workerItems.length === 0 && (
          <div className="text-sm text-secondary">{t('fanout.drillin.empty')}</div>
        )}
        {!loading && !notFound && workerItems && workerItems.map((item, idx) => {
          if (
            item.type === 'user_message' ||
            item.type === 'agent_message' ||
            item.type === 'sub_agent_message'
          ) {
            return <ChatMessage key={`wi-${idx}`} message={item as WorkerMessageItem} />;
          }
          if (item.type === 'agent_run') {
            return <WorkerCapsule key={`wi-${idx}`} capsule={item} t={t} />;
          }
          // Other item types (fanout_card, approval_card, budget_event, …)
          // are not expected inside a worker's own transcript in v1 — skip
          // rather than render a mismatched component (D2).
          return null;
        })}
        {!loading && !notFound && liveStream && (
          <div data-testid="drillin-live-stream">
            <ChatMessage
              message={{
                type: 'agent_message',
                content: liveStream,
                source: handle,
                timestamp: liveStreamStartedAtRef.current,
              } as WorkerMessageItem}
            />
          </div>
        )}
        {!loading && !notFound && !workerItems && transcript && transcript.entries.length === 0 && (
          <div className="text-sm text-secondary">{t('fanout.drillin.empty')}</div>
        )}
        {!loading && !notFound && !workerItems && transcript?.entries.map((entry, idx) => (
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
