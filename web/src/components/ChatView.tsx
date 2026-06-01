// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Send, Square, Loader2, Plus, ChevronRight, ChevronDown } from 'lucide-react';
import { api, apiWithTotal, BASE_URL, isRelayMode } from '../config';
import { useWebSocket } from '../hooks/useWebSocket';
import { useAgent } from '../hooks/useAgent';
import { useQueue } from '../hooks/useQueue';
import { transformChatHistory, truncateResult } from '../utils/chatTransform';
import type { DisplayItem } from '../utils/chatTransform';
import AttachmentChip from './AttachmentChip';
import { uploadFile } from '../lib/attachment-upload';
import { buildAttachmentsBlock } from '../lib/attachment-parsing';
import ComposerDisabledPrompt from './ComposerDisabledPrompt';

const MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024;
const MAX_ATTACHMENTS = 10;

interface PendingAttachment {
  id: string;
  file: File;
  filename: string;
  mime: string;
  size: number;
  status: 'pending' | 'uploading' | 'done' | 'error';
  uploadedPath?: string;
  thumbnailUrl?: string;
  errorMessage?: string;
}

function genId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

type AgentRunItem = Extract<DisplayItem, { type: 'agent_run' }>;
type CapsuleChild = AgentRunItem['items'][number];

function formatToolBreakdown(counts: Record<string, number>): string {
  const entries = Object.entries(counts);
  if (entries.length === 0) return '';
  entries.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return entries.map(([n, c]) => (c === 1 ? n : `${c} ${n}s`)).join(', ');
}

function formatDuration(startedAt: number, endedAt: number | null): string {
  if (!endedAt || endedAt <= startedAt) return '<1s';
  const ms = endedAt - startedAt;
  if (ms < 1000) return '<1s';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  return rs ? `${m}m ${rs}s` : `${m}m`;
}

function capsuleSummaryText(capsule: AgentRunItem): string {
  const breakdown = formatToolBreakdown(capsule.tool_call_count_by_name);
  const dur = formatDuration(capsule.started_at, capsule.ended_at);
  const head = breakdown || (capsule.has_thinking ? 'thinking' : 'agent step');
  let line = `${head} · ${dur}`;
  if (capsule.status === 'error' || capsule.status === 'stopped') {
    line += ' · stopped at error';
  }
  return line;
}

function getLiveRunningCapsule(
  items: DisplayItem[],
): { idx: number; capsule: AgentRunItem } | null {
  if (items.length === 0) return null;
  const last = items[items.length - 1];
  if (last.type === 'agent_run' && last.status === 'running') {
    return { idx: items.length - 1, capsule: last };
  }
  return null;
}

function appendToLiveCapsule(
  prev: DisplayItem[],
  child: CapsuleChild,
  timestamp: string,
): DisplayItem[] {
  const ms = Date.parse(timestamp);
  const live = getLiveRunningCapsule(prev);
  if (!live) {
    const id = `cap:live:${ms}:${Math.random().toString(36).slice(2, 8)}`;
    const counts: Record<string, number> = {};
    let hasThinking = false;
    if (child.type === 'tool_call_row') counts[child.tool_name] = 1;
    if (child.type === 'reasoning_block') hasThinking = true;
    const fresh: AgentRunItem = {
      type: 'agent_run',
      capsule_id: id,
      status: 'running',
      items: [child],
      tool_call_count_by_name: counts,
      has_thinking: hasThinking,
      started_at: ms,
      ended_at: null,
    };
    return [...prev, fresh];
  }
  const counts = { ...live.capsule.tool_call_count_by_name };
  let hasThinking = live.capsule.has_thinking;
  if (child.type === 'tool_call_row') {
    counts[child.tool_name] = (counts[child.tool_name] ?? 0) + 1;
  }
  if (child.type === 'reasoning_block') hasThinking = true;
  const updated: AgentRunItem = {
    ...live.capsule,
    items: [...live.capsule.items, child],
    tool_call_count_by_name: counts,
    has_thinking: hasThinking,
    ended_at: ms,
  };
  const next = [...prev];
  next[live.idx] = updated;
  return next;
}

function finalizeLiveCapsule(
  prev: DisplayItem[],
  status: 'completed' | 'error' | 'stopped',
): DisplayItem[] {
  const live = getLiveRunningCapsule(prev);
  if (!live) return prev;
  if (live.capsule.items.length === 0) {
    return prev.slice(0, live.idx);
  }
  const updated: AgentRunItem = { ...live.capsule, status };
  const next = [...prev];
  next[live.idx] = updated;
  return next;
}

// Live tool_result events carry only the placeholder description
// "Tool result received" and the originating tool_call_id (in the
// tool_name field per activity_translator.py:189) — no real content is
// on the wire. Mark the most recent pending tool_call_row inside the
// live capsule as received with empty content; the JSONL reload will
// surface the actual content on next mount via chatTransform pairing.
type ToolCallRowItem = Extract<CapsuleChild, { type: 'tool_call_row' }>;

function ToolCallRow({ row }: { row: ToolCallRowItem }): React.ReactNode {
  const [expanded, setExpanded] = useState(false);
  const expandable = row.result_status !== 'pending';
  const Chevron = expanded ? ChevronDown : ChevronRight;

  return (
    <div className="mb-1 font-mono text-[11.5px] text-secondary">
      <button
        type="button"
        onClick={expandable ? () => setExpanded(e => !e) : undefined}
        disabled={!expandable}
        className={`flex items-center gap-2 w-full text-left ${expandable ? 'cursor-pointer hover:text-primary' : 'cursor-default'}`}
      >
        {expandable ? (
          <Chevron size={12} className="shrink-0 opacity-70" />
        ) : (
          <span className="shrink-0 w-3" aria-hidden />
        )}
        <span className="text-primary font-medium">{row.tool_name}</span>
        <span className="text-muted" aria-hidden>·</span>
        <span className="truncate">{row.target_description}</span>
      </button>
      {expanded && expandable && (() => {
        const raw = row.result_content;
        if (raw === null || raw === '') {
          return (
            <div className="mt-1 ml-5 px-3 py-2 rounded bg-background border border-border/40 text-[12px] italic text-secondary/70">
              no result content
            </div>
          );
        }
        const { text, footer } = truncateResult(raw);
        return (
          <div className="mt-1 ml-5">
            <pre className="px-3 py-2 rounded bg-background border border-border/40 text-[12px] text-secondary leading-relaxed whitespace-pre-wrap break-words font-mono">
              {text}
            </pre>
            {footer && (
              <div className="mt-1 px-3 text-[11px] text-secondary/60 italic">{footer}</div>
            )}
          </div>
        );
      })()}
    </div>
  );
}

function markLatestLiveCallResultReceived(
  prev: DisplayItem[],
  timestamp: string,
): DisplayItem[] {
  const live = getLiveRunningCapsule(prev);
  if (!live) return prev;
  const items = live.capsule.items;
  for (let k = items.length - 1; k >= 0; k--) {
    const item = items[k];
    if (item.type === 'tool_call_row' && item.result_status === 'pending') {
      const updatedRow = { ...item, result_content: '', result_status: 'received' as const };
      const newItems = [...items];
      newItems[k] = updatedRow;
      const updatedCapsule: AgentRunItem = {
        ...live.capsule,
        items: newItems,
        ended_at: Date.parse(timestamp),
      };
      const next = [...prev];
      next[live.idx] = updatedCapsule;
      return next;
    }
  }
  return prev;
}

// FE-1/FE-3: the legacy `reconcileTrailingRunning` sweep is gone. With the
// transform-once approach (full raw history in one pass) there are no
// non-trailing 'running' capsules to sweep, and the trailing capsule's status
// is now decided at source by `transformChatHistory`'s `isActivelyRunning`
// flag — so no post-pass reconciliation is needed.

// Messages (raw JSONL lines) fetched per page on initial load and per
// "Load earlier" click. Larger page → fewer paginations; sessions at or under
// this size load fully with no "Load earlier" button. Render cost is modest:
// tool activity collapses into capsules, so 100 raw messages render ~30
// markdown bubbles, well within the already-reachable full-session ceiling.
const CHAT_PAGE_SIZE = 100;
const REST_FALLBACK_DELAY_MS = 500;
const SLASH_COMMANDS = [
  { name: '/new', description: 'Start a fresh session' },
];
import type {
  AgentRunStatus,
  ChatMessage as ChatMessageType,
  StreamDeltaEvent,
  ActivityEvent,
  ApprovalRequestEvent,
  ApprovalResolvedEvent,
  SubAgentMessageEvent,
  UserMessageEvent,
  AgentNotifyEvent,
  StateRefreshLifecycleEvent,
  WorkspaceClaudemdWarningEvent,
  WebSocketEvent,
  Project,
} from '../types';
import ChatMessage from './ChatMessage';
import StreamingMessage from './StreamingMessage';
import MessageAvatar from './MessageAvatar';
import ApprovalCard from './ApprovalCard';
import CredentialCard from './CredentialCard';
import RefreshTurnStatus from './RefreshTurnStatus';
import ClaudemdWarningBanner, { type ClaudemdWarning } from './ClaudemdWarningBanner';
import SlotHeldNotice from './SlotHeldNotice';

interface ChatViewProps {
  projectId: string;
  project: Project;
  agentStatus: AgentRunStatus;
  statusTick?: number;
  /**
   * Installed sub-agents available for @-mention. Lifted to App-level state
   * so tab switches don't refetch /agents/available. Empty array while
   * App is still resolving the initial fetch — the dropdown then shows no
   * matches, which is fine: `@` was typed before sub-agents loaded, the
   * keystrokes that follow re-evaluate against the populated list.
   */
  mentionAgents: Array<{ slug: string; name: string }>;
  /**
   * The F1 session_id currently being viewed (the active session for this
   * project). The conversation, history fetch, draft, and inject target are
   * all scoped to this session. `undefined` while the active session is
   * still being resolved by the parent (ChatTab) — ChatView renders an empty
   * state and skips history load until a sessionId arrives.
   *
   * Single active-loop slot model: only ONE session in a project executes at
   * a time. Live WS events (stream/activity/approvals) carry only project_id,
   * not session_id, so they always belong to the slot holder. ChatView
   * appends those live events to the viewed conversation ONLY when the viewed
   * `sessionId` matches the holder (resolved via run-status
   * `current_holder_session_id`). See §5 of the T5 task brief.
   */
  sessionId?: string;
  /**
   * Re-fetch this project's runtime fields (e.g. `budget_spent_usd`) and merge
   * them into the App-level projects list. Called on the running→idle
   * transition so the header's budget/cost reflects the just-finished turn.
   */
  onRefreshProject?: (id: string) => void;
}

interface StreamState {
  text: string;
  source: string;
  isComplete: boolean;
}

interface PendingApproval {
  what: string;
  tool_name: string;
  tool_call_id: string;
  tool_args: Record<string, unknown>;
  recent_activity: ChatMessageType[];
  reasoning?: string;
  resolved?: 'approved' | 'denied';
}

export default function ChatView({ projectId, project, agentStatus, statusTick, mentionAgents, sessionId, onRefreshProject }: ChatViewProps) {
  // FE-1 (transform-once): loaded chat history is stored as RAW messages
  // across all paginated pages (initial page + each "Load earlier" prepend),
  // then transformed in a SINGLE pass via the useMemo below. This eliminates
  // the per-page transform's page-boundary tool-result drops and stranded
  // pending tool-calls — the transform always sees the complete conversation.
  //
  // `items` remains the render/live state: it is seeded from the memoized
  // history transform whenever history (re)loads, and live WS handlers
  // continue to mutate it incrementally (live capsule appends, optimistic
  // user messages, sub-agent messages, finalize-on-idle, etc.). The live tail
  // therefore layers on top of the transform-once history.
  const [rawMessages, setRawMessages] = useState<ChatMessageType[]>([]);
  const [items, setItems] = useState<DisplayItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [totalMessages, setTotalMessages] = useState(0);
  const [loadedOffset, setLoadedOffset] = useState(0);
  const [stream, setStream] = useState<StreamState | null>(null);
  const [approvals, setApprovals] = useState<Map<string, PendingApproval>>(new Map());
  const [expandedCapsules, setExpandedCapsules] = useState<Set<string>>(new Set());
  const [inputText, setInputText] = useState('');
  // Slot holder: the F1 session_id currently holding the project's
  // active-loop slot (from run-status `current_holder_session_id`), or null
  // if no session is running. Single-slot model — only ONE session runs at a
  // time. Live WS events carry only project_id, so they belong to this
  // holder. ChatView appends them to the viewed conversation ONLY when the
  // viewed session IS the holder (viewing a non-holder session shows its
  // static history; live events for the holder are dropped here).
  const [holderSessionId, setHolderSessionId] = useState<string | null>(null);
  const [showMentionDropdown, setShowMentionDropdown] = useState(false);
  const [mentionFilter, setMentionFilter] = useState('');
  const [selectedMentionIndex, setSelectedMentionIndex] = useState(0);
  const [subAgentLoading, setSubAgentLoading] = useState<string | null>(null);
  const [showThinking, setShowThinking] = useState(false);
  const [showCommandDropdown, setShowCommandDropdown] = useState(false);
  const [commandFilter, setCommandFilter] = useState('');
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [injectError, setInjectError] = useState<string | null>(null);
  // Track J Phase 1: intermediate "slot held" notice. Set when /inject
  // returns 202 with `slot_held` because the caller's session_id didn't
  // hold the project's active-loop slot. See SlotHeldNotice.tsx.
  const [slotHeldNotice, setSlotHeldNotice] = useState<
    | {
        holdingSessionId: string;
        pendingContent: string;
        pendingTarget?: string;
        pendingNonce: string;
        pendingAttachments?: Array<{ path: string; mime: string; size: number }>;
      }
    | null
  >(null);
  // Workspace CLAUDE.md interference banner: latest unhandled warning
  // for this project (cleared on dismiss).
  const [claudemdWarning, setClaudemdWarning] = useState<ClaudemdWarning | null>(null);
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const [dragCounter, setDragCounter] = useState(0);
  // Stop-button optimistic state. Set true synchronously on click so the
  // input row immediately shows a loading affordance; cleared when
  // agentStatus transitions to 'idle' (loop actually exited) or after a
  // 10s timeout falls back to a retry notice. The backend
  // agent.status:idle broadcast is authoritative — this just covers the
  // in-flight gap between click and the WS event arriving.
  const [isCancelling, setIsCancelling] = useState(false);
  const [cancelTimeoutNotice, setCancelTimeoutNotice] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const localNoncesRef = useRef<Map<string, number>>(new Map());
  const wasRunningRef = useRef(false);
  const { on, off, connectionState } = useWebSocket();
  const { injectMessage, cancelMessage, newSession } = useAgent();
  // Queue-active gating: when the queue is running (actively dispatching a
  // task), the chat composer is replaced by ComposerDisabledPrompt — the user
  // must pause the queue first. ('idle'/'paused' leave the composer enabled.)
  const { snapshot: queueSnapshot, stopQueue } = useQueue(projectId);
  const queueActive = queueSnapshot?.state === 'running';
  // Per-session composer drafts. Keyed by F1 sessionId so switching to
  // another session and back restores the original session's unsent text.
  // In-memory only (drafts don't survive a full reload — acceptable).
  const draftsRef = useRef<Map<string, string>>(new Map());
  // Live mirror of inputText, read synchronously when persisting a draft on
  // session switch (the effect closes over a stale inputText otherwise).
  const inputTextRef = useRef('');
  // Latest sessionId, readable synchronously inside WS handlers (which close
  // over the value at registration time otherwise). The holder-comparison
  // and inject-target paths read this ref to stay current.
  const sessionIdRef = useRef<string | undefined>(sessionId);
  sessionIdRef.current = sessionId;
  // Latest holder, readable synchronously inside WS handlers.
  const holderSessionIdRef = useRef<string | null>(holderSessionId);
  holderSessionIdRef.current = holderSessionId;
  // Seam 3 / Phase 3: live WS events are routed STRICTLY by session_id. Every
  // live event now carries the canonical session_id (Phases 1+2), so a handler
  // renders an event into the viewed conversation iff
  // ``event.session_id === sessionIdRef.current``. The old ``viewingHolder``
  // heuristic (with its lenient null-holder "presume the viewed session is the
  // holder" fallback) is gone: it defaulted to SHOW, which leaked events across
  // sessions whenever the holder was momentarily unresolved. There is no
  // default-to-show path anymore — an event with no/other session_id is dropped.
  //
  // ``holderSessionId`` is retained, but ONLY as a fact (which session holds the
  // project's active-loop slot), never as an event-routing gate. It drives:
  //   - the cancel target (Stop cancels the RUNNING session, not the viewed one),
  //   - child props that need the live session, and
  //   - ``isActivelyRunning`` below (a RENDER concern: is the VIEWED session the
  //     one currently executing — used to keep the trailing capsule spinning).

  // FE-3 / FE-A1: the trailing open capsule is only genuinely "running" when
  // the viewed session is the one actively executing — i.e. the viewed session
  // IS the holder and the project is running. Applied at RENDER time below (the
  // agent_run case), NOT fed into the transform, so a status flip never
  // invalidates the memo (FE-A1).
  const isActivelyRunning =
    sessionId !== undefined &&
    sessionId === holderSessionId &&
    agentStatus === 'running';

  // FE-1 / FE-A1: single-pass transform of the FULL accumulated raw history.
  // Deps are purely the source data (rawMessages + workspace) — status flips
  // do NOT trigger a re-seed. The result seeds `items` only when history
  // (re)loads; live WS events then mutate `items` on top of this baseline.
  const historyItems = useMemo(
    () => transformChatHistory(rawMessages, project.workspace),
    [rawMessages, project.workspace],
  );

  // Seed the render/live `items` from the transform-once history whenever that
  // baseline changes (initial load, session switch, "Load earlier" prepend, or
  // an isActivelyRunning flip). Live handlers mutate `items` afterward; the
  // catch-up fetch on idle recovers any tail not yet persisted to history.
  //
  // Gate on a non-empty history: when there is no loaded history (empty
  // session, or the post-/new reset), `items` is owned entirely by the live
  // overlay (e.g. the "New session started" notice). Clearing `items` for an
  // empty session is handled explicitly in the load effect, so the seed here
  // must not stomp a live-only overlay back to [].
  // Seed the expanded-capsules Set with every capsule the transform marked
  // `defaultExpanded` (content-null turns whose opening reasoning should be
  // visible without a click). Membership in this Set is the SOLE source of
  // truth for whether a capsule is expanded — so a user chevron-click that
  // removes the id can actually collapse it. (Previously `defaultExpanded`
  // was OR'd into the render-time check, which made the toggle a no-op.)
  // Seeded in the SAME effect as setItems so a default-expanded capsule never
  // flashes collapsed for a frame before a follow-up effect expands it.
  useEffect(() => {
    if (historyItems.length === 0) return;
    setItems(historyItems);
    setExpandedCapsules((prev) => {
      let changed = false;
      const next = new Set(prev);
      for (const item of historyItems) {
        if (item.type === 'agent_run' && item.defaultExpanded && !next.has(item.capsule_id)) {
          next.add(item.capsule_id);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [historyItems]);

  /**
   * REST fallback: fetch the latest assistant message from the REST API.
   * Used when streaming deltas are missed (tunnel drop, late subscribe, etc.)
   * to recover the agent's response. Scoped to the viewed session so a
   * recovered message lands in the right conversation.
   */
  const fetchLatestMessage = useCallback(() => {
    const sid = sessionIdRef.current;
    if (sid === undefined) return;
    const sessionParam = `&session_id=${encodeURIComponent(sid)}`;
    setTimeout(() => {
      api<Array<{ role: string; content: string; timestamp?: string }>>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=1${sessionParam}`,
      )
        .then((messages) => {
          if (!messages || messages.length === 0) return;
          const latest = messages[messages.length - 1];
          if (latest.role !== 'assistant' || !latest.content) return;
          const restText = latest.content;
          setItems((prevItems) => {
            // Check if the message is already present (by content match on last item)
            const last = prevItems[prevItems.length - 1];
            if (
              last &&
              last.type === 'agent_message' &&
              'content' in last &&
              last.content === restText
            ) {
              return prevItems;
            }
            // Also check if the last agent_message has shorter content (stream was truncated)
            if (
              last &&
              last.type === 'agent_message' &&
              'content' in last &&
              restText.length > last.content.length
            ) {
              const updated = [...prevItems];
              updated[prevItems.length - 1] = { ...last, content: restText };
              return updated;
            }
            // Message not present at all — insert before any trailing user messages
            // so recovered agent responses appear before follow-up questions
            const newMsg = {
              type: 'agent_message' as const,
              content: restText,
              source: 'assistant',
              timestamp: latest.timestamp ?? new Date().toISOString(),
            };
            let insertIdx = prevItems.length;
            while (insertIdx > 0 && prevItems[insertIdx - 1].type === 'user_message') {
              insertIdx--;
            }
            const updated = [...prevItems];
            updated.splice(insertIdx, 0, newMsg);
            return updated;
          });
        })
        .catch(() => {
          // REST fetch failed — best effort
        });
    }, REST_FALLBACK_DELAY_MS);
  }, [projectId]);

  // Track the loaded-offset synchronously so the idle-refresh below sees the
  // current pagination depth without depending on state from a closure.
  // Mirrored from `loadedOffset` state on every render.
  const loadedOffsetRef = useRef(loadedOffset);
  loadedOffsetRef.current = loadedOffset;

  /**
   * FE-A1: refresh the latest N raw messages from the backend after a
   * running→idle transition so the just-finished turn's persisted lines
   * (assistant w/ tool_calls, tool results, [Sub-agent] lifecycle markers)
   * land in `rawMessages` before the memo re-runs. Without this, the seed
   * effect overwrites `items` with stale historyItems and the live overlay
   * from WS handlers is wiped.
   *
   * Uses `limit=max(loadedOffset, CHAT_PAGE_SIZE)` so any pages the user
   * loaded via "Load earlier" stay covered. Scoped to the session captured
   * at call time — if the user switches sessions mid-fetch, the response is
   * dropped.
   */
  const refreshRawMessages = useCallback(() => {
    const sid = sessionIdRef.current;
    if (sid === undefined) return;
    const targetLimit = Math.max(loadedOffsetRef.current, CHAT_PAGE_SIZE);
    const url = `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=${targetLimit}&session_id=${encodeURIComponent(sid)}`;
    apiWithTotal<ChatMessageType[]>(url)
      .then(({ data, total }) => {
        // Guard against session switch during the in-flight fetch.
        if (sessionIdRef.current !== sid) return;
        setRawMessages(data);
        setTotalMessages(total);
        setLoadedOffset(Math.min(targetLimit, data.length || targetLimit));
      })
      .catch(() => {
        // best-effort refresh; the seed remains the prior history
      });
  }, [projectId]);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) {
      requestAnimationFrame(() => {
        el.scrollTop = el.scrollHeight;
      });
    }
  }, []);

  /**
   * Shared approval-recovery fetch used by:
   *  1. Component mount (page reload)
   *  2. WebSocket reconnect (relay tunnel drop, mobile foreground)
   *  3. agentStatus transitioning to 'pending_approval'
   *  4. Existing 5s steady-state polls
   *
   * Calls GET /pending-approval and adds the approval to the Map if not
   * already present (dedup by tool_call_id). No-ops on network error.
   */
  const fetchPendingApproval = useCallback(() => {
    // Scope the recovery to the viewed session. Without this, the backend
    // resolves to the default-session sentinel and silently misses approvals
    // pending in non-default sessions.
    const viewed = sessionIdRef.current;
    const qs = viewed ? `?session_id=${encodeURIComponent(viewed)}` : '';
    api<{
      pending: boolean;
      tool_call_id?: string;
      tool_name?: string;
      tool_args?: Record<string, unknown>;
      what?: string;
      recent_activity?: ChatMessageType[];
      reasoning?: string;
    }>(`/api/v2/agents/${encodeURIComponent(projectId)}/pending-approval${qs}`)
      .then((result) => {
        if (!result.pending || !result.tool_call_id) return;
        // Seam 3 / Phase 3 carryover: route this display path strictly by
        // session_id, never by the holder. The fetch above is already scoped to
        // the viewed session (?session_id=<viewed>), and the backend
        // get_pending_approval resolves the (project, viewed) handle — it
        // returns an approval ONLY when the viewed session itself is paused, and
        // None for a non-holder viewed session. So a non-null result here
        // already belongs to the viewed session; reading holderSessionId to gate
        // display would be a latent cross-session leak (the old lenient
        // null-holder branch could surface another session's card). Trust the
        // session-scoped response.
        setApprovals((prev) => {
          // Dedup: skip if a card already exists for this tool_call_id
          if (prev.has(result.tool_call_id!)) return prev;
          const next = new Map(prev);
          next.set(result.tool_call_id!, {
            what: result.what ?? '',
            tool_name: result.tool_name ?? '',
            tool_call_id: result.tool_call_id!,
            tool_args: result.tool_args ?? {},
            recent_activity: result.recent_activity ?? [],
            reasoning: result.reasoning,
          });
          return next;
        });
        scrollToBottom();
      })
      .catch(() => {
        // REST fetch failed — best effort
      });
  }, [projectId, scrollToBottom]);

  /**
   * Resolve the project's current active-loop slot holder (the F1 session_id
   * running right now, or null). run-status returns `current_holder_session_id`.
   * Live WS events belong to this holder; the holder gates whether they are
   * appended to the viewed conversation (see viewingHolder above). Fetched on
   * mount, on agentStatus/sessionId change, and via the existing 5s poll.
   */
  const fetchHolder = useCallback(() => {
    api<{ project_id: string; status: string; current_holder_session_id?: string | null }>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/run-status`,
    )
      .then((result) => {
        setHolderSessionId(result.current_holder_session_id ?? null);
      })
      .catch(() => {
        // best effort — leave the prior holder value in place
      });
  }, [projectId]);

  // Keep the holder fresh: on mount/project change, on every agentStatus
  // transition (a status change always implies the slot may have been
  // acquired/released), and whenever the viewed session changes (so the
  // viewingHolder comparison is correct for the newly-viewed session).
  useEffect(() => {
    fetchHolder();
  }, [fetchHolder, agentStatus, statusTick, sessionId]);

  // Per-session history load. Re-runs when the viewed sessionId changes so
  // switching sessions shows that session's own history (and its own loading
  // state). Skips while sessionId is undefined (active session not yet
  // resolved) — the empty state renders until one arrives.
  useEffect(() => {
    let cancelled = false;

    if (sessionId === undefined) {
      // No active session resolved yet — clear and show the (non-loading)
      // empty state. Reset pagination so a later session load starts fresh.
      // Clearing rawMessages re-seeds `items` to [] via the history memo.
      setRawMessages([]);
      setItems([]);
      setStream(null);
      setApprovals(new Map());
      setTotalMessages(0);
      setLoadedOffset(0);
      setLoading(false);
      return;
    }

    const sessionParam = `&session_id=${encodeURIComponent(sessionId)}`;

    async function loadData() {
      setLoading(true);
      // Clear prior session's live state so it never bleeds into the new one.
      setStream(null);
      setApprovals(new Map());
      try {
        const chatResult = await apiWithTotal<ChatMessageType[]>(
          `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=${CHAT_PAGE_SIZE}${sessionParam}`,
        ).catch((err) => {
          console.error('[ChatView] Failed to load chat history:', err);
          return null;
        });
        if (cancelled) return;

        if (chatResult) {
          const { data: messages, total } = chatResult;
          // FE-1: store RAW messages; the transform-once memo turns them into
          // DisplayItems and seeds `items`. No per-page transform here.
          setRawMessages(messages);
          // Empty history: seed effect is gated on non-empty, so clear `items`
          // here explicitly. Non-empty history is seeded by the effect.
          if (messages.length === 0) setItems([]);
          setTotalMessages(total);
          setLoadedOffset(CHAT_PAGE_SIZE);
        } else {
          setRawMessages([]);
          setItems([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadData();
    return () => {
      cancelled = true;
    };
  }, [projectId, sessionId, project.workspace]);

  const hasMore = totalMessages > loadedOffset;

  // Pending scroll-position compensation for prepend operations.
  // Set BEFORE setItems(...) prepends history; the layout effect below
  // reads it, applies scrollTop += (newHeight - prevHeight) synchronously
  // after DOM commit (before paint), and clears it. Prevents the visible
  // snap-to-bottom that would otherwise occur via the items auto-scroll
  // effect.
  const pendingPrependPrevHeightRef = useRef<number | null>(null);

  async function loadOlderMessages() {
    if (loadingMore || !hasMore || sessionId === undefined) return;
    setLoadingMore(true);
    const el = scrollRef.current;
    const prevHeight = el?.scrollHeight ?? 0;
    const sessionParam = `&session_id=${encodeURIComponent(sessionId)}`;
    try {
      const { data: messages } = await apiWithTotal<ChatMessageType[]>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=${CHAT_PAGE_SIZE}&offset=${loadedOffset}${sessionParam}`,
      );
      if (messages.length === 0) return;
      // FE-1: prepend the older RAW messages to the accumulated history. The
      // transform-once memo re-runs on the FULL concatenated list, so a
      // tool-call/result pair straddling the page seam is paired correctly
      // (no orphan drops, no stranded pending rows). The memo result re-seeds
      // `items` via its effect.
      // Stash prevHeight; the useLayoutEffect on `items` will compensate
      // synchronously after the prepended DOM has committed.
      pendingPrependPrevHeightRef.current = prevHeight;
      setRawMessages(prev => [...messages, ...prev]);
      setLoadedOffset(prev => prev + CHAT_PAGE_SIZE);
    } catch (err) {
      console.error('[ChatView] Failed to load older messages:', err);
    } finally {
      setLoadingMore(false);
    }
  }

  // Per-session composer draft (§4). Live mirror of inputText for synchronous
  // reads in the swap effect. Set during render (not in an effect) so the swap
  // effect — which runs in the same commit when sessionId changes — always
  // sees the OUTGOING session's text, never a stale value. The draft MAP is
  // written only by the swap effect (on switch), keeping ownership simple: the
  // live `inputText` is the source of truth for the active session, and we
  // persist-on-switch from this mirror.
  inputTextRef.current = inputText;

  const prevSessionForDraftRef = useRef<string | undefined>(sessionId);
  useEffect(() => {
    const prev = prevSessionForDraftRef.current;
    if (prev === sessionId) return;
    // Save the outgoing session's draft from the live mirror.
    if (prev !== undefined) {
      draftsRef.current.set(prev, inputTextRef.current);
    }
    // Load the incoming session's draft (default to empty).
    const incoming = sessionId !== undefined ? (draftsRef.current.get(sessionId) ?? '') : '';
    prevSessionForDraftRef.current = sessionId;
    setInputText(incoming);
    // Reset the textarea height to fit the loaded draft on the next frame.
    // Cancel on unmount / re-run so a pending frame can't fire after teardown.
    const raf = requestAnimationFrame(() => adjustTextareaHeight());
    return () => cancelAnimationFrame(raf);
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fix 3C: Show "Thinking..." indicator when agent starts running.
  // Live machinery rows accumulate inside an open agent_run capsule; on
  // terminal status the capsule is finalized so it collapses.
  //
  // agentStatus is PROJECT-wide (it reflects the slot holder). Conversation
  // mutations here (finalize capsule, thinking indicator, catch-up fetch,
  // new_session reset) therefore only apply when the VIEWED session is the
  // holder. The Stop-button affordance (isCancelling) is project-wide and is
  // always cleared on a terminal status regardless of which session is viewed.
  useEffect(() => {
    // "viewing" = the viewed session IS the running/holder session. This is a
    // render-state concern (finalize capsule, thinking indicator), NOT event
    // routing — live events route strictly by session_id in the handlers below.
    const viewing =
      sessionIdRef.current !== undefined &&
      sessionIdRef.current === holderSessionIdRef.current;
    if (agentStatus === 'idle' || agentStatus === 'error') {
      // Loop has actually exited — clear the in-flight cancel affordance.
      // Effect re-runs on agentStatus change, so this naturally coincides
      // with the agent.status:idle WS broadcast from _on_loop_done. This is
      // composer-global and must run even when viewing a non-holder session.
      // (A stopped/cancelled agent reports as `idle` — there is no separate
      // `stopped` run status.)
      setIsCancelling(false);
      // Clear the thinking indicator and catch up the latest message even
      // when `viewing` is false. Reason: under multi-session, the holder
      // fetch racing with the running→idle transition can collapse
      // `viewingHolder` to false in the same tick the idle event arrives —
      // leaving the spinner stuck and the final reply unfetched on the
      // session that actually ran. These two cleanups are about loop state,
      // not the visible chat surface, so they are safe to run unconditionally.
      // See docs/investigations/INVESTIGATION-thinking-stuck-after-dispatch.md
      // (Option 1).
      setShowThinking(false);
      if (wasRunningRef.current) {
        wasRunningRef.current = false;
        fetchLatestMessage();
        // FE-A1: refresh raw history so the just-finished turn's persisted
        // assistant/tool/[Sub-agent] lines reach the memo. Without this, the
        // live overlay items get wiped on the next seed without a backing
        // history line to replace them.
        refreshRawMessages();
        // Refresh the project's runtime fields (budget_spent_usd) so the
        // header reflects the spend from the just-completed turn.
        onRefreshProject?.(projectId);
      }
      if (!viewing) return;
      const finalStatus = agentStatus === 'idle' ? 'completed' : 'error';
      setItems((prev) => finalizeLiveCapsule(prev, finalStatus));
    } else if (agentStatus === 'running') {
      if (!viewing) return;
      wasRunningRef.current = true;
      setShowThinking(true);
    } else if (agentStatus === 'pending_approval') {
      if (!viewing) return;
      // Fetch pending approval via REST in case the WS event was missed.
      // Covers status poll discovering a stale approval after reconnect.
      fetchPendingApproval();
    } else if (agentStatus === 'new_session') {
      if (!viewing) return;
      // WS event is the single source of truth for session swap
      wasRunningRef.current = false;
      // Clear the accumulated raw history first so the transform-once memo
      // re-seeds `items` to [] (matching the reset), then overlay the notice.
      setRawMessages([]);
      setItems([{
        type: 'agent_notify' as const,
        title: 'New session started',
        body: 'Workspace memory preserved. The agent remembers your project.',
        urgency: 'low' as const,
        timestamp: new Date().toISOString(),
      }]);
      setStream(null);
      setApprovals(new Map());
      setShowThinking(false);
      setTotalMessages(0);
      setLoadedOffset(0);
    }
  }, [agentStatus, statusTick, projectId, fetchLatestMessage, fetchPendingApproval, refreshRawMessages, onRefreshProject]);

  // Cancel timeout fallback: if the loop hasn't actually idled within 10s
  // of the Stop click, clear the in-flight affordance and surface a
  // retry-able notice. Covers degenerate cases (network drop after the
  // POST returned, daemon stall, etc.) so the input row doesn't lock up.
  useEffect(() => {
    if (!isCancelling) return;
    const timer = setTimeout(() => {
      setIsCancelling(false);
      setCancelTimeoutNotice('Cancel took longer than expected — try again if needed.');
    }, 10_000);
    return () => clearTimeout(timer);
  }, [isCancelling]);

  // On mount, always check for pending approvals via REST.
  // Handles the case where ChatView was unmounted (tab switch to files)
  // and remounted with a stale agentStatus that doesn't trigger the
  // status-change effect above. Also covers page reload — the initial
  // fetch runs before the first WS event arrives.
  useEffect(() => {
    fetchPendingApproval();
  }, [projectId, fetchPendingApproval]);

  // Reconnect-triggered fetch: whenever the WebSocket transitions into
  // the 'connected' state (relay tunnel drop / mobile foreground), fetch
  // pending approvals immediately so approval cards surface without
  // waiting up to 5 seconds for the next steady-state poll cycle.
  //
  // A ref tracks the prior connection state so the first mount (which
  // also runs the effect above) doesn't issue a duplicate fetch — we
  // only fire on an actual transition INTO 'connected'.
  const prevConnectionStateRef = useRef<typeof connectionState>(connectionState);
  useEffect(() => {
    const prev = prevConnectionStateRef.current;
    prevConnectionStateRef.current = connectionState;
    if (connectionState === 'connected' && prev !== 'connected') {
      // Skip the very first render's transition from initial
      // 'disconnected' → 'connected': the mount effect above already
      // issued a fetch, and the setApprovals dedup would no-op anyway.
      // But firing it here is still safe (idempotent), and covers the
      // case where the mount fetch raced ahead of the server becoming
      // ready. Let it run — the dedup Map guarantees no duplicates.
      fetchPendingApproval();
    }
  }, [connectionState, fetchPendingApproval]);

  // Fix 2B: Status poll fallback — if agent appears stuck as "running"
  // with no stream activity for 15 seconds, poll REST for actual status.
  const lastEventTimeRef = useRef<number>(Date.now());
  useEffect(() => {
    // Reset timer whenever we get any stream delta
    lastEventTimeRef.current = Date.now();
  }, [stream]);

  useEffect(() => {
    if (!agentStatus) return;
    const timer = setInterval(() => {
      api<{ project_id: string; status: string; current_holder_session_id?: string | null }>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/run-status`,
      )
        .then((result) => {
          // Keep the slot holder current from the same poll response.
          setHolderSessionId(result.current_holder_session_id ?? null);
          if (result.status !== agentStatus) {
            window.dispatchEvent(
              new CustomEvent('agent-status-override', {
                detail: { project_id: projectId, status: result.status },
              }),
            );
          } else if (result.status === 'pending_approval') {
            // Same status but might be a NEW approval — fetch if needed
            fetchPendingApproval();
          }
        })
        .catch(() => {});
    }, 5_000);
    return () => clearInterval(timer);
  }, [agentStatus, projectId, fetchPendingApproval]);

  // Polling safety net: while the agent is running, poll /pending-approval
  // every 5 seconds so approvals surface even if the WS event was missed
  // (relay disconnect, transient drop). This is a fallback — the WebSocket
  // handler `handleApprovalRequest` remains the primary path. The reconnect
  // and mount effects above handle the immediate-recovery case without
  // waiting up to 5 seconds for this cycle.
  useEffect(() => {
    if (agentStatus !== 'running') return;
    const timer = setInterval(() => {
      fetchPendingApproval();
    }, 5_000);
    return () => clearInterval(timer);
  }, [agentStatus, fetchPendingApproval]);

  useEffect(() => {
    // Single active-loop slot model: WS events carry only project_id, never
    // session_id, so any live agent event belongs to the project's slot
    // holder. We append it to the VIEWED conversation only when the viewed
    // session IS the holder. When viewing a non-holder session, these live
    // events pertain to the holder and must not bleed into the static
    // history being shown (see §5 of the T5 brief). The user's own typed
    // message (handleUserMessage nonce path) is the one exception — it always
    // renders optimistically for the session it was sent into.
    function handleStreamDelta(event: WebSocketEvent) {
      const e = event as StreamDeltaEvent;
      if (e.project_id !== projectId) return;
      // Strict session routing: render only deltas for the viewed session.
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      if (e.is_final) {
        setStream((prev) => {
          if (!prev) {
            return null;
          }
          const finalText = prev.text + e.text;
          if (finalText.trim()) {
            setItems((prevItems) => {
              // Visible text closes any open live capsule first, then appends.
              const afterCapsule = finalizeLiveCapsule(prevItems, 'completed');
              const last = afterCapsule[afterCapsule.length - 1];
              if (last && last.type === 'agent_message' && 'content' in last && last.content === finalText) {
                return afterCapsule;
              }
              return [
                ...afterCapsule,
                {
                  type: 'agent_message',
                  content: finalText,
                  source: e.source,
                  timestamp: new Date().toISOString(),
                },
              ];
            });
          }
          return null;
        });
        fetchLatestMessage();
        scrollToBottom();
        return;
      }

      setShowThinking(false);
      setStream((prev) => ({
        text: (prev?.text ?? '') + e.text,
        source: e.source,
        isComplete: false,
      }));
      scrollToBottom();
    }

    function handleActivity(event: WebSocketEvent) {
      const e = event as ActivityEvent;
      if (e.project_id !== projectId) return;
      // Strict session routing (seam 3 / Phase 3): render only activity for the
      // viewed session. Activity events carry session_id; the old viewingHolder
      // fallback (for legacy no-session_id events) is gone — an event without a
      // matching session_id is dropped, never shown by default. (A rare
      // project-scoped activity with no session_id, e.g. network_blocked, is
      // dropped rather than risk leaking into every session.)
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      // agent_output activities duplicate sub-agent messages already
      // delivered via chat.sub_agent_message — drop them.
      if (e.category === 'agent_output') return;

      // tool_result activity: the broadcast carries tool_call_id in the
      // tool_name field (per activity_translator.py:189) but no result
      // content — only a "Tool result received" placeholder we never
      // surface. Mark the most recent pending row in the capsule as
      // received with empty content; the actual content arrives via
      // JSONL on next mount through chatTransform pairing.
      if (e.category === 'tool_result') {
        setItems((prev) => markLatestLiveCallResultReceived(prev, e.timestamp));
        scrollToBottom();
        return;
      }

      // Tool-use family: route into the live capsule. The live
      // ActivityEvent does not carry tool_call_id for tool_use; the
      // event id is used as a synthetic key — pairing with tool_result
      // on the live path is positional, not by id.
      setItems((prev) =>
        appendToLiveCapsule(
          prev,
          {
            type: 'tool_call_row',
            tool_name: e.tool_name,
            target_description: e.description,
            tool_call_id: e.id,
            category: e.category,
            timestamp: e.timestamp,
            result_content: null,
            result_status: 'pending',
          },
          e.timestamp,
        ),
      );
      scrollToBottom();
    }

    function handleApprovalRequest(event: WebSocketEvent) {
      const e = event as ApprovalRequestEvent;
      if (e.project_id !== projectId) return;
      // Strict session routing (seam 3 / Phase 3): surface the approval card
      // only for the viewed session. Management approvals now carry session_id
      // (Phase 2), so the old always-on viewingHolder gate is removed — an
      // approval for another session never leaks into this pane.
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      setApprovals((prev) => {
        const next = new Map(prev);
        next.set(e.tool_call_id, {
          what: e.what,
          tool_name: e.tool_name,
          tool_call_id: e.tool_call_id,
          tool_args: e.tool_args,
          recent_activity: e.recent_activity,
          reasoning: e.reasoning,
        });
        return next;
      });

      if (document.hidden && 'Notification' in window && Notification.permission === 'granted') {
        new Notification(`Orbital: ${project.name} needs your approval`, {
          body: e.what,
        });
      } else if (
        document.hidden &&
        'Notification' in window &&
        Notification.permission === 'default'
      ) {
        Notification.requestPermission();
      }

      scrollToBottom();
    }

    function handleApprovalResolved(event: WebSocketEvent) {
      const e = event as ApprovalResolvedEvent;
      if (e.project_id !== projectId) return;
      if (e.session_id && e.session_id !== sessionIdRef.current) return;

      setApprovals((prev) => {
        const next = new Map(prev);
        const existing = next.get(e.tool_call_id);
        if (existing) {
          next.set(e.tool_call_id, { ...existing, resolved: e.resolution });
        }
        return next;
      });
    }

    function handleSubAgentMessage(event: WebSocketEvent) {
      const e = event as SubAgentMessageEvent;
      if (e.project_id !== projectId) return;
      // Strict session routing (seam 3 / Phase 3): sub-agent replies carry the
      // parent session_id; render only those for the viewed session.
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      // Strip ANSI codes and filter empty / "(no response)" content
      const cleaned = (e.content ?? '').replace(/\x1b\[[0-9;]*m/g, '').trim();
      if (!cleaned || cleaned === '(no response)') return;

      setItems((prev) => [
        ...prev,
        {
          type: 'sub_agent_message',
          content: cleaned,
          source: e.source,
          timestamp: e.timestamp,
        },
      ]);
      scrollToBottom();
    }

    function handleUserMessage(event: WebSocketEvent) {
      const e = event as UserMessageEvent;
      if (e.project_id !== projectId) return;

      // Evict nonces older than 30s on each incoming message
      const now = Date.now();
      for (const [n, ts] of localNoncesRef.current) {
        if (ts > 0 && now - ts > 30_000) localNoncesRef.current.delete(n);
      }

      // Skip if this is our own message (nonce matches a local send). The
      // optimistic append in handleSend already rendered it into the viewed
      // session, regardless of holder — so this dedup path is the one place
      // a user's own message survives even when not viewing the holder.
      if (e.nonce && localNoncesRef.current.has(e.nonce)) {
        // Mark as received with timestamp instead of deleting, so relay
        // retries of the same event are still deduped within the TTL window.
        localNoncesRef.current.set(e.nonce, Date.now());
        return;
      }

      // A non-own user_message echo (no matching nonce — e.g. injected from
      // another client). Strict session routing (seam 3 / Phase 3): render it
      // only when it belongs to the viewed session.
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      setItems((prev) => {
        const afterCapsule = finalizeLiveCapsule(prev, 'completed');
        return [
          ...afterCapsule,
          {
            type: 'user_message',
            content: e.content,
            timestamp: e.timestamp,
          },
        ];
      });
      scrollToBottom();
    }

    function handleAgentNotify(event: WebSocketEvent) {
      const e = event as AgentNotifyEvent;
      if (e.project_id !== projectId) return;
      // Strict session routing (seam 3 / Phase 3): agent.notify now carries
      // session_id (Phase 2); render only for the viewed session — the old
      // always-on viewingHolder gate is removed so a notify for another session
      // never leaks into this pane.
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      setItems((prev) => [
        ...prev,
        {
          type: 'agent_notify' as const,
          title: e.title,
          body: e.body,
          urgency: e.urgency,
          timestamp: e.timestamp,
        },
      ]);

      // Browser notification for high/normal urgency
      if (e.urgency !== 'low' && 'Notification' in window && Notification.permission === 'granted') {
        new Notification(e.title, { body: e.body });
      }

      scrollToBottom();
    }

    function handleStateRefresh(event: WebSocketEvent) {
      const e = event as StateRefreshLifecycleEvent;
      if (e.project_id !== projectId) return;
      // Strict session routing (seam 3 / Phase 3): state_refresh.lifecycle
      // carries session_id; render only for the viewed session (was: ignored
      // e.session_id and gated on viewingHolder).
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      setItems((prev) => {
        // Find the last refresh_status item to update in-place (in_progress → done/failed)
        const lastRefreshIdx = [...prev].reverse().findIndex((item) => item.type === 'refresh_status');
        const absIdx = lastRefreshIdx >= 0 ? prev.length - 1 - lastRefreshIdx : -1;

        if (e.status === 'in_progress') {
          // Always append a new in_progress card
          return [
            ...prev,
            {
              type: 'refresh_status' as const,
              status: e.status,
              trigger: e.trigger,
              timestamp: e.timestamp,
            },
          ];
        }

        if (absIdx >= 0) {
          // Update existing card to terminal state
          const updated = [...prev];
          updated[absIdx] = {
            type: 'refresh_status' as const,
            status: e.status,
            trigger: e.trigger,
            timestamp: e.timestamp,
          };
          return updated;
        }

        // No prior in_progress card — still surface terminal state
        return [
          ...prev,
          {
            type: 'refresh_status' as const,
            status: e.status,
            trigger: e.trigger,
            timestamp: e.timestamp,
          },
        ];
      });

      scrollToBottom();
    }

    function handleClaudemdWarning(event: WebSocketEvent) {
      const e = event as WorkspaceClaudemdWarningEvent;
      if (e.project_id !== projectId) return;
      setClaudemdWarning({
        project_id: e.project_id,
        claudemd_path: e.claudemd_path,
        content_hash: e.content_hash,
        matched_token: e.matched_token,
      });
    }

    on('chat.stream_delta', handleStreamDelta);
    on('agent.activity', handleActivity);
    on('approval.request', handleApprovalRequest);
    on('approval.resolved', handleApprovalResolved);
    on('chat.sub_agent_message', handleSubAgentMessage);
    on('chat.user_message', handleUserMessage);
    on('agent.notify', handleAgentNotify);
    on('state_refresh.lifecycle', handleStateRefresh);
    on('workspace_claudemd_warning', handleClaudemdWarning);

    return () => {
      off('chat.stream_delta', handleStreamDelta);
      off('agent.activity', handleActivity);
      off('approval.request', handleApprovalRequest);
      off('approval.resolved', handleApprovalResolved);
      off('chat.sub_agent_message', handleSubAgentMessage);
      off('chat.user_message', handleUserMessage);
      off('agent.notify', handleAgentNotify);
      off('state_refresh.lifecycle', handleStateRefresh);
      off('workspace_claudemd_warning', handleClaudemdWarning);
    };
  }, [projectId, project.name, on, off, scrollToBottom]);

  // Fingerprint for the last item. DisplayItem has no stable id; use a
  // composite of type + a discriminating field per variant.
  function lastItemKey(item: DisplayItem | undefined): string | null {
    if (!item) return null;
    if (item.type === 'approval_card') return `approval_card:${item.tool_call_id}`;
    if (item.type === 'session_separator') return `session_separator:${item.timestamp}`;
    if (item.type === 'agent_notify') {
      return `agent_notify:${item.timestamp}:${item.title}`;
    }
    if (item.type === 'refresh_status') {
      return `refresh_status:${item.timestamp}:${item.status}`;
    }
    if (item.type === 'tool_call_row') {
      // Include result_status so transitions pending → received re-fingerprint.
      return `tool_call_row:${item.tool_call_id}:${item.timestamp}:${item.result_status}`;
    }
    if (item.type === 'reasoning_block') {
      return `reasoning_block:${item.turn_id}:${item.content.slice(0, 32)}`;
    }
    if (item.type === 'agent_run') {
      // status + items.length lets the auto-scroll fire on each child append
      // and again on status flip from running → completed.
      return `agent_run:${item.capsule_id}:${item.status}:${item.items.length}`;
    }
    if (item.type === 'sub_agent_activity') {
      return `sub_agent_activity:${item.timestamp}:${item.handle}:${item.action}`;
    }
    // user_message, agent_message, sub_agent_message — use timestamp +
    // first 32 chars of content as a stable-enough fingerprint.
    const contentPrefix = item.content.slice(0, 32);
    return `${item.type}:${item.timestamp}:${contentPrefix}`;
  }

  // Compensate scroll position synchronously after a history prepend.
  // Runs after DOM commit, before paint — so the user never sees the
  // chat snap to bottom. Must run before the auto-scroll useEffect
  // below would have fired (it's now gated and won't fire on prepend
  // anyway, but the layout-vs-effect ordering also guarantees it).
  useLayoutEffect(() => {
    const prevHeight = pendingPrependPrevHeightRef.current;
    if (prevHeight === null) return;
    pendingPrependPrevHeightRef.current = null;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop += el.scrollHeight - prevHeight;
  }, [items]);

  // Auto-scroll to bottom only when a NEW message arrives at the tail.
  // Gating on the last-item fingerprint prevents a snap-to-bottom when
  // history is prepended at the head (loadOlderMessages).
  const lastItemKeyRef = useRef<string | null>(null);
  useEffect(() => {
    const key = lastItemKey(items[items.length - 1]);
    if (key !== lastItemKeyRef.current) {
      lastItemKeyRef.current = key;
      scrollToBottom();
    }
  }, [items, scrollToBottom]);

  function adjustTextareaHeight() {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
    }
  }

  const filteredAgents = mentionAgents.filter(a =>
    a.slug.toLowerCase().includes(mentionFilter) ||
    a.name.toLowerCase().includes(mentionFilter)
  );

  function handleInputChange(value: string) {
    setInputText(value);
    adjustTextareaHeight();

    // Check for /command trigger (only when input starts with /)
    if (value.startsWith('/')) {
      setShowCommandDropdown(true);
      setCommandFilter(value.slice(1).toLowerCase());
      setSelectedCommandIndex(0);
      // Hide @mention dropdown when in command mode
      setShowMentionDropdown(false);
      setMentionFilter('');
      return;
    }
    setShowCommandDropdown(false);
    setCommandFilter('');

    // Check for @mention trigger
    const atMatch = value.match(/@(\S*)$/);
    if (atMatch) {
      setShowMentionDropdown(true);
      setMentionFilter(atMatch[1].toLowerCase());
      setSelectedMentionIndex(0);
    } else {
      setShowMentionDropdown(false);
      setMentionFilter('');
    }
  }

  function selectMention(slug: string) {
    // Replace @partial with @slug
    const newText = inputText.replace(/@\S*$/, `@${slug} `);
    setInputText(newText);
    setShowMentionDropdown(false);
    setMentionFilter('');
    textareaRef.current?.focus();
  }

  const filteredCommands = SLASH_COMMANDS.filter(c =>
    c.name.toLowerCase().startsWith('/' + commandFilter)
  );

  async function executeNewSession() {
    setInputText('');
    // Echo the command so user sees it was received
    setItems((prev) => {
      const afterCapsule = finalizeLiveCapsule(prev, 'completed');
      return [...afterCapsule, {
        type: 'user_message' as const,
        content: '/new',
        timestamp: new Date().toISOString(),
      }];
    });
    setShowCommandDropdown(false);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    try {
      const result = await newSession(projectId, sessionId);
      if (result.status === 'no_active_session') {
        setItems((prev) => [...prev, {
          type: 'agent_notify' as const,
          title: 'No active session',
          body: 'Start the agent first, then use /new to reset the session.',
          urgency: 'high' as const,
          timestamp: new Date().toISOString(),
        }]);
      }
      // Otherwise, state clearing happens via WS new_session event
    } catch (err) {
      console.error('[ChatView] /new failed:', err);
    }
  }

  function selectCommand(name: string) {
    setShowCommandDropdown(false);
    setCommandFilter('');
    if (name === '/new') {
      executeNewSession();
      return;
    }
    setInputText(name);
    setTimeout(() => handleSend(), 0);
  }

  // ---- Attachment helpers ------------------------------------------------

  const uploadAttachment = useCallback(
    async (att: PendingAttachment) => {
      setAttachments((prev) =>
        prev.map((a) =>
          a.id === att.id ? { ...a, status: 'uploading' as const } : a,
        ),
      );
      try {
        const baseUrl = isRelayMode ? window.location.origin : BASE_URL;
        const { path, size } = await uploadFile({
          projectId,
          file: att.file,
          baseUrl,
          isRelayMode,
        });
        setAttachments((prev) =>
          prev.map((a) =>
            a.id === att.id
              ? { ...a, status: 'done' as const, uploadedPath: path, size }
              : a,
          ),
        );
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Upload failed';
        setAttachments((prev) =>
          prev.map((a) =>
            a.id === att.id
              ? { ...a, status: 'error' as const, errorMessage: message }
              : a,
          ),
        );
      }
    },
    [projectId],
  );

  const addAttachments = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;
      setAttachments((prev) => {
        const room = MAX_ATTACHMENTS - prev.length;
        if (room <= 0) {
          setInjectError(`Maximum of ${MAX_ATTACHMENTS} attachments per message.`);
          return prev;
        }
        const usable = files.slice(0, room);
        if (files.length > usable.length) {
          setInjectError(`Only ${MAX_ATTACHMENTS} attachments allowed per message.`);
        }
        const additions: PendingAttachment[] = usable.map((file) => {
          const id = genId();
          const isImage = file.type.startsWith('image/');
          const thumbnailUrl = isImage ? URL.createObjectURL(file) : undefined;
          const tooBig = file.size > MAX_ATTACHMENT_BYTES;
          const att: PendingAttachment = {
            id,
            file,
            filename: file.name,
            mime: file.type || 'application/octet-stream',
            size: file.size,
            status: tooBig ? 'error' : 'pending',
            thumbnailUrl,
            errorMessage: tooBig
              ? 'File exceeds 10 MB limit'
              : undefined,
          };
          return att;
        });

        // Kick off uploads for the valid ones on the next tick so React has
        // committed the new chip state.
        setTimeout(() => {
          for (const a of additions) {
            if (a.status === 'pending') {
              uploadAttachment(a);
            }
          }
        }, 0);

        return [...prev, ...additions];
      });
    },
    [uploadAttachment],
  );

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => {
      const target = prev.find((a) => a.id === id);
      if (target?.thumbnailUrl) {
        try {
          URL.revokeObjectURL(target.thumbnailUrl);
        } catch {
          // ignore
        }
      }
      return prev.filter((a) => a.id !== id);
    });
  }, []);

  const retryAttachment = useCallback(
    (id: string) => {
      setAttachments((prev) => {
        const target = prev.find((a) => a.id === id);
        if (!target) return prev;
        const reset: PendingAttachment = {
          ...target,
          status: 'pending',
          errorMessage: undefined,
        };
        setTimeout(() => uploadAttachment(reset), 0);
        return prev.map((a) => (a.id === id ? reset : a));
      });
    },
    [uploadAttachment],
  );

  const clearAttachments = useCallback(() => {
    setAttachments((prev) => {
      for (const a of prev) {
        if (a.thumbnailUrl) {
          try {
            URL.revokeObjectURL(a.thumbnailUrl);
          } catch {
            // ignore
          }
        }
      }
      return [];
    });
  }, []);

  function handleFilePickerChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) addAttachments(files);
    e.target.value = '';
  }

  function handlePaste(e: React.ClipboardEvent<HTMLTextAreaElement>) {
    const items = Array.from(e.clipboardData?.items ?? []);
    const files = items
      .filter((it) => it.kind === 'file')
      .map((it) => it.getAsFile())
      .filter((f): f is File => f !== null);
    if (files.length > 0) {
      e.preventDefault();
      addAttachments(files);
    }
  }

  function handleDragEnter(e: React.DragEvent<HTMLDivElement>) {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
    setDragCounter((n) => n + 1);
  }

  function handleDragOver(e: React.DragEvent<HTMLDivElement>) {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
  }

  function handleDragLeave(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragCounter((n) => Math.max(0, n - 1));
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragCounter(0);
    const files = Array.from(e.dataTransfer?.files ?? []);
    if (files.length > 0) addAttachments(files);
  }

  const anyUploading = attachments.some((a) => a.status === 'uploading');
  const anyDone = attachments.some((a) => a.status === 'done');
  const allError =
    attachments.length > 0 && attachments.every((a) => a.status === 'error');
  const hasText = inputText.trim().length > 0;
  // Send is enabled when there is text OR a done chip, AND no chip is uploading.
  const canSend =
    !anyUploading &&
    !allError &&
    (hasText || anyDone) &&
    !(attachments.length > 0 && !hasText && !anyDone);
  const disabledReason = anyUploading
    ? 'Waiting for uploads…'
    : allError && !hasText
      ? 'Remove failed attachments or retry'
      : '';

  async function handleSend() {
    const text = inputText.trim();
    const doneAttachments = attachments.filter((a) => a.status === 'done');
    if (!text && doneAttachments.length === 0) return;
    if (anyUploading) return;

    // Slash command: /new
    if (text === '/new') {
      executeNewSession();
      return;
    }

    let target: string | undefined;
    let content = text;
    const atMatch = text.match(/^@([\w-]+)\s+([\s\S]*)/);
    if (atMatch) {
      target = atMatch[1];
      content = atMatch[2];
    }

    // Generate nonce so we can deduplicate the WS echo of our own message
    // crypto.randomUUID() requires secure context (HTTPS/localhost) — use fallback for LAN HTTP
    const nonce = typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    localNoncesRef.current.set(nonce, 0);

    const attachmentsPayload = doneAttachments.map((a) => ({
      path: a.uploadedPath!,
      mime: a.mime,
      size: a.size,
    }));
    // Build the same prefix the backend will emit, so the optimistic local
    // user_message has the identical wire content as the WS echo will.
    const wireContent = buildAttachmentsBlock(attachmentsPayload) + content;

    setInputText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    clearAttachments();

    const optimisticTimestamp = new Date().toISOString();
    setItems((prev) => {
      const afterCapsule = finalizeLiveCapsule(prev, 'completed');
      return [
        ...afterCapsule,
        {
          type: 'user_message',
          content: wireContent,
          timestamp: optimisticTimestamp,
          ...(target && { target }),
        },
      ];
    });
    // F1: also push into rawMessages so the transform-once memo (which the
    // seed effect re-runs whenever isActivelyRunning flips on idle→running)
    // reproduces this bubble instead of stomping it. Without this, the very
    // next memo run would overwrite the optimistic tail and the just-typed
    // message would vanish until the WS echo / catch-up fetch lands.
    setRawMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: wireContent,
        source: 'user',
        timestamp: optimisticTimestamp,
        ...(target && { target }),
      },
    ]);

    if (target) setSubAgentLoading(target);
    setInjectError(null);
    try {
      const result = await injectMessage(
        projectId,
        content,
        target,
        nonce,
        attachmentsPayload.length > 0 ? attachmentsPayload : undefined,
        sessionId,
      );
      // Track J Phase 1: backend returned 202 with `slot_held` because
      // another session in this project holds the active-loop slot.
      // Strip the optimistic user_message we just appended (it didn't
      // land), restore the typed text to the composer, and surface the
      // intermediate notice with [Wait] / [Cancel-and-send] affordances.
      if (
        result &&
        typeof result === 'object' &&
        result.status === 'slot_held' &&
        result.holding_session_id
      ) {
        setItems((prev) => {
          // Remove only the latest user_message we just appended (by
          // nonce-free identity: type + content + most recent timestamp).
          // Walk from the end so concurrent state updates leave older
          // user_messages intact.
          for (let i = prev.length - 1; i >= 0; i--) {
            const it = prev[i];
            if (it.type === 'user_message' && it.content === wireContent) {
              return [...prev.slice(0, i), ...prev.slice(i + 1)];
            }
          }
          return prev;
        });
        // F1: also strip the optimistic entry we pushed into rawMessages so
        // the transform-once memo doesn't re-emit a phantom user bubble.
        setRawMessages((prev) => {
          for (let i = prev.length - 1; i >= 0; i--) {
            const m = prev[i];
            if (m.role === 'user' && m.content === wireContent && m.timestamp === optimisticTimestamp) {
              return [...prev.slice(0, i), ...prev.slice(i + 1)];
            }
          }
          return prev;
        });
        setInputText(text);
        setSlotHeldNotice({
          holdingSessionId: result.holding_session_id,
          pendingContent: content,
          pendingTarget: target,
          pendingNonce: nonce,
          pendingAttachments:
            attachmentsPayload.length > 0 ? attachmentsPayload : undefined,
        });
        return;
      }
      // If the backend auto-denied a pending approval because we sent this
      // message while paused, immediately mark the approval card as denied
      // so the user sees the transition without waiting for the WS echo.
      // The system message ("[Pending approval ... was dismissed]") arrives
      // via the normal message stream — no extra UI work needed here.
      if (
        result &&
        typeof result === 'object' &&
        'approval_dismissed' in result &&
        result.approval_dismissed &&
        result.dismissed_tool_call_id
      ) {
        const dismissedId = result.dismissed_tool_call_id;
        setApprovals((prev) => {
          const next = new Map(prev);
          const existing = next.get(dismissedId);
          if (existing) {
            next.set(dismissedId, { ...existing, resolved: 'denied' });
          }
          return next;
        });
      }
    } catch {
      setInjectError('Failed to send message. Please try again.');
    } finally {
      setSubAgentLoading(null);
    }

    scrollToBottom();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (showCommandDropdown) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedCommandIndex(i => Math.min(i + 1, filteredCommands.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedCommandIndex(i => Math.max(i - 1, 0));
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        if (filteredCommands.length > 0) {
          selectCommand(filteredCommands[selectedCommandIndex].name);
        }
        return;
      }
      if (e.key === 'Escape') {
        setShowCommandDropdown(false);
        return;
      }
    }
    if (showMentionDropdown) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedMentionIndex(i => Math.min(i + 1, filteredAgents.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedMentionIndex(i => Math.max(i - 1, 0));
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        if (filteredAgents.length > 0) {
          selectMention(filteredAgents[selectedMentionIndex].slug);
        }
        return;
      }
      if (e.key === 'Escape') {
        setShowMentionDropdown(false);
        return;
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const isDragActive = dragCounter > 0;

  return (
    <div
      className="flex flex-col h-full relative"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragActive && (
        <div
          className="absolute inset-0 z-50 bg-background/80 border-2 border-dashed border-accent rounded flex items-center justify-center pointer-events-none"
          data-testid="chat-drop-overlay"
        >
          <span className="text-lg font-medium">Drop files to attach</span>
        </div>
      )}
      {claudemdWarning && (
        <ClaudemdWarningBanner
          projectId={projectId}
          warning={claudemdWarning}
          onDismiss={() => setClaudemdWarning(null)}
        />
      )}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-7 py-5 max-md:pb-20 flex flex-col gap-4">
        {loading && (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 bg-sidebar rounded-lg animate-pulse" />
            ))}
          </div>
        )}

        {!loading && hasMore && (
          <button
            onClick={loadOlderMessages}
            disabled={loadingMore}
            className="w-full py-2 text-xs text-secondary hover:text-primary transition-colors disabled:opacity-50"
          >
            {loadingMore ? 'Loading...' : `Load earlier messages (${totalMessages - loadedOffset} more)`}
          </button>
        )}

        {!loading && items.length === 0 && !stream && (
          <div className="text-secondary text-sm text-center mt-12">
            No messages yet. Send a message to get started.
          </div>
        )}

        {!loading &&
          items.map((item, index) => {
            const historical = 'isHistorical' in item && item.isHistorical;
            let rendered: React.ReactNode = null;

            if (item.type === 'user_message') {
              rendered = <ChatMessage key={`msg-${index}`} message={item} />;
            } else if (item.type === 'agent_message' || item.type === 'sub_agent_message') {
              rendered = <ChatMessage key={`msg-${index}`} message={item} agentName={project.agent_name} />;
            } else if (item.type === 'sub_agent_activity') {
              // FE-A2: compact one-line marker for [Sub-agent] lifecycle.
              const label =
                item.action === 'started' ? 'started' :
                item.action === 'sent' ? `sent: ${item.preview ?? ''}` :
                item.action === 'completed' ? `completed${item.summary ? `: ${item.summary}` : ''}` :
                /* failed */ `failed${item.error ? `: ${item.error}` : ''}`;
              const tone = item.action === 'failed' ? 'text-error/80' : 'text-secondary';
              rendered = (
                <div
                  key={`sa-${index}`}
                  data-testid="sub-agent-activity"
                  data-action={item.action}
                  className={`flex items-baseline gap-2 px-2 py-1 text-[11.5px] font-mono ${tone}`}
                >
                  <span className="text-primary">{item.handle}</span>
                  <span className="truncate">{label}</span>
                </div>
              );
            } else if (item.type === 'session_separator') {
              return (
                <div key={`sep-${index}`} className="flex items-center gap-3 px-2 opacity-50">
                  <div className="flex-1 border-t border-border" />
                  <span className="text-xs text-secondary whitespace-nowrap">
                    Previous session
                  </span>
                  <div className="flex-1 border-t border-border" />
                </div>
              );
            } else if (item.type === 'agent_run') {
              // FE-A1: trailing-capsule "running" status is a render-time
              // derivation. The transform always emits `completed`; we
              // upgrade to `running` only when this is the LAST item AND the
              // viewed session is actively running. This keeps the transform
              // pure (no status-flip re-runs that wipe the live overlay) and
              // still shows a spinner when one is appropriate.
              const isLastItem = index === items.length - 1;
              const derivedStatus =
                isLastItem && isActivelyRunning && item.status === 'completed'
                  ? 'running'
                  : item.status;
              const isExpanded =
                derivedStatus === 'running' ||
                expandedCapsules.has(item.capsule_id);
              const isLocked = derivedStatus === 'running';
              const summary = capsuleSummaryText({ ...item, status: derivedStatus });
              const Chevron = isExpanded ? ChevronDown : ChevronRight;
              rendered = (
                <div
                  key={`run-${item.capsule_id}`}
                  data-testid="agent_run"
                  data-capsule-id={item.capsule_id}
                  data-capsule-status={derivedStatus}
                  className="ml-9 flex gap-[10px]"
                >
                  <div className="w-0.5 shrink-0 rounded-sm bg-border" aria-hidden />
                  <div className="min-w-0 flex-1">
                  <button
                    type="button"
                    onClick={
                      isLocked
                        ? undefined
                        : () => {
                            setExpandedCapsules((prev) => {
                              const next = new Set(prev);
                              if (next.has(item.capsule_id)) next.delete(item.capsule_id);
                              else next.add(item.capsule_id);
                              return next;
                            });
                          }
                    }
                    disabled={isLocked}
                    className={`flex items-center gap-2 w-full text-left font-mono text-[11.5px] text-secondary ${isLocked ? 'cursor-default' : 'cursor-pointer hover:text-primary'}`}
                  >
                    <Chevron size={13} className="shrink-0" />
                    {derivedStatus === 'running' && (
                      <Loader2 size={12} className="shrink-0 animate-spin" />
                    )}
                    <span className="truncate text-primary font-medium">{summary}</span>
                  </button>
                  {isExpanded && item.items.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-border/30">
                      {item.items.map((child, ci) => {
                        if (child.type === 'reasoning_block') {
                          return (
                            <div
                              key={`rc-r-${ci}`}
                              className="mb-2 pl-3 border-l-2 border-accent/40 italic text-secondary text-[13px] leading-relaxed whitespace-pre-wrap"
                            >
                              {child.content}
                            </div>
                          );
                        }
                        if (child.type === 'tool_call_row') {
                          return (
                            <ToolCallRow
                              key={`rc-t-${ci}-${child.tool_call_id}`}
                              row={child}
                            />
                          );
                        }
                        if (child.type === 'agent_message') {
                          // Empty-content marker delimits silent tool batches.
                          return (
                            <div
                              key={`rc-m-${ci}`}
                              className="my-2 border-t border-border/40"
                              aria-hidden
                            />
                          );
                        }
                        return null;
                      })}
                    </div>
                  )}
                  </div>
                </div>
              );
            } else if (item.type === 'agent_notify') {
              const urgencyColor = item.urgency === 'high' ? 'border-error/40 bg-error/5' : 'border-accent/30 bg-accent/5';
              rendered = (
                <div key={`notify-${index}`} className={`rounded-lg border ${urgencyColor} px-4 py-3`}>
                  <p className="text-sm font-medium text-primary">{item.title}</p>
                  <p className="text-sm text-secondary mt-1">{item.body}</p>
                </div>
              );
            } else if (item.type === 'refresh_status') {
              rendered = (
                <RefreshTurnStatus
                  key={`refresh-${index}-${item.timestamp}`}
                  status={item.status}
                  trigger={item.trigger}
                  timestamp={item.timestamp}
                />
              );
            } else if (item.type === 'approval_card') {
              const resolved = approvals.get(item.tool_call_id)?.resolved ?? item.resolved;
              rendered = item.tool_name === 'request_credential' ? (
                <CredentialCard
                  key={`cred-${item.tool_call_id}`}
                  credential={{
                    tool_call_id: item.tool_call_id,
                    name: item.tool_args?.name as string ?? '',
                    domain: item.tool_args?.domain as string ?? '',
                    fields: (item.tool_args?.fields as string[]) ?? [],
                    reason: item.tool_args?.reason as string ?? '',
                    resolved: !!resolved,
                  }}
                  projectId={projectId}
                  sessionId={holderSessionId ?? sessionId}
                  onResolve={(toolCallId: string) => {
                    setApprovals((prev) => {
                      const next = new Map(prev);
                      const existing = next.get(toolCallId);
                      if (existing) {
                        next.set(toolCallId, { ...existing, resolved: 'approved' });
                      }
                      return next;
                    });
                  }}
                />
              ) : (
                <ApprovalCard
                  key={`apr-${item.tool_call_id}`}
                  approval={item}
                  projectId={projectId}
                  sessionId={holderSessionId ?? sessionId}
                  resolved={resolved}
                  onResolve={(toolCallId: string, resolution: 'approved' | 'denied') => {
                    setApprovals((prev) => {
                      const next = new Map(prev);
                      const existing = next.get(toolCallId);
                      if (existing) {
                        next.set(toolCallId, { ...existing, resolved: resolution });
                      }
                      return next;
                    });
                  }}
                />
              );
            }

            if (!rendered) return null;

            return historical ? (
              <div key={`hist-${index}`} className="opacity-50">
                {rendered}
              </div>
            ) : (
              rendered
            );
          })}

        {!loading &&
          Array.from(approvals.values())
            .filter(
              (a) =>
                !a.resolved &&
                !items.some(
                  (i) => i.type === 'approval_card' && i.tool_call_id === a.tool_call_id,
                ),
            )
            .map((a) =>
              a.tool_name === 'request_credential' ? (
                <CredentialCard
                  key={`rt-cred-${a.tool_call_id}`}
                  credential={{
                    tool_call_id: a.tool_call_id,
                    name: a.tool_args?.name as string ?? '',
                    domain: a.tool_args?.domain as string ?? '',
                    fields: (a.tool_args?.fields as string[]) ?? [],
                    reason: a.tool_args?.reason as string ?? '',
                  }}
                  projectId={projectId}
                  sessionId={holderSessionId ?? sessionId}
                  onResolve={(toolCallId: string) => {
                    setApprovals((prev) => {
                      const next = new Map(prev);
                      const existing = next.get(toolCallId);
                      if (existing) {
                        next.set(toolCallId, { ...existing, resolved: 'approved' });
                      }
                      return next;
                    });
                  }}
                />
              ) : (
                <ApprovalCard
                  key={`rt-apr-${a.tool_call_id}`}
                  approval={a}
                  projectId={projectId}
                  sessionId={holderSessionId ?? sessionId}
                  resolved={a.resolved}
                  onResolve={(toolCallId: string, resolution: 'approved' | 'denied') => {
                    setApprovals((prev) => {
                      const next = new Map(prev);
                      const existing = next.get(toolCallId);
                      if (existing) {
                        next.set(toolCallId, { ...existing, resolved: resolution });
                      }
                      return next;
                    });
                  }}
                />
              ),
            )}

        {subAgentLoading && (
          <div className="flex gap-[10px]">
            <MessageAvatar variant="agent" />
            <div className="flex-1 min-w-0">
              <div className="font-mono text-[11px] mb-1">
                <span className="text-secondary">{subAgentLoading}</span>
              </div>
              <div className="text-[13px] leading-[1.55]">
                <span className="inline-flex gap-1 text-secondary">
                  <span className="animate-pulse">●</span>
                  <span className="animate-pulse" style={{animationDelay: '0.2s'}}>●</span>
                  <span className="animate-pulse" style={{animationDelay: '0.4s'}}>●</span>
                </span>
              </div>
            </div>
          </div>
        )}

        {showThinking && !stream && !subAgentLoading && (
          <div className="flex items-center gap-2 px-2 py-1 text-secondary text-sm">
            <Loader2 size={14} className="animate-spin" />
            <span>Thinking...</span>
          </div>
        )}

        {stream && (
          <StreamingMessage
            text={stream.text}
            source={stream.source}
            isComplete={stream.isComplete}
          />
        )}
      </div>

      {injectError && (
        <div className="shrink-0 px-4 py-1">
          <p className="text-xs text-error">{injectError}</p>
        </div>
      )}

      {slotHeldNotice && (
        <div className="shrink-0 px-4">
          <SlotHeldNotice
            holdingSessionId={slotHeldNotice.holdingSessionId}
            onWait={() => setSlotHeldNotice(null)}
            onCancelAndSend={async () => {
              // Cancel the running turn on the holding session, then re-inject
              // the previously-typed message. The backend /cancel now accepts a
              // session_id so it stops the SPECIFIC session holding the slot
              // (slotHeldNotice.holdingSessionId), not just the default one.
              await cancelMessage(projectId, slotHeldNotice.holdingSessionId);
              const pending = slotHeldNotice;
              // Replace the notice with the actual user message (per
              // DISPATCH-2026-05-22 §5.4). After /cancel the slot is freed, so
              // re-inject targeting the VIEWED session — it now takes the slot
              // and the message lands in the conversation the user is looking
              // at (single active-loop slot model).
              setSlotHeldNotice(null);
              setInputText('');
              const wireContent = pending.pendingContent;
              setItems((prev) => [
                ...prev,
                {
                  type: 'user_message' as const,
                  content: wireContent,
                  timestamp: new Date().toISOString(),
                  ...(pending.pendingTarget && { target: pending.pendingTarget }),
                },
              ]);
              await injectMessage(
                projectId,
                pending.pendingContent,
                pending.pendingTarget,
                pending.pendingNonce,
                pending.pendingAttachments,
                sessionId,
              );
            }}
          />
        </div>
      )}

      {cancelTimeoutNotice && (
        <div className="shrink-0 px-4 py-1">
          <p className="text-xs text-secondary" role="status">{cancelTimeoutNotice}</p>
        </div>
      )}

      <div className="shrink-0 px-4 pb-4 pt-2 max-md:fixed max-md:bottom-0 max-md:left-0 max-md:right-0 max-md:bg-background max-md:z-[60] max-md:pb-[env(safe-area-inset-bottom,12px)]">
        {queueActive ? (
          <ComposerDisabledPrompt onPauseQueue={stopQueue} />
        ) : (
        <div className="relative flex flex-col gap-2 bg-background border border-border rounded-[10px] shadow-lg px-3 py-2">
          {showCommandDropdown && filteredCommands.length > 0 && (
            <div className="absolute bottom-full left-0 mb-1 w-64 bg-zinc-800 border border-zinc-700 rounded-lg shadow-lg overflow-hidden z-50">
              {filteredCommands.map((cmd, i) => (
                <button
                  key={cmd.name}
                  className={`w-full text-left px-3 py-2 text-sm hover:bg-zinc-700 max-md:min-h-[44px] ${
                    i === selectedCommandIndex ? 'bg-zinc-700' : ''
                  }`}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    selectCommand(cmd.name);
                  }}
                >
                  <span className="font-medium text-zinc-200">{cmd.name}</span>
                  <span className="ml-2 text-zinc-500">{cmd.description}</span>
                </button>
              ))}
            </div>
          )}
          {showMentionDropdown && (
            <div className="absolute bottom-full left-0 mb-1 w-64 bg-zinc-800 border border-zinc-700 rounded-lg shadow-lg overflow-hidden z-50">
              {filteredAgents.length === 0 ? (
                <div className="px-3 py-2 text-sm text-zinc-500">No agents available</div>
              ) : (
                filteredAgents.map((agent, i) => (
                  <button
                    key={agent.slug}
                    className={`w-full text-left px-3 py-2 text-sm hover:bg-zinc-700 max-md:min-h-[44px] ${
                      i === selectedMentionIndex ? 'bg-zinc-700' : ''
                    }`}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      selectMention(agent.slug);
                    }}
                  >
                    <span className="font-medium text-zinc-200">@{agent.slug}</span>
                    <span className="ml-2 text-zinc-500">{agent.name}</span>
                  </button>
                ))
              )}
            </div>
          )}
          {attachments.length > 0 && (
            <div
              className="flex flex-wrap gap-2 max-h-[100px] overflow-y-auto"
              data-testid="chip-strip"
            >
              {attachments.map((a) => (
                <AttachmentChip
                  key={a.id}
                  filename={a.filename}
                  mime={a.mime}
                  size={a.size}
                  status={a.status}
                  thumbnailUrl={a.thumbnailUrl}
                  errorMessage={a.errorMessage}
                  onRemove={() => removeAttachment(a.id)}
                  onRetry={a.status === 'error' ? () => retryAttachment(a.id) : undefined}
                />
              ))}
            </div>
          )}
          <div className="flex items-center gap-2">
            <input
              type="file"
              multiple
              ref={fileInputRef}
              className="hidden"
              onChange={handleFilePickerChange}
              data-testid="attachment-file-input"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              aria-label="Attach files"
              className="shrink-0 p-2 text-secondary hover:text-primary rounded max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
            >
              <Plus size={18} />
            </button>
            <textarea
              ref={textareaRef}
              value={inputText}
              onChange={(e) => handleInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              placeholder="Send a message..."
              rows={1}
              disabled={isCancelling}
              className="flex-1 resize-none text-[13px] max-md:text-base bg-transparent focus:outline-none leading-relaxed disabled:opacity-50"
            />
            {(agentStatus === 'running' || agentStatus === 'waiting') ? (
              <>
                <button
                  type="button"
                  onClick={() => {
                    if (isCancelling) return;
                    setCancelTimeoutNotice(null);
                    setIsCancelling(true);
                    // Cancel the SPECIFIC running session: prefer the resolved
                    // slot holder, falling back to the viewed session.
                    cancelMessage(projectId, holderSessionId ?? sessionId).catch(() => {
                      // POST failed (offline, daemon down, etc.) — drop the
                      // optimistic state immediately so the user can retry
                      // instead of waiting for the 10s timeout.
                      setIsCancelling(false);
                      setCancelTimeoutNotice('Cancel request failed — try again if needed.');
                    });
                  }}
                  onTouchEnd={(e) => {
                    e.preventDefault();
                    if (isCancelling) return;
                    setCancelTimeoutNotice(null);
                    setIsCancelling(true);
                    cancelMessage(projectId, holderSessionId ?? sessionId).catch(() => {
                      setIsCancelling(false);
                      setCancelTimeoutNotice('Cancel request failed — try again if needed.');
                    });
                  }}
                  disabled={isCancelling}
                  aria-label={isCancelling ? 'Cancelling' : 'Stop'}
                  className="shrink-0 p-1.5 rounded-lg transition-colors duration-150 cursor-pointer text-red-500 hover:bg-red-500/10 disabled:cursor-default disabled:hover:bg-transparent max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
                >
                  {isCancelling
                    ? <Loader2 size={18} className="animate-spin" />
                    : <Square size={18} />}
                </button>
                <button
                  type="button"
                  onClick={handleSend}
                  onTouchEnd={(e) => { e.preventDefault(); handleSend(); }}
                  disabled={!canSend || isCancelling}
                  title={disabledReason || undefined}
                  className={`shrink-0 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide transition-colors duration-150 max-md:min-h-[44px] max-md:flex max-md:items-center max-md:justify-center ${
                    canSend && !isCancelling
                      ? 'bg-accent text-white hover:bg-accent/85 cursor-pointer'
                      : 'bg-secondary/20 text-secondary/40 cursor-default'
                  }`}
                >
                  Queue
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                onTouchEnd={(e) => { e.preventDefault(); handleSend(); }}
                aria-disabled={!canSend}
                aria-label="Send"
                disabled={!canSend}
                title={disabledReason || undefined}
                className={`shrink-0 p-1.5 rounded-lg transition-colors duration-150 cursor-pointer max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center ${
                  canSend
                    ? 'text-accent hover:bg-accent/10'
                    : 'text-secondary opacity-40'
                }`}
              >
                <Send size={18} />
              </button>
            )}
          </div>
        </div>
        )}
      </div>
    </div>
  );
}
