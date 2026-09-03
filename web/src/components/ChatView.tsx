// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Send, Loader2, Plus, ChevronRight, ChevronDown, ArrowDown } from 'lucide-react';
import { api, apiWithTotal } from '../config';
import { useWebSocket } from '../hooks/useWebSocket';
import { useAgent } from '../hooks/useAgent';
import { useQueue } from '../hooks/useQueue';
import { useAutoScroll } from '../hooks/useAutoScroll';
import {
  transformChatHistory,
  truncateResult,
  mergeRecoveredAssistantMessage,
  describeLiveActivity,
} from '../utils/chatTransform';
import type { DisplayItem } from '../utils/chatTransform';
import { isWorkerHandle } from '../utils/subAgentHandle';
import type { ChatMessage as ChatMessageRow, AgentStatusEvent } from '../types';
import AgentErrorNotice from './AgentErrorNotice';
import { parseProviderError, providerErrorKey } from '../utils/providerError';
import AttachmentChip from './AttachmentChip';
import { useAttachments } from '../hooks/useAttachments';
import { useAnnotations } from '../hooks/useAnnotations';
import {
  annotationFilename,
  formatQuotes,
  renderAnnotatedPng,
  type Annotation,
} from '../utils/annotations';
import { uploadFile } from '../lib/attachment-upload';
import { BASE_URL, isRelayMode } from '../config';
import { buildAttachmentsBlock, parseAttachmentsBlock } from '../lib/attachment-parsing';
import ComposerDisabledPrompt from './ComposerDisabledPrompt';
import PinTargetSelect, { resolveSendTarget } from './PinTargetSelect';
import { useSessions } from '../hooks/useSessions';
import { StopGlyph } from './StopGlyph';
import ContextStrip from './ContextStrip';
import { useContextUsage } from '../hooks/useContextUsage';
import { useT, translate } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import type { StringKey } from '../i18n/strings';
import { publishSessionMessages, sessionMessagesKey } from '../utils/sessionMessagesStore';
import { budgetTimelineText } from '../budget/timelineText';

type AgentRunItem = Extract<DisplayItem, { type: 'agent_run' }>;
type CapsuleChild = AgentRunItem['items'][number];

function formatToolBreakdown(counts: Record<string, number>): string {
  const entries = Object.entries(counts);
  if (entries.length === 0) return '';
  entries.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return entries.map(([n, c]) => (c === 1 ? n : `${c} ${n}s`)).join(', ');
}

/**
 * One-line label for a staged annotation in the composer chip's expanded list.
 * Deliberately NOT translated: every part of it is dynamic content the agent
 * produced (a page title, a workspace path, a quoted span), which the i18n
 * rules exclude. Only the bare "Browser" fallback is chrome.
 */
function annotationSummary(a: Annotation, tr: (key: StringKey) => string): string {
  switch (a.kind) {
    case 'browser':
      return a.pageTitle || tr('panel.browser.label');
    case 'image':
      return a.path;
    case 'text':
      return a.lines ? `${a.path}:${a.lines[0]}-${a.lines[1]}` : a.path;
    case 'file':
      return a.path;
  }
}

// Locale-aware translator for capsule summaries; defaults to English so any
// caller that omits it (and any non-localized path) gets identical output.
type CapsuleTr = (key: StringKey, vars?: Record<string, string | number>) => string;
const EN_CAPSULE: CapsuleTr = (k, v) => translate('en', k, v);

/** Parse an ISO timestamp to epoch ms, falling back to "now" for anything
 *  unparseable — used for the fanout_card's batch-duration start point. */
function tsToMsSafe(ts: string): number {
  const n = Date.parse(ts);
  return Number.isFinite(n) ? n : Date.now();
}

function formatDuration(startedAt: number, endedAt: number | null, tr: CapsuleTr = EN_CAPSULE): string {
  if (!endedAt || endedAt <= startedAt) return tr('duration.lessThan1s');
  const ms = endedAt - startedAt;
  if (ms < 1000) return tr('duration.lessThan1s');
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  return rs ? `${m}m ${rs}s` : `${m}m`;
}

function capsuleSummaryText(capsule: AgentRunItem, tr: CapsuleTr = EN_CAPSULE): string {
  const breakdown = formatToolBreakdown(capsule.tool_call_count_by_name);
  const dur = formatDuration(capsule.started_at, capsule.ended_at, tr);
  const head = breakdown || (capsule.has_thinking ? tr('chat.capsule.thinking') : tr('chat.capsule.agentStep'));
  let line = `${head} · ${dur}`;
  if (capsule.status === 'error' || capsule.status === 'stopped') {
    line += ` ${tr('chat.capsule.stoppedAtError')}`;
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

/**
 * Live reasoning accumulation. During the model's <think> phase the WS
 * stream_delta event carries `reasoning_content` with EMPTY `text`. We render
 * it as a single reasoning_block inside the running live capsule, accumulating
 * successive deltas into that one block (rather than emitting one block per
 * delta). This mirrors how persisted reasoning is rendered (a reasoning_block
 * within the capsule) so the live and persisted views stay consistent.
 */
export function appendLiveReasoning(
  prev: DisplayItem[],
  reasoning: string,
  timestamp: string,
  source: string,
): DisplayItem[] {
  const live = getLiveRunningCapsule(prev);
  // Find a trailing reasoning_block in the running capsule to extend.
  if (live) {
    const items = live.capsule.items;
    const lastChild = items[items.length - 1];
    if (lastChild && lastChild.type === 'reasoning_block') {
      const merged: CapsuleChild = {
        ...lastChild,
        content: lastChild.content + reasoning,
      };
      const newItems = [...items];
      newItems[newItems.length - 1] = merged;
      const updated: AgentRunItem = {
        ...live.capsule,
        items: newItems,
        has_thinking: true,
        ended_at: Date.parse(timestamp),
      };
      const next = [...prev];
      next[live.idx] = updated;
      return next;
    }
  }
  // Live capsule exists but has no trailing reasoning_block (e.g. a tool row
  // was last) — append a new reasoning_block to that SAME capsule. It is
  // already anchored under its agent header, so no new header is needed.
  if (live) {
    return appendToLiveCapsule(
      prev,
      { type: 'reasoning_block', content: reasoning, timestamp, turn_id: timestamp },
      timestamp,
    );
  }

  // No running capsule — this reasoning starts a fresh agent turn. Anchor it
  // with a header-only agent_message (mirrors chatTransform's FE-A3) so the
  // capsule renders the agent avatar and does NOT visually attach to the
  // preceding user message. The agent_run capsule draws no avatar of its own
  // (it is indented to sit under a header); without this anchor the live
  // thinking floats unattributed — and in cold-start (no preceding user
  // message) it is not attributed to the management agent at all.
  const lastItem = prev[prev.length - 1];
  const anchored =
    !!lastItem &&
    (lastItem.type === 'agent_message' ||
      lastItem.type === 'sub_agent_message' ||
      lastItem.type === 'agent_run');
  const base: DisplayItem[] = anchored
    ? prev
    : [
        ...prev,
        { type: 'agent_message', content: '', source, timestamp, isHeaderOnly: true },
      ];
  return appendToLiveCapsule(
    base,
    { type: 'reasoning_block', content: reasoning, timestamp, turn_id: timestamp },
    timestamp,
  );
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
  const t = useT();
  const { locale } = useLocale();
  const [expanded, setExpanded] = useState(false);
  const expandable = row.result_status !== 'pending';
  const Chevron = expanded ? ChevronDown : ChevronRight;

  return (
    <div className="mb-1 font-mono text-[11px] text-secondary">
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
            <div className="mt-1 ml-5 px-3 py-2 rounded bg-background border border-border/40 text-xs italic text-secondary/70">
              {t('chat.toolRow.noResult')}
            </div>
          );
        }
        const { text, footer } = truncateResult(raw, (k, v) => translate(locale, k, v));
        return (
          <div className="mt-1 ml-5">
            <pre className="px-3 py-2 rounded bg-background border border-border/40 text-xs text-secondary leading-relaxed whitespace-pre-wrap break-words font-mono">
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

// Bug #48 (fix C): per-session chat-history cache, mirroring useSessions'
// module-level `sessionsCache`. Switching back to a recently-viewed session
// paints its last-known transcript immediately (no skeleton flash) while a
// background refetch revalidates. Never authoritative — every hit still
// refetches; the cache only removes the blank-first-paint. Keyed
// `${projectId}:${sessionId}` (session ids are only unique per project).
interface ChatHistoryCacheEntry {
  messages: ChatMessageType[];
  total: number;
  loadedOffset: number;
}
const chatHistoryCache = new Map<string, ChatHistoryCacheEntry>();

/** Test-only: the module-level cache would otherwise leak between mounts. */
export function __clearChatHistoryCacheForTests() {
  chatHistoryCache.clear();
}
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
  SubAgentLifecycleEvent,
  UserMessageEvent,
  AgentNotifyEvent,
  StateRefreshLifecycleEvent,
  WorkspaceClaudemdWarningEvent,
  PendingEnqueuedEvent,
  PendingDispatchedEvent,
  PendingCancelledEvent,
  FanoutStartedEvent,
  FanoutTaskUpdateEvent,
  FanoutCompletedEvent,
  WebSocketEvent,
  Project,
} from '../types';
import ChatMessage from './ChatMessage';
import { OpenPathContext } from './filePreviewContext';
import StreamingMessage from './StreamingMessage';
import MessageAvatar from './MessageAvatar';
import ApprovalCard from './ApprovalCard';
import CredentialCard from './CredentialCard';
import RefreshTurnStatus from './RefreshTurnStatus';
import ClaudemdWarningBanner, { type ClaudemdWarning } from './ClaudemdWarningBanner';
import SlotHeldNotice from './SlotHeldNotice';
import PendingInputNotice from './PendingInputNotice';
import { ColdStartCard } from './ColdStartCard';
import SubAgentStatusBar from './SubAgentStatusBar';
import FanoutCard, { isTerminal, type FanoutTaskState } from './FanoutCard';
import SubAgentDrillIn from './SubAgentDrillIn';

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
   * One-shot composer prefill (Workbench card-tap doorway, spec 2026-07-24
   * §5.3 — route.draft, threaded down by ChatTab). Applied at most once:
   * seeded directly into the composer's initial state (visible immediately,
   * even before `sessionId` resolves), and re-asserted by the per-session
   * draft-swap effect the first time `sessionId` transitions away from
   * `undefined` — otherwise that effect's per-session draft map (which has
   * no entry for a session never visited before) would blank it out from
   * under the seeded value. Never auto-sent; never spawns a session by
   * itself. `onDraftConsumed` fires once, immediately on mount, so the
   * caller can clear its copy (route.draft) and the prefill won't reappear
   * on a later remount (e.g. tab switch away and back).
   */
  initialDraft?: string;
  onDraftConsumed?: () => void;
  /**
   * Re-fetch this project's runtime fields and merge
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

// Pending-input queue (spec 006 · v3). One queued message awaiting its turn,
// keyed by nonce in `pendingInputs`. Two kinds (§11c):
//   - `'cross'` — slot held by ANOTHER session (`_pending_inject`); `holder` is
//     that session. Offers [Run now].
//   - `'same'`  — THIS session's own turn is mid-flight (`session._queue`);
//     `holder` is unused. No Run-now (it drains automatically).
// `content`/`timestamp` mirror the kept optimistic bubble so cancel/dispatch can
// locate and remove it. `rawText` is the user's RAW typed text (BEFORE
// buildAttachmentsBlock) — what ↑/tap-recall loads back into the composer (§12
// R4). `hasAttachments` gates recall off (don't half-restore chips).
interface PendingInputEntry {
  kind: 'cross' | 'same';
  sessionId: string;
  holder: string;
  content: string;
  rawText: string;
  hasAttachments: boolean;
  timestamp: string;
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

export default function ChatView({ projectId, project, agentStatus, statusTick, mentionAgents, sessionId, initialDraft, onDraftConsumed, onRefreshProject }: ChatViewProps) {
  const t = useT();
  const { locale } = useLocale();
  // Chrome-only translator for the annotation chip's fallback label (the
  // rest of each summary is dynamic content and stays untranslated).
  const chromeTr = useMemo(() => (key: StringKey) => translate(locale, key), [locale]);
  // The WS effect's handler closures don't re-subscribe on locale change —
  // read the live locale through a ref (same staleness fix as sessionIdRef).
  const localeRef = useRef(locale);
  useEffect(() => { localeRef.current = locale; }, [locale]);
  // Spec 002: open a clicked workspace path in the FilePreviewDrawer. Provided
  // by ProjectDetail (which owns setRoute); null when no provider is present.
  const onOpenPath = useContext(OpenPathContext) ?? undefined;
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
  // Spec 078 §11.5: the workspace panel derives touched files and the
  // fallback screenshot from this same transcript instead of fetching it
  // again. Published by reference; ChatTab subscribes via useSessionMessages.
  useEffect(() => {
    publishSessionMessages(sessionMessagesKey(projectId, sessionId), rawMessages);
  }, [projectId, sessionId, rawMessages]);
  const [items, setItems] = useState<DisplayItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [totalMessages, setTotalMessages] = useState(0);
  // Cold-start consent card (imported non-empty workspace, first session).
  const [coldStartBusy, setColdStartBusy] = useState(false);
  const [coldStartDismissed, setColdStartDismissed] = useState(false);
  // Translated inline error on the cold-start card (scan failed, e.g.
  // missing API key). Previously the failure was console-only.
  const [coldStartError, setColdStartError] = useState<string | null>(null);
  // Classified agent/provider error for the viewed session (credential-error
  // surfacing): fed by agent.status error broadcasts and run-status
  // last_terminal_event hydration; rendered as AgentErrorNotice.
  const [agentError, setAgentError] = useState<{
    code?: string;
    message?: string;
  } | null>(null);
  const [loadedOffset, setLoadedOffset] = useState(0);
  const [stream, setStream] = useState<StreamState | null>(null);
  const [approvals, setApprovals] = useState<Map<string, PendingApproval>>(new Map());
  const [expandedCapsules, setExpandedCapsules] = useState<Set<string>>(new Set());
  // Seeded from initialDraft (Workbench prefill doorway) so it's visible on
  // the very first paint — useState's initializer runs once, at mount, so
  // this cannot re-apply on a later re-render with a changed prop.
  const [inputText, setInputText] = useState(initialDraft ?? '');
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
  // Spec 074 — the composer "Talking to" pin. Backend truth comes from the
  // session list (`pinned_target` on the entry); `localPin` is the
  // optimistic overlay for the VIEWED session so the dropdown responds
  // immediately and keeps working for a brand-new session whose JSONL has
  // not materialized yet (the PATCH 404s until the first message lands —
  // handleSend re-persists the pin after a successful pinned send).
  const { sessions: pinSessions, pinAgent } = useSessions(projectId);
  const [localPin, setLocalPin] = useState<{ sessionId: string; slug: string | null } | null>(null);
  // The pin PATCH write-locks the session JSONL on the backend for a
  // load+meta-rewrite burst; an inject landing inside that burst gets
  // rejected (2026-08-31 bug: pick agent → send immediately → "Failed to
  // send message"). Track the in-flight PATCH so handleSend can serialize
  // behind it; outcome is irrelevant (failure keeps the optimistic pin).
  const pinPatchInFlightRef = useRef<Promise<void> | null>(null);
  const persistPin = useCallback((sid: string, slug: string | null) => {
    const done: Promise<void> = pinAgent(sid, slug)
      .then(() => undefined, () => undefined)
      .finally(() => {
        if (pinPatchInFlightRef.current === done) {
          pinPatchInFlightRef.current = null;
        }
      });
    pinPatchInFlightRef.current = done;
  }, [pinAgent]);
  const backendPin = useMemo(() => {
    if (sessionId === undefined) return null;
    const entry = pinSessions.find(
      (s) => s.session_id === sessionId || s.session_uuid === sessionId,
    );
    return entry?.pinned_target ?? null;
  }, [pinSessions, sessionId]);
  const pinnedTarget =
    localPin && localPin.sessionId === sessionId ? localPin.slug : backendPin;
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
  // Pending-input queue (spec 006, Path A). A NONCE-KEYED OVERLAY — NOT a flag
  // in `items` (the historyItems memo reseed would stomp that). Each entry is a
  // message the user sent to a session while another session held the project's
  // single active-loop slot; the backend accepted + queued it (202
  // `queued_pending_slot`) and auto-dispatches it when the slot frees. The map
  // is PROJECT-scoped; the waiting affordance renders only for entries whose
  // `sessionId` is the viewed session. `content`/`timestamp` identify the kept
  // optimistic bubble so cancel/dispatch can locate it. Cleared per-nonce on
  // `chat.pending_dispatched` (sole clear trigger) / `chat.pending_cancelled`,
  // and rebuilt from GET /pending on session load + WS reconnect.
  const [pendingInputs, setPendingInputs] = useState<Map<string, PendingInputEntry>>(new Map());
  // Workspace CLAUDE.md interference banner: latest unhandled warning
  // for this project (cleared on dismiss).
  const [claudemdWarning, setClaudemdWarning] = useState<ClaudemdWarning | null>(null);
  // Composer attachments live in a shared hook (the queue composer stages
  // files the same way). Cap-exceeded copy lands in the inject-error banner.
  const {
    attachments,
    anyUploading,
    anyDone,
    allError,
    removeAttachment,
    retryAttachment,
    clearAttachments,
    handleFilePickerChange,
    handlePaste,
    handleDrop: dropAttachments,
  } = useAttachments(projectId, { onError: setInjectError });
  // Spec 078 §5.4 — annotation drafts the panel staged for THIS session. The
  // store is module-level so the panel (which lives in ChatTab) and the
  // composer share one list without prop drilling through the shell.
  const {
    annotations,
    remove: removeAnnotation,
    clear: clearAnnotations,
    updateNote: updateAnnotationNote,
  } = useAnnotations(sessionId);
  const [annotationsOpen, setAnnotationsOpen] = useState(false);
  const [dragCounter, setDragCounter] = useState(0);
  // Stop-button optimistic state. Set true synchronously on click so the
  // input row immediately shows a loading affordance; cleared when
  // agentStatus transitions to 'idle' (loop actually exited) or after a
  // 10s timeout falls back to a retry notice. The backend
  // agent.status:idle broadcast is authoritative — this just covers the
  // in-flight gap between click and the WS event arriving.
  const [isCancelling, setIsCancelling] = useState(false);
  const [cancelTimeoutNotice, setCancelTimeoutNotice] = useState<string | null>(null);
  // Fanout live overlay (spec 009 §0.5): per-fanout, per-handle status —
  // NOT baked into the fanout_card DisplayItem itself, same pattern as the
  // `approvals` Map overlaying `approval_card` items. Fed by
  // fanout.task_update; fanoutCompletedAt records when fanout.completed
  // fired so FanoutCard can freeze its (batch-level) duration display.
  //
  // Round 2 (issue 1, per-task countdown): each handle's entry is now
  // {status, completedAtMs} rather than a bare status string, so FanoutCard
  // can freeze each row's OWN duration independently instead of sharing one
  // never-freezing batch countdown.
  const [fanoutStatuses, setFanoutStatuses] = useState<Map<string, Record<string, FanoutTaskState>>>(new Map());
  const [fanoutCompletedAt, setFanoutCompletedAt] = useState<Map<string, number>>(new Map());
  // Row-click drill-in target (spec 009 §0.5): replaces the chat message area
  // (NOT a modal) with SubAgentDrillIn. null = showing the normal chat.
  const [drillIn, setDrillIn] = useState<{ handle: string; label: string } | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const localNoncesRef = useRef<Map<string, number>>(new Map());
  const wasRunningRef = useRef(false);
  const { on, off, connectionState } = useWebSocket();
  const { injectMessage, cancelMessage, newSession, coldStartScan, cancelPendingInput, getPending } = useAgent();
  // Queue-active gating: when the queue is running (actively dispatching a
  // task), the chat composer is replaced by ComposerDisabledPrompt — the user
  // must pause the queue first. ('idle'/'paused' leave the composer enabled.)
  const { snapshot: queueSnapshot, stopQueue } = useQueue(projectId);
  // Ambient context meter above the composer. Event-driven off the ledger's
  // existing budget.spend_updated (context only moves on an LLM call), so no
  // polling and no new WS type. Renders nothing until compaction is close.
  const { usage: contextUsage } = useContextUsage(projectId, sessionId);
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
  // Latest pending-input map, readable synchronously inside WS handlers and the
  // nonce-eviction sweep (which must EXEMPT pending nonces — spec §3h F4 — so a
  // long wait can't evict the nonce before the dispatch echo dedups against the
  // kept optimistic bubble). Never read a value mutated inside a setState
  // updater (React 19 batching) — read this committed ref instead.
  const pendingInputsRef = useRef<Map<string, PendingInputEntry>>(pendingInputs);
  pendingInputsRef.current = pendingInputs;
  // Bug29 D2: committed mirrors read by the seed effect, which stays keyed ONLY
  // on [historyItems]. Adding status/holder/items as effect deps would reseed on
  // every flip and re-wipe the very overlay the skip protects — so these values
  // are read from refs, never effect-closure state (React-19 batching rule,
  // CLAUDE.md "React Anti-Patterns"). `itemsSessionRef` records which session the
  // current `items` overlay belongs to, so the mid-turn skip protects ONLY the
  // viewed session's in-flight overlay and never strands a tab-switch on the
  // prior session's content.
  const agentStatusRef = useRef(agentStatus);
  agentStatusRef.current = agentStatus;
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const itemsSessionRef = useRef<string | undefined>(undefined);
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
    () => transformChatHistory(rawMessages, project.workspace, (k, v) => translate(locale, k, v)),
    [rawMessages, project.workspace, locale],
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
    // Bug29 D2: an in-flight turn's intermediate content — live capsule tool
    // rows, streamed reasoning, sub-agent bubbles, and any final JUST finalized
    // by handleSend — lives ONLY in `items`; it is never mirrored into
    // rawMessages. A mid-turn send pushes the optimistic user bubble into
    // rawMessages, which recomputes historyItems and would trigger the wholesale
    // reseed below, wiping that overlay. Skip the reseed while the VIEWED
    // session's turn is in flight and there is rendered content to protect.
    //
    // All gates read committed refs, not effect-closure state (deps stay
    // [historyItems]). The "content to protect" gate is `items` non-empty — NOT
    // "a live capsule exists": handleSend finalizes the live capsule to
    // 'completed' BEFORE the rawMessages push that triggers this reseed, so a
    // live-only gate would fail to protect the just-finalized capsule. A fresh
    // mount has items === [] so it still seeds; `itemsSessionRef` makes a
    // TAB-SWITCH to the running holder reseed the NEW session's history instead
    // of stranding the prior session's stale overlay. On idle the catch-up
    // (refreshRawMessages → historyItems change) re-runs this with the turn no
    // longer in flight, restoring canonical server history.
    const viewedIsHolder =
      sessionIdRef.current !== undefined &&
      sessionIdRef.current === holderSessionIdRef.current;
    const turnInFlight =
      agentStatusRef.current === 'running' || agentStatusRef.current === 'waiting';
    const itemsAreForViewedSession = itemsSessionRef.current === sessionIdRef.current;
    if (viewedIsHolder && turnInFlight && itemsAreForViewedSession && itemsRef.current.length > 0) {
      return;
    }
    itemsSessionRef.current = sessionIdRef.current;
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
      // Typed as the SAME row shape normal ingest consumes (ChatMessage):
      // the old inline type omitted `source`, which is how a hardcoded
      // source: 'assistant' crept in and split one management turn into
      // two speakers (INVESTIGATION-two-speakers).
      api<ChatMessageRow[]>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=1${sessionParam}`,
      )
        .then((messages) => {
          if (!messages || messages.length === 0) return;
          const latest = messages[messages.length - 1];
          if (latest.role !== 'assistant') return;
          const restReasoning = (latest.reasoning_content ?? '').trim();
          // Carry reasoning_content like the primary refreshRawMessages path:
          // a recovered reasoning turn is a COMPLETED turn, so emit it as a
          // closed (collapsed) agent_run capsule containing a reasoning_block,
          // mirroring transformChatHistory's persisted handling. Without this
          // the REST fallback silently drops reasoning.
          if (restReasoning) {
            setItems((prevItems) => {
              const ts = latest.timestamp ?? new Date().toISOString();
              const ms = Date.parse(ts);
              // Dedup against the primary refetch / live append.
              const already = prevItems.some(
                (it) =>
                  it.type === 'agent_run' &&
                  it.items.some(
                    (c) => c.type === 'reasoning_block' && c.content.trim() === restReasoning,
                  ),
              );
              if (already) return prevItems;
              // Close any still-running live capsule first.
              const closed = finalizeLiveCapsule(prevItems, 'completed');
              const capsule: AgentRunItem = {
                type: 'agent_run',
                capsule_id: `cap:rest:${ms}:${Math.random().toString(36).slice(2, 8)}`,
                status: 'completed',
                items: [
                  {
                    type: 'reasoning_block',
                    content: restReasoning,
                    timestamp: ts,
                    turn_id: ts,
                  },
                ],
                tool_call_count_by_name: {},
                has_thinking: true,
                started_at: Number.isFinite(ms) ? ms : Date.now(),
                ended_at: Number.isFinite(ms) ? ms : Date.now(),
              };
              return [...closed, capsule];
            });
          }
          if (!latest.content) return;
          // Identity + dedup live in mergeRecoveredAssistantMessage: source
          // is taken verbatim from the persisted row (same as normal
          // ingest), and dedup scans the whole list — the old inline
          // version hardcoded source: 'assistant' and only checked the
          // last item, rendering one management turn as two speakers.
          setItems((prevItems) => mergeRecoveredAssistantMessage(prevItems, latest));
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
        const nextOffset = Math.min(targetLimit, data.length || targetLimit);
        setLoadedOffset(nextOffset);
        // Bug #58 (hygiene): keep chatHistoryCache in step with the catch-up.
        // The cache used to be written ONLY by the session-load effect, so
        // every completed turn left the entry a little further behind — this
        // is the running→idle catch-up, the moment the freshest canonical
        // history is in hand. Same key format as the load effect. Not the fix
        // for the mid-run drop on its own (the in-flight rounds still are not
        // in the cache while the turn runs), but it stops the entry drifting
        // stale by construction between turns.
        chatHistoryCache.set(`${projectId}:${sid}`, {
          messages: data,
          total,
          loadedOffset: nextOffset,
        });
      })
      .catch(() => {
        // best-effort refresh; the seed remains the prior history
      });
  }, [projectId]);

  // Auto-follow arbiter (backlog #44). scrollToBottom now follows the bottom
  // only while the user is pinned there; when they scroll up to read earlier
  // content it no-ops (and surfaces the jump-to-latest pill) instead of yanking
  // them back on every stream token. Gating lives inside scrollToBottom, so the
  // ~17 existing call sites keep calling it unchanged — only the user's own
  // send, the pill, and a session switch pass { force: true }.
  const {
    scrollToBottom,
    onScroll: handleScroll,
    showJumpButton,
    reset: resetAutoScroll,
  } = useAutoScroll(scrollRef);

  // On session switch (and mount), snap to the newest message and re-arm
  // auto-follow — a freshly opened conversation should start pinned to the
  // bottom, not wherever the previous session's scroll happened to be.
  useEffect(() => {
    resetAutoScroll();
  }, [sessionId, resetAutoScroll]);

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
  const holderAbortRef = useRef<AbortController | null>(null);
  const fetchHolder = useCallback(() => {
    // Bug #48 (fix C): rapid session switching re-fires this on every switch.
    // Abort the superseded request so responses can't pile up or resolve out
    // of order (a slow older response would overwrite a newer holder value).
    holderAbortRef.current?.abort();
    const controller = new AbortController();
    holderAbortRef.current = controller;
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    api<{
      project_id: string;
      status: string;
      current_holder_session_id?: string | null;
      last_terminal_event?: {
        type: string;
        details?: string | null;
        error_code?: string;
      } | null;
    }>(`/api/v2/agents/${encodeURIComponent(projectId)}/run-status${qs}`, {
      signal: controller.signal,
    })
      .then((result) => {
        setHolderSessionId(result.current_holder_session_id ?? null);
        // Hydrate a classified error after reload (the agent.status broadcast
        // that carried it is ephemeral). Only fill in when nothing fresher is
        // already showing.
        const ev = result.last_terminal_event;
        if (ev && ev.type === 'error') {
          setAgentError((prev) =>
            prev ?? { code: ev.error_code, message: ev.details ?? undefined },
          );
        }
      })
      .catch(() => {
        // best effort — leave the prior holder value in place
      });
  }, [projectId, sessionId]);

  // Pending-input queue (spec 006). Remove a kept optimistic bubble (identified
  // by its content + timestamp) from BOTH the live `items` and `rawMessages`
  // (so the transform-once reseed doesn't re-emit it). Walks from the end so
  // concurrent appends of unrelated bubbles are left intact.
  const removeOptimisticBubble = useCallback((content: string, timestamp: string) => {
    setItems((prev) => {
      for (let i = prev.length - 1; i >= 0; i--) {
        const it = prev[i];
        if (it.type === 'user_message' && it.content === content && it.timestamp === timestamp) {
          return [...prev.slice(0, i), ...prev.slice(i + 1)];
        }
      }
      return prev;
    });
    setRawMessages((prev) => {
      for (let i = prev.length - 1; i >= 0; i--) {
        const m = prev[i];
        if (m.role === 'user' && m.content === content && m.timestamp === timestamp) {
          return [...prev.slice(0, i), ...prev.slice(i + 1)];
        }
      }
      return prev;
    });
  }, []);

  // Pending-input queue (spec 006 v3 §11c + §12 R2). Recall a queued message
  // (cross- OR same-session) into the composer for editing — shared by the
  // desktop ↑ accelerator and the mobile tap-to-edit affordance. This is
  // SERVER-AUTHORITATIVE: we await the cancel/dequeue and only load the text
  // when the backend confirms it removed a STILL-QUEUED entry (`removed===true`).
  // If `removed===false` the message already dispatched (its turn is running) —
  // we DO NOT load the text (that would double-send); we just drop the stale
  // overlay entry and the kept bubble becomes a normal sent message. If the
  // entry carries attachments we no-op entirely (recall can't restore chips;
  // §12 R4).
  const recallQueuedMessage = useCallback(
    async (nonce: string) => {
      const entry = pendingInputsRef.current.get(nonce);
      if (!entry) return;
      if (entry.hasAttachments) return;
      let res: { removed?: boolean } | undefined;
      // Surface a network failure to the caller (Finding 3): the tap path shows
      // pending.cancelError; the ↑ accelerator swallows it. Leaves the overlay
      // in place so the user can retry.
      res = await cancelPendingInput(projectId, entry.sessionId, nonce);
      if (res?.removed === true) {
        // Still queued — safe to pull it back into the composer for editing.
        setInputText(entry.rawText);
        removeOptimisticBubble(entry.content, entry.timestamp);
        localNoncesRef.current.delete(nonce);
        setPendingInputs((prev) => {
          if (!prev.has(nonce)) return prev;
          const next = new Map(prev);
          next.delete(nonce);
          return next;
        });
        textareaRef.current?.focus();
      } else if (res?.removed === false) {
        // Already dispatched/drained — DO NOT load the text (no double-send).
        // Drop the stale overlay; the kept bubble is now a normal sent message.
        setPendingInputs((prev) => {
          if (!prev.has(nonce)) return prev;
          const next = new Map(prev);
          next.delete(nonce);
          return next;
        });
      }
    },
    [projectId, cancelPendingInput, removeOptimisticBubble],
  );

  // Pending-input queue (spec 006 §3h). Reconcile the pending overlay from the
  // server (GET /pending) on session load + WS reconnect. Rebuilds the
  // project-scoped `pendingInputs` map from server truth (preserving an existing
  // entry's bubble identity so the kept optimistic bubble isn't duplicated), and
  // re-appends optimistic bubbles for the VIEWED session into `rawMessages` (the
  // transform-once memo + seed then render them — surviving the reseed). Fixes
  // switch-away-and-back stale/lost-affordance (F3).
  const reconcilePending = useCallback(() => {
    getPending(projectId)
      .then((res) => {
        if (!res) return;
        const list = res.pending ?? [];
        const holder = res.holder ?? '';
        const viewed = sessionIdRef.current;
        // Build the next map from server truth, preserving any existing entry's
        // content/timestamp (its bubble identity) via the committed ref.
        const rebuilt: Array<[string, PendingInputEntry]> = [];
        for (const p of list) {
          if (!p.nonce) continue;
          const existing = pendingInputsRef.current.get(p.nonce);
          if (existing) {
            rebuilt.push([p.nonce, existing]);
          } else {
            // Persist-at-dispatch means we recover only the wire `content`, not
            // the raw typed text. Detect an attachments-block prefix: if present
            // we can't reconstruct the chips, so flag hasAttachments and let
            // recall no-op (§12 R4); otherwise the content IS the raw text.
            const hasAttachments =
              parseAttachmentsBlock(p.content).attachments.length > 0;
            rebuilt.push([
              p.nonce,
              {
                kind: p.kind,
                sessionId: p.session_id,
                holder,
                content: p.content,
                rawText: hasAttachments ? '' : p.content,
                hasAttachments,
                timestamp: new Date().toISOString(),
              },
            ]);
          }
          // Keep the nonce alive so the eventual dispatch echo dedups against
          // the rebuilt bubble (don't clobber an existing received-timestamp).
          if (!localNoncesRef.current.has(p.nonce)) {
            localNoncesRef.current.set(p.nonce, 0);
          }
        }
        setPendingInputs(() => new Map(rebuilt));
        // Ensure a bubble exists in rawMessages for each viewed-session entry.
        if (viewed === undefined) return;
        setRawMessages((prev) => {
          let next = prev;
          for (const [, entry] of rebuilt) {
            if (entry.sessionId !== viewed) continue;
            const has = next.some(
              (m) => m.role === 'user' && m.content === entry.content,
            );
            if (!has) {
              next = [
                ...next,
                {
                  role: 'user',
                  content: entry.content,
                  source: 'user',
                  timestamp: entry.timestamp,
                },
              ];
            }
          }
          return next;
        });
      })
      .catch(() => {
        // best effort — leave the prior overlay in place
      });
  }, [projectId, getPending]);

  // Keep the holder fresh: on mount/project change, on every agentStatus
  // transition (a status change always implies the slot may have been
  // acquired/released), and whenever the viewed session changes (so the
  // viewingHolder comparison is correct for the newly-viewed session).
  useEffect(() => {
    fetchHolder();
  }, [fetchHolder, agentStatus, statusTick, sessionId]);

  // A session/project switch shows a different conversation — drop the
  // previous session's error notice (hydration re-fills it if it applies).
  useEffect(() => {
    setAgentError(null);
    setColdStartError(null);
  }, [projectId, sessionId]);

  // Per-session history load. Re-runs when the viewed sessionId changes so
  // switching sessions shows that session's own history (and its own loading
  // state). Skips while sessionId is undefined (active session not yet
  // resolved) — the empty state renders until one arrives.
  useEffect(() => {
    let cancelled = false;
    // Bug #48 (fix C): abort the in-flight history fetch when the user
    // switches away — the `cancelled` boolean only suppressed the setState,
    // leaving discarded requests executing against the backend.
    const controller = new AbortController();

    // Reset the "was running while viewed" latch on every session (re)load so a
    // stale latch from a previous session can't trigger a full-history
    // reconcile against the newly-viewed session on the next idle transition.
    wasRunningRef.current = false;

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
    const cacheKey = `${projectId}:${sessionId}`;

    async function loadData() {
      // Bug #48 (fix C): on a cache hit, paint the last-known transcript
      // immediately and revalidate in the background — no skeleton flash on
      // switch-back. On a miss, keep the original loading gate.
      //
      // Bug #58: two cases must NOT paint from the cache.
      //
      // 1. A turn is in flight. The cache is written only here, from a settled
      //    fetch — `refreshRawMessages` and the live WS appends never write
      //    back — so mid-turn the entry is stale by construction. Painting it
      //    fills `items` with the stale snapshot, which arms the seed effect's
      //    mid-turn skip above: the skip then refuses to let the revalidate
      //    replace `items` for the REST OF THE TURN, so every round persisted
      //    after the entry was written vanishes until the next running→idle
      //    refresh or a full reload (which clears this module-level cache —
      //    that is why refreshing "brings the messages back"). Show the
      //    skeleton and wait for the real fetch instead. Nothing the skip
      //    exists to protect is lost: a remount has already destroyed the live
      //    overlay, and the fresh fetch lands with `items` empty, so the skip
      //    fails its own non-empty gate and seeds correctly. The switch-back
      //    skeleton flash is an accepted cost, scoped strictly to turn-in-flight.
      // 2. The entry is empty. The backend legitimately returns [] for a
      //    freshly minted session id that has not materialized yet, so an entry
      //    cached at that moment is `messages: []`; repainting it hard-blanked
      //    a pane that has since filled up. Same site, same root — treat an
      //    empty entry as a miss and let the fetch decide.
      const cached = chatHistoryCache.get(cacheKey);
      // Committed ref, not effect-closure state: this effect is deliberately
      // NOT keyed on agentStatus (see the seed effect's dep-array note).
      const turnInFlight =
        agentStatusRef.current === 'running' || agentStatusRef.current === 'waiting';
      const paintCached = cached !== undefined && cached.messages.length > 0 && !turnInFlight;
      if (paintCached) {
        setRawMessages(cached.messages);
        setTotalMessages(cached.total);
        setLoadedOffset(cached.loadedOffset);
        setLoading(false);
      } else {
        setLoading(true);
      }
      // Clear prior session's live state so it never bleeds into the new one.
      setStream(null);
      setApprovals(new Map());
      try {
        // Refetch enough to cover any "Load earlier" pages the user had
        // already opened (same idiom as the catch-up refetch above).
        const targetLimit = cached
          ? Math.max(cached.loadedOffset, CHAT_PAGE_SIZE)
          : CHAT_PAGE_SIZE;
        const chatResult = await apiWithTotal<ChatMessageType[]>(
          `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=${targetLimit}${sessionParam}`,
          { signal: controller.signal },
        ).catch((err) => {
          // Abort = the user switched away; not a failure worth logging.
          if ((err as Error)?.name !== 'AbortError') {
            console.error('[ChatView] Failed to load chat history:', err);
          }
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
          setLoadedOffset(targetLimit);
          chatHistoryCache.set(cacheKey, {
            messages,
            total,
            loadedOffset: targetLimit,
          });
        } else if (!paintCached) {
          // Fetch failed with nothing painted — show the empty state. Keyed on
          // `paintCached`, not `cached`: when the cached entry was withheld
          // (mid-turn, or empty) there is no stale transcript on screen to
          // keep, so the pane must fall back to the empty state rather than
          // sit on a skeleton forever. On a real painted hit, keep the stale
          // transcript instead of blanking it.
          setRawMessages([]);
          setItems([]);
        }
        // Pending-input queue (spec 006 §3h): after the session's history is
        // loaded, rebuild the pending overlay + re-append the kept optimistic
        // bubble(s) for this session from GET /pending. Runs after the history
        // setState above so the bubble append lands on top (switch-away-and-back
        // restores the affordance; the queued message is NOT in JSONL yet).
        if (!cancelled) reconcilePending();
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadData();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [projectId, sessionId, project.workspace, reconcilePending]);

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
      // Bug #48 (fix C): remember the deeper pagination so a switch-back
      // revalidates the full loaded range, not just the first page.
      const cacheKey = `${projectId}:${sessionId}`;
      const entry = chatHistoryCache.get(cacheKey);
      if (entry) {
        chatHistoryCache.set(cacheKey, {
          ...entry,
          loadedOffset: entry.loadedOffset + CHAT_PAGE_SIZE,
        });
      }
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

  // Workbench (or other) one-shot prefill (§ initialDraft prop doc). Only
  // captured when sessionId is UNRESOLVED at mount — the scenario the
  // doorway actually produces (route.draft navigates without a sessionId;
  // ChatTab resolves one a tick later). Read once by the swap effect below,
  // the first time sessionId transitions away from undefined, so that
  // transition's normal "load this session's draft from the map" (empty, for
  // a session never visited) doesn't blank the seeded initial text out from
  // under it. If sessionId is already resolved at mount, the useState
  // initializer above already has it right and this stays unused.
  const pendingPrefillRef = useRef(sessionId === undefined ? initialDraft : undefined);

  const prevSessionForDraftRef = useRef<string | undefined>(sessionId);
  useEffect(() => {
    const prev = prevSessionForDraftRef.current;
    if (prev === sessionId) return;
    // Save the outgoing session's draft from the live mirror.
    if (prev !== undefined) {
      draftsRef.current.set(prev, inputTextRef.current);
    }
    // The pending-prefill transition (undefined → resolved, first time only):
    // the composer is ALREADY correct — either still showing the seeded
    // draft, or holding whatever the user typed into it in the meantime (the
    // resolution can take a while: session-list fetch, etc.) — so this must
    // NOT call setInputText at all here. Doing so with pendingPrefillRef's
    // (now possibly stale) captured string would silently clobber a user
    // edit made during that window — the bug this comment used to gloss
    // over. Just consume the ref and adopt the session id; leave inputText
    // untouched.
    if (sessionId !== undefined && pendingPrefillRef.current !== undefined) {
      pendingPrefillRef.current = undefined; // consumed — never re-applied
      prevSessionForDraftRef.current = sessionId;
      return;
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

  // Mount-only half of the prefill: adjusts height, focuses the textarea
  // with the cursor at the end (spec 2026-07-24 §5.3 — "cursor lands at the
  // end, textarea focused"), and tells the caller the draft was consumed so
  // it can clear its copy (route.draft) — otherwise the prefill would
  // reappear on a later remount (e.g. switching tabs away from Chat and
  // back). Runs exactly once regardless of whether sessionId is resolved yet
  // ([] deps — must not re-fire on prop churn).
  useEffect(() => {
    if (initialDraft === undefined) return;
    const ta = textareaRef.current;
    if (ta) {
      adjustTextareaHeight();
      // preventScroll: WKWebView scrolls focus() targets by their VISUAL
      // position (CLAUDE.md quirk) — invisible in Chrome/jsdom.
      ta.focus({ preventScroll: true });
      const end = ta.value.length;
      ta.setSelectionRange(end, end);
    }
    onDraftConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
        // Refresh the project's runtime fields so the
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
        title: t('chat.notify.newSession.title'),
        body: t('chat.notify.newSession.body'),
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
      setCancelTimeoutNotice(t('chat.cancelTimeout'));
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
      // Pending-input queue (spec 006 §3h): a relay tunnel drop / mobile
      // foreground may have missed pending_enqueued/dispatched/cancelled events.
      // Rebuild the overlay from GET /pending so waiting affordances survive.
      reconcilePending();
    }
  }, [connectionState, fetchPendingApproval, reconcilePending]);

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

      // A live delta for the VIEWED session is definitive proof it is the
      // session actively executing. Latch the "was running while viewed" flag
      // so the running->idle catch-up reconciles full persisted history — even
      // when the holder/viewing latch never fired (holder unresolved at the
      // running transition, the captured bug). Reset on session switch (load
      // effect) so this never reconciles a session navigated away from.
      wasRunningRef.current = true;

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

      const reasoning = e.reasoning_content ?? '';
      const hasVisibleText = !!e.text;

      // Reasoning-only phase: the delta carries reasoning with empty text. Keep
      // the thinking indicator alive and render the streaming reasoning into the
      // live capsule. Do NOT clear the spinner — that only happens once visible
      // answer text begins (below) or the turn completes (is_final, above).
      if (reasoning && !hasVisibleText) {
        setItems((prev) => appendLiveReasoning(prev, reasoning, new Date().toISOString(), e.source));
        scrollToBottom();
        return;
      }

      // If this delta also carried reasoning alongside visible text, capture it.
      if (reasoning) {
        setItems((prev) => appendLiveReasoning(prev, reasoning, new Date().toISOString(), e.source));
      }

      // Visible answer text has begun — transition out of the thinking phase.
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

      // The fanout tool call never renders as a capsule row — the
      // fanout.started card IS its representation (spec 009 §0.5), and the
      // persisted-history transform skips it the same way, so live and
      // reloaded views agree. Its ack (a tool_result event) is harmless:
      // markLatestLiveCallResultReceived no-ops when nothing is pending.
      if (e.tool_name === 'fanout') {
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
            // Localized when the daemon ships parsed arguments; the wire
            // description (backend English) is only the old-daemon fallback.
            target_description: describeLiveActivity(
              e.tool_name,
              e.arguments,
              project.workspace,
              (k, v) => translate(localeRef.current, k, v),
              e.description,
            ),
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

      // Fanout workers are ephemeral task executors, not persistent
      // collaborators (spec 009 §0.5-8) — their live turn output must not
      // leak into the main chat as bubbles. The progress card (fanout tool
      // call render) and the join summary are the only sanctioned surfaces;
      // mirrors SubAgentStatusBar.tsx's own `worker:` filter.
      if (isWorkerHandle(e.source)) return;

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

    // Spec 074 hidden-response fix: the bubble appended above lives ONLY in
    // `items` — never mirrored into rawMessages — and a pinned dispatch keeps
    // the management loop idle/holderless, so the Bug29-D2 reseed gate does
    // not protect it. The next send's optimistic rawMessages push then
    // reseeds `items` wholesale and the previous response vanishes. A
    // sub-agent terminal event for the viewed session refreshes the canonical
    // history instead: by broadcast time the backend has persisted both the
    // session terminal row and the transcript turn boundary (no awaits in
    // between on the daemon's loop), so the refetched /chat page carries the
    // server-synthesized response bubble and the reseed becomes harmless.
    function handleSubAgentTerminal(event: WebSocketEvent) {
      const e = event as SubAgentLifecycleEvent;
      if (e.project_id !== projectId) return;
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;
      refreshRawMessages();
    }

    function handleUserMessage(event: WebSocketEvent) {
      const e = event as UserMessageEvent;
      if (e.project_id !== projectId) return;

      // Evict nonces older than 30s on each incoming message. EXEMPT pending
      // nonces (spec 006 §3h F4): a queued message may wait far longer than 30s,
      // and the nonce must survive so the dispatch-time persisted echo dedups
      // against the kept optimistic bubble (no duplicate on long waits).
      const now = Date.now();
      for (const [n, ts] of localNoncesRef.current) {
        if (pendingInputsRef.current.has(n)) continue;
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

    // Pending-input queue (spec 006 §3h). A message was accepted + queued behind
    // the slot holder. Dedup by nonce (mirroring handleUserMessage): the origin
    // tab already rendered the optimistic bubble + map entry in handleSend, so it
    // ignores its own echo; OTHER tabs render the optimistic bubble + add the
    // map entry. The bubble renders only for the viewed session, but the map
    // entry is project-scoped (the affordance gates on the viewed session).
    function handlePendingEnqueued(event: WebSocketEvent) {
      const e = event as PendingEnqueuedEvent;
      if (e.project_id !== projectId) return;
      if (!e.nonce) return;
      // Our own send (origin) or a relay retry already handled — ignore.
      if (localNoncesRef.current.has(e.nonce)) return;
      // Register so the eventual dispatch echo dedups against the bubble we
      // render here (and exempt from eviction while it sits in pendingInputs).
      localNoncesRef.current.set(e.nonce, 0);
      const ts = new Date().toISOString();
      // pending_enqueued is always cross-session (it carries a holder). We only
      // have the wire `content` from another tab — detect an attachments-block
      // prefix to gate recall (§12 R4); otherwise content IS the raw text.
      const hasAttachments =
        parseAttachmentsBlock(e.content).attachments.length > 0;
      setPendingInputs((prev) => {
        const next = new Map(prev);
        next.set(e.nonce, {
          kind: 'cross',
          sessionId: e.session_id,
          holder: e.holder,
          content: e.content,
          rawText: hasAttachments ? '' : e.content,
          hasAttachments,
          timestamp: ts,
        });
        return next;
      });
      // Optimistic bubble only for the viewed session — pushed to BOTH items
      // and rawMessages so the transform-once reseed reproduces it.
      if (e.session_id !== sessionIdRef.current) return;
      setItems((prev) => {
        const afterCapsule = finalizeLiveCapsule(prev, 'completed');
        return [
          ...afterCapsule,
          { type: 'user_message', content: e.content, timestamp: ts },
        ];
      });
      setRawMessages((prev) => [
        ...prev,
        { role: 'user', content: e.content, source: 'user', timestamp: ts },
      ]);
      scrollToBottom();
    }

    // Pending-input queue (spec 006 §3h). The queued message started dispatching
    // — clear ITS waiting affordance by removing the nonce from pendingInputs.
    // This is the SOLE clear trigger (agent.status carries no session_id). The
    // kept optimistic bubble stays; the real turn streams via the per-session
    // path and the persisted echo dedups against the bubble by nonce.
    function handlePendingDispatched(event: WebSocketEvent) {
      const e = event as PendingDispatchedEvent;
      if (e.project_id !== projectId) return;
      if (!e.nonce) return;
      const nonce = e.nonce;
      setPendingInputs((prev) => {
        if (!prev.has(nonce)) return prev;
        const next = new Map(prev);
        next.delete(nonce);
        return next;
      });
    }

    // Pending-input queue (spec 006 §3h). The queued message was cancelled
    // ("Stop waiting" / orphan cleanup). Remove the optimistic bubble + map
    // entry, and drop the nonce from the dedup map.
    function handlePendingCancelled(event: WebSocketEvent) {
      const e = event as PendingCancelledEvent;
      if (e.project_id !== projectId) return;
      if (!e.nonce) return;
      const nonce = e.nonce;
      const entry = pendingInputsRef.current.get(nonce);
      localNoncesRef.current.delete(nonce);
      if (entry) removeOptimisticBubble(entry.content, entry.timestamp);
      setPendingInputs((prev) => {
        if (!prev.has(nonce)) return prev;
        const next = new Map(prev);
        next.delete(nonce);
        return next;
      });
    }

    // Spec 009 §0.5: fanout.started arrives with the real, backend-assigned
    // tasks (handle + label) — append the card directly, no need to wait for
    // the tool-result ack text the way the persisted-history transform does.
    function handleFanoutStarted(event: WebSocketEvent) {
      const e = event as FanoutStartedEvent;
      if (e.project_id !== projectId) return;
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      setItems((prev) => {
        // Duplicate guard (mirrors the approval_card dedup precedent below):
        // a relay retry or a reconnect replaying the same event must not
        // append a second card for the same fanout_id.
        if (prev.some((it) => it.type === 'fanout_card' && it.fanout_id === e.fanout_id)) {
          return prev;
        }
        const afterCapsule = finalizeLiveCapsule(prev, 'completed');
        return [
          ...afterCapsule,
          {
            type: 'fanout_card' as const,
            fanout_id: e.fanout_id,
            tasks: e.tasks,
            timestamp: new Date().toISOString(),
          },
        ];
      });
      scrollToBottom();
    }

    function handleFanoutTaskUpdate(event: WebSocketEvent) {
      const e = event as FanoutTaskUpdateEvent;
      if (e.project_id !== projectId) return;
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      // Round 2 (issue 1): the backend stamps completed_at_ms on terminal
      // transitions; older daemons that predate the field send none, so a
      // terminal status with no timestamp still needs to freeze — arrival
      // time is the best available stand-in for "whenever it actually
      // finished" (unknowable precisely, but far closer than never-freezing).
      const completedAtMs = e.completed_at_ms ?? (isTerminal(e.status) ? Date.now() : undefined);
      setFanoutStatuses((prev) => {
        const next = new Map(prev);
        next.set(e.fanout_id, {
          ...(next.get(e.fanout_id) ?? {}),
          [e.handle]: { status: e.status, completedAtMs },
        });
        return next;
      });
    }

    function handleFanoutCompleted(event: WebSocketEvent) {
      const e = event as FanoutCompletedEvent;
      if (e.project_id !== projectId) return;
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      setFanoutCompletedAt((prev) => {
        if (prev.has(e.fanout_id)) return prev;
        const next = new Map(prev);
        next.set(e.fanout_id, Date.now());
        return next;
      });

      // Round 2: fanout.completed now carries each task's terminal snapshot
      // too (previously it carried none) — merge it into the same overlay
      // handleFanoutTaskUpdate feeds, in case any task's own task_update was
      // missed (e.g. a client that (re)connected mid-batch).
      if (e.tasks && e.tasks.length > 0) {
        setFanoutStatuses((prev) => {
          const next = new Map(prev);
          const merged = { ...(next.get(e.fanout_id) ?? {}) };
          for (const task of e.tasks) {
            const completedAtMs = task.completed_at_ms ?? (isTerminal(task.status) ? Date.now() : undefined);
            merged[task.handle] = { status: task.status, completedAtMs };
          }
          next.set(e.fanout_id, merged);
          return next;
        });
      }
    }

    on('chat.stream_delta', handleStreamDelta);
    on('agent.activity', handleActivity);
    on('approval.request', handleApprovalRequest);
    on('approval.resolved', handleApprovalResolved);
    on('chat.sub_agent_message', handleSubAgentMessage);
    // Terminal family only — 'started' persists nothing a reseed could hide.
    on('sub_agent.completed', handleSubAgentTerminal);
    on('sub_agent.error', handleSubAgentTerminal);
    on('sub_agent.failed', handleSubAgentTerminal);
    on('sub_agent.stopped', handleSubAgentTerminal);
    on('sub_agent.turn_interrupted', handleSubAgentTerminal);
    on('chat.user_message', handleUserMessage);
    on('agent.notify', handleAgentNotify);
    on('state_refresh.lifecycle', handleStateRefresh);
    on('workspace_claudemd_warning', handleClaudemdWarning);
    on('chat.pending_enqueued', handlePendingEnqueued);
    on('chat.pending_dispatched', handlePendingDispatched);
    on('chat.pending_cancelled', handlePendingCancelled);
    on('fanout.started', handleFanoutStarted);
    on('fanout.task_update', handleFanoutTaskUpdate);
    on('fanout.completed', handleFanoutCompleted);

    return () => {
      off('chat.stream_delta', handleStreamDelta);
      off('agent.activity', handleActivity);
      off('approval.request', handleApprovalRequest);
      off('approval.resolved', handleApprovalResolved);
      off('chat.sub_agent_message', handleSubAgentMessage);
      off('sub_agent.completed', handleSubAgentTerminal);
      off('sub_agent.error', handleSubAgentTerminal);
      off('sub_agent.failed', handleSubAgentTerminal);
      off('sub_agent.stopped', handleSubAgentTerminal);
      off('sub_agent.turn_interrupted', handleSubAgentTerminal);
      off('chat.user_message', handleUserMessage);
      off('agent.notify', handleAgentNotify);
      off('state_refresh.lifecycle', handleStateRefresh);
      off('workspace_claudemd_warning', handleClaudemdWarning);
      off('chat.pending_enqueued', handlePendingEnqueued);
      off('chat.pending_dispatched', handlePendingDispatched);
      off('chat.pending_cancelled', handlePendingCancelled);
      off('fanout.started', handleFanoutStarted);
      off('fanout.task_update', handleFanoutTaskUpdate);
      off('fanout.completed', handleFanoutCompleted);
    };
  }, [projectId, project.name, on, off, scrollToBottom, removeOptimisticBubble, refreshRawMessages]);

  // Credential-error surfacing: agent.status error events for the viewed
  // session feed the AgentErrorNotice. Events without a meaningful session
  // match too (trigger failures broadcast project-level). Any *active* status
  // clears the notice (a new run started); idle does NOT (error → idle
  // transitions must not wipe the message). Separate effect from the main
  // WS block because the filter depends on sessionId.
  useEffect(() => {
    function handleAgentStatusError(event: WebSocketEvent) {
      const e = event as AgentStatusEvent & {
        session_id?: string;
        error_code?: string;
      };
      if (e.project_id !== projectId) return;
      if (e.session_id && sessionId !== undefined && e.session_id !== sessionId) {
        return;
      }
      if (e.status === 'error') {
        setAgentError({ code: e.error_code, message: e.reason });
      } else if (e.status !== 'idle') {
        setAgentError(null);
      }
    }
    on('agent.status', handleAgentStatusError);
    return () => off('agent.status', handleAgentStatusError);
  }, [projectId, sessionId, on, off]);

  // Fingerprint for the last item. DisplayItem has no stable id; use a
  // composite of type + a discriminating field per variant.
  function lastItemKey(item: DisplayItem | undefined): string | null {
    if (!item) return null;
    if (item.type === 'approval_card') return `approval_card:${item.tool_call_id}`;
    if (item.type === 'session_separator') return `session_separator:${item.timestamp}`;
    // Marker only — no content field, so it must be keyed before the content
    // fallthrough at the end (which would crash on undefined.slice).
    if (item.type === 'compaction_marker') return `compaction_marker:${item.timestamp}`;
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
    if (item.type === 'budget_event') {
      // No content field — key off timestamp + action BEFORE the content
      // fallthrough below (which would crash on undefined.slice).
      return `budget_event:${item.timestamp}:${item.action}`;
    }
    if (item.type === 'fanout_card') {
      // Identity-only, like approval_card — live status ticks flow through
      // the separate fanoutStatuses overlay, not the item itself.
      return `fanout_card:${item.fanout_id}`;
    }
    if (item.type === 'fanout_summary') {
      return `fanout_summary:${item.fanout_id}`;
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
          title: t('chat.notify.noSession.title'),
          body: t('chat.notify.noSession.body'),
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
    setDragCounter(0);
    dropAttachments(e);
  }

  const hasText = inputText.trim().length > 0;
  // Send is enabled when there is text OR a done chip, AND no chip is uploading.
  // A staged annotation is sendable on its own: the quote block + the note
  // carry the whole question ("this one, not the ad") without typed text.
  const canSend =
    !anyUploading &&
    !allError &&
    (hasText || anyDone || annotations.length > 0) &&
    !(attachments.length > 0 && !hasText && !anyDone);
  const disabledReason = anyUploading
    ? t('chat.disabled.waitingUploads')
    : allError && !hasText
      ? t('chat.disabled.removeFailed')
      : '';

  /**
   * Spec 078 §5.4 step 2 — draw each box onto a copy of its source image and
   * upload the result through the composer's own upload path, so the marked-up
   * PNG arrives in the `<attached_files>` prefix and a vision model sees it.
   *
   * Failure is not fatal: the coordinates and the note are already in the
   * quotes block, which is what a text-only model gets anyway. The caller
   * surfaces `failed` through the existing inject-error banner and sends.
   */
  async function uploadAnnotationImages(anns: Annotation[]): Promise<{
    attachments: { path: string; mime: string; size: number }[];
    failed: boolean;
  }> {
    const out: { path: string; mime: string; size: number }[] = [];
    let failed = false;
    const baseUrl = isRelayMode ? window.location.origin : BASE_URL;
    for (const a of anns) {
      if (a.kind !== 'browser' && a.kind !== 'image') continue;
      if (!a.imageDataUrl) continue;
      try {
        const blob = await renderAnnotatedPng(a.imageDataUrl, [{ n: a.n, box: a.box }]);
        const file = new File([blob], annotationFilename(a.n), { type: 'image/png' });
        const { path, size } = await uploadFile({ projectId, file, baseUrl, isRelayMode });
        out.push({ path, mime: 'image/png', size });
      } catch {
        failed = true;
      }
    }
    return { attachments: out, failed };
  }

  async function handleSend() {
    const text = inputText.trim();
    const doneAttachments = attachments.filter((a) => a.status === 'done');
    const pendingAnnotations = annotations;
    if (!text && doneAttachments.length === 0 && pendingAnnotations.length === 0) return;
    if (anyUploading) return;

    // Slash command: /new
    if (text === '/new') {
      executeNewSession();
      return;
    }

    // Spec 074 target precedence: a leading @mention wins for this one
    // message; otherwise the sticky "Talking to" pin applies; otherwise the
    // management agent. `@orbital` is the reserved one-message manager aside
    // — it routes down the management branch WITHOUT unpinning.
    const resolved = resolveSendTarget(text, pinnedTarget);
    const target = resolved.target;
    let content = resolved.content;

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

    // Spec 078 §5.4 — the annotated PNGs ride the attachments list, the quotes
    // block is appended to the text. Both must be settled BEFORE the optimistic
    // bubble is painted, so its wire content matches the WS echo exactly.
    let annotationUploadFailed = false;
    if (pendingAnnotations.length > 0) {
      const uploaded = await uploadAnnotationImages(pendingAnnotations);
      attachmentsPayload.push(...uploaded.attachments);
      annotationUploadFailed = uploaded.failed;
      const quotes = formatQuotes(pendingAnnotations);
      content = content ? `${content}\n\n${quotes}` : quotes;
    }

    // Build the same prefix the backend will emit, so the optimistic local
    // user_message has the identical wire content as the WS echo will.
    const wireContent = buildAttachmentsBlock(attachmentsPayload) + content;

    setInputText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    clearAttachments();
    if (pendingAnnotations.length > 0) {
      clearAnnotations();
      setAnnotationsOpen(false);
    }

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
    // The user's own send always re-arms auto-follow and jumps to the bottom,
    // even if they had scrolled up to read history before sending.
    scrollToBottom({ force: true });

    if (target) setSubAgentLoading(target);
    // An annotated PNG that failed to render/upload still gets sent — the
    // coordinates + note in the quotes block carry the meaning — but the user
    // is told the image did not make it.
    setInjectError(annotationUploadFailed ? t('chat.uploadError') : null);
    try {
      // Serialize behind a just-fired pin PATCH: it holds the session file's
      // write lock on the backend, and an inject racing that burst gets
      // rejected with the message dropped. Waiting out the PATCH (success or
      // failure alike) removes the race at its source.
      if (pinPatchInFlightRef.current) await pinPatchInFlightRef.current;
      const result = await injectMessage(
        projectId,
        content,
        target,
        nonce,
        attachmentsPayload.length > 0 ? attachmentsPayload : undefined,
        sessionId,
        resolved.pinned,
      );
      // A pinned send materialized the session on the backend — if the pin
      // PATCH had failed earlier (brand-new session, 404 before the first
      // message), persist it now so it survives reloads. `pinAgent`'s
      // optimistic update keeps `backendPin` current on success, so this
      // fires only while the stored value is actually behind.
      if (
        resolved.pinned &&
        sessionId !== undefined &&
        target !== undefined &&
        backendPin !== target
      ) {
        persistPin(sessionId, target);
      }
      // Pending-input queue (spec 006 §3h): the happy-path 202. The backend
      // ACCEPTED + queued this message behind the slot holder; it auto-dispatches
      // when the slot frees. KEEP the optimistic bubble (unlike slot_held below,
      // which removes it) and add a nonce→entry to `pendingInputs` so the bubble
      // shows a "Waiting for {holder}…" affordance. The kept bubble's nonce is
      // exempt from the 30s eviction so the dispatch echo dedups against it.
      if (
        result &&
        typeof result === 'object' &&
        result.status === 'queued_pending_slot' &&
        result.holding_session_id
      ) {
        const holder = result.holding_session_id;
        const queuedSession = result.queued_session_id ?? sessionId;
        if (queuedSession !== undefined) {
          setPendingInputs((prev) => {
            const next = new Map(prev);
            next.set(nonce, {
              kind: 'cross',
              sessionId: queuedSession,
              holder,
              content: wireContent,
              rawText: text,
              hasAttachments: attachmentsPayload.length > 0,
              timestamp: optimisticTimestamp,
            });
            return next;
          });
        }
        scrollToBottom();
        return;
      }
      // Pending-input queue (spec 006 v3 §11d.4): the same-session 200. The
      // backend ACCEPTED + queued this message behind the VIEWED session's own
      // in-flight turn (`session._queue`); it drains automatically when that
      // turn ends. KEEP the optimistic bubble and add a same-session
      // `pendingInputs` entry so the bubble shows the "Waiting for the current
      // response to finish." line and is ↑/tap-recallable. No holder, no
      // Run-now (there is no other session to cancel).
      if (
        result &&
        typeof result === 'object' &&
        result.status === 'queued_same_session' &&
        sessionId !== undefined
      ) {
        setPendingInputs((prev) => {
          const next = new Map(prev);
          next.set(nonce, {
            kind: 'same',
            sessionId,
            holder: '',
            content: wireContent,
            rawText: text,
            hasAttachments: attachmentsPayload.length > 0,
            timestamp: optimisticTimestamp,
          });
          return next;
        });
        scrollToBottom();
        return;
      }
      // Track J Phase 1 (now the ENQUEUE-FAILURE fallback only): backend
      // returned 202 with `slot_held` because enqueue itself raised. Strip the
      // optimistic user_message we just appended (it didn't land), restore the
      // typed text to the composer, and surface the intermediate notice with
      // [Wait] / [Cancel-and-send] affordances.
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
    } catch (err) {
      // A classified provider/credential 400 (auto-start could not build the
      // LLM provider) gets the actionable AgentErrorNotice; anything else
      // keeps the generic inline error line.
      const info = parseProviderError(err);
      if (info) {
        setAgentError(info);
      } else {
        setInjectError(t('chat.injectError'));
      }
    } finally {
      setSubAgentLoading(null);
    }

    scrollToBottom();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // While an IME (e.g. Pinyin) is composing, Enter/Arrow/Tab belong to the
    // candidate window — don't submit or hijack them for our dropdowns.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
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
    // Pending-input queue (spec 006 v3 §11c + §12). ↑ in an EMPTY composer (no
    // command/mention dropdown open — those guards above already handled ↑ and
    // returned) recalls the NEWEST queued message for the VIEWED session into
    // the composer and dequeues it. No cycling, no input history (§12 R5). When
    // the composer is non-empty, or no queued message exists for this session,
    // do NOT preventDefault so ↑ still moves the caret normally.
    if (e.key === 'ArrowUp' && inputText.trim() === '' && sessionId !== undefined) {
      let newestNonce: string | null = null;
      let newestTs = '';
      for (const [nonce, entry] of pendingInputsRef.current) {
        if (entry.sessionId !== sessionId) continue;
        if (newestNonce === null || entry.timestamp > newestTs) {
          newestNonce = nonce;
          newestTs = entry.timestamp;
        }
      }
      if (newestNonce !== null) {
        const entry = pendingInputsRef.current.get(newestNonce);
        // Finding 4: a newest entry with attachments can't be recalled (chips
        // can't be restored), so recall would no-op — let ↑ move the caret
        // instead of swallowing the key.
        if (entry && !entry.hasAttachments) {
          e.preventDefault();
          // ↑ accelerator: swallow recall errors (the tap affordance surfaces
          // them via pending.cancelError). Finding 3.
          recallQueuedMessage(newestNonce).catch(() => {});
          return;
        }
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
          <span className="text-lg font-medium">{t('chat.dropFiles')}</span>
        </div>
      )}
      {claudemdWarning && (
        <ClaudemdWarningBanner
          projectId={projectId}
          warning={claudemdWarning}
          onDismiss={() => setClaudemdWarning(null)}
        />
      )}
      {/* Piece 3 Part D: honest sub-agent status badge + user stop control */}
      <SubAgentStatusBar projectId={projectId} sessionId={sessionId} />
      {drillIn ? (
        <SubAgentDrillIn
          projectId={projectId}
          sessionId={sessionId}
          handle={drillIn.handle}
          displayName={drillIn.label}
          onBack={() => setDrillIn(null)}
        />
      ) : (
      <>
      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto px-7 py-5 max-md:pb-20 flex flex-col gap-4">
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
            {loadingMore ? t('chat.loadingMore') : t('chat.loadEarlier', { n: totalMessages - loadedOffset })}
          </button>
        )}

        {!loading && items.length === 0 && !stream && (
          project?.is_empty_workspace === false &&
          sessionId === undefined &&
          !coldStartDismissed ? (
            <div className="mt-8">
              <ColdStartCard
                folderName={(project.workspace || '').replace(/\\/g, '/').split('/').filter(Boolean).pop() || t('coldStart.folderFallback')}
                busy={coldStartBusy}
                error={coldStartError}
                onScan={async () => {
                  setColdStartBusy(true);
                  setColdStartError(null);
                  try {
                    // Navigation to the new session is WS-driven, like /new.
                    await coldStartScan(project.project_id);
                  } catch (err) {
                    console.error('[ChatView] cold-start scan failed:', err);
                    // Classified 400 (e.g. missing_api_key) → actionable
                    // message on the card; anything else → generic copy.
                    // Never fail silently (the original bug).
                    const info = parseProviderError(err);
                    setColdStartError(t(providerErrorKey(info?.code)));
                    setColdStartBusy(false);
                  }
                }}
                onSkip={() => setColdStartDismissed(true)}
              />
            </div>
          ) : (
            <div className="text-secondary text-sm text-center mt-12">
              {t('chat.empty')}
            </div>
          )
        )}

        {!loading &&
          items.map((item, index) => {
            const historical = 'isHistorical' in item && item.isHistorical;
            let rendered: React.ReactNode = null;

            if (item.type === 'user_message') {
              rendered = <ChatMessage key={`msg-${index}`} message={item} workspace={project.workspace} onOpenPath={onOpenPath} />;
            } else if (item.type === 'agent_message' || item.type === 'sub_agent_message') {
              rendered = <ChatMessage key={`msg-${index}`} message={item} agentName={project.agent_name} workspace={project.workspace} onOpenPath={onOpenPath} />;
            } else if (item.type === 'sub_agent_activity') {
              // FE-A2: compact one-line marker for [Sub-agent] lifecycle.
              // 'error' (backlog #23 D2 — the "stopped with error:" marker)
              // mirrors 'failed''s styling/severity treatment below; it is
              // a distinct terminal outcome (a crash mid-turn vs. a clean
              // failure report), so it gets its own copy key.
              //
              // backlog #27: this was a ternary chain ending in an unguarded
              // `/* failed */` fallback, so the four action values added for
              // #27 would each have rendered as "failed" — the timeline
              // telling the user their own stop request was a failure, with
              // no type error to catch it. It is now an exhaustive switch:
              // the next action added to the union fails `tsc` at the
              // `never` assignment instead of shipping a plausible lie.
              const label = ((): string => {
                // Switch on a captured copy, not `item.action` directly:
                // narrowing the discriminant would narrow `item` itself to
                // `never` in the default branch and make the exhaustiveness
                // proof below unwritable. Every field read here lives on the
                // one sub_agent_activity shape, so nothing needs `item`
                // narrowed per action anyway.
                const action = item.action;
                switch (action) {
                  case 'started':
                    return t('chat.subActivity.started');
                  case 'sent':
                    return t('chat.subActivity.sent', { preview: item.preview ?? '' });
                  case 'completed':
                    return item.summary
                      ? t('chat.subActivity.completed', { summary: item.summary })
                      : t('chat.subActivity.completed', { summary: '' }).replace(/[:：]\s*$/, '');
                  case 'error':
                    return item.error
                      ? t('chat.subActivity.error', { error: item.error })
                      : t('chat.subActivity.error', { error: '' }).replace(/[:：]\s*$/, '');
                  case 'failed':
                    return item.error
                      ? t('chat.subActivity.failed', { error: item.error })
                      : t('chat.subActivity.failed', { error: '' }).replace(/[:：]\s*$/, '');
                  case 'stopped':
                    // The stop itself is unremarkable — the user asked for
                    // it. The background work it destroyed is the news.
                    return item.count
                      ? t('chat.subActivity.stoppedWithBackground', {
                          n: item.count,
                          commands: item.detail ?? '',
                        })
                      : t('chat.subActivity.stopped');
                  case 'interrupted':
                    return t('chat.subActivity.interrupted');
                  case 'background_lost':
                    return t('chat.subActivity.backgroundLost', {
                      n: item.count ?? 0,
                      commands: item.detail ?? '',
                    });
                  case 'interaction_required':
                    return item.prompt
                      ? t('chat.subActivity.interactionRequired', { prompt: item.prompt })
                      : t('chat.subActivity.interactionRequired', { prompt: '' }).replace(/[:：]\s*$/, '');
                  case 'queue_dropped':
                    // backlog #35a: this is what the exhaustive switch is
                    // for. Under the old ternary chain a new action fell
                    // through to the `failed` branch, which is exactly the
                    // lie #26 shipped by borrowing on_failed's shape.
                    return item.reason
                      ? t('chat.subActivity.queueDropped', { reason: item.reason })
                      : t('chat.subActivity.queueDropped', { reason: '' }).replace(/[:：]\s*$/, '');
                  default: {
                    // Unreachable while the switch stays exhaustive (this
                    // assignment is the compile-time proof). It still runs
                    // if a session written by a NEWER build is replayed
                    // here, so it stays deliberately generic — an honest
                    // "don't know" beats guessing a severity.
                    const unhandled: never = action;
                    return t('chat.subActivity.unknown', { action: String(unhandled) });
                  }
                }
              })();
              // Severity, not decoration. Red: something the user was
              // waiting on will never arrive. Amber: work was destroyed or
              // is blocked on them. Accent: needs their answer. Neutral: the
              // user's own deliberate action — a plain stop is not a fault,
              // but one that killed live background work is amber, because
              // that part they did not ask for.
              //
              // 'queue_dropped' is amber, not red (backlog #35a): a message
              // the user sent was thrown away, which they need to know, but
              // the most common cause is their own stop and nothing
              // malfunctioned.
              const tone =
                item.action === 'failed' || item.action === 'error'
                  ? 'text-error/80'
                  : item.action === 'background_lost' ||
                      item.action === 'interrupted' ||
                      item.action === 'queue_dropped' ||
                      (item.action === 'stopped' && !!item.count)
                    ? 'text-warning'
                    : item.action === 'interaction_required'
                      ? 'text-accent'
                      : 'text-secondary';
              rendered = (
                <div
                  key={`sa-${index}`}
                  data-testid="sub-agent-activity"
                  data-action={item.action}
                  className={`flex items-baseline gap-2 px-2 py-1 text-[11px] font-mono ${tone}`}
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
                    {t('chat.previousSession')}
                  </span>
                  <div className="flex-1 border-t border-border" />
                </div>
              );
            } else if (item.type === 'compaction_marker') {
              // The agent ran out of room and summarized everything above
              // this line. Same divider shape as the session boundary — both
              // are "the conversation above here is not what it looks like" —
              // but deliberately a different item type, since a compaction
              // does not start a new session and must not grey out history.
              // No action here: the summarization has already happened, so
              // the new-session offer lives in ContextStrip, BEFORE the fact.
              return (
                <div
                  key={`compact-${index}`}
                  data-testid="compaction-marker"
                  className="flex items-center gap-3 px-2 opacity-50"
                >
                  <div className="flex-1 border-t border-border" />
                  <span className="text-xs text-secondary whitespace-nowrap">
                    {t('chat.compacted')}
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
              const summary = capsuleSummaryText({ ...item, status: derivedStatus }, t);
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
                    className={`flex items-center gap-2 w-full text-left font-mono text-[11px] text-secondary ${isLocked ? 'cursor-default' : 'cursor-pointer hover:text-primary'}`}
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
            } else if (item.type === 'budget_event') {
              // Budget-trip timeline row (P3-G). Codes/numbers from the payload
              // only — localized client-side (en + zh) via the shared pure
              // composer in budget/timelineText.ts (also unit-tested there).
              const text = budgetTimelineText(item, locale, t);
              rendered = (
                <div
                  key={`budget-${index}-${item.timestamp}`}
                  data-testid="budget-timeline-row"
                  className="rounded-lg border border-error/30 bg-error/5 px-4 py-2.5"
                >
                  <p className="text-xs font-medium text-error">{text}</p>
                </div>
              );
            } else if (item.type === 'fanout_card') {
              // Merge the persisted-history backfill (item.statuses /
              // item.completedAtMs — set by chatTransform when a join-summary
              // system message was found later in history) with the live WS
              // overlay, which wins per-key when present. Without this merge
              // a card rebuilt from a COMPLETED fanout's history would show
              // every row defaulting to "running" forever (no live events
              // ever arrive for a session that's already done).
              //
              // Round 2 (issue 1): item.statuses is still plain
              // Record<handle, FanoutTaskStatus> (chatTransform's
              // parseFanoutSummary carries no per-task times — the join
              // summary only records the batch's terminal moment), so the
              // historical side of the merge maps each into the rich
              // {status, completedAtMs} shape using the batch-level
              // item.completedAtMs (every historical row freezes together —
              // a summary backfill has no finer-grained timing to offer).
              const liveStatuses = fanoutStatuses.get(item.fanout_id);
              const historicalStatuses: Record<string, FanoutTaskState> = {};
              for (const [handle, status] of Object.entries(item.statuses ?? {})) {
                historicalStatuses[handle] = { status, completedAtMs: item.completedAtMs };
              }
              const mergedStatuses = { ...historicalStatuses, ...(liveStatuses ?? {}) };
              const mergedCompletedAtMs =
                fanoutCompletedAt.get(item.fanout_id) ?? item.completedAtMs ?? null;
              rendered = (
                <FanoutCard
                  key={`fanout-${item.fanout_id}`}
                  fanoutId={item.fanout_id}
                  tasks={item.tasks}
                  statuses={mergedStatuses}
                  startedAtMs={tsToMsSafe(item.timestamp)}
                  completedAtMs={mergedCompletedAtMs}
                  onSelectTask={(handle, label) => setDrillIn({ handle, label })}
                />
              );
            } else if (item.type === 'fanout_summary') {
              // Join-summary system message (spec 009 §0.5 item 4): a plain
              // system row, no dedicated component — v1 just shows the
              // persisted text verbatim (mirrors sub_agent_activity's inline
              // rendering, which also has no separate component file).
              rendered = (
                <div
                  key={`fanout-summary-${item.fanout_id}-${index}`}
                  data-testid="fanout-summary"
                  className="px-2 py-1 text-[11px] font-mono text-secondary whitespace-pre-wrap"
                >
                  {item.content}
                </div>
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
            <span>{t('chat.thinking')}</span>
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

      {/* backlog #44: jump-to-latest pill. Shown ONLY when the user has
          scrolled up AND new content has since arrived. Arrow-only, no count —
          clicking it jumps to the bottom and resumes auto-follow. */}
      {showJumpButton && (
        <button
          type="button"
          onClick={() => scrollToBottom({ force: true })}
          aria-label={t('chat.jumpToLatest')}
          data-testid="jump-to-latest"
          className="absolute left-1/2 bottom-24 -translate-x-1/2 z-30 flex items-center justify-center w-9 h-9 rounded-full bg-accent text-white shadow-lg hover:opacity-90 transition-opacity"
        >
          <ArrowDown size={18} />
        </button>
      )}

      {agentError && (
        <div className="shrink-0 px-4">
          <AgentErrorNotice
            code={agentError.code}
            message={agentError.message}
            onOpenSettings={() =>
              window.dispatchEvent(new CustomEvent('open-global-settings'))
            }
            onDismiss={() => setAgentError(null)}
          />
        </div>
      )}

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

      {/* Pending-input queue (spec 006 v3 §11b/§11c/§12): a muted waiting line
          under each queued message bubble for the VIEWED session. Cross-session
          entries also get a single [Run now] (cancel the holder ONLY — the
          queued message self-dispatches, no re-inject). The line is tappable to
          recall/edit (mobile equivalent of ↑; §12 R1) — no [Stop waiting]. */}
      {Array.from(pendingInputs.entries())
        .filter(([, entry]) => sessionId !== undefined && entry.sessionId === sessionId)
        .map(([nonce, entry]) => (
          <div key={`pending-${nonce}`} className="shrink-0 px-4">
            <PendingInputNotice
              kind={entry.kind}
              canEdit={!entry.hasAttachments}
              onRunNow={
                entry.kind === 'cross'
                  ? async () => {
                      // Run now = cancel the HOLDER only (spec §3h F5). NO
                      // re-inject: the queued message dispatches itself when the
                      // slot frees.
                      await cancelMessage(projectId, entry.holder);
                    }
                  : undefined
              }
              onEdit={() => recallQueuedMessage(nonce)}
            />
          </div>
        ))}

      {cancelTimeoutNotice && (
        <div className="shrink-0 px-4 py-1">
          <p className="text-xs text-secondary" role="status">{cancelTimeoutNotice}</p>
        </div>
      )}

      <div className="shrink-0 px-4 pb-4 pt-2 max-md:fixed max-md:bottom-0 max-md:left-0 max-md:right-0 max-md:bg-card max-md:z-[60] max-md:pb-[env(safe-area-inset-bottom,12px)]">
        {queueActive ? (
          <ComposerDisabledPrompt onPauseQueue={stopQueue} />
        ) : (
        <>
        {/* Context meter. Sits above the composer rather than in the project
            header because it is per-SESSION, and the header is per-project —
            it would report the wrong session while browsing the session list.
            Reuses executeNewSession verbatim: the offer is the same /new flow
            the slash command runs, not a parallel path. */}
        <ContextStrip usage={contextUsage} onNewSession={executeNewSession} />
        <div className="relative flex flex-col gap-2 bg-card border border-border rounded-lg shadow-[0_1px_2px_rgba(0,0,0,0.04)] px-3 py-2 transition-[border-color,box-shadow] duration-150 focus-within:border-accent focus-within:shadow-[0_0_0_3px_color-mix(in_oklab,var(--color-accent)18%,transparent)] motion-reduce:transition-none">
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
                  <span className="font-medium text-zinc-200">{cmd.name === '/new' ? t('chat.slash.new') : cmd.name}</span>
                  <span className="ml-2 text-zinc-500">{cmd.name === '/new' ? t('chat.slash.new.desc') : cmd.description}</span>
                </button>
              ))}
            </div>
          )}
          {showMentionDropdown && (
            <div className="absolute bottom-full left-0 mb-1 w-64 bg-zinc-800 border border-zinc-700 rounded-lg shadow-lg overflow-hidden z-50">
              {filteredAgents.length === 0 ? (
                <div className="px-3 py-2 text-sm text-zinc-500">{t('chat.noAgents')}</div>
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
          {/* Spec 078 §5.4 — the annotation chip. Collapsed it is just the
              count; expanded it lists each quote with an editable note and a
              remove control, so a note typed in a hurry on the panel can be
              fixed before the message goes. */}
          {annotations.length > 0 && (
            <div className="flex flex-col gap-1" data-testid="annotation-strip">
              <button
                type="button"
                onClick={() => setAnnotationsOpen((open) => !open)}
                aria-expanded={annotationsOpen}
                data-testid="annotation-chip"
                className="self-start inline-flex items-center gap-1 rounded-full border border-border px-2 py-0.5 text-[11px] text-secondary hover:text-primary transition-colors"
              >
                {annotations.length === 1
                  ? t('composer.annotations.one')
                  : t('composer.annotations.other', { n: annotations.length })}
                {annotationsOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              </button>
              {annotationsOpen && (
                <ul
                  className="flex flex-col gap-1 max-h-[140px] overflow-y-auto"
                  data-testid="annotation-list"
                >
                  {annotations.map((a) => (
                    <li key={a.n} className="flex items-center gap-2">
                      <span className="font-mono text-[11px] text-secondary shrink-0">[{a.n}]</span>
                      <span
                        className="text-[11px] text-secondary truncate max-w-[40%]"
                        title={annotationSummary(a, chromeTr)}
                      >
                        {annotationSummary(a, chromeTr)}
                      </span>
                      <input
                        value={a.note}
                        onChange={(e) => updateAnnotationNote(a.n, e.target.value)}
                        placeholder={t('panel.annotation.note')}
                        aria-label={`${t('panel.annotation.note')} [${a.n}]`}
                        className="flex-1 min-w-0 text-[11px] bg-transparent border border-border rounded px-1.5 py-0.5 text-primary outline-none focus:border-accent"
                      />
                      <button
                        type="button"
                        onClick={() => removeAnnotation(a.n)}
                        aria-label={`${t('panel.annotation.remove')} [${a.n}]`}
                        className="shrink-0 text-secondary hover:text-primary text-xs leading-none px-1"
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
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
            {/* Spec 074 — the logo-only pin mark fused to the row's left edge
                IS the sub-agent pin (Orbital's mark at rest, the worker's
                while pinned). Renders nothing when no sub-agents are
                installed (PinTargetSelect returns null). A PATCH failure
                keeps the optimistic localPin; handleSend re-persists after
                the first successful pinned send materializes a brand-new
                session. */}
            <PinTargetSelect
              agents={mentionAgents}
              value={pinnedTarget}
              onChange={(slug) => {
                if (sessionId === undefined) return;
                setLocalPin({ sessionId, slug });
                persistPin(sessionId, slug);
              }}
              disabled={sessionId === undefined}
            />
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
              aria-label={t('chat.attachFiles')}
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
              placeholder={pinnedTarget
                ? t('chat.composer.placeholderPinned', { agent: pinnedTarget })
                : t('chat.composer.placeholder')}
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
                      setCancelTimeoutNotice(t('chat.cancelFailed'));
                    });
                  }}
                  onTouchEnd={(e) => {
                    e.preventDefault();
                    if (isCancelling) return;
                    setCancelTimeoutNotice(null);
                    setIsCancelling(true);
                    cancelMessage(projectId, holderSessionId ?? sessionId).catch(() => {
                      setIsCancelling(false);
                      setCancelTimeoutNotice(t('chat.cancelFailed'));
                    });
                  }}
                  disabled={isCancelling}
                  aria-label={isCancelling ? t('chat.cancelling') : t('chat.stop')}
                  // Muted error, not raw `text-red-500`. Two changes: it goes
                  // through the --color-error token so it moves with the
                  // design system, and it rests at 70% so it stops being the
                  // only fully saturated element in a row where Plus is
                  // text-secondary. Stopping your own agent is not a fault
                  // (see the sub-agent 'stopped' handling above), so it should
                  // not shout like one — the queue's own pause control already
                  // reads this quietly. rounded-lg joins the radius ladder the
                  // rest of the composer is built from.
                  className="group shrink-0 p-1.5 rounded-lg transition-colors duration-150 cursor-pointer text-error/70 hover:text-error hover:bg-error/8 disabled:cursor-default disabled:hover:bg-transparent max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
                >
                  <StopGlyph cancelling={isCancelling} />
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
                  {t('chat.queue')}
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                onTouchEnd={(e) => { e.preventDefault(); handleSend(); }}
                aria-disabled={!canSend}
                aria-label={t('chat.send')}
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
        </>
        )}
      </div>
      </>
      )}
    </div>
  );
}
