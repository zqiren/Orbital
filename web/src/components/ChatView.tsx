// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { memo, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  Loader2, Plus, ChevronRight, ChevronDown, ArrowDown,
  Clock, CornerDownLeft, FolderSearch, ListChecks, MessageSquareQuote, type LucideIcon,
} from 'lucide-react';
import ExamplePrompt from './ExamplePrompt';
import LeftOffCards from './LeftOffCards';
import PrivateNotesCard from './PrivateNotesCard';
import { sendTelemetry } from '../utils/telemetry';
import OrbitalMark from './OrbitalMark';
import { api, apiWithTotal, ApiError } from '../config';
import { useWebSocket } from '../hooks/useWebSocket';
import { useAgent, type InjectResult } from '../hooks/useAgent';
import { useQueue } from '../hooks/useQueue';
import { projectAgents, useAgentAvailability } from '../hooks/useAgentAvailability';
import { summarizeQuota, switchTarget } from '../utils/quotaDisplay';
import UsageLimitRow from './UsageLimitRow';
import { useAutoScroll } from '../hooks/useAutoScroll';
import {
  transformChatHistory,
  truncateResult,
  mergeRecoveredAssistantMessage,
  describeLiveActivity,
  describeWorkerTool,
  dispatchFromToolCall,
  workerToolCategory,
} from '../utils/chatTransform';
import type { DisplayItem, ResultTotals } from '../utils/chatTransform';
import { isWorkerHandle } from '../utils/subAgentHandle';
import {
  applyWorkerEvent,
  endCard,
  ensureLiveCard,
  findCardIndex,
  hasDispatchCards,
  isCardedInitiator,
  unwrapCard,
  type DispatchCardItem,
  type WorkerEventRoute,
} from '../utils/dispatchCards';
import { publishDispatchCards, type WorkerPanelTarget } from '../utils/dispatchCardsStore';
import DispatchCard from './DispatchCard';
import { createFrameTextBuffer, type FrameTextBuffer } from '../utils/frameTextBuffer';
import type { ChatMessage as ChatMessageRow, AgentStatusEvent } from '../types';
import AgentErrorNotice from './AgentErrorNotice';
import { parseProviderError } from '../utils/providerError';
import { markFreshSession, consumeFreshSession } from '../utils/freshSession';
import AttachmentChip from './AttachmentChip';
import SkillChip from './SkillChip';
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
import { buildInlineSkillBlock, buildSkillBlock, parseSkillBlock, skillAwarePreview } from '../lib/skill-parsing';
import { skillSendParts, type SkillCatalogEntry } from '../lib/skill-catalog';
import SlashMenu, { useSlashPicker } from './SlashMenu';
import ComposerDisabledPrompt from './ComposerDisabledPrompt';
import AskCard from './AskCard';
import { useSessionAsks, type SessionAsk } from '../hooks/useSessionAsks';
import { formatAskAnswer, stripTurnNote } from '../utils/askAnswer';
import { changeOptionPrefill, isChangeOption } from '../utils/changeOptionPrefill';
import PinTargetSelect, { resolveSendTarget } from './PinTargetSelect';
import { useConversationModels } from '../hooks/useConversationModels';
import { resolveCard } from '../hooks/useCredentialCards';
import { useSessions } from '../hooks/useSessions';
import { effectiveChecked, markChecked } from '../hooks/useSeenReplies';
import { seenKey } from '../utils/sessionUnread';
import ContextStrip from './ContextStrip';
import { useContextUsage } from '../hooks/useContextUsage';
import { useT, translate } from '../i18n/useT';
import { formatCapsuleDuration, formatToolBreakdown } from '../utils/toolBreakdown';
import { useLocale } from '../i18n/LocaleContext';
import type { StringKey } from '../i18n/strings';
import type { BuiltinSkillKey } from '../route';
import { publishSessionMessages, sessionMessagesKey } from '../utils/sessionMessagesStore';
import { budgetTimelineText } from '../budget/timelineText';
import TurnChangedLine from './history/TurnChangedLine';
import { isMemoryFile, lastCapsuleOfTurns, stripFileEditCounts } from './history/historyModel';
import { isStatusFold, isStatusOnly, latestStatus, statusOf, statusSteps, stripStatusTags, type StatusStep } from '../utils/statusTag';
import { historyApi, historyVisible, requestHistoryFocus, useCapsuleTurn, useHistoryStatus, useSeenOnce } from '../hooks/useFileHistory';
import type { TurnRef } from './history/TurnUndoSheet';
import { classifySessionName } from '../lib/sessionLabel';
import { parseTriggerHeader } from '../lib/trigger-parsing';
import { parseSourceTask } from '../lib/source-task-parsing';
import { forkSessionName, rebuildEditedContent, splitUserWire, stashForkSend, takeForkSend } from '../utils/editMessage';

type AgentRunItem = Extract<DisplayItem, { type: 'agent_run' }>;
type CapsuleChild = AgentRunItem['items'][number];

// Example prompts offered in a project's empty chat (see the empty state).
const EMPTY_CHAT_EXAMPLES: { key: StringKey; icon: LucideIcon }[] = [
  { key: 'chat.empty.example.explore', icon: FolderSearch },
  { key: 'chat.empty.example.automate', icon: Clock },
  { key: 'chat.empty.example.plan', icon: ListChecks },
];

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


export function capsuleSummaryText(
  capsule: AgentRunItem,
  tr: CapsuleTr = EN_CAPSULE,
  opts: { stripFileEdits?: boolean } = {},
): string {
  // Spec 124 §3.7 follow-up: the tidy pass's capsule says what the pass did
  // (counts from the row, not a tool breakdown) and carries no duration —
  // the row is a one-line report, not a timed step.
  if (capsule.memory_tidy) {
    const t = capsule.memory_tidy;
    if (t.outcome === 'failed') return tr('chat.capsule.memoryTidyFailed', { error: t.error ?? t.outcome });
    if (t.archived === 0 && t.restored === 0 && capsule.items.length === 0) {
      return tr('chat.capsule.memoryTidyNothing');
    }
    return tr('chat.capsule.memoryTidy', { archived: t.archived, restored: t.restored });
  }
  // Spec 065 §5.9: on a turn's last capsule whose turn changed files, the
  // changed-files line under it says what was written (shell and worker
  // writes included), so the breakdown drops its own file-writing counts.
  const breakdown = formatToolBreakdown(
    opts.stripFileEdits
      ? stripFileEditCounts(capsule.tool_call_count_by_name)
      : capsule.tool_call_count_by_name,
    tr,
  );
  const dur = formatCapsuleDuration(capsule.started_at, capsule.ended_at, tr);
  // Spec 112 A6: the memory record turn's capsule says what it is.
  const head = capsule.memory_record
    ? (breakdown ? `${tr('chat.capsule.memoryRecord')} · ${breakdown}` : tr('chat.capsule.memoryRecord'))
    : breakdown || (capsule.has_thinking ? tr('chat.capsule.thinking') : tr('chat.capsule.agentStep'));
  let line = `${head} · ${dur}`;
  if (capsule.status === 'error' || capsule.status === 'stopped') {
    line += ` ${tr('chat.capsule.stoppedAtError')}`;
  }
  return line;
}

// Spec 100: every live-capsule helper below takes an `owner`: undefined for
// the management agent (its capsules carry no `source`), or a sub-agent
// handle (its capsules carry `source: handle`). A running capsule at the tail
// is extended only by events of its own owner.
function getLiveRunningCapsule(
  items: DisplayItem[],
  owner?: string,
): { idx: number; capsule: AgentRunItem } | null {
  // Spec 195: dispatch cards opened at the tail do not end the running
  // capsule before them. Orbital's sends land in its capsule first and their
  // cards follow it, as a reload draws them (the capsule, then each marker's
  // card) even when a card opened between two of its tool rows.
  let k = items.length - 1;
  while (k >= 0 && items[k].type === 'dispatch_card') k--;
  if (k < 0) return null;
  const last = items[k];
  if (last.type === 'agent_run' && last.status === 'running' && last.source === owner) {
    return { idx: k, capsule: last };
  }
  return null;
}

function appendToLiveCapsule(
  prev: DisplayItem[],
  child: CapsuleChild,
  timestamp: string,
  owner?: string,
): DisplayItem[] {
  const ms = Date.parse(timestamp);
  const live = getLiveRunningCapsule(prev, owner);
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
      ...(owner ? { source: owner } : {}),
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
  owner?: string,
): DisplayItem[] {
  const live = getLiveRunningCapsule(prev, owner);
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
      owner,
    );
  }

  // No running capsule — this reasoning starts a fresh agent turn. Anchor it
  // with a header-only agent_message (mirrors chatTransform's FE-A3) so the
  // capsule renders the agent avatar and does NOT visually attach to the
  // preceding user message. The agent_run capsule draws no avatar of its own
  // (it is indented to sit under a header); without this anchor the live
  // thinking floats unattributed — and in cold-start (no preceding user
  // message) it is not attributed to the management agent at all.
  return appendToLiveCapsule(
    anchorAgentHeader(prev, source, timestamp, owner),
    { type: 'reasoning_block', content: reasoning, timestamp, turn_id: timestamp },
    timestamp,
    owner,
  );
}

/** Prepend-anchor for a fresh live capsule: the header-only agent_message
 * that attributes it. The management agent's capsule is anchored by any
 * preceding agent row; a sub-agent's only by its OWN header or bubble —
 * otherwise its run would sit under the manager's (or another worker's)
 * avatar. */
function anchorAgentHeader(
  prev: DisplayItem[],
  source: string,
  timestamp: string,
  owner?: string,
): DisplayItem[] {
  const lastItem = prev[prev.length - 1];
  const anchored = owner
    ? !!lastItem &&
      (lastItem.type === 'agent_message' || lastItem.type === 'sub_agent_message') &&
      lastItem.source === owner
    : !!lastItem &&
      (lastItem.type === 'agent_message' ||
        lastItem.type === 'sub_agent_message' ||
        lastItem.type === 'agent_run');
  return anchored
    ? prev
    : [
        ...prev,
        { type: 'agent_message', content: '', source, timestamp, isHeaderOnly: true },
      ];
}

/** Spec 100: a sub-agent's live tool call. Opens (and anchors) the worker's
 * capsule on the first call; a call already in it — a codex completion or
 * an ACP progress update adding a title or arguments — updates its row in
 * place, never appends a second one. */
export function upsertWorkerToolRow(
  prev: DisplayItem[],
  owner: string,
  row: ToolCallRowItem,
): DisplayItem[] {
  const live = getLiveRunningCapsule(prev, owner);
  if (live) {
    const items = live.capsule.items;
    const k = items.findIndex(
      (c) => c.type === 'tool_call_row' && c.tool_call_id === row.tool_call_id,
    );
    if (k >= 0) {
      const existing = items[k] as ToolCallRowItem;
      const counts = { ...live.capsule.tool_call_count_by_name };
      if (existing.tool_name !== row.tool_name) {
        counts[existing.tool_name] = (counts[existing.tool_name] ?? 1) - 1;
        if (counts[existing.tool_name] <= 0) delete counts[existing.tool_name];
        counts[row.tool_name] = (counts[row.tool_name] ?? 0) + 1;
      }
      const newItems = [...items];
      newItems[k] = {
        ...existing,
        tool_name: row.tool_name,
        target_description: row.target_description,
        category: row.category,
      };
      const next = [...prev];
      next[live.idx] = { ...live.capsule, items: newItems, tool_call_count_by_name: counts };
      return next;
    }
    return appendToLiveCapsule(prev, row, row.timestamp, owner);
  }
  return appendToLiveCapsule(
    anchorAgentHeader(prev, owner, row.timestamp, owner),
    row,
    row.timestamp,
    owner,
  );
}

function finalizeLiveCapsule(
  prev: DisplayItem[],
  status: 'completed' | 'error' | 'stopped',
  owner?: string,
): DisplayItem[] {
  const live = getLiveRunningCapsule(prev, owner);
  if (!live) return prev;
  // Nothing ran in it (at most a status marker no tool followed): no capsule.
  if (!live.capsule.items.some((c) => c.type !== 'agent_message')) {
    return prev.slice(0, live.idx);
  }
  const updated: AgentRunItem = { ...live.capsule, status };
  const next = [...prev];
  next[live.idx] = updated;
  return next;
}

/**
 * A manager reply carried a [STATUS: …]: its marker opens the reply's batch
 * in the running capsule (opened, and anchored under the agent's header, when
 * there is none), so the tools that follow are that step and the fold shows
 * the status live. Mirrors chatTransform's status marker.
 */
export function appendLiveStatus(
  prev: DisplayItem[],
  status: string,
  source: string,
  timestamp: string,
  afterText = false,
): DisplayItem[] {
  const marker: CapsuleChild = {
    type: 'agent_message', content: '', source, timestamp, status,
    ...(afterText ? { statusAfterText: true } : {}),
  };
  const live = getLiveRunningCapsule(prev);
  if (live) {
    // Idempotent: the is_final handler can run twice (React re-runs the
    // stream updater in StrictMode); the same status already opening the
    // tail batch is this reply's.
    const tail = live.capsule.items[live.capsule.items.length - 1];
    if (tail && tail.type === 'agent_message' && tail.status === status) return prev;
    return appendToLiveCapsule(prev, marker, timestamp);
  }
  return appendToLiveCapsule(anchorAgentHeader(prev, source, timestamp), marker, timestamp);
}

type ToolCallRowItem = Extract<CapsuleChild, { type: 'tool_call_row' }>;

function ToolCallRow({ row }: { row: ToolCallRowItem }): React.ReactNode {
  const t = useT();
  const { locale } = useLocale();
  const [expanded, setExpanded] = useState(false);
  const { dispatch } = row;
  // An agent_message row can open before its ack arrives: the message it
  // sent is already known from the call's arguments.
  const hasResult = row.result_status !== 'pending';
  const expandable = hasResult || !!dispatch;
  const Chevron = expanded ? ChevronDown : ChevronRight;

  return (
    <div
      data-testid="tool-call-row"
      className="mb-1 font-mono text-micro text-secondary"
    >
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
      {expanded && dispatch && (
        <div className="mt-1 ml-5">
          <div className="mb-1 px-3 text-micro text-secondary">
            {t('chat.toolRow.messageTo', { agent: dispatch.agent })}
          </div>
          <pre className="max-h-80 overflow-y-auto px-3 py-2 rounded bg-background border border-border/40 text-caption text-secondary whitespace-pre-wrap break-words font-mono">
            {dispatch.message}
          </pre>
        </div>
      )}
      {expanded && hasResult && (() => {
        const raw = row.result_content;
        const label = dispatch && (
          <div className="mt-2 mb-1 px-3 text-micro text-secondary">{t('chat.toolRow.result')}</div>
        );
        if (raw === null || raw === '') {
          return (
            <div className="mt-1 ml-5">
              {label}
              <div className="px-3 py-2 rounded bg-background border border-border/40 text-caption italic text-secondary">
                {t('chat.toolRow.noResult')}
              </div>
            </div>
          );
        }
        const { text, footer } = truncateResult(raw, (k, v) => translate(locale, k, v), row.result_totals);
        return (
          <div className="mt-1 ml-5">
            {label}
            <pre className="px-3 py-2 rounded bg-background border border-border/40 text-caption text-secondary whitespace-pre-wrap break-words font-mono">
              {text}
            </pre>
            {footer && (
              <div className="mt-1 px-3 text-micro text-secondary italic">{footer}</div>
            )}
          </div>
        );
      })()}
    </div>
  );
}

// Spec 100 §3.4.8: the transcript renders every item on every setItems (no
// virtualization), so the heavy rows are memoized: a live event that
// replaces one item re-renders that row only. Items are immutable (each
// update copies the changed item), so identity is the change signal.
const MemoChatMessage = memo(ChatMessage);

interface AgentRunCapsuleProps {
  item: AgentRunItem;
  derivedStatus: AgentRunItem['status'];
  isExpanded: boolean;
  onToggle: (capsuleId: string) => void;
  /**
   * Spec 065 §5.9 — set on a turn's LAST capsule only: the turn's span (from
   * its user row to the next) and whether it has ended. The capsule then
   * asks the history turn route what the turn changed and, when it changed
   * something, shows the changed-files line. Primitives, so memo holds.
   */
  projectId?: string;
  sessionId?: string;
  turnFrom?: string;
  turnTo?: string | null;
  turnEnded?: boolean;
  onUndoTurn?: (turn: { sessionId: string; fromTs: string; toTs: string | null }) => void;
  onOpenHistory?: (path: string, v: number) => void;
  /** A status fold right under its agent's header-only anchor sits close
   *  under it, as one message (the anchor has no copy row to clear). */
  tight?: boolean;
}

/** A capsule's children as its expanded body draws them. */
function CapsuleChildren({ items, keyPrefix }: { items: CapsuleChild[]; keyPrefix: string }): React.ReactNode {
  return (
    <>
      {items.map((child, ci) => {
        if (child.type === 'reasoning_block') {
          return (
            <div
              key={`${keyPrefix}-r-${ci}`}
              className="mb-2 pl-3 border-l-2 border-accent/40 italic text-secondary text-body whitespace-pre-wrap"
            >
              {child.content}
            </div>
          );
        }
        if (child.type === 'tool_call_row') {
          return <ToolCallRow key={`${keyPrefix}-t-${ci}-${child.tool_call_id}`} row={child} />;
        }
        if (child.type === 'agent_message') {
          // Empty-content marker delimits silent tool batches.
          return <div key={`${keyPrefix}-m-${ci}`} className="my-2 border-t border-border/40" aria-hidden />;
        }
        return null;
      })}
    </>
  );
}

// The status line's parts, in the tool-row language: chevron, the status in
// caption type, then the quiet mono part (tool names, steps, time).
const STATUS_LINE_BTN = 'flex items-start gap-2 w-full min-h-5 py-0.5 text-left text-secondary cursor-pointer hover:text-primary';
const STATUS_CHEV_CLS = 'shrink-0 mt-[calc((var(--text-caption--line-height)-13px)/2)]';
const STATUS_TEXT_CLS = 'min-w-0 text-caption text-primary';
const STATUS_DOT_CLS = 'shrink-0 font-mono text-micro leading-[var(--text-caption--line-height)] text-muted';
const STATUS_QUIET_CLS = 'shrink-0 whitespace-nowrap font-mono text-micro leading-[var(--text-caption--line-height)] text-secondary';

/** The status itself: one line with an ellipsis (whole on hover and when
 *  open); while live, the Workbench shimmer, rising in when it changes. */
function StatusText({ text, open, live }: { text: string; open: boolean; live: boolean }): React.ReactNode {
  return (
    <span
      key={live ? text : undefined}
      title={text}
      data-testid="status-line-text"
      className={`${STATUS_TEXT_CLS} ${open ? 'whitespace-normal break-words' : 'truncate'} ${live ? 'status-shim status-rise' : ''}`}
    >
      {text}
    </span>
  );
}

function StatusStepLine({
  step,
  live,
  endMs,
  tr,
}: {
  step: StatusStep;
  live: boolean;
  endMs: number | null;
  tr: CapsuleTr;
}): React.ReactNode {
  const [open, setOpen] = useState(false);
  const Chevron = open ? ChevronDown : ChevronRight;
  const breakdown = formatToolBreakdown(step.tool_call_count_by_name, tr);
  const dur = formatCapsuleDuration(step.started_at, step.ended_at ?? endMs, tr);
  const label = step.status ?? (breakdown || tr('chat.capsule.agentStep'));
  const tools = step.status && breakdown ? breakdown : '';
  return (
    <div data-testid="status-step" data-open={open || undefined}>
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className={STATUS_LINE_BTN}>
        <Chevron size={13} className={STATUS_CHEV_CLS} />
        {live && <Loader2 size={12} className="shrink-0 animate-spin mt-[calc((var(--text-caption--line-height)-12px)/2)]" />}
        <StatusText text={label} open={open} live={live} />
        <span className={STATUS_DOT_CLS} aria-hidden>·</span>
        <span className={STATUS_QUIET_CLS}>
          {tools && <span className="max-md:hidden">{tools} · </span>}
          {dur}
        </span>
      </button>
      {open && step.children.length > 0 && (
        <div className="pt-1 pb-1.5 pl-[21px]">
          <CapsuleChildren items={step.children} keyPrefix="st" />
        </div>
      )}
    </div>
  );
}

/**
 * Variant B: a run of status steps as one quiet line — the latest status,
 * the step count and the run's time. Opens to one line per step, each
 * opening to its tool rows. Running, it stays closed and the line is the
 * live status (spinner + shimmer); the user can still open it.
 */
function StatusFold({
  item,
  steps,
  status,
  isExpanded,
  onToggle,
  tr,
}: {
  item: AgentRunItem;
  steps: StatusStep[];
  status: AgentRunItem['status'];
  isExpanded: boolean;
  onToggle: (capsuleId: string) => void;
  tr: CapsuleTr;
}): React.ReactNode {
  const running = status === 'running';
  const stopped = status === 'error' || status === 'stopped' ? ` ${tr('chat.capsule.stoppedAtError')}` : '';
  if (steps.length === 1) {
    return <StatusStepLine step={steps[0]} live={running} endMs={item.ended_at} tr={tr} />;
  }
  const Chevron = isExpanded ? ChevronDown : ChevronRight;
  const label = latestStatus(steps) ?? tr('chat.capsule.agentStep');
  const count = tr(steps.length === 1 ? 'chat.capsule.steps.one' : 'chat.capsule.steps.other', { n: steps.length });
  const dur = formatCapsuleDuration(item.started_at, item.ended_at, tr);
  return (
    <div data-testid="status-fold" data-open={isExpanded || undefined}>
      <button
        type="button"
        onClick={() => onToggle(item.capsule_id)}
        aria-expanded={isExpanded}
        className={STATUS_LINE_BTN}
      >
        <Chevron size={13} className={STATUS_CHEV_CLS} />
        {running && <Loader2 size={12} className="shrink-0 animate-spin mt-[calc((var(--text-caption--line-height)-12px)/2)]" />}
        <StatusText text={label} open={isExpanded} live={running} />
        <span className={STATUS_DOT_CLS} aria-hidden>·</span>
        <span className={STATUS_QUIET_CLS} data-testid="status-fold-meta">{`${count} · ${dur}${stopped}`}</span>
      </button>
      {isExpanded && (
        <div className="flex flex-col gap-0.5 pl-[21px] pb-1">
          {steps.map((step, k) => (
            <StatusStepLine
              key={`step-${k}`}
              step={step}
              live={running && k === steps.length - 1}
              endMs={item.ended_at}
              tr={tr}
            />
          ))}
        </div>
      )}
    </div>
  );
}

const AgentRunCapsule = memo(function AgentRunCapsule({
  item,
  derivedStatus,
  isExpanded,
  onToggle,
  projectId,
  sessionId,
  turnFrom,
  turnTo = null,
  turnEnded = false,
  onUndoTurn,
  onOpenHistory,
  tight = false,
}: AgentRunCapsuleProps): React.ReactNode {
  const t = useT();
  const isLocked = derivedStatus === 'running';
  const [seenRef, seen] = useSeenOnce<HTMLDivElement>();
  const turn = useCapsuleTurn(projectId, sessionId, turnFrom, turnTo, turnEnded && !isLocked, seen);
  const summary = capsuleSummaryText({ ...item, status: derivedStatus }, t, { stripFileEdits: turn !== null });
  const Chevron = isExpanded ? ChevronDown : ChevronRight;
  const steps = useMemo(() => statusSteps(item, { running: isLocked }), [item, isLocked]);
  const changed = turn && sessionId && turnFrom && (
    <TurnChangedLine
      turn={turn}
      onUndo={onUndoTurn ? () => onUndoTurn({ sessionId, fromTs: turnFrom, toTs: turnTo }) : undefined}
      onOpenFile={onOpenHistory}
    />
  );
  // A status that no step has shown yet (it came with a text bubble and no
  // tool has run under it): nothing to draw.
  if (!steps && isStatusFold(item)) return null;
  if (steps) {
    return (
      <div
        ref={seenRef}
        data-testid="agent_run"
        data-capsule-id={item.capsule_id}
        data-capsule-status={derivedStatus}
        data-status-fold=""
        className={`ml-9 flex gap-[10px] ${tight ? '-mt-3' : ''}`}
      >
        <div className="w-0.5 shrink-0 rounded-sm bg-border" aria-hidden />
        <div className="min-w-0 flex-1">
          <StatusFold item={item} steps={steps} status={derivedStatus} isExpanded={isExpanded} onToggle={onToggle} tr={t} />
          {changed}
        </div>
      </div>
    );
  }
  return (
    <div
      ref={seenRef}
      data-testid="agent_run"
      data-capsule-id={item.capsule_id}
      data-capsule-status={derivedStatus}
      className="ml-9 flex gap-[10px]"
    >
      <div className="w-0.5 shrink-0 rounded-sm bg-border" aria-hidden />
      <div className="min-w-0 flex-1">
      <button
        type="button"
        onClick={isLocked ? undefined : () => onToggle(item.capsule_id)}
        disabled={isLocked}
        className={`flex items-center gap-2 w-full text-left font-mono text-micro text-secondary ${isLocked ? 'cursor-default' : 'cursor-pointer hover:text-primary'}`}
      >
        <Chevron size={13} className="shrink-0" />
        {derivedStatus === 'running' && (
          <Loader2 size={12} className="shrink-0 animate-spin" />
        )}
        <span className="truncate text-primary font-medium">{summary}</span>
      </button>
      {changed}
      {isExpanded && item.items.length > 0 && (
        <div className="mt-2 pt-2 border-t border-border/30">
          <CapsuleChildren items={item.items} keyPrefix="rc" />
        </div>
      )}
      </div>
    </div>
  );
});

/**
 * Spec 195: a dispatched agent's steps — the same header, capsules and
 * bubbles its run renders inline — read-only. The docked panel shows them
 * (WorkerPanel); without a panel the card opens them in place. A running
 * capsule is open, a finished one opens on click, as in the conversation.
 */
export function WorkerSteps({
  steps,
  workspace,
  onOpenPath,
}: {
  steps: DisplayItem[];
  workspace?: string;
  onOpenPath?: (path: string) => void;
}): React.ReactNode {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggle = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);
  return (
    <>
      {steps.map((step, i) => {
        if (step.type === 'agent_message' || step.type === 'sub_agent_message') {
          return <MemoChatMessage key={`ws-${i}`} message={step} workspace={workspace} onOpenPath={onOpenPath} />;
        }
        if (step.type === 'agent_run') {
          return (
            <AgentRunCapsule
              key={`ws-${step.capsule_id}`}
              item={step}
              derivedStatus={step.status}
              isExpanded={step.status === 'running' || expanded.has(step.capsule_id)}
              onToggle={toggle}
            />
          );
        }
        return null;
      })}
    </>
  );
}

// A live tool_result event, as far as the capsule row needs it. Newer
// daemons send the tool_call_id plus the part of the result the row shows
// (activity_translator.py `_result_preview`); older ones send neither — only
// the "Tool result received" placeholder, with the id in `tool_name`.
interface LiveToolResult {
  toolCallId?: string;
  content: string;
  totals?: ResultTotals;
}

function markLatestLiveCallResultReceived(
  prev: DisplayItem[],
  timestamp: string,
  result: LiveToolResult = { content: '' },
  owner?: string,
): DisplayItem[] {
  const live = getLiveRunningCapsule(prev, owner);
  if (!live) return prev;
  const items = live.capsule.items;
  // Pair by id when the daemon sends one: parallel calls finish in any
  // order, and a result with no row here (the fanout ack) must not land on
  // another call's row. Without an id (older daemon) fall back to the most
  // recent pending row, as before; the content is then empty and the JSONL
  // reload fills it in on next mount via chatTransform pairing.
  for (let k = items.length - 1; k >= 0; k--) {
    const item = items[k];
    if (item.type !== 'tool_call_row') continue;
    const isTarget = result.toolCallId
      ? item.tool_call_id === result.toolCallId
      : item.result_status === 'pending';
    if (isTarget) {
      const updatedRow: ToolCallRowItem = {
        ...item,
        result_content: result.content,
        result_status: 'received',
        ...(result.totals ? { result_totals: result.totals } : {}),
      };
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

/** Spec 100: a sub-agent's live tool result. Pairs by id with the row in
 * the worker's most recent capsule holding it — not only a running tail
 * capsule: the worker's own text bubble may have closed that capsule
 * between the call and its result. */
export function markWorkerResult(
  prev: DisplayItem[],
  owner: string,
  timestamp: string,
  result: LiveToolResult,
): DisplayItem[] {
  if (!result.toolCallId) {
    return markLatestLiveCallResultReceived(prev, timestamp, result, owner);
  }
  for (let i = prev.length - 1; i >= 0; i--) {
    const item = prev[i];
    if (item.type !== 'agent_run' || item.source !== owner) continue;
    const k = item.items.findIndex(
      (c) => c.type === 'tool_call_row' && c.tool_call_id === result.toolCallId,
    );
    if (k < 0) continue;
    const row = item.items[k] as ToolCallRowItem;
    const newItems = [...item.items];
    newItems[k] = {
      ...row,
      result_content: result.content,
      result_status: 'received',
      ...(result.totals ? { result_totals: result.totals } : {}),
    };
    const next = [...prev];
    next[i] = {
      ...item,
      items: newItems,
      ...(item.status === 'running' ? { ended_at: Date.parse(timestamp) } : {}),
    };
    return next;
  }
  return prev;
}

/** Spec 195: the terminal broadcasts that end a dispatch card, and how. */
const CARD_END_BY_EVENT: Partial<Record<string, Exclude<DispatchCardItem['status'], 'running'>>> = {
  'sub_agent.completed': 'completed',
  'sub_agent.error': 'error',
  'sub_agent.failed': 'error',
  'sub_agent.stopped': 'stopped',
  'sub_agent.turn_interrupted': 'stopped',
};

// Spec 100: worker thinking reaches the pane at most once per frame.
function requestFrame(fn: () => void): number {
  return typeof requestAnimationFrame === 'function'
    ? requestAnimationFrame(fn)
    : window.setTimeout(fn, 16);
}

function cancelFrame(id: number): void {
  if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(id);
  else window.clearTimeout(id);
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
/**
 * Spec 131 A: rows the idle catch-up asks for. The end-of-run memory record
 * turn (spec 112 A6) appends tagged rows behind the real reply — the
 * `[MEMORY]` ask, the model's edit call, one or two edit results and a
 * closing text, five rows in the live evidence — so the tail must reach past
 * them for the catch-up to still find an UNTAGGED reply when the stream was
 * lost mid-turn. Six covers the "one or two edits" the record prompt asks
 * for; a longer record turn just means nothing is recovered (never a wrong
 * row), and the full refreshRawMessages that follows covers that case.
 */
const REST_FALLBACK_LIMIT = 6;

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

/**
 * True when a queued message's wire content carries an attachments or a
 * skill block: recall can't rebuild those chips, so it stays off (§12 R4;
 * spec 132 treats the skill block like an attachment block).
 */
function hasMachineBlocks(content: string): boolean {
  const { strippedContent, attachments } = parseAttachmentsBlock(content);
  return attachments.length > 0 || parseSkillBlock(strippedContent).skill !== null;
}

/** The composer grows with its text up to this height, then scrolls. */
const COMPOSER_MAX_HEIGHT_PX = 160;

/**
 * Fit the composer to its text (spec 105 P2).
 *
 * Collapsing `height` to `auto` before reading `scrollHeight` is the usual
 * auto-grow trick, but in WebKit the collapse is one synchronous full-page
 * layout and the pixel write after it is a second one, and both move every
 * height-dependent block on the page (the docked file preview paid ~87 ms
 * per keystroke for it). Growth never needs the collapse: against the
 * current fixed height `scrollHeight` already reports the taller content.
 * Only a shrink hides behind the fixed height, so callers collapse first
 * (`reset`) only when the text may have got shorter. The write is skipped
 * when the height is unchanged, so a same-line keystroke does no layout work.
 *
 * While collapsed for the measure, a bottom margin of the old height holds
 * the composer's footprint, so the chat above never gets taller for that one
 * forced layout. Without it the browser clamps the chat's scrollTop to the
 * briefly taller view, the restore leaves the transcript displaced (93px with
 * a 5-line draft) and the clamp's scroll event reads as the user scrolling
 * away from the bottom. Style writes only: no extra layout.
 */
export function fitComposerHeight(ta: HTMLTextAreaElement, reset: boolean): void {
  const held = reset ? ta.style.height : '';
  if (reset) {
    ta.style.height = 'auto';
    if (held) ta.style.marginBottom = held;
  }
  const next = Math.min(ta.scrollHeight, COMPOSER_MAX_HEIGHT_PX) + 'px';
  if (held) ta.style.marginBottom = '';
  if (ta.style.height !== next) ta.style.height = next;
}

function countNewlines(text: string): number {
  let n = 0;
  for (let i = text.indexOf('\n'); i !== -1; i = text.indexOf('\n', i + 1)) n++;
  return n;
}
import type {
  AgentRunStatus,
  ChatMessage as ChatMessageType,
  StreamDeltaEvent,
  ActivityEvent,
  ApprovalRequestEvent,
  ApprovalResolvedEvent,
  SubAgentMessageEvent,
  SubAgentLifecycleEvent,
  SubAgentQueueDroppedEvent,
  UserMessageEvent,
  AgentNotifyEvent,
  StateRefreshLifecycleEvent,
  WorkspaceClaudemdWarningEvent,
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
import QueueBlock from './QueueBlock';
import SendButton from './SendButton';
import {
  fromPendingWire, mergeIntoDraft, queueRows, type QueueRow, type QueuedChip,
} from '../utils/composerQueue';
import SubAgentStatusBar from './SubAgentStatusBar';
import { useSubAgentStatus } from '../hooks/useSubAgentStatus';
import { isTurnInFlight } from '../utils/turnInFlight';
import FanoutCard, { isTerminal, type FanoutTaskState } from './FanoutCard';
import SubAgentDrillIn from './SubAgentDrillIn';
import InlineMarkdown from './InlineMarkdown';
import { projectDisplayName } from '../utils/projectDisplayName';

interface ChatViewProps {
  projectId: string;
  project: Project;
  agentStatus: AgentRunStatus;
  statusTick?: number;
  /**
   * Installed sub-agents for the composer pin menu. Lifted to App-level
   * state so tab switches don't refetch /agents/available. Empty array while
   * App is still resolving the initial fetch — the pin control renders
   * nothing until the list arrives.
   */
  agents: Array<{ slug: string; name: string }>;
  /**
   * The F1 session_id currently being viewed (the active session for this
   * project). The conversation, history fetch, draft, and inject target are
   * all scoped to this session. `undefined` while the active session is
   * still being resolved by the parent (ChatTab) — ChatView renders an empty
   * state and skips history load until a sessionId arrives.
   *
   * Sessions of a project run in parallel (spec 155). `agentStatus` is THIS
   * session's status, and live WS events render only when their
   * `session_id` is this one.
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
  /** Spec 146: open with this ask quoted in the composer (a Workbench card's
   *  Reply). Consumed once, through `onDraftConsumed`. */
  initialAskReply?: { id: string; text: string };
  /**
   * Spec 137: a built-in skill to have picked along with `initialDraft`
   * (Tasks → "New automation" opens chat with `/automate` chipped). Same
   * one-shot lifetime; `onDraftConsumed` clears both.
   */
  initialSkill?: BuiltinSkillKey;
  onDraftConsumed?: () => void;
  /**
   * Re-fetch this project's runtime fields and merge
   * them into the App-level projects list. Called on the running→idle
   * transition so the header's budget/cost reflects the just-finished turn.
   */
  onRefreshProject?: (id: string) => void;
  /** Open another session of this project (the left-off cards, spec 138). */
  onOpenSession?: (sessionId: string) => void;
  /**
   * Spec 065 §5.9 — open the reviewable undo sheet for one turn (ChatTab puts
   * it in the docked panel slot, a bottom sheet on the phone). Absent: the
   * changed-files line shows no undo glyph.
   */
  onUndoTurn?: (turn: TurnRef | null) => void;
  /**
   * Workbench reading pane (2026-11-04): false = being on screen does not
   * check the session's reply. The pane is a glance, not the desk: the card
   * stays until ✓, and comes back to 等你 when a turn ends while the pane is
   * open. Default true (the session's own page checks it).
   */
  checksOnView?: boolean;
  /** Called once the user's own message went out (the pane moves its card
   *  to 进行中 at once), with the inject response: a `recorded` reply
   *  started no turn, so the pane lets its card go (spec 169 B). */
  onUserSend?: (result: InjectResult | undefined) => void;
  /** Spec 180: the daemon delivered a send into another project (the
   *  session had moved): the open chat follows it there. */
  onSessionMoved?: (sessionId: string, toProjectId: string) => void;
  /**
   * Spec 195: open a dispatched agent's steps (or a fanout worker's) in the
   * docked panel ChatTab owns. Absent (the Workbench reading pane): a card
   * opens its steps in place under it, a fanout row the full-area drill-in.
   */
  onOpenWorker?: (target: WorkerPanelTarget) => void;
  /** What the panel shows now, so its card reads as selected. */
  openWorker?: WorkerPanelTarget | null;
}

/** Spec 195: a pinned message waiting in a worker's queue. `held` (the
 *  composer queue block, 2026-11-24): its chat row is not written until it
 *  is dispatched, so it shows as a queue row, not a bubble; without it (a
 *  send that raced a spawn) it is a bubble with the clock, as before. */
interface QueuedSend {
  dispatchId: string;
  handle: string;
  sessionId: string;
  content: string;
  held?: boolean;
  nonce?: string;
  /** The row's words, and what edit puts back in the composer. */
  text: string;
  chips: QueuedChip[];
  editable: boolean;
  queuedAt: string;
  skill?: SkillCatalogEntry | null;
}

/**
 * Esc stops the running turn from the composer (owner, 2026-11-24) unless
 * something else on screen owns Esc: an open dialog, menu, list or popup
 * (their own document listeners close them), the turn-undo sheet, or a
 * maximized panel (ChatTab restores it on Esc).
 */
function escapeOwnedElsewhere(): boolean {
  const owners = document.querySelectorAll(
    '[role="dialog"], [role="menu"], [role="listbox"], [aria-haspopup][aria-expanded="true"], '
    + '[data-testid="turn-undo-sheet"], .panel-max-window',
  );
  // A closed drawer stays mounted, hidden (aria-hidden / inert): not an owner.
  return [...owners].some((el) => !el.closest('[aria-hidden="true"], [inert]'));
}

/**
 * Spec 195 item 7: the worker tiles under the session title repeat each
 * card's dot and stop, so they hide while cards are in the conversation.
 * One switch, to revert this alone.
 */
const HIDE_WORKER_TILES_WITH_CARDS = true;

interface StreamState {
  text: string;
  source: string;
  isComplete: boolean;
}

// Pending-input queue (spec 006 · v3). One queued message awaiting its turn,
// keyed by nonce in `pendingInputs`: THIS session's own turn is mid-flight
// (`session._queue`); the agent takes it at its next step. (Spec 155 removed
// the cross-session kind: no session waits for another.) The composer queue
// block (2026-11-24) shows it as a row; its bubble is drawn when
// `chat.pending_dispatched` says it went out. `content` is the wire string
// (the bubble's), `text` the row's words and what edit (↑, the pencil) puts
// back, `chips` its attachments, restored as chips on edit.
interface PendingInputEntry {
  sessionId: string;
  content: string;
  text: string;
  chips: QueuedChip[];
  editable: boolean;
  queuedAt: string;
  skill?: SkillCatalogEntry | null;
}

/** One attachment on the /inject wire. `local` (desktop app): `path` is the
 *  user's own file, by absolute path. */
type AttachmentPayload = { path: string; mime: string; size: number; local?: boolean };

/**
 * Spec 156 §5.7 / 164 §2.3 — one message for `postToInject`, already composed.
 * The composer builds it from its draft; the editor from the rebuilt wire
 * content (no `attachments` / `skill` fields: the blocks are in `content`);
 * Retry from the latest input's raw content and its original target.
 */
interface InjectSend {
  /** Where it goes. Explicit: an edit sends into the fork it just opened. */
  sessionId: string | undefined;
  /** The /inject `content` field. */
  content: string;
  /** What the optimistic bubble carries: the wire string the echo will. */
  wireContent: string;
  target?: string;
  pinned: boolean;
  attachments?: AttachmentPayload[];
  skill?: { key: string; path: string };
  answering?: { id: string; words: string };
  /** The typed words, put back by edit if the message queues. */
  rawText: string;
  /** Its attachments as chips, restored by edit if the message queues. */
  chips?: QueuedChip[];
  /** The picked skill, restored by edit if the message queues. */
  skillEntry?: SkillCatalogEntry | null;
  /** Shown on the error line while sending (else the line is cleared). */
  notice?: string | null;
  /** The send failed (the editor puts its text back in the composer). */
  onFailed?: () => void;
}

type MessageRow = Extract<DisplayItem, { type: 'user_message' | 'agent_message' | 'sub_agent_message' }>;

const QUEUE_ITEM_RE = /^\[QUEUE ITEM/;

/** Spec 156 §5.1: a row 修改 / 重试 may act on — a human input persisted in
 *  this session's log. */
function isPersistedHumanInput(
  item: Extract<DisplayItem, { type: 'user_message' }>,
  optimistic: ReadonlySet<string>,
): boolean {
  if (item.isHistorical) return false;
  if (optimistic.has(item.timestamp)) return false;
  if (QUEUE_ITEM_RE.test(item.content)) return false;
  // Spec 188: a routed task's framing is Orbital's, not the user's words.
  if (parseSourceTask(stripTurnNote(item.content)) !== null) return false;
  return parseTriggerHeader(item.content).trigger === null;
}

interface PendingApproval {
  what: string;
  tool_name: string;
  tool_call_id: string;
  tool_args: Record<string, unknown>;
  recent_activity: ChatMessageType[];
  reasoning?: string;
  resolved?: 'approved' | 'denied';
  /** Spec 117 D4/D5: `"unsandboxed_run"` with the blocked command and the
   *  detected denial; the card offers once / always / turn off safe mode. */
  kind?: string;
  command?: string;
  executable?: string;
  reason?: string;
  /** Spec 195: the asking agent's handle (a worker's approval), for its logo. */
  source?: string;
}

/** The asking worker of an approval: the payload's `source`, else the handle
 *  in the daemon's own "Sub-agent <handle> requests approval" line. */
function approvalSource(source: unknown, what: string | undefined): string | undefined {
  if (typeof source === 'string' && source && source !== 'management') return source;
  const m = (what ?? '').match(/^Sub-agent ([\w.-]+) requests approval/);
  return m ? m[1] : undefined;
}

export default function ChatView({ projectId, project, agentStatus, statusTick, agents, sessionId, initialAskReply, initialDraft, initialSkill, onDraftConsumed, onRefreshProject, onOpenSession, onUndoTurn, checksOnView = true, onUserSend, onSessionMoved, onOpenWorker, openWorker = null }: ChatViewProps) {
  const t = useT();
  const { locale } = useLocale();
  // Chrome-only translator for the annotation chip's fallback label (the
  // rest of each summary is dynamic content and stays untranslated).
  const chromeTr = useMemo(() => (key: StringKey) => translate(locale, key), [locale]);
  // The WS effect's handler closures don't re-subscribe on locale change —
  // read the live locale through a ref (same staleness fix as sessionIdRef).
  const localeRef = useRef(locale);
  // Spec 100: worker thinking waiting for the next frame, per worker handle.
  const workerReasoningRef = useRef(new Map<string, { text: string; route: WorkerEventRoute }>());
  // Spec 195: each handle's last `sub_agent.dispatched` initiator, until its
  // turn ends — how an event from a daemon that does not stamp its own
  // initiator finds its card.
  const liveInitiatorRef = useRef(new Map<string, string>());
  const workerFrameRef = useRef<number | null>(null);
  // Spec 171 A7: the manager's reply text waiting for the next frame. Lives
  // as long as the component (not the WS effect, which can re-run mid-stream);
  // the session-reset paths discard it.
  const streamBufferRef = useRef<FrameTextBuffer<string, string> | null>(null);
  useEffect(() => { localeRef.current = locale; }, [locale]);
  // One live sub-agent status pipeline for this session (spec 099): the
  // status bar's chips and the pin control's running dot both read it.
  const subAgentStatus = useSubAgentStatus(projectId, sessionId);
  const runningSubAgents = useMemo(
    // The pin dot pulses for a worker producing output; one waiting on the
    // user's approval is not producing.
    () => subAgentStatus.agents
      .filter((a) => a.status === 'running' && a.awaiting_approval !== true)
      .map((a) => a.handle),
    [subAgentStatus.agents],
  );
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
  // Classified agent/provider error for the viewed session (credential-error
  // surfacing): fed by agent.status error broadcasts and run-status
  // last_terminal_event hydration; rendered as AgentErrorNotice.
  const [agentError, setAgentError] = useState<{
    code?: string;
    message?: string;
    agent?: string;
    agent_name?: string;
  } | null>(null);
  const [loadedOffset, setLoadedOffset] = useState(0);
  const [stream, setStream] = useState<StreamState | null>(null);
  // The live text without its [STATUS: …] (a tag still arriving included),
  // so the raw tag never flashes; nothing to show means no bubble yet.
  const streamShown = useMemo(
    () => (stream ? stripStatusTags(stream.text, { streaming: true }) : ''),
    [stream],
  );
  const [approvals, setApprovals] = useState<Map<string, PendingApproval>>(new Map());
  const [expandedCapsules, setExpandedCapsules] = useState<Set<string>>(new Set());
  const toggleCapsule = useCallback((capsuleId: string) => {
    setExpandedCapsules((prev) => {
      const next = new Set(prev);
      if (next.has(capsuleId)) next.delete(capsuleId);
      else next.add(capsuleId);
      return next;
    });
  }, []);
  // Spec 065 §5.9: a path chip on the changed-files line opens that file
  // with its history column on the turn's version.
  const handleOpenHistory = useCallback(
    (path: string, v: number) => {
      requestHistoryFocus(projectId, path, v);
      onOpenPath?.(path);
    },
    [projectId, onOpenPath],
  );
  // Seeded from initialDraft (Workbench prefill doorway) so it's visible on
  // the very first paint — useState's initializer runs once, at mount, so
  // this cannot re-apply on a later re-render with a changed prop.
  const [inputText, setInputText] = useState(initialDraft ?? '');
  // The pinned-send "●●●" bubble, stamped with the session that sent it so a
  // switch while the inject is in flight doesn't carry it along (spec 096).
  const [subAgentLoading, setSubAgentLoading] = useState<
    { sessionId: string | undefined; target: string } | null
  >(null);
  // Spec 074 — the composer "Talking to" pin. Backend truth comes from the
  // session list (`pinned_target` on the entry); `localPin` is the
  // optimistic overlay for the VIEWED session so the dropdown responds
  // immediately and keeps working for a brand-new session whose JSONL has
  // not materialized yet (the PATCH 404s until the first message lands —
  // handleSend re-persists the pin after a successful pinned send).
  const { sessions: pinSessions, pinAgent } = useSessions(projectId);
  // Spec 154 §5.1: viewing a session checks its reply, on every device. The
  // project column's session list does the same on desktop; this covers the
  // phone, which has no session list (markChecked dedupes the two).
  useEffect(() => {
    if (!sessionId || !checksOnView) return;
    const row = pinSessions.find((s) => s.session_id === sessionId);
    if (!row?.last_reply_at) return;
    const key = seenKey(row);
    const checked = effectiveChecked(projectId, key, row.checked_reply_at);
    if (!checked || Date.parse(row.last_reply_at) > Date.parse(checked)) {
      void markChecked(projectId, key, row.last_reply_at);
    }
  }, [projectId, sessionId, pinSessions, checksOnView]);
  // Keyed by the viewed session; `sessionId: undefined` is the project with no
  // session minted yet (the choice is the project's sticky pin then).
  const [localPin, setLocalPin] = useState<{ sessionId: string | undefined; slug: string | null } | null>(null);
  // The pin PATCH write-locks the session JSONL on the backend for a
  // load+meta-rewrite burst; an inject landing inside that burst gets
  // rejected (2026-08-31 bug: pick agent → send immediately → "Failed to
  // send message"). Track the in-flight PATCH so handleSend can serialize
  // behind it; outcome is irrelevant (failure keeps the optimistic pin).
  const pinPatchInFlightRef = useRef<Promise<void> | null>(null);
  // Spec 138: the session last opened from a left-off card, until its first
  // send — a pin change in between is the "switched agent after opening" signal.
  const leftOffOpenedRef = useRef<string | null>(null);
  const persistPin = useCallback((sid: string, slug: string | null) => {
    const done: Promise<void> = pinAgent(sid, slug)
      .then(() => undefined, () => undefined)
      .finally(() => {
        if (pinPatchInFlightRef.current === done) {
          pinPatchInFlightRef.current = null;
        }
        // The PATCH also moved the project's sticky pin (last choice wins),
        // which seeds the next new session — re-read the project row.
        onRefreshProject?.(projectId);
      });
    pinPatchInFlightRef.current = done;
  }, [pinAgent, onRefreshProject, projectId]);
  // The viewed session's list entry (absent for a minted id whose JSONL has
  // not materialized yet, and while there is no session at all).
  const viewedEntry = useMemo(() => {
    if (sessionId === undefined) return undefined;
    return pinSessions.find(
      (s) => s.session_id === sessionId || s.session_uuid === sessionId,
    );
  }, [pinSessions, sessionId]);
  // The session's own pin record; undefined = no record yet.
  const recordedPin = viewedEntry?.pinned_target;
  // Keyless onboarding (contract §2/§3): a session with no pin record starts
  // on the project's effective pin — its sticky choice, or with no key the
  // first available worker. An older daemon sends neither (→ Orbital).
  const projectPin = project.effective_pinned_target ?? null;
  const managerReady = project.manager_ready !== false;
  const pinnedTarget =
    localPin && localPin.sessionId === sessionId
      ? localPin.slug
      : recordedPin !== undefined
        ? recordedPin
        : projectPin;
  // Before the first session exists a pick is the project's sticky pin
  // (handlePinChange) and the local pin steers the first send, so the
  // control is always live — a new project's first message can go to a
  // worker without 新建会话 first.
  // Spec 171 A8: a stable handler, so the memoised pin control skips the
  // renders a keystroke or a stream frame cause.
  const handlePinChange = useCallback((slug: string | null) => {
    if (leftOffOpenedRef.current !== null && leftOffOpenedRef.current === sessionId && slug !== pinnedTarget) {
      leftOffOpenedRef.current = null;
      sendTelemetry('agent_switch_after_open');
    }
    setLocalPin({ sessionId, slug });
    if (sessionId === undefined) {
      // No session yet: the choice is the project's sticky pin, which the
      // first send's session then starts on. Best effort — an older daemon
      // ignores or rejects the field and the local pin still steers that
      // first send.
      void api(`/api/v2/projects/${encodeURIComponent(projectId)}`, {
        method: 'PUT',
        body: JSON.stringify({ default_pinned_target: slug }),
      })
        .then(() => onRefreshProject?.(projectId))
        .catch(() => {});
      return;
    }
    persistPin(sessionId, slug);
  }, [sessionId, pinnedTarget, projectId, onRefreshProject, persistPin]);
  const pinnedAgentName = pinnedTarget
    ? (agents.find((a) => a.slug === pinnedTarget)?.name ?? pinnedTarget)
    : null;
  // Spec 149: the pin menu's model line per worker (this conversation) and
  // Orbital's credential card (this project — the same PUT Project Settings
  // makes). Loaded when the menu opens.
  const convModels = useConversationModels(projectId, sessionId);
  const [cardPick, setCardPick] = useState<{ projectId: string; cardId: string | null } | null>(null);
  const projectCardId = cardPick && cardPick.projectId === projectId
    ? cardPick.cardId : (project.card_id ?? null);
  const pinModels = useMemo(() => ({
    workers: convModels.workers,
    onWorkerChange: convModels.setWorkerParams,
    onMenuOpen: convModels.refresh,
    manager: convModels.cards.length > 0 ? {
      cards: convModels.cards.map((c) => ({ id: c.id, name: c.name })),
      cardId: projectCardId,
      defaultName: resolveCard(convModels.cards, null, convModels.defaultCardId)?.name ?? null,
      onChange: async (cardId: string | null) => {
        const before = projectCardId;
        setCardPick({ projectId, cardId });
        try {
          await api(`/api/v2/projects/${encodeURIComponent(projectId)}`, {
            method: 'PUT',
            body: JSON.stringify({ card_id: cardId }),
          });
          onRefreshProject?.(projectId);
          return true;
        } catch {
          setCardPick({ projectId, cardId: before });
          return false;
        }
      },
    } : undefined,
  }), [convModels.workers, convModels.setWorkerParams, convModels.refresh, convModels.cards,
    convModels.defaultCardId, projectCardId, projectId, onRefreshProject]);
  // Session import: an imported row whose transcript the first /chat converts.
  const viewedIsUnimportedImport =
    viewedEntry?.origin === 'imported' && viewedEntry.imported?.materialized === false;
  const [showThinking, setShowThinking] = useState(false);
  // Spec 132: the `/` skill picker (shared with the Workbench composer, see
  // SlashMenu.tsx). Spec 166: with a worker pinned the list also holds that
  // worker's own commands, so it is read per pin; null = Orbital. Spec 137:
  // the built-in `/automate` sits in the same flat list, not offered keyless
  // (no manager to run it); typed text still goes to the pin verbatim.
  const managerMissing = project?.manager_ready === false;
  const slash = useSlashPicker({
    projectId, worker: pinnedTarget ?? null, managerMissing, text: inputText,
  });
  const builtins = slash.builtins;
  // The pin menu offers only the workers this project keeps on.
  const pinAgents = useMemo(() => projectAgents(agents, project), [agents, project]);
  // Spec 197: who the usage-limit row's one button switches to — only while
  // this conversation still talks to the limited agent and it is limited now.
  const agentAvailability = useAgentAvailability();
  const limitSwitchFor = useCallback((handle: string) => {
    if (pinnedTarget !== handle) return null;
    const now = Date.now();
    if (summarizeQuota(agentAvailability[handle]?.quota, now).kind !== 'limited') return null;
    const slug = switchTarget(pinAgents.map((a) => a.slug), handle, agentAvailability, now);
    if (!slug) return null;
    return { slug, name: pinAgents.find((a) => a.slug === slug)?.name ?? slug };
  }, [pinnedTarget, agentAvailability, pinAgents]);
  // One skill per message, scoped to the session it was picked in (a chip
  // must not follow the user into another conversation). Seeded from
  // `initialSkill` (the Tasks page's "New automation" doorway).
  const [pickedSkillState, setPickedSkillState] = useState<
    { sessionId: string | undefined; skill: SkillCatalogEntry } | null
  >(() => {
    const seeded = initialSkill ? builtins.find((b) => b.name === initialSkill) : undefined;
    return seeded ? { sessionId, skill: seeded } : null;
  });
  const pickedSkill =
    pickedSkillState && pickedSkillState.sessionId === sessionId ? pickedSkillState.skill : null;

  const [injectError, setInjectError] = useState<string | null>(null);
  // Pending-input queue (spec 006). A NONCE-KEYED OVERLAY — NOT a flag in
  // `items` (the historyItems memo reseed would stomp that). Each entry is a
  // message the user sent to a session while that session's own turn was
  // running (`queued_same_session`); it drains when the turn ends. The map is
  // PROJECT-scoped; the waiting affordance renders only for entries whose
  // `sessionId` is the viewed session. `content`/`timestamp` identify the kept
  // optimistic bubble so cancel/dispatch can locate it. Cleared per-nonce on
  // `chat.pending_dispatched` (sole clear trigger) / `chat.pending_cancelled`,
  // and rebuilt from GET /pending on session load + WS reconnect.
  const [pendingInputs, setPendingInputs] = useState<Map<string, PendingInputEntry>>(new Map());
  // Spec 195: pinned messages waiting behind a worker's current task; the
  // held ones are queue-block rows, rebuilt from GET /pending after a reload.
  // Cleared by the dispatch that sends it (`sub_agent.dispatched`), its drop
  // (`sub_agent.queue_dropped`) or its removal (`chat.pending_cancelled`).
  const [queuedSends, setQueuedSends] = useState<QueuedSend[]>([]);
  const clearQueuedClock = useCallback((dispatchId: string) => {
    setQueuedSends((prev) => (prev.some((q) => q.dispatchId === dispatchId)
      ? prev.filter((q) => q.dispatchId !== dispatchId)
      : prev));
  }, []);
  // Cards opened in place (no docked panel: the Workbench reading pane).
  const [inPlaceCards, setInPlaceCards] = useState<Set<string>>(new Set());
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
    openPicker,
    restoreAttachments,
  } = useAttachments(projectId, { onError: setInjectError, localPaths: true });
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

  // The composer's inline capsules (2026-10-28): the picked skill and the
  // staged annotations sit at the start of the input's first line, and the
  // typed text flows after them. The textarea stays a plain textarea (IME,
  // paste, undo and every key handler depend on it): the capsule row is
  // positioned over its top-left corner and the textarea's FIRST line is
  // indented by the row's measured width (`text-indent` only ever indents the
  // first line). The row carries its own trailing gap, so indent == width.
  const hasCapsule = pickedSkill !== null || annotations.length > 0;
  const capsuleRowRef = useRef<HTMLDivElement>(null);
  const [capsuleWidth, setCapsuleWidth] = useState(0);
  useLayoutEffect(() => {
    const el = capsuleRowRef.current;
    if (!hasCapsule || !el) {
      setCapsuleWidth(0);
      return;
    }
    const measure = () => setCapsuleWidth(Math.ceil(el.getBoundingClientRect().width));
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasCapsule, pickedSkill, annotations.length]);
  // The annotation list opens as a popover anchored to its capsule; a click
  // outside it or Esc closes it.
  const annotationPopoverRef = useRef<HTMLDivElement>(null);
  // Anchored to the capsule, but never past the viewport: on a phone the
  // capsule sits mid-input and the list is wider than what is left to its
  // right, so it slides left just enough to stay 12px inside the screen.
  const annotationListBoxRef = useRef<HTMLDivElement>(null);
  const [popoverShift, setPopoverShift] = useState(0);
  useLayoutEffect(() => {
    const el = annotationListBoxRef.current;
    if (!annotationsOpen || !el) {
      setPopoverShift(0);
      return;
    }
    const anchorLeft = el.parentElement?.getBoundingClientRect().left ?? 0;
    const width = el.getBoundingClientRect().width;
    const overflow = anchorLeft + width - (window.innerWidth - 12);
    setPopoverShift(overflow > 0 ? Math.min(overflow, Math.max(0, anchorLeft - 12)) : 0);
  }, [annotationsOpen]);
  useEffect(() => {
    if (!annotationsOpen) return;
    const onDown = (e: MouseEvent) => {
      const root = annotationPopoverRef.current;
      if (root && !e.composedPath().includes(root)) setAnnotationsOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setAnnotationsOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [annotationsOpen]);
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
  const { injectMessage, cancelMessage, newSession, cancelPendingInput, getPending, sendPendingNow } = useAgent();

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
  // over the value at registration time otherwise). The inject-target paths
  // read this ref to stay current.
  const sessionIdRef = useRef<string | undefined>(sessionId);
  sessionIdRef.current = sessionId;
  // Latest pending-input map, readable synchronously inside WS handlers and the
  // nonce-eviction sweep (which must EXEMPT pending nonces — spec §3h F4 — so a
  // long wait can't evict the nonce before the dispatch echo dedups against the
  // kept optimistic bubble). Never read a value mutated inside a setState
  // updater (React 19 batching) — read this committed ref instead.
  const pendingInputsRef = useRef<Map<string, PendingInputEntry>>(pendingInputs);
  pendingInputsRef.current = pendingInputs;
  const queuedSendsRef = useRef<QueuedSend[]>(queuedSends);
  queuedSendsRef.current = queuedSends;
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
  // Spec 185: the viewed session's workers, read by the same two gates as
  // `agentStatusRef`. The hook reports only the viewed session's list (empty
  // until it is known after a switch), so a worker of another session never
  // counts. A pinned worker's turn keeps the manager idle, so without this a
  // mid-turn send's reseed wiped the worker's whole running turn.
  const subAgentsRef = useRef(subAgentStatus.agents);
  subAgentsRef.current = subAgentStatus.agents;
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

  // Spec 155: sessions run in parallel and `agentStatus` is the VIEWED
  // session's own status (App keeps one per session), so "the viewed session
  // is the one running" is simply "a session is viewed".
  //
  // FE-3 / FE-A1: the trailing open capsule is only genuinely "running" while
  // the viewed session runs. Applied at RENDER time below (the agent_run
  // case), NOT fed into the transform, so a status flip never invalidates the
  // memo (FE-A1).
  const viewingSession = sessionId !== undefined;
  const isActivelyRunning = viewingSession && agentStatus === 'running';
  // Spec 065 §5.9: the open turn of the viewed session has not ended while
  // it is mid-run (waiting on a worker or an approval counts): its
  // changed-files line would still be growing.
  const turnInFlight =
    viewingSession &&
    (agentStatus === 'running' || agentStatus === 'waiting' || agentStatus === 'pending_approval');
  // Each turn's last capsule → the turn's span (user row to next user row).
  const turnSpans = useMemo(() => lastCapsuleOfTurns(items), [items]);
  // Spec 195: the conversation's dispatch cards, for the docked panel.
  const dispatchCards = useMemo(
    () => items.filter((it): it is DispatchCardItem => it.type === 'dispatch_card'),
    [items],
  );
  useEffect(() => {
    publishDispatchCards(sessionMessagesKey(projectId, sessionId), dispatchCards);
  }, [projectId, sessionId, dispatchCards]);
  const showWorkerTiles = !(HIDE_WORKER_TILES_WITH_CARDS && hasDispatchCards(dispatchCards));
  // A card's amber dot: its agent waits on the user's approval.
  const needsYouHandles = useMemo(() => {
    const out = new Set<string>();
    for (const a of subAgentStatus.agents) if (a.awaiting_approval === true) out.add(a.handle);
    for (const a of approvals.values()) if (!a.resolved && a.source) out.add(a.source);
    return out;
  }, [subAgentStatus.agents, approvals]);
  const agentNameOf = useCallback(
    (handle: string) => agents.find((a) => a.slug === handle)?.name ?? handle,
    [agents],
  );
  const queuedHintFor = (item: Extract<DisplayItem, { type: 'user_message' }>): string | undefined => {
    if (!item.target || queuedSends.length === 0) return undefined;
    const q = queuedSends.find(
      (x) => x.sessionId === sessionId && x.handle === item.target && x.content === item.content,
    );
    return q ? t('chat.queuedBehind', { agent: agentNameOf(q.handle) }) : undefined;
  };
  const handleOpenCard = useCallback((card: DispatchCardItem) => {
    if (onOpenWorker) {
      onOpenWorker({ kind: 'card', cardId: card.card_id, dispatchId: card.dispatch_id, handle: card.handle });
      return;
    }
    setInPlaceCards((prev) => {
      const next = new Set(prev);
      if (next.has(card.card_id)) next.delete(card.card_id);
      else next.add(card.card_id);
      return next;
    });
  }, [onOpenWorker]);
  // Spec 197: only the pinned agent's latest usage-limit row carries the
  // switch button. Batch 1124: a later limit row of another agent (an
  // Orbital dispatch still running when the user pinned, spec 195) must not
  // take it away, so the search is for the pinned agent's rows only.
  const lastLimitIndex = useMemo(() => {
    if (!pinnedTarget) return -1;
    for (let i = items.length - 1; i >= 0; i--) {
      const it = items[i];
      if (it.type === 'sub_agent_activity' && it.limit && it.handle === pinnedTarget) return i;
    }
    return -1;
  }, [items, pinnedTarget]);
  // Spec 096: `showThinking` follows the viewed session's status (this
  // instance outlives session switches). Not isActivelyRunning: the spinner
  // stays through running → waiting on purpose.
  const thinkingVisible = showThinking && viewingSession;
  const subAgentLoadingTarget =
    subAgentLoading && subAgentLoading.sessionId === sessionId
      ? subAgentLoading.target
      : null;

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
    //
    // Spec 185: "in flight" is the shared definition: the manager running or
    // waiting, OR a worker of the viewed session running (a pinned worker's
    // turn keeps the manager idle). The worker's catch-up is the terminal-
    // event refetch (handleSubAgentTerminal), which lands with it idle.
    const viewedIsHolder = sessionIdRef.current !== undefined;
    const turnInFlight = isTurnInFlight(agentStatusRef.current, subAgentsRef.current);
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
   *
   * Kind-aware (spec 131 A): only an UNTAGGED row is ever recovered. Rows
   * tagged `_kind` (the daemon's memory record / tidy turns) are rendered
   * by the persisted transform alone — they stream nothing live by design,
   * and recovering their closing text here is exactly what painted a
   * second reply after the memory capsule. The merge itself gates on
   * `_kind` too; the pick below is the belt that still finds the real reply
   * when the record turn has already appended behind it.
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
        `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=${REST_FALLBACK_LIMIT}${sessionParam}`,
      )
        .then((messages) => {
          if (!messages || messages.length === 0) return;
          // The last row without a `_kind` tag — the newest row the live
          // path was ever going to show. A window of only tagged rows
          // recovers nothing, never a wrong row.
          let latest: ChatMessageRow | undefined;
          for (let i = messages.length - 1; i >= 0; i--) {
            if (!messages[i]._kind) {
              latest = messages[i];
              break;
            }
          }
          if (!latest || latest.role !== 'assistant') return;
          // Gated on the tag as well (spec 131 A4): a record-turn row with
          // reasoning_content must never spawn a stray thinking capsule.
          const restReasoning = latest._kind ? '' : (latest.reasoning_content ?? '').trim();
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

  // Spec 171 A7: the manager's visible reply, joined per frame and committed
  // once (one render, one markdown parse) instead of once per delta. Keyed by
  // session; a commit for a session no longer viewed is dropped. Unmount
  // cancels the pending frame.
  const scrollToBottomRef = useRef(scrollToBottom);
  useEffect(() => { scrollToBottomRef.current = scrollToBottom; }, [scrollToBottom]);
  useEffect(() => {
    const buffer = createFrameTextBuffer<string, string>(
      (sid, text, source) => {
        if (sid !== sessionIdRef.current) return;
        setShowThinking(false);
        setStream((prev) => ({
          text: (prev?.text ?? '') + text,
          source,
          isComplete: false,
        }));
        scrollToBottomRef.current();
      },
      requestFrame,
      cancelFrame,
    );
    streamBufferRef.current = buffer;
    return () => {
      buffer.discard();
      if (streamBufferRef.current === buffer) streamBufferRef.current = null;
    };
  }, []);

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
      kind?: string;
      command?: string;
      executable?: string;
      reason?: string;
      source?: string;
    }>(`/api/v2/agents/${encodeURIComponent(projectId)}/pending-approval${qs}`)
      .then((result) => {
        if (!result.pending || !result.tool_call_id) return;
        // The viewed session changed while this request was in flight: the
        // answer belongs to the session we left, not the one on screen (the
        // switch already re-fetched for the new one).
        if (sessionIdRef.current !== viewed) return;
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
            kind: result.kind,
            command: result.command,
            executable: result.executable,
            reason: result.reason,
            source: approvalSource(result.source, result.what),
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
   * The viewed session's run-status (spec 155: per session). Hydrates a
   * classified error after reload and hands the session's status to App on a
   * fresh view, so a session running when the page opened shows as running.
   * Fetched on mount and on agentStatus/sessionId change.
   */
  const runStatusAbortRef = useRef<AbortController | null>(null);
  const fetchRunStatus = useCallback(() => {
    // Bug #48 (fix C): rapid session switching re-fires this on every switch.
    // Abort the superseded request so responses can't pile up or resolve out
    // of order (a slow older response would overwrite a newer value).
    runStatusAbortRef.current?.abort();
    if (!sessionId) return;
    const controller = new AbortController();
    runStatusAbortRef.current = controller;
    const qs = `?session_id=${encodeURIComponent(sessionId)}`;
    api<{
      project_id: string;
      status: string;
      last_terminal_event?: {
        type: string;
        details?: string | null;
        error_code?: string;
      } | null;
    }>(`/api/v2/agents/${encodeURIComponent(projectId)}/run-status${qs}`, {
      signal: controller.signal,
    })
      .then((result) => {
        if (result.status !== agentStatusRef.current) {
          window.dispatchEvent(
            new CustomEvent('agent-status-override', {
              detail: { project_id: projectId, session_id: sessionId, status: result.status },
            }),
          );
        }
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
        // best effort — leave the prior status in place
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

  // The composer queue block (2026-11-24). A queued message's bubble is drawn
  // only when it is really sent: `chat.pending_dispatched` (this chat's own
  // agent took it) or `sub_agent.dispatched` (an outside agent did). Drawn the
  // way a send draws its optimistic bubble, so the history refetch replaces it.
  const paintUserBubble = useCallback((content: string, target?: string) => {
    const timestamp = new Date().toISOString();
    optimisticTsRef.current.add(timestamp);
    setItems((prev) => [
      ...finalizeLiveCapsule(prev, 'completed'),
      { type: 'user_message', content, timestamp, ...(target && { target }) },
    ]);
    setRawMessages((prev) => [
      ...prev,
      { role: 'user', content, source: 'user', timestamp, ...(target && { target }) },
    ]);
  }, []);

  const dropOwnQueued = useCallback((nonce: string) => {
    setPendingInputs((prev) => {
      if (!prev.has(nonce)) return prev;
      const next = new Map(prev);
      next.delete(nonce);
      return next;
    });
  }, []);

  // A queue row's text and chips go back into the composer (edit). A draft
  // already there keeps its place; the message follows on a new line.
  const restoreIntoComposer = useCallback(
    (text: string, chips: readonly QueuedChip[], skill: SkillCatalogEntry | null | undefined) => {
      setInputText((prev) => mergeIntoDraft(prev, text));
      if (chips.length > 0) restoreAttachments(chips);
      if (skill) setPickedSkillState({ sessionId, skill });
      textareaRef.current?.focus();
    },
    [restoreAttachments, sessionId],
  );

  // Server-authoritative, as recall always was (§12 R2): the message is
  // loaded back only when the daemon confirms it removed it while it was
  // still waiting (`removed: true`). `removed: false` means it already went
  // out: nothing is loaded (no double send) and its row just leaves.
  const [queueBusyKey, setQueueBusyKey] = useState<string | null>(null);
  const takeQueued = useCallback(
    async (row: QueueRow, mode: 'edit' | 'delete') => {
      if (sessionId === undefined) return;
      setQueueBusyKey(row.key);
      try {
        if (row.kind === 'own') {
          const nonce = row.key.slice(2);
          const entry = pendingInputsRef.current.get(nonce);
          const res = await cancelPendingInput(projectId, entry?.sessionId ?? sessionId, nonce);
          if (res?.removed === true) {
            localNoncesRef.current.delete(nonce);
            if (mode === 'edit' && entry) restoreIntoComposer(entry.text, entry.chips, entry.skill);
          }
          if (res?.removed === true || res?.removed === false) dropOwnQueued(nonce);
        } else {
          const dispatchId = row.key.slice(2);
          const q = queuedSendsRef.current.find((x) => x.dispatchId === dispatchId);
          const res = await cancelPendingInput(projectId, q?.sessionId ?? sessionId, undefined, dispatchId);
          if (res?.removed === true) {
            if (q?.nonce) localNoncesRef.current.delete(q.nonce);
            if (mode === 'edit' && q) restoreIntoComposer(q.text, q.chips, q.skill);
          }
          if (res?.removed === true || res?.removed === false) clearQueuedClock(dispatchId);
        }
      } catch {
        setInjectError(t('pending.cancelError'));
      } finally {
        setQueueBusyKey((cur) => (cur === row.key ? null : cur));
      }
    },
    // t is unstable; the error copy is read at call time.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [projectId, sessionId, cancelPendingInput, restoreIntoComposer, dropOwnQueued, clearQueuedClock],
  );

  // Send now: the daemon stops what is running and sends this one next. The
  // row stays until the dispatch event says it went out; `sent: false` means
  // it had already gone (or was removed elsewhere), so its row just leaves.
  const sendQueuedNow = useCallback(
    async (row: QueueRow) => {
      if (sessionId === undefined) return;
      setQueueBusyKey(row.key);
      try {
        const id = row.key.slice(2);
        const res = row.kind === 'own'
          ? await sendPendingNow(projectId, pendingInputsRef.current.get(id)?.sessionId ?? sessionId, { nonce: id })
          : await sendPendingNow(projectId, queuedSendsRef.current.find((x) => x.dispatchId === id)?.sessionId ?? sessionId, { dispatchId: id });
        if (res?.sent === false) {
          if (row.kind === 'own') dropOwnQueued(id);
          else clearQueuedClock(id);
        }
      } catch {
        setInjectError(t('pending.cancelError'));
      } finally {
        setQueueBusyKey((cur) => (cur === row.key ? null : cur));
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [projectId, sessionId, sendPendingNow, dropOwnQueued, clearQueuedClock],
  );

  // Pending-input queue (spec 006 §3h). Reconcile the queue block from the
  // server (GET /pending) on session load + WS reconnect: both kinds, rebuilt
  // from server truth (an entry this client already holds keeps its chips,
  // which may still carry the picked File for a thumbnail). Nothing is drawn
  // in the chat: a queued message is not there until it is sent.
  const reconcilePending = useCallback(() => {
    getPending(projectId)
      .then((res) => {
        if (!res) return;
        const all = res.pending ?? [];
        const own: Array<[string, PendingInputEntry]> = [];
        const held: QueuedSend[] = [];
        for (const p of all) {
          if (p.kind === 'worker') {
            if (!p.dispatch_id || !p.handle) continue;
            const existing = queuedSendsRef.current.find((q) => q.dispatchId === p.dispatch_id);
            if (existing) {
              held.push({ ...existing, held: true });
            } else {
              const parsed = fromPendingWire(p);
              held.push({
                dispatchId: p.dispatch_id,
                handle: p.handle,
                sessionId: p.session_id,
                content: p.content,
                held: true,
                nonce: p.nonce ?? undefined,
                text: parsed.text,
                chips: parsed.chips,
                editable: parsed.editable,
                queuedAt: p.queued_at || new Date().toISOString(),
              });
            }
          } else if (p.kind === undefined || p.kind === 'same') {
            if (!p.nonce) continue;
            const existing = pendingInputsRef.current.get(p.nonce);
            if (existing) {
              own.push([p.nonce, existing]);
            } else {
              const parsed = fromPendingWire(p);
              own.push([p.nonce, {
                sessionId: p.session_id,
                content: p.content,
                text: parsed.text,
                chips: parsed.chips,
                editable: parsed.editable,
                queuedAt: p.queued_at || new Date(Date.now() - 1000 + p.position).toISOString(),
              }]);
            }
          } else {
            continue;
          }
          // Keep the nonce alive so the eventual dispatch echo dedups against
          // the bubble drawn at dispatch (don't clobber a received-timestamp).
          const n = p.nonce;
          if (n && !localNoncesRef.current.has(n)) localNoncesRef.current.set(n, 0);
        }
        setPendingInputs(() => new Map(own));
        // Held worker rows come from the server; a not-held one (a bubble
        // with the clock) is this client's own memory and stays.
        setQueuedSends((prev) => [...prev.filter((q) => !q.held), ...held]);
      })
      .catch(() => {});
  }, [projectId, getPending]);

  // Keep the viewed session's status fresh: on mount/project change, on every
  // agentStatus transition, and whenever the viewed session changes.
  useEffect(() => {
    fetchRunStatus();
  }, [fetchRunStatus, agentStatus, statusTick, sessionId]);

  // A session/project switch shows a different conversation — drop the
  // previous session's error notice (hydration re-fills it if it applies).
  useEffect(() => {
    setAgentError(null);
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
      streamBufferRef.current?.discard();
      setStream(null);
      setApprovals(new Map());
      setTotalMessages(0);
      setLoadedOffset(0);
      setLoading(false);
      setHistoryFor(undefined);
      return;
    }

    const sessionParam = `&session_id=${encodeURIComponent(sessionId)}`;
    const cacheKey = `${projectId}:${sessionId}`;

    // Spec 107 FE-1: this tab just minted this id ("+ new session" / `/new`).
    // Its history is empty by construction — the file does not exist until
    // the first message — so seed the empty state directly instead of
    // fetching /chat and sitting on a skeleton while the backend looks
    // through every session log for a session that cannot be there. Live
    // state is still cleared and the pending overlay still reconciled, as on
    // the fetch path. Consumed once: a later visit to the same id (by then
    // the file may exist) takes the normal fetch below.
    if (consumeFreshSession(projectId, sessionId)) {
      setRawMessages([]);
      setItems([]);
      streamBufferRef.current?.discard();
      setStream(null);
      setApprovals(new Map());
      setTotalMessages(0);
      setLoadedOffset(CHAT_PAGE_SIZE);
      setLoading(false);
      setHistoryFor(sessionId);
      reconcilePending();
      return;
    }

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
      // Committed refs, not effect-closure state: this effect is deliberately
      // NOT keyed on agentStatus (see the seed effect's dep-array note). Same
      // definition as the seed guard (spec 185).
      const turnInFlight = isTurnInFlight(agentStatusRef.current, subAgentsRef.current);
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
      streamBufferRef.current?.discard();
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
          // Spec 185 R1: this fetch answers a session open, so it is
          // authoritative over the cached page painted above. A worker's
          // status can arrive between that paint and this answer (the hook
          // reads it asynchronously), which would arm the seed guard and pin
          // the stale page for the rest of the turn. Forget whose `items`
          // these are so the seed effect replaces them. Only after a cached
          // paint: otherwise `items` are empty or another session's (both
          // seed anyway), and a same-session re-run keeps the guard as before.
          if (paintCached) itemsSessionRef.current = undefined;
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
        if (!cancelled) {
          setLoading(false);
          setHistoryFor(sessionId);
        }
      }
    }

    loadData();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [projectId, sessionId, project.workspace, reconcilePending]);

  const hasMore = totalMessages > loadedOffset;

  // ── Spec 156 / 164: the action row (copy · 重试 · 修改) ───────────────────
  // Optimistic bubbles by their client timestamp: not a row in the session
  // log yet (the history refetch replaces them with the persisted row), so
  // neither 修改 (the fork names the row by its timestamp) nor 重试.
  const optimisticTsRef = useRef<Set<string>>(new Set());
  // Which bubble is the inline editor, by session + row timestamp. Not state
  // inside the bubble: rows are keyed by list index, so a flag there would
  // follow the index into another session (and a fork's copied rows carry the
  // parent's timestamps).
  const [editingRow, setEditingRow] = useState<{ sessionId: string; ts: string } | null>(null);
  const editingTs = editingRow && editingRow.sessionId === sessionId ? editingRow.ts : null;
  const [editBusy, setEditBusy] = useState(false);
  const [actionToast, setActionToast] = useState<string | null>(null);
  useEffect(() => {
    if (!actionToast) return;
    const id = setTimeout(() => setActionToast(null), 4000);
    return () => clearTimeout(id);
  }, [actionToast]);
  // The session a history load last finished for: the edited message waits
  // for its fork's history before it is sent there (§5.7).
  const [historyFor, setHistoryFor] = useState<string | undefined>(undefined);

  // 修改 and 重试 hide while this session works: a turn in flight (waiting on
  // a worker or an approval counts), a worker running, a pinned send not yet
  // answered, a message queued behind the turn. Copy stays. Another session
  // running hides nothing.
  const queuedHere = useMemo(
    () => sessionId !== undefined && Array.from(pendingInputs.values()).some((e) => e.sessionId === sessionId),
    [pendingInputs, sessionId],
  );
  // The composer queue block's rows for the viewed session (both kinds).
  const viewedQueueRows = useMemo(
    () => queueRows(pendingInputs, queuedSends.filter((q) => q.held), sessionId),
    [pendingInputs, queuedSends, sessionId],
  );
  const sessionBusy = turnInFlight || runningSubAgents.length > 0 || subAgentLoadingTarget !== null || queuedHere;
  // Edit opens a fork (it navigates) and may offer 065's sheet (the parent
  // mounts it): both must be wired. The Workbench pane wires neither.
  const editWired = sessionId !== undefined && onOpenSession !== undefined && onUndoTurn !== undefined;
  // Spec 196 D1: 撤销这条消息之后的修改 under the user's own messages, by the
  // same gates as 修改 (idle, persisted human input), once the project's
  // history is watching and has recorded a change. Opens 065's sheet for the
  // message to now; no age or session limit, no rewind, no second dialog.
  const undoWired = sessionId !== undefined && onUndoTurn !== undefined;
  const historyStatus = useHistoryStatus(undoWired ? projectId : undefined);
  const undoSinceOk = undoWired && historyStatus?.status === 'watching' && historyVisible(projectId, historyStatus);

  // The latest turn: the last user row in the loaded items, when it is a
  // persisted human input (not optimistic or queued, not an automation run or
  // a queue item, not another session's history). 重试 sits once, under that
  // turn's last agent body, or under the input itself when the attempt left
  // no body (failed, stopped). An input not loaded yet (pagination) has no
  // row here, so there is no 重试 until it is.
  const retryPlan = useMemo(() => {
    let lastUser = -1;
    for (let i = items.length - 1; i >= 0; i--) {
      if (items[i].type === 'user_message') {
        lastUser = i;
        break;
      }
    }
    if (lastUser < 0) return null;
    const input = items[lastUser] as Extract<DisplayItem, { type: 'user_message' }>;
    if (!isPersistedHumanInput(input, optimisticTsRef.current)) return null;
    let anchor = lastUser;
    for (let i = items.length - 1; i > lastUser; i--) {
      const it = items[i];
      if (
        (it.type === 'agent_message' && !it.isHeaderOnly && it.content.length > 0) ||
        (it.type === 'sub_agent_message' && it.content.length > 0)
      ) {
        anchor = i;
        break;
      }
    }
    return { anchor, input };
  }, [items]);

  // Handlers read the latest render through a ref, so the callbacks handed
  // to the memoized rows stay stable.
  const actionStateRef = useRef({ sessionBusy, retryPlan, editBusy });
  actionStateRef.current = { sessionBusy, retryPlan, editBusy };
  const actionHandlersRef = useRef<{
    edit: (m: MessageRow) => void;
    cancel: () => void;
    send: (m: MessageRow, body: string) => void;
    retry: () => Promise<unknown>;
    undoSince: (m: MessageRow) => void;
  } | null>(null);
  const onEditMessage = useCallback((m: MessageRow) => actionHandlersRef.current?.edit(m), []);
  const onEditCancel = useCallback(() => actionHandlersRef.current?.cancel(), []);
  const onEditSend = useCallback((m: MessageRow, body: string) => actionHandlersRef.current?.send(m, body), []);
  const onRetryMessage = useCallback(() => actionHandlersRef.current?.retry(), []);
  const onUndoSinceMessage = useCallback((m: MessageRow) => actionHandlersRef.current?.undoSince(m), []);
  // An edit whose 065 sheet is open: cancelling the editor closes it too.
  const editSheetOpenRef = useRef(false);

  /** §5.7: fork at the edited row, open the fork, and leave the message to
   *  be sent there once its history has loaded (the effect below). */
  async function forkAndSend(fromSession: string, m: MessageRow, content: string, body: string) {
    editSheetOpenRef.current = false;
    setEditBusy(true);
    const parent = pinSessions.find((s) => s.session_id === fromSession || s.session_uuid === fromSession);
    const parentName =
      classifySessionName(parent?.name, parent?.origin).displayName ?? t('sessionHeading.new');
    const name = forkSessionName(parentName, m.timestamp, t);
    try {
      const res = await api<{ session_id: string; session_uuid?: string; target?: string | null }>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(fromSession)}/fork`,
        { method: 'POST', body: JSON.stringify({ at: m.timestamp, name }) },
      );
      const target = res.target ?? null;
      stashForkSend(projectId, res.session_id, { content, target, body });
      setEditingRow(null);
      setLocalPin({ sessionId: res.session_id, slug: target });
      onOpenSession?.(res.session_id);
    } catch (err) {
      // 409 session_running: a turn started since the row was drawn. The
      // editor stays open with the text; anything else says the send failed.
      setActionToast(
        err instanceof ApiError && err.status === 409 ? t('chat.edit.busy') : t('chat.injectError'),
      );
    } finally {
      setEditBusy(false);
    }
  }

  actionHandlersRef.current = {
    edit: (m) => {
      if (sessionId === undefined) return;
      setEditingRow({ sessionId, ts: m.timestamp });
    },
    cancel: () => {
      setEditingRow(null);
      if (editSheetOpenRef.current) {
        editSheetOpenRef.current = false;
        onUndoTurn?.(null);
      }
    },
    send: (m, body) => {
      void (async () => {
        const fromSession = sessionId;
        if (fromSession === undefined || actionStateRef.current.editBusy) return;
        if (actionStateRef.current.sessionBusy) {
          setActionToast(t('chat.edit.busy'));
          return;
        }
        const content = rebuildEditedContent(m.content, body);
        const go = () => void forkAndSend(fromSession, m, content, body);
        // §5.6: what this session changed in the workspace from the edited
        // message to now. Files to offer → 065's sheet first (restore, then
        // fork, then send); none, or history unavailable / still building →
        // straight to the fork. Memory files alone are never offered (the
        // sheet never pre-checks them, as on a capsule's line).
        const now = new Date().toISOString();
        let offer = false;
        setEditBusy(true);
        try {
          const span = await historyApi.turn(projectId, fromSession, m.timestamp, now);
          offer = span.status === 'watching' && span.files.some((f) => !isMemoryFile(f.path));
        } catch {
          offer = false;
        } finally {
          setEditBusy(false);
        }
        if (sessionIdRef.current !== fromSession) return;
        if (offer && onUndoTurn) {
          editSheetOpenRef.current = true;
          onUndoTurn({
            sessionId: fromSession,
            fromTs: m.timestamp,
            toTs: now,
            edit: {
              onProceed: go,
              onSendOnly: () => {
                onUndoTurn(null);
                go();
              },
            },
          });
          return;
        }
        go();
      })();
    },
    undoSince: (m) => {
      if (sessionId === undefined || actionStateRef.current.sessionBusy) return;
      onUndoTurn?.({ sessionId, fromTs: m.timestamp, toTs: null, since: true });
    },
    retry: async () => {
      // 164 §2.3: re-checked at click time; the latest input, raw, to its
      // original target, into this session. No fork, no name, no sheet; the
      // composer's draft and chips are not touched.
      const { sessionBusy: busy, retryPlan: plan } = actionStateRef.current;
      if (busy || !plan || sessionId === undefined) return;
      const input = plan.input;
      const parts = splitUserWire(input.content);
      await postToInject({
        sessionId,
        content: input.content,
        wireContent: input.content,
        target: input.target,
        pinned: input.target !== undefined,
        rawText: parts.body,
      });
    },
  };

  // §5.7: the fork is open and its history loaded — send the edited message
  // there through the same path as the composer, with the fork's pin. A
  // failure lands on this session's error line with the words back in the
  // composer, so nothing typed is lost.
  useEffect(() => {
    if (sessionId === undefined || historyFor !== sessionId) return;
    const pending = takeForkSend(projectId, sessionId);
    if (!pending) return;
    const target = pending.target ?? undefined;
    setLocalPin({ sessionId, slug: pending.target });
    void postToInject({
      sessionId,
      content: pending.content,
      wireContent: pending.content,
      target,
      pinned: target !== undefined,
      rawText: pending.body,
      onFailed: () => setInputText(pending.body),
    });
    // postToInject is a render-scoped function; this runs once per arrival.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [historyFor, sessionId, projectId]);

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
      // A chip seeded with the prefill (spec 137 `initialSkill`) was picked
      // before the session resolved; it belongs to this one.
      setPickedSkillState((prev) => (prev && prev.sessionId === undefined ? { ...prev, sessionId } : prev));
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
  // agentStatus is the VIEWED session's status (spec 155). Conversation
  // mutations here (finalize capsule, catch-up fetch, new_session reset)
  // apply while a session is viewed; live events route strictly by
  // session_id in the handlers below.
  useEffect(() => {
    const viewing = sessionIdRef.current !== undefined;
    if (agentStatus === 'idle' || agentStatus === 'error') {
      // Loop has actually exited — clear the in-flight cancel affordance.
      // Effect re-runs on agentStatus change, so this naturally coincides
      // with the agent.status:idle WS broadcast from _on_loop_done. (A
      // stopped/cancelled agent reports as `idle` — there is no separate
      // `stopped` run status.)
      setIsCancelling(false);
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
      // wasRunningRef stays behind the guard — it drives the viewed
      // session's idle catch-up.
      setShowThinking(true);
      if (!viewing) return;
      wasRunningRef.current = true;
    } else if (agentStatus === 'pending_approval') {
      // Fetch pending approval via REST in case the WS event was missed.
      // Covers status poll discovering a stale approval after reconnect.
      // Not gated on `viewing`: the fetch is scoped to the viewed session
      // (?session_id=), so it can only surface that session's own card.
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
      streamBufferRef.current?.discard();
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

  // On mount AND on every viewed-session change, check for pending approvals
  // via REST. Handles the case where ChatView was unmounted (tab switch to
  // files, Project Settings open) and remounted with a stale agentStatus that
  // doesn't trigger the status-change effect above. Also covers page reload —
  // the initial fetch runs before the first WS event arrives.
  //
  // Keyed on sessionId because the per-session history load above clears
  // `approvals` on every session change (so one session's card never bleeds
  // into another). Without re-fetching here, a session that resolves after
  // mount (ChatTab can mount ChatView before route.sessionId is set) or a
  // switch back to the session paused on an approval left the header saying
  // pending_approval with no card until a full reload (Windows bench
  // 2026-10-19). Declared after the history-load effect, so in the same
  // commit the clear runs first and this fetch repopulates.
  useEffect(() => {
    fetchPendingApproval();
  }, [projectId, sessionId, fetchPendingApproval]);

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
      // foreground may have missed pending dispatched/cancelled events.
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
    if (!agentStatus || !sessionId) return;
    const timer = setInterval(() => {
      // Spec 155: the VIEWED session's status, never the project's.
      api<{ project_id: string; status: string }>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/run-status?session_id=${encodeURIComponent(sessionId)}`,
      )
        .then((result) => {
          if (result.status !== agentStatus) {
            window.dispatchEvent(
              new CustomEvent('agent-status-override', {
                detail: { project_id: projectId, session_id: sessionId, status: result.status },
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
  }, [agentStatus, projectId, sessionId, fetchPendingApproval]);

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
    // Live agent events carry their session_id and render only into the
    // VIEWED session (seam 3 strict routing; spec 155 runs sessions in
    // parallel). The user's own typed message (handleUserMessage nonce path)
    // renders optimistically for the session it was sent into.
    // Spec 100: a sub-agent's thinking, buffered per worker and applied at
    // most once per frame (one setItems for everything that arrived). The
    // buffer is read and cleared here, outside the updater.
    function flushWorkerReasoning() {
      if (workerFrameRef.current !== null) {
        cancelFrame(workerFrameRef.current);
        workerFrameRef.current = null;
      }
      const pending = workerReasoningRef.current;
      if (pending.size === 0) return;
      const entries = [...pending];
      pending.clear();
      const ts = new Date().toISOString();
      setItems((prev) =>
        entries.reduce(
          (acc, [owner, { text, route }]) =>
            applyWorkerEvent(acc, owner, route, (list) => appendLiveReasoning(list, text, ts, owner, owner)),
          prev,
        ),
      );
      scrollToBottom();
    }

    // Spec 195: where a worker event goes — its card, or inline.
    function routeOf(owner: string, e: { initiator?: string; dispatch_id?: string }): WorkerEventRoute {
      return {
        initiator: e.initiator || undefined,
        dispatchId: e.dispatch_id || undefined,
        fallbackInitiator: liveInitiatorRef.current.get(owner),
      };
    }

    function queueWorkerReasoning(owner: string, text: string, route: WorkerEventRoute) {
      const pending = workerReasoningRef.current;
      const held = pending.get(owner);
      pending.set(owner, { text: (held?.text ?? '') + text, route });
      if (workerFrameRef.current === null) {
        workerFrameRef.current = requestFrame(flushWorkerReasoning);
      }
    }

    function handleStreamDelta(event: WebSocketEvent) {
      const e = event as StreamDeltaEvent;
      if (e.project_id !== projectId) return;
      // Strict session routing: render only deltas for the viewed session.
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;

      // Spec 100: a sub-agent's thinking goes into its own capsule. It is not
      // the manager's stream: no thinking indicator, no running latch.
      if (e.worker) {
        const reasoning = e.reasoning_content ?? '';
        if (reasoning && e.source && e.source !== 'management' && !isWorkerHandle(e.source)) {
          queueWorkerReasoning(e.source, reasoning, routeOf(e.source, e));
        }
        return;
      }

      // A live delta for the VIEWED session is definitive proof it is the
      // session actively executing. Latch the "was running while viewed" flag
      // so the running->idle catch-up reconciles full persisted history — even
      // when the holder/viewing latch never fired (holder unresolved at the
      // running transition, the captured bug). Reset on session switch (load
      // effect) so this never reconciles a session navigated away from.
      wasRunningRef.current = true;

      if (e.is_final) {
        // Text still waiting for its frame lands first, so the final message
        // is the whole concatenation.
        streamBufferRef.current?.flush();
        setStream((prev) => {
          if (!prev) {
            return null;
          }
          const finalText = prev.text + e.text;
          if (finalText.trim()) {
            // [STATUS: …]: a reply that is only its status is a step of the
            // running capsule, not a bubble; a reply with text keeps its
            // bubble and its status labels the tools that follow.
            const status = statusOf(finalText);
            const statusOnly = isStatusOnly(finalText);
            const ts = new Date().toISOString();
            setItems((prevItems) => {
              if (status && statusOnly) return appendLiveStatus(prevItems, status, e.source, ts);
              // Already applied (this handler can run twice): the bubble,
              // maybe followed by the capsule its status opened.
              let k = prevItems.length - 1;
              const tailRun = prevItems[k];
              if (tailRun && tailRun.type === 'agent_run' && tailRun.status === 'running' &&
                  tailRun.items.every((c) => c.type === 'agent_message')) k--;
              const tailMsg = prevItems[k];
              if (status && tailMsg && tailMsg.type === 'agent_message' && tailMsg.content === finalText) return prevItems;
              // Visible text closes any open live capsule first, then appends.
              const afterCapsule = finalizeLiveCapsule(prevItems, 'completed');
              const last = afterCapsule[afterCapsule.length - 1];
              if (last && last.type === 'agent_message' && 'content' in last && last.content === finalText) {
                return afterCapsule;
              }
              const withBubble: DisplayItem[] = [
                ...afterCapsule,
                {
                  type: 'agent_message',
                  content: finalText,
                  source: e.source,
                  timestamp: ts,
                },
              ];
              return status ? appendLiveStatus(withBubble, status, e.source, ts, true) : withBubble;
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

      // Visible answer text has begun — the next frame's commit leaves the
      // thinking phase and appends it.
      streamBufferRef.current?.push(e.session_id, e.text, e.source);
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

      // Spec 100: a sub-agent's live tool call or result, into its own
      // capsule (anchored under its header). Fanout workers stay out of the
      // main chat (spec 009 §0.5-8), as their bubbles do.
      if (e.worker_event) {
        const owner = e.source;
        if (!owner || owner === 'management' || isWorkerHandle(owner)) return;
        flushWorkerReasoning(); // thinking that arrived first renders first
        if (e.worker_event === 'tool_result') {
          const totals =
            e.result_total_chars !== undefined && e.result_total_lines !== undefined
              ? { chars: e.result_total_chars, lines: e.result_total_lines }
              : undefined;
          const route = routeOf(owner, e);
          setItems((prev) =>
            applyWorkerEvent(prev, owner, route, (list) =>
              markWorkerResult(list, owner, e.timestamp, {
                toolCallId: e.tool_call_id || undefined,
                content: e.result_preview ?? '',
                totals,
              }),
            ),
          );
        } else {
          const toolName = e.tool_name || 'tool';
          const route = routeOf(owner, e);
          setItems((prev) =>
            applyWorkerEvent(prev, owner, route, (list) => upsertWorkerToolRow(list, owner, {
              type: 'tool_call_row',
              tool_name: toolName,
              target_description: describeWorkerTool(
                toolName,
                e.arguments,
                project.workspace,
                (k, v) => translate(localeRef.current, k, v),
              ),
              tool_call_id: e.tool_call_id || e.id,
              category: workerToolCategory(toolName),
              timestamp: e.timestamp,
              result_content: null,
              result_status: 'pending',
            })),
          );
        }
        scrollToBottom();
        return;
      }

      // agent_output activities duplicate sub-agent messages already
      // delivered via chat.sub_agent_message — drop them.
      if (e.category === 'agent_output') return;

      // tool_result activity: newer daemons send the tool_call_id and the
      // part of the result the row shows (plus the full size when cut), so
      // a row expanded mid-turn has content. Older daemons send only the
      // "Tool result received" placeholder — the row is then marked
      // received with empty content until a reload.
      if (e.category === 'tool_result') {
        const totals =
          e.result_total_chars !== undefined && e.result_total_lines !== undefined
            ? { chars: e.result_total_chars, lines: e.result_total_lines }
            : undefined;
        setItems((prev) =>
          markLatestLiveCallResultReceived(prev, e.timestamp, {
            toolCallId: e.tool_call_id || undefined,
            content: e.result_preview ?? '',
            totals,
          }),
        );
        scrollToBottom();
        return;
      }

      // The fanout tool call never renders as a capsule row — the
      // fanout.started card IS its representation (spec 009 §0.5), and the
      // persisted-history transform skips it the same way, so live and
      // reloaded views agree. Its ack (a tool_result event) is harmless:
      // no row carries its tool_call_id, so markLatestLiveCallResultReceived
      // no-ops (an older daemon sends no id; then it no-ops when nothing is
      // pending).
      if (e.tool_name === 'fanout') {
        return;
      }

      // Tool-use family: route into the live capsule. Newer daemons send
      // the tool_call_id, which pairs the row with its tool_result by id;
      // for older ones the event id is a synthetic key and pairing is
      // positional.
      const dispatch = dispatchFromToolCall(e.tool_name, e.arguments);
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
            tool_call_id: e.tool_call_id || e.id,
            category: e.category,
            timestamp: e.timestamp,
            result_content: null,
            result_status: 'pending',
            ...(dispatch ? { dispatch } : {}),
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
          kind: e.kind,
          command: e.command,
          executable: e.executable,
          reason: e.reason,
          source: approvalSource(e.source, e.what),
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

      flushWorkerReasoning();
      const route = routeOf(e.source, e);
      setItems((prev) =>
        applyWorkerEvent(prev, e.source, route, (list) => [
          // Spec 100: the worker's text closes its live capsule, as the
          // manager's visible text closes its own.
          ...finalizeLiveCapsule(list, 'completed', e.source),
          {
            type: 'sub_agent_message',
            content: cleaned,
            source: e.source,
            timestamp: e.timestamp,
          },
        ]),
      );
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
    //
    // Spec 185: while the worker runs, the seed guard now counts its turn as
    // in flight, so a send DURING the turn no longer wipes it. This refetch
    // stays the only catch-up: the worker's `idle` status is pushed before the
    // terminal row is written, so only this event sees complete history. If
    // the worker is running again when it lands (a chained turn), the guard
    // skips it; the chain's last terminal seeds.
    function handleSubAgentTerminal(event: WebSocketEvent) {
      const e = event as SubAgentLifecycleEvent;
      if (e.project_id !== projectId) return;
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;
      // Spec 195: the handle's turn is over — close its card now (the
      // refetch below replaces it with the persisted one once nothing else
      // in the session is in flight). A spec-129 own turn has no card.
      if (e.handle) liveInitiatorRef.current.delete(e.handle);
      const end = CARD_END_BY_EVENT[e.type];
      if (e.handle && end) {
        const handle = e.handle;
        const summary = typeof e.summary === 'string' && e.summary.trim() && e.summary.trim() !== '(no output)'
          ? e.summary : undefined;
        flushWorkerReasoning();
        setItems((prev) => {
          const idx = findCardIndex(prev, { handle });
          if (idx < 0) return prev;
          return endCard(prev, idx, end, new Date().toISOString(),
            (steps, status) => finalizeLiveCapsule(steps, status === 'completed' ? 'completed' : status, handle),
            summary);
        });
      }
      refreshRawMessages();
    }

    // Spec 195: Orbital (or the pinned worker) sent a worker a task — its
    // card opens now, titled with the brief. The user's own pinned send
    // stays inline (a card an early event opened for it is unwrapped).
    function handleSubAgentDispatched(event: WebSocketEvent) {
      const e = event as SubAgentLifecycleEvent;
      if (e.project_id !== projectId) return;
      if (e.dispatch_id) {
        // A held queue row went out: its bubble enters the chat now (its row
        // was written to the session just before this dispatch).
        const sent = queuedSendsRef.current.find((q) => q.dispatchId === e.dispatch_id);
        if (sent?.held && sent.sessionId === sessionIdRef.current) {
          paintUserBubble(sent.content, sent.handle);
          scrollToBottom();
        }
        clearQueuedClock(e.dispatch_id);
      }
      if (!e.session_id || e.session_id !== sessionIdRef.current) return;
      if (!e.handle || !e.initiator || isWorkerHandle(e.handle)) return;
      const { handle, initiator } = e;
      liveInitiatorRef.current.set(handle, initiator);
      if (isCardedInitiator(initiator)) {
        setItems((prev) => ensureLiveCard(prev, {
          handle,
          dispatchId: e.dispatch_id,
          initiator,
          brief: e.preview,
          timestamp: new Date().toISOString(),
        }).items);
        scrollToBottom();
        return;
      }
      setItems((prev) => {
        const idx = findCardIndex(prev, { handle });
        if (idx < 0) return prev;
        const card = prev[idx] as DispatchCardItem;
        return card.dispatch_id && card.dispatch_id !== e.dispatch_id ? prev : unwrapCard(prev, idx);
      });
    }

    function handleQueueDropped(event: WebSocketEvent) {
      const e = event as SubAgentQueueDroppedEvent;
      if (e.project_id !== projectId) return;
      if (!e.dispatch_id) return;
      const dropped = queuedSendsRef.current.find((q) => q.dispatchId === e.dispatch_id);
      clearQueuedClock(e.dispatch_id);
      // A held row was written just before its "not sent" marker: fetch both.
      if (dropped?.held && dropped.sessionId === sessionIdRef.current) refreshRawMessages();
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
        if (queuedSendsRef.current.some((q) => q.nonce === n)) continue;
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
      // The queue block: it is sent now, so its bubble enters the chat.
      const entry = pendingInputsRef.current.get(nonce);
      if (entry && entry.sessionId === sessionIdRef.current) {
        paintUserBubble(entry.content);
        scrollToBottom();
      }
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
      if (e.dispatch_id) {
        clearQueuedClock(e.dispatch_id);
        return;
      }
      if (!e.nonce) return;
      const nonce = e.nonce;
      // A queued message has no bubble (the queue block), so only its row goes.
      localNoncesRef.current.delete(nonce);
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
    // Spec 129: a self-initiated worker turn ended; its row and reply are
    // on disk, so reseed from history like any other terminal.
    on('sub_agent.continued', handleSubAgentTerminal);
    on('sub_agent.error', handleSubAgentTerminal);
    on('sub_agent.failed', handleSubAgentTerminal);
    on('sub_agent.stopped', handleSubAgentTerminal);
    on('sub_agent.turn_interrupted', handleSubAgentTerminal);
    on('sub_agent.dispatched', handleSubAgentDispatched);
    on('sub_agent.queue_dropped', handleQueueDropped);
    on('chat.user_message', handleUserMessage);
    on('agent.notify', handleAgentNotify);
    on('state_refresh.lifecycle', handleStateRefresh);
    on('workspace_claudemd_warning', handleClaudemdWarning);
    on('chat.pending_dispatched', handlePendingDispatched);
    on('chat.pending_cancelled', handlePendingCancelled);
    on('fanout.started', handleFanoutStarted);
    on('fanout.task_update', handleFanoutTaskUpdate);
    on('fanout.completed', handleFanoutCompleted);

    return () => {
      // Buffered worker thinking belongs to the view being torn down; the
      // transcript still has it for the next load.
      if (workerFrameRef.current !== null) {
        cancelFrame(workerFrameRef.current);
        workerFrameRef.current = null;
      }
      workerReasoningRef.current.clear();
      off('chat.stream_delta', handleStreamDelta);
      off('agent.activity', handleActivity);
      off('approval.request', handleApprovalRequest);
      off('approval.resolved', handleApprovalResolved);
      off('chat.sub_agent_message', handleSubAgentMessage);
      off('sub_agent.completed', handleSubAgentTerminal);
      off('sub_agent.continued', handleSubAgentTerminal);
      off('sub_agent.error', handleSubAgentTerminal);
      off('sub_agent.failed', handleSubAgentTerminal);
      off('sub_agent.stopped', handleSubAgentTerminal);
      off('sub_agent.turn_interrupted', handleSubAgentTerminal);
      off('sub_agent.dispatched', handleSubAgentDispatched);
      off('sub_agent.queue_dropped', handleQueueDropped);
      off('chat.user_message', handleUserMessage);
      off('agent.notify', handleAgentNotify);
      off('state_refresh.lifecycle', handleStateRefresh);
      off('workspace_claudemd_warning', handleClaudemdWarning);
      off('chat.pending_dispatched', handlePendingDispatched);
      off('chat.pending_cancelled', handlePendingCancelled);
      off('fanout.started', handleFanoutStarted);
      off('fanout.task_update', handleFanoutTaskUpdate);
      off('fanout.completed', handleFanoutCompleted);
    };
  }, [projectId, project.name, on, off, scrollToBottom, removeOptimisticBubble, refreshRawMessages, clearQueuedClock, paintUserBubble]);

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
    if (item.type === 'ask_line') {
      return `ask_line:${item.timestamp}:${item.text.slice(0, 32)}`;
    }
    if (item.type === 'daemon_row') {
      return `daemon_row:${item.event}:${item.timestamp}:${item.handle}`;
    }
    if (item.type === 'session_moved') {
      return `session_moved:${item.timestamp}:${item.toProjectId ?? item.to}`;
    }
    if (item.type === 'dispatch_card') {
      // A card's steps grow in the panel, not in the conversation: only its
      // own opening and its end move the conversation.
      return `dispatch_card:${item.card_id}:${item.status}`;
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

  /** Fit the composer to its text; see `fitComposerHeight`. */
  function adjustTextareaHeight({ reset = true }: { reset?: boolean } = {}) {
    const ta = textareaRef.current;
    if (ta) fitComposerHeight(ta, reset);
  }

  function handleInputChange(value: string) {
    // `inputText` is the text before this edit (each onChange commits
    // synchronously before the next one fires). Growth is detected from
    // scrollHeight alone; anything that may shrink (shorter text, a newline
    // gone, a same-length replacement) collapses to `auto` first.
    const prev = inputText;
    const mayShrink =
      value.length <= prev.length || countNewlines(value) < countNewlines(prev);
    setInputText(value);
    adjustTextareaHeight({ reset: mayShrink });

    // Spec 132 `/` skill picker: the sticky Esc and the catalog load.
    slash.onTextChange(value);
  }

  // Spec 137: the built-ins keep the list from ever being empty, so the old
  // "no skills installed" hint is gone — a bare `/` lists `/automate` (keyed;
  // keyless with no skills installed, `/` opens nothing).

  /** Pick a skill: the `/query` text goes, the chip replaces any earlier pick. */
  function pickSkill(entry: SkillCatalogEntry) {
    slash.resetIndex();
    if (entry.invoke) {
      // Spec 166: the pinned worker's own command. Its invocation goes in the
      // composer, caret after it for arguments, and travels as plain text:
      // no chip, and any earlier chip goes (it would add a block, or route
      // the message to the manager).
      const invoke = entry.invoke;
      setPickedSkillState(null);
      setInputText(invoke);
      requestAnimationFrame(() => {
        const ta = textareaRef.current;
        if (!ta) return;
        adjustTextareaHeight();
        ta.focus({ preventScroll: true });
        ta.setSelectionRange(invoke.length, invoke.length);
      });
      return;
    }
    setPickedSkillState({ sessionId, skill: entry });
    if (entry.builtin) {
      // Spec 137: a built-in prefills its phrasing, selected, so typing
      // replaces the example and Enter sends it as is.
      const prefill = entry.builtin.prefill;
      setInputText(prefill);
      requestAnimationFrame(() => {
        const ta = textareaRef.current;
        if (!ta) return;
        adjustTextareaHeight();
        ta.focus({ preventScroll: true });
        ta.setSelectionRange(0, prefill.length);
      });
      return;
    }
    setInputText('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  }

  // Reused by the context strip's "new session" offer.
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
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    try {
      const result = await newSession(projectId, sessionId);
      // Spec 107 FE-1: the minted id has no history until its first message;
      // mark it so the load effect seeds the empty state instead of fetching.
      if (result.session_id) markFreshSession(projectId, result.session_id);
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
  // A picked skill is sendable on its own too ("run this skill").
  // Spec 146 / item 160: while this conversation has an open ask, the
  // composer becomes the ask (Claude Code style): the question and its
  // options head the composer, and what is typed under it answers the ask.
  // One at a time, oldest first; 稍后 (Later) puts the plain composer back
  // for this session so a plain message can go out. It sets aside the asks
  // open at that moment only: one the agent opens afterwards takes the
  // composer again, since the point is that a wait is never invisible.
  const { asks: sessionAsks, close: closeAsk, dismiss: dismissAsk } = useSessionAsks(projectId, sessionId, agentStatus);
  const [asksLater, setAsksLater] = useState<{ sessionId: string | undefined; ids: ReadonlySet<string> } | null>(null);
  const laterIds = asksLater && asksLater.sessionId === sessionId ? asksLater.ids : null;
  // An ask carried in from the Workbench (a session-less ask opened into a
  // fresh session, spec 154): the same card, until answered or dismissed.
  const [askReply, setAskReply] = useState<{ id: string; text: string; sessionId?: string } | null>(null);
  const carriedAsk = askReply && askReply.sessionId === sessionId ? askReply : null;
  const cardAsk: SessionAsk | null = carriedAsk
    ? { id: carriedAsk.id, text: carriedAsk.text, due: null, options: [], detail: '' }
    : sessionAsks.find((a) => !laterIds?.has(a.id)) ?? null;
  // The composer becomes the card while an ask shows: the question, its
  // detail and options, then one bare text line (no pin, no attach).
  const bareAsk = cardAsk !== null;

  const askCardNode = cardAsk ? (
    <AskCard
      ask={cardAsk}
      workspace={project.workspace}
      onOpenPath={onOpenPath}
      onOption={(option) => {
        // Batch 1110 item 2: the goals card's 需要改 asks for the correction.
        if (isChangeOption(option, cardAsk.change_option)) {
          const prefill = changeOptionPrefill(option);
          setInputText(prefill);
          requestAnimationFrame(() => {
            const ta = textareaRef.current;
            if (!ta) return;
            adjustTextareaHeight();
            ta.focus({ preventScroll: true });
            ta.setSelectionRange(prefill.length, prefill.length);
          });
          return;
        }
        void sendComposer(option);
      }}
      onDone={(id) => { setAskReply(null); void closeAsk(id, 'fulfilled'); }}
      onNotNeeded={(id) => { setAskReply(null); void closeAsk(id, 'irrelevant'); }}
      onLater={() => {
        if (carriedAsk) setAskReply(null);
        else setAsksLater({ sessionId, ids: new Set(sessionAsks.map((a) => a.id)) });
      }}
    />
  ) : null;
  useEffect(() => {
    if (!initialAskReply) return;
    setAskReply({ ...initialAskReply, sessionId });
    onDraftConsumed?.();
    textareaRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialAskReply]);

  const canSend =
    !anyUploading &&
    !allError &&
    (hasText || anyDone || annotations.length > 0 || pickedSkill !== null) &&
    !(attachments.length > 0 && !hasText && !anyDone && pickedSkill === null);
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

  function handleSend() {
    return sendComposer();
  }

  /** The composer's send. `answerText` is an ask option the user tapped: it
   *  goes out as the message (and the answer) in place of the typed text,
   *  which stays in the field. */
  async function sendComposer(answerText?: string) {
    leftOffOpenedRef.current = null;
    const text = (answerText ?? inputText).trim();
    const doneAttachments = attachments.filter((a) => a.status === 'done');
    const pendingAnnotations = annotations;
    const skill = pickedSkill;
    if (!text && doneAttachments.length === 0 && pendingAnnotations.length === 0 && !skill) return;
    if (anyUploading) return;

    // Spec 091: the sticky "Talking to" pin is the only direct-to-worker send;
    // otherwise the management agent. Text goes out verbatim — a leading
    // `@codex` is part of the message, never a target.
    const resolvedPin = resolveSendTarget(text, pinnedTarget);
    // Spec 137: `/automate` goes to the manager whatever the pin — only it
    // has the trigger tools. Not on a keyless project (no manager to run
    // it): there the pin stands, as for any other message.
    const resolved =
      skill?.builtin?.managerOnly && project?.manager_ready !== false
        ? { target: undefined, content: resolvedPin.content, pinned: false }
        : resolvedPin;
    const target = resolved.target;
    let content = resolved.content;
    // Spec 146 / item 160: a message typed under the ask card answers that
    // ask; the inject route closes it in the user's words.
    const answering = cardAsk && text ? { id: cardAsk.id, words: text } : undefined;
    if (answering && cardAsk) {
      content = formatAskAnswer(cardAsk.id, cardAsk.text, content);
      setAskReply(null);
      dismissAsk(cardAsk.id);
    }

    const attachmentsPayload: AttachmentPayload[] =
      doneAttachments.map((a) => (a.localPath
        ? { path: a.localPath, mime: a.mime, size: a.size, local: true }
        : { path: a.uploadedPath!, mime: a.mime, size: a.size }));
    // If the message queues, edit puts these chips back (the composer queue
    // block): the files are already uploaded or attached by path.
    const queuedChips: QueuedChip[] = doneAttachments.map((a) => ({
      filename: a.filename,
      mime: a.mime,
      size: a.size,
      ...(a.localPath ? { localPath: a.localPath } : { uploadedPath: a.uploadedPath }),
      ...(a.file && { file: a.file }),
    }));

    // Spec 078 §5.4 — the annotated PNGs ride the attachments list, the quotes
    // block is appended to the text. Both must be settled BEFORE the optimistic
    // bubble is painted, so its wire content matches the WS echo exactly.
    let annotationUploadFailed = false;
    if (pendingAnnotations.length > 0) {
      const uploaded = await uploadAnnotationImages(pendingAnnotations);
      attachmentsPayload.push(...uploaded.attachments);
      for (const u of uploaded.attachments) {
        queuedChips.push({
          filename: u.path.split(/[\\/]/).pop() || u.path, mime: u.mime, size: u.size,
          uploadedPath: u.path,
        });
      }
      annotationUploadFailed = uploaded.failed;
      const quotes = formatQuotes(pendingAnnotations);
      content = content ? `${content}\n\n${quotes}` : quotes;
    }

    // Build the same prefix the backend will emit, so the optimistic local
    // user_message has the identical wire content as the WS echo will.
    // The skill block follows the attachments block, as on the backend. (A
    // safe-mode manager gets the body inlined server-side; the echo then
    // differs from this bubble, which the nonce dedup tolerates.)
    // A built-in skill (spec 137) is not in the backend catalog: its text
    // rides inside `content` as an inline block, with no skill payload.
    const skillParts = skillSendParts(skill, buildSkillBlock, buildInlineSkillBlock);
    const skillPayload = skillParts.payload;
    content = skillParts.contentPrefix + content;
    const wireContent =
      buildAttachmentsBlock(attachmentsPayload) + skillParts.pointerBlock + content;

    if (answerText === undefined) {
      setInputText('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
    if (skill) setPickedSkillState(null);
    clearAttachments();
    if (pendingAnnotations.length > 0) {
      clearAnnotations();
      setAnnotationsOpen(false);
    }

    await postToInject({
      sessionId,
      content,
      wireContent,
      target,
      pinned: resolved.pinned,
      attachments: attachmentsPayload.length > 0 ? attachmentsPayload : undefined,
      skill: skillPayload,
      answering,
      // An annotated send keeps its quotes in the text edit puts back.
      rawText: pendingAnnotations.length > 0 ? content.slice(skillParts.contentPrefix.length) : text,
      chips: queuedChips,
      skillEntry: skill,
      // An annotated PNG that failed to render/upload still gets sent — the
      // coordinates + note in the quotes block carry the meaning — but the
      // user is told the image did not make it.
      notice: annotationUploadFailed ? t('chat.uploadError') : null,
    });
  }

  /**
   * Spec 156 §5.7 / 164 §2.3 — the post-to-/inject half of every send: the
   * composer, the message editor (into the fork it just opened) and Retry
   * (back into this session). It paints the optimistic bubble, posts with a
   * fresh nonce, and handles the queued / approval-dismissed / error answers.
   * It never touches the composer's draft or chips: that is the caller's.
   * Resolves true when the daemon accepted the message.
   */
  async function postToInject(send: InjectSend): Promise<boolean> {
    const { sessionId: sid, content, wireContent, target } = send;
    // Generate nonce so we can deduplicate the WS echo of our own message
    // crypto.randomUUID() requires secure context (HTTPS/localhost) — use fallback for LAN HTTP
    const nonce = typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    localNoncesRef.current.set(nonce, 0);

    const optimisticTimestamp = new Date().toISOString();
    // The composer queue block (2026-11-24): a message that will wait for
    // its turn is not drawn in the chat. When the send is likely to queue
    // (this chat's agent is mid-turn, or the pinned worker is busy) the
    // bubble waits for the daemon's answer; otherwise it is drawn at once,
    // as always, and taken back if the answer says it queued.
    const likelyQueued = target ? runningSubAgents.includes(target) : agentStatus === 'running';
    let painted = false;
    const paint = () => {
      if (painted) return;
      painted = true;
      // Not a persisted row until the history refetch replaces it: no edit or
      // retry on it (its timestamp names no row in the session log).
      optimisticTsRef.current.add(optimisticTimestamp);
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
    };
    const unpaint = () => {
      if (!painted) return;
      painted = false;
      removeOptimisticBubble(wireContent, optimisticTimestamp);
    };
    // What a queue row keeps, to show it and to put it back on edit. A send
    // whose blocks came inside `content` (retry, the editor) has no chips to
    // restore; it stays editable only when it carries no attachments.
    const queuedChips = send.chips ?? [];
    const queuedEditable = send.chips !== undefined || !hasMachineBlocks(wireContent);
    if (!likelyQueued) paint();
    // The user's own send always re-arms auto-follow and jumps to the bottom,
    // even if they had scrolled up to read history before sending.
    scrollToBottom({ force: true });

    const loading = target ? { sessionId: sid, target } : null;
    if (loading) setSubAgentLoading(loading);
    setInjectError(send.notice ?? null);
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
        send.attachments,
        sid,
        send.pinned,
        send.skill,
        send.answering,
      );
      onUserSend?.(result);
      // Spec 180: the session had moved (another device, or a moment ago) and
      // the daemon delivered this into its new project: follow it there.
      if (sid !== undefined && result?.project_id && result.project_id !== projectId) {
        onSessionMoved?.(sid, result.project_id);
      }
      // A pinned send materialized the session on the backend — if the pin
      // PATCH had failed earlier (brand-new session, 404 before the first
      // message), persist it now so it survives reloads. `pinAgent`'s
      // optimistic update keeps `backendPin` current on success, so this
      // fires only while the stored value is actually behind.
      if (
        send.pinned &&
        sid !== undefined &&
        target !== undefined &&
        (recordedPin ?? null) !== target
      ) {
        persistPin(sid, target);
      }
      // Spec 195: the pinned message waits behind the worker's current task
      // (Orbital's, a worker's, or the user's own earlier one). Held (its chat
      // row not written yet): a queue-block row, no bubble. Not held (a send
      // that raced the worker's start): a bubble with the clock, as before.
      if (
        result &&
        typeof result === 'object' &&
        result.status === 'queued_behind' &&
        result.dispatch_id &&
        sid !== undefined
      ) {
        const held = result.held === true;
        const queued: QueuedSend = {
          dispatchId: result.dispatch_id,
          handle: result.handle ?? target ?? '',
          sessionId: sid,
          content: wireContent,
          held,
          nonce,
          text: send.rawText,
          chips: queuedChips,
          editable: queuedEditable,
          queuedAt: optimisticTimestamp,
          skill: send.skillEntry ?? null,
        };
        if (held) unpaint();
        else paint();
        setQueuedSends((prev) => [...prev.filter((q) => q.dispatchId !== queued.dispatchId), queued]);
        if (held) {
          scrollToBottom();
          return true;
        }
      }
      // Pending-input queue (spec 006 v3 §11d.4): the same-session 200. The
      // backend ACCEPTED + queued this message behind the VIEWED session's own
      // in-flight turn (`session._queue`); the agent takes it at its next
      // step. It becomes a queue-block row (no bubble until
      // `chat.pending_dispatched`), editable with the pencil or ↑.
      if (
        result &&
        typeof result === 'object' &&
        result.status === 'queued_same_session' &&
        sid !== undefined
      ) {
        unpaint();
        setPendingInputs((prev) => {
          const next = new Map(prev);
          next.set(nonce, {
            sessionId: sid,
            content: wireContent,
            text: send.rawText,
            chips: queuedChips,
            editable: queuedEditable,
            queuedAt: optimisticTimestamp,
            skill: send.skillEntry ?? null,
          });
          return next;
        });
        scrollToBottom();
        return true;
      }
      // Delivered or started: it is in the chat now.
      paint();
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
      // The failed message stays visible in the chat, as it always did.
      paint();
      // A classified provider/credential 400 (auto-start could not build the
      // LLM provider) gets the actionable AgentErrorNotice; anything else
      // keeps the generic inline error line.
      const info = parseProviderError(err);
      if (info) {
        setAgentError(info);
      } else {
        setInjectError(t('chat.injectError'));
      }
      // A failed pinned send: the daemon persisted a
      // `sub_agent_dispatch_failed` row (the translated reason) before
      // answering 404, and broadcasts `sub_agent.failed` for it. Refetch too,
      // so the row still lands when that event was missed (relay drop).
      if (target) refreshRawMessages();
      send.onFailed?.();
      return false;
    } finally {
      // Only this send's bubble — a later send may have replaced it.
      if (loading) setSubAgentLoading((cur) => (cur === loading ? null : cur));
    }

    scrollToBottom();
    return true;
  }

  // The Stop square and Esc.
  function stopCurrentTurn() {
    if (isCancelling) return;
    setCancelTimeoutNotice(null);
    setIsCancelling(true);
    // Stop THIS session only (spec 155: others keep running).
    cancelMessage(projectId, sessionId).catch(() => {
      // POST failed (offline, daemon down, etc.) — drop the optimistic state
      // immediately so the user can retry instead of waiting for the 10s
      // timeout.
      setIsCancelling(false);
      setCancelTimeoutNotice(t('chat.cancelFailed'));
    });
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // While an IME (e.g. Pinyin) is composing, Enter/Arrow/Tab belong to the
    // candidate window — don't submit or hijack them for our dropdowns.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (slash.onKeyDown(e, pickSkill)) return;
    // Pending-input queue (spec 006 v3 §11c + §12). ↑ in an EMPTY composer (no
    // command dropdown open — the guard above already handled ↑ and
    // returned) edits the NEWEST queue-block row of the VIEWED session: it
    // leaves the queue and comes back into the composer, chips included. No
    // cycling, no input history (§12 R5). When the composer is non-empty, or
    // no editable row exists, do NOT preventDefault so ↑ still moves the caret.
    if (e.key === 'ArrowUp' && inputText.trim() === '' && sessionId !== undefined) {
      const newest = viewedQueueRows[viewedQueueRows.length - 1];
      if (newest && newest.editable) {
        e.preventDefault();
        void takeQueued(newest, 'edit');
        return;
      }
    }
    // Esc stops the running turn (the Stop square's shortcut). Not while an
    // IME composes (returned above), not when the slash menu took it (above),
    // and not when another surface owns Esc.
    if (e.key === 'Escape' && !e.defaultPrevented && !bareAsk
      && (agentStatus === 'running' || agentStatus === 'waiting')
      && !isCancelling && !annotationsOpen && !escapeOwnedElsewhere()) {
      e.preventDefault();
      stopCurrentTurn();
      return;
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const isDragActive = dragCounter > 0;

  // Spec 078 §5.4 — the annotation chip. Collapsed it is just the count;
  // expanded it lists each quote with an editable note and a remove control,
  // so a note typed in a hurry on the panel can be fixed before the message
  // goes. It renders in BOTH composer states: in the composer it is an inline
  // capsule at the start of the input (its list a popover); while the queue
  // owns the chat the card is replaced by ComposerDisabledPrompt, so it falls
  // back to a strip above the notice — otherwise annotations would be staged
  // with nothing on screen to show for them (2026-09-05: "when i do the
  // browser annotation, where did it go?").
  const annotationList = (
    <ul
      className="flex flex-col gap-1 max-h-[140px] overflow-y-auto"
      data-testid="annotation-list"
    >
      {annotations.map((a) => (
        <li key={a.n} className="flex items-center gap-2">
          <span className="font-mono text-micro text-secondary shrink-0">[{a.n}]</span>
          <span
            className="text-micro text-secondary truncate max-w-[40%]"
            title={annotationSummary(a, chromeTr)}
          >
            {annotationSummary(a, chromeTr)}
          </span>
          <input
            value={a.note}
            onChange={(e) => updateAnnotationNote(a.n, e.target.value)}
            placeholder={t('panel.annotation.note')}
            aria-label={`${t('panel.annotation.note')} [${a.n}]`}
            className="flex-1 min-w-0 text-micro bg-transparent border border-border rounded px-1.5 py-0.5 text-primary outline-none focus:border-accent"
          />
          <button
            type="button"
            onClick={() => removeAnnotation(a.n)}
            aria-label={`${t('panel.annotation.remove')} [${a.n}]`}
            className="shrink-0 text-secondary hover:text-primary text-caption leading-none px-1"
          >
            ×
          </button>
        </li>
      ))}
    </ul>
  );

  const annotationCount =
    annotations.length === 1
      ? t('composer.annotations.one')
      : t('composer.annotations.other', { n: annotations.length });

  const annotationStrip = annotations.length > 0 && (
    <div className="flex flex-col gap-1" data-testid="annotation-strip">
      <button
        type="button"
        onClick={() => setAnnotationsOpen((open) => !open)}
        aria-expanded={annotationsOpen}
        data-testid="annotation-chip"
        className="self-start inline-flex items-center gap-1 rounded-full border border-border px-2 py-0.5 text-micro text-secondary hover:text-primary transition-colors"
      >
        {annotationCount}
        {annotationsOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
      {annotationsOpen && annotationList}
    </div>
  );

  // The composer's capsule row: skill first, then the annotation capsule
  // (its list opens as a popover above it). Positioned over the textarea's
  // first line; see `capsuleWidth`.
  const capsuleRow = hasCapsule && (
    <div
      ref={capsuleRowRef}
      data-testid="composer-capsules"
      // The row covers the start of the input's first line: a tap on a
      // capsule's body (not its × or the annotation toggle) goes to the
      // input, as a tap on that spot always did.
      onMouseDown={(e) => {
        if ((e.target as HTMLElement).closest('button')) return;
        e.preventDefault();
        textareaRef.current?.focus();
      }}
      className="absolute left-0 top-0 z-10 flex max-w-[75%] items-center gap-1 pr-1.5 h-[1.625em] text-body max-md:text-base"
    >
      {pickedSkill && (
        <SkillChip
          inline
          name={pickedSkill.name}
          title={pickedSkill.description || pickedSkill.name}
          onRemove={() => setPickedSkillState(null)}
          data-testid="composer-skill-chip"
        />
      )}
      {annotations.length > 0 && (
        <div ref={annotationPopoverRef} className="relative min-w-0 flex">
          <button
            type="button"
            onClick={() => setAnnotationsOpen((open) => !open)}
            aria-expanded={annotationsOpen}
            aria-haspopup="dialog"
            aria-label={annotationCount}
            title={annotationCount}
            data-testid="annotation-chip"
            className="inline-flex min-w-0 items-center gap-0.5 rounded-full border border-border bg-card pl-1.5 pr-1 h-[18px] text-micro leading-none text-secondary hover:text-primary transition-colors whitespace-nowrap"
          >
            <MessageSquareQuote size={11} aria-hidden="true" className="shrink-0" />
            {/* The words where there is room; on a phone (173px of input) the
                count alone, so the skill capsule keeps its name. */}
            <span className="truncate max-md:hidden">{annotationCount}</span>
            <span className="md:hidden">{annotations.length}</span>
            {annotationsOpen ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
          </button>
          {annotationsOpen && (
            <div
              ref={annotationListBoxRef}
              role="dialog"
              aria-label={annotationCount}
              data-testid="annotation-popover"
              style={popoverShift ? { transform: `translateX(-${popoverShift}px)` } : undefined}
              className="absolute bottom-full left-0 mb-2 z-50 w-[min(26rem,calc(100vw-3rem))] surface-float-pop p-2 text-body"
            >
              {annotationList}
            </div>
          )}
        </div>
      )}
    </div>
  );

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
          <span className="text-title font-medium">{t('chat.dropFiles')}</span>
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
      {showWorkerTiles && (
        <SubAgentStatusBar projectId={projectId} sessionId={sessionId} live={subAgentStatus} />
      )}
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
      {/* data-chat-state: a session with messages is showing, so no left-off
          cards will mount — the tour reads that as its answer. */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        data-chat-state={!loading && (items.length > 0 || stream) ? 'open' : undefined}
        className="flex-1 overflow-y-auto px-7 py-5 max-md:pb-20 flex flex-col gap-4"
      >
        {loading && (
          <div className="space-y-3">
            {viewedIsUnimportedImport && (
              <p data-testid="chat-importing" className="text-caption text-secondary px-1">
                {t('chat.importing')}
              </p>
            )}
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 bg-sidebar rounded-lg animate-pulse" />
            ))}
          </div>
        )}

        {!loading && hasMore && (
          <button
            onClick={loadOlderMessages}
            disabled={loadingMore}
            className="w-full py-2 text-caption text-secondary hover:text-primary transition-colors disabled:opacity-50"
          >
            {loadingMore ? t('chat.loadingMore') : t('chat.loadEarlier', { n: totalMessages - loadedOffset })}
          </button>
        )}

        {!loading && items.length === 0 && !stream && (
            <div className="mx-auto mt-10 flex w-full max-w-[460px] flex-col items-center px-2 text-center">
              {/* The mark, quietly: this is the agent you are about to talk to
                  (the same logo MessageAvatar gives its replies). On Graphite
                  it sits on a light tile like the app icon, in its own colours. */}
              <div className="flex h-14 w-14 items-center justify-center dark:rounded-[14px] dark:bg-logo-tile">
                <OrbitalMark keepColours className="h-14 w-14 opacity-90 dark:h-12 dark:w-12 dark:opacity-100" />
              </div>
              <p className="mt-3 text-body text-secondary">{t('chat.empty')}</p>
              {/* Example prompts for a project's empty chat — what to ask is
                  the first thing a new user does not know, and the middle one
                  shows that automations are created by just asking. A click
                  FILLS the composer and never sends: the user reads, edits
                  and sends it themselves. Not in Quick Tasks, whose scratch
                  workspace makes "this folder" / "this project" meaningless. */}
              {/* Spec 153: what Claude Code knew about this project, now in it
                  for every agent (imported projects only; hides itself). */}
              <PrivateNotesCard
                projectId={projectId}
                project={project}
                agents={agents}
                onOpenPath={onOpenPath}
              />
              {/* Spec 138: with sessions to pick up (its own or imported), the
                  left-off cards take the examples' place. */}
              <LeftOffCards
                projectId={projectId}
                agents={agents}
                onOpen={(sid) => {
                  leftOffOpenedRef.current = sid;
                  onOpenSession?.(sid);
                }}
                fallback={!project?.is_scratch && (
                <div
                  role="group"
                  aria-label={t('chat.empty.examples')}
                  data-testid="chat-empty-examples"
                  className="mt-6 flex w-full flex-col gap-2"
                >
                  <span
                    aria-hidden="true"
                    className="type-label text-secondary"
                  >
                    {t('chat.empty.examples')}
                  </span>
                  {EMPTY_CHAT_EXAMPLES.map(({ key, icon }) => (
                    <ExamplePrompt
                      key={key}
                      icon={icon}
                      text={t(key)}
                      actionIcon={CornerDownLeft}
                      onClick={() => {
                        handleInputChange(t(key));
                        textareaRef.current?.focus();
                      }}
                    />
                  ))}
                </div>
              )}
              />
            </div>
        )}

        {!loading &&
          items.map((item, index) => {
            const historical = 'isHistorical' in item && item.isHistorical;
            let rendered: React.ReactNode = null;

            if (item.type === 'user_message') {
              const isEditing = editingTs !== null && editingTs === item.timestamp;
              rendered = (
                <MemoChatMessage
                  key={`msg-${index}`}
                  message={item}
                  workspace={project.workspace}
                  onOpenPath={onOpenPath}
                  canEdit={editWired && !sessionBusy && isPersistedHumanInput(item, optimisticTsRef.current)}
                  canRetry={!sessionBusy && retryPlan?.anchor === index}
                  editing={isEditing}
                  editBusy={isEditing && editBusy}
                  onEdit={onEditMessage}
                  onEditCancel={onEditCancel}
                  onEditSend={onEditSend}
                  onRetry={onRetryMessage}
                  queuedHint={queuedHintFor(item)}
                  canUndoSince={undoSinceOk && !sessionBusy && isPersistedHumanInput(item, optimisticTsRef.current)}
                  onUndoSince={onUndoSinceMessage}
                />
              );
            } else if (item.type === 'dispatch_card') {
              const open =
                openWorker?.kind === 'card' &&
                (openWorker.cardId === item.card_id ||
                  (!!item.dispatch_id && openWorker.dispatchId === item.dispatch_id));
              const inPlace = !onOpenWorker && inPlaceCards.has(item.card_id);
              rendered = (
                <div key={`card-${item.card_id}`} className="flex flex-col gap-2">
                  <DispatchCard
                    card={item}
                    name={agentNameOf(item.handle)}
                    needsYou={needsYouHandles.has(item.handle)}
                    open={open || inPlace}
                    onOpen={handleOpenCard}
                  />
                  {inPlace && (
                    <div className="ml-9 pl-3 flex flex-col gap-3" data-testid="dispatch-card-inplace">
                      <WorkerSteps steps={item.steps} workspace={project.workspace} onOpenPath={onOpenPath} />
                    </div>
                  )}
                </div>
              );
            } else if (item.type === 'agent_message' || item.type === 'sub_agent_message') {
              rendered = (
                <MemoChatMessage
                  key={`msg-${index}`}
                  message={item}
                  agentName={project.agent_name}
                  workspace={project.workspace}
                  onOpenPath={onOpenPath}
                  canRetry={!sessionBusy && retryPlan?.anchor === index}
                  onRetry={onRetryMessage}
                />
              );
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
                    return t('chat.subActivity.sent', { preview: skillAwarePreview(item.preview ?? '') });
                  case 'completed':
                    return item.summary
                      ? t('chat.subActivity.completed', { summary: item.summary })
                      : t('chat.subActivity.completed', { summary: '' }).replace(/[:：]\s*$/, '');
                  case 'continued':
                    // Spec 129: the worker's own turn (a background command
                    // it waited on finished) ended — not a reply to anyone.
                    return item.summary
                      ? t('chat.subActivity.continued', { summary: item.summary })
                      : t('chat.subActivity.continued', { summary: '' }).replace(/[:：]\s*$/, '');
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
                  ? 'text-error-text'
                  : item.action === 'background_lost' ||
                      item.action === 'interrupted' ||
                      item.action === 'queue_dropped' ||
                      (item.action === 'stopped' && !!item.count)
                    ? 'text-warning-text'
                    : item.action === 'interaction_required'
                      ? 'text-primary'
                      : 'text-secondary';
              rendered = item.limit ? (
                <UsageLimitRow
                  key={`sa-${index}`}
                  handle={item.handle}
                  name={agents.find((a) => a.slug === item.handle)?.name ?? item.handle}
                  resetsAt={item.limit.resetsAt}
                  vendorText={item.error ?? ''}
                  switchTo={index === lastLimitIndex ? limitSwitchFor(item.handle) : null}
                  onSwitch={handlePinChange}
                />
              ) : (
                <div
                  key={`sa-${index}`}
                  data-testid="sub-agent-activity"
                  data-action={item.action}
                  className={`flex items-baseline gap-2 px-2 py-1 text-micro font-mono ${tone}`}
                >
                  <span className="text-primary">{item.handle}</span>
                  <span className="truncate">{label}</span>
                </div>
              );
            } else if (item.type === 'ask_line') {
              rendered = (
                <p key={`ask-${index}`} className="m-0 pl-9 text-body text-secondary break-words" data-testid="ask-line">
                  {t('chat.ask.title')} · <InlineMarkdown text={item.text} />
                </p>
              );
            } else if (item.type === 'daemon_row') {
              // Keyless onboarding / session import divider: the session
              // separator's shape at full strength, translated here from the
              // row's fields (the daemon's English content is not shown).
              const agentName =
                agents.find((a) => a.slug === item.handle)?.name ?? item.handle;
              const date = item.lastActivity ? new Date(item.lastActivity) : null;
              const dateText = date && !Number.isNaN(date.getTime())
                ? date.toLocaleDateString(locale === 'zh' ? 'zh-CN' : 'en-US', { year: 'numeric', month: 'short', day: 'numeric' })
                : '';
              const text = item.event === 'worker_onboarding'
                ? t('chat.daemonRow.workerOnboarding', { agent: agentName })
                : t(item.truncated ? 'chat.daemonRow.importedTruncated' : 'chat.daemonRow.imported', {
                    agent: agentName,
                    date: dateText,
                  });
              rendered = (
                <div
                  key={`daemon-${index}`}
                  className="flex items-center gap-3 px-2"
                  data-event={item.event}
                >
                  <div className="flex-1 border-t border-border" />
                  <span data-testid="chat-daemon-row" className="text-caption text-secondary text-center">
                    {text}
                  </span>
                  <div className="flex-1 border-t border-border" />
                </div>
              );
            } else if (item.type === 'session_moved') {
              // Spec 180: one quiet divider, 「从 A 移到 B」, names in medium
              // weight and ellipsised. This project's own name is the live one.
              const toName = item.toProjectId === project.project_id
                ? projectDisplayName(project, t)
                : item.toScratch ? t('project.scratchName') : item.to;
              const fromName = item.fromScratch ? t('project.scratchName') : item.from;
              const parts = t('chat.sessionMoved', { a: '\u0001', b: '\u0002' }).split(/(\u0001|\u0002)/);
              rendered = (
                <div key={`moved-${index}`} className="flex items-center gap-3 px-2" data-testid="chat-session-moved">
                  <div className="flex-1 min-w-4 border-t border-border" />
                  <span className="inline-flex items-center min-w-0 max-w-[82%] whitespace-nowrap text-caption text-secondary">
                    {parts.map((part, i) => {
                      if (part !== '\u0001' && part !== '\u0002') return part ? <span key={i} className="whitespace-pre">{part}</span> : null;
                      const name = part === '\u0001' ? fromName : toName;
                      return (
                        <b key={i} title={name} className="font-medium inline-block truncate align-bottom max-w-[14em] max-md:max-w-[7.5em]">
                          {name}
                        </b>
                      );
                    })}
                  </span>
                  <div className="flex-1 min-w-4 border-t border-border" />
                </div>
              );
            } else if (item.type === 'session_separator') {
              return (
                <div key={`sep-${index}`} className="flex items-center gap-3 px-2 opacity-50">
                  <div className="flex-1 border-t border-border" />
                  <span className="text-caption text-secondary whitespace-nowrap">
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
                  <span className="text-caption text-secondary whitespace-nowrap">
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
              // A status fold stays one closed line while it runs (its line
              // is the live status); a plain capsule opens while running.
              const isExpanded =
                (derivedStatus === 'running' && !isStatusFold(item)) ||
                expandedCapsules.has(item.capsule_id);
              const span = turnSpans.get(item.capsule_id);
              rendered = (
                <AgentRunCapsule
                  key={`run-${item.capsule_id}`}
                  item={item}
                  derivedStatus={derivedStatus}
                  isExpanded={isExpanded}
                  onToggle={toggleCapsule}
                  tight={(() => {
                    const before = items[index - 1];
                    return before?.type === 'agent_message' && before.isHeaderOnly === true && isStatusFold(item);
                  })()}
                  {...(span && sessionId
                    ? {
                        projectId,
                        sessionId,
                        turnFrom: span.fromTs,
                        turnTo: span.toTs,
                        turnEnded: span.toTs !== null || !turnInFlight,
                        onUndoTurn,
                        onOpenHistory: handleOpenHistory,
                      }
                    : {})}
                />
              );
            } else if (item.type === 'agent_notify') {
              const urgencyColor = item.urgency === 'high' ? 'border-error/40 bg-error/5' : 'border-accent/30 bg-accent/5';
              rendered = (
                <div key={`notify-${index}`} className={`rounded-lg border ${urgencyColor} px-4 py-3`}>
                  <p className="text-body font-medium text-primary">{item.title}</p>
                  <p className="text-body text-secondary mt-1">{item.body}</p>
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
                  <p className="text-caption font-medium text-error">{text}</p>
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
                  onSelectTask={(handle, label) =>
                    onOpenWorker ? onOpenWorker({ kind: 'fanout', handle, label }) : setDrillIn({ handle, label })}
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
                  className="px-2 py-1 text-micro font-mono text-secondary whitespace-pre-wrap"
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
                  sessionId={sessionId}
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
                  asker={approvals.get(item.tool_call_id)?.source}
                  askerName={(() => { const h = approvals.get(item.tool_call_id)?.source; return h ? agentNameOf(h) : undefined; })()}
                  approval={item}
                  projectId={projectId}
                  sessionId={sessionId}
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
                  sessionId={sessionId}
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
                  asker={a.source}
                  askerName={a.source ? agentNameOf(a.source) : undefined}
                  approval={a}
                  projectId={projectId}
                  sessionId={sessionId}
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

        {subAgentLoadingTarget && (
          <div className="flex gap-[10px]">
            <MessageAvatar variant="agent" />
            <div className="flex-1 min-w-0">
              <div className="font-mono text-micro mb-1">
                <span className="text-secondary">{subAgentLoadingTarget}</span>
              </div>
              <div className="text-body">
                <span className="inline-flex gap-1 text-secondary">
                  <span className="animate-pulse">●</span>
                  <span className="animate-pulse" style={{animationDelay: '0.2s'}}>●</span>
                  <span className="animate-pulse" style={{animationDelay: '0.4s'}}>●</span>
                </span>
              </div>
            </div>
          </div>
        )}

        {thinkingVisible && !stream && !subAgentLoadingTarget && (
          <div className="flex items-center gap-2 px-2 py-1 text-secondary text-body">
            <Loader2 size={14} className="animate-spin" />
            <span>{t('chat.thinking')}</span>
          </div>
        )}

        {stream && streamShown && (
          <StreamingMessage
            text={streamShown}
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
            agent={agentError.agent}
            agentName={agentError.agent_name}
            // Every non-internal code here is about the provider/key: land
            // on the LLM providers section, as the pin switcher's pill does.
            onOpenSettings={() =>
              window.dispatchEvent(new CustomEvent('open-global-settings', { detail: { section: 'llm' } }))
            }
            onDismiss={() => setAgentError(null)}
          />
        </div>
      )}

      {/* Spec 156: an edit that could not fork (正在运行，停止后再修改). The
          editor stays open with the text; this only says why. Above the
          composer, clear of 065's restore toast at the window's foot. */}
      {actionToast && (
        <div
          role="status"
          data-testid="chat-action-toast"
          className="absolute bottom-24 left-1/2 z-40 -translate-x-1/2 whitespace-nowrap rounded-[10px] bg-[#18181B] px-3.5 py-2 text-body text-white shadow-[0_10px_30px_rgb(0_0_0/0.25)] dark:bg-[#F4F6F9] dark:text-[#18181B] max-md:fixed max-md:bottom-[112px]"
        >
          {actionToast}
        </div>
      )}

      {injectError && (
        <div className="shrink-0 px-4 py-1">
          <p className="text-caption text-error">{injectError}</p>
        </div>
      )}

      {cancelTimeoutNotice && (
        <div className="shrink-0 px-4 py-1">
          <p className="text-caption text-secondary" role="status">{cancelTimeoutNotice}</p>
        </div>
      )}

      <div data-tour="composer" className="shrink-0 px-4 pb-4 pt-2 max-md:fixed max-md:bottom-0 max-md:left-0 max-md:right-0 max-md:bg-card max-md:z-[60] max-md:pb-[env(safe-area-inset-bottom,12px)]">
        {queueActive ? (
          // The panel stays usable while the queue runs, so annotations can
          // still be staged here. Show them above the notice — pausing the
          // queue (the notice's own button) hands them back to the composer.
          <div className="flex flex-col gap-2">
            {annotationStrip}
            <ComposerDisabledPrompt onPauseQueue={stopQueue} />
          </div>
        ) : (
        <>
        {/* Context meter. Sits above the composer rather than in the project
            header because it is per-SESSION, and the header is per-project —
            it would report the wrong session while browsing the session list.
            Its offer runs executeNewSession (the flow the removed /new slash
            command used to run). */}
        <ContextStrip usage={contextUsage} onNewSession={executeNewSession} />
        <div className="relative flex flex-col gap-2 surface-float-static surface-float-focus px-3 py-2 transition-[box-shadow] duration-150 motion-reduce:transition-none">
          {/* Spec 132: the `/` skill menu. Skills only, one flat list (no
              origin badge); at most 8 rows visible, then it scrolls. */}
          <SlashMenu picker={slash} onPick={pickSkill} className="absolute bottom-full left-0 mb-1" />
          {/* The composer queue block (2026-11-24, layout 1): messages that
              wait for their turn, inside this surface above the input. */}
          <QueueBlock
            rows={viewedQueueRows}
            agentName={agentNameOf}
            onEdit={(row) => void takeQueued(row, 'edit')}
            onDelete={(row) => void takeQueued(row, 'delete')}
            onSendNow={(row) => void sendQueuedNow(row)}
            busyKey={queueBusyKey}
          />
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
                  localPath={a.localPath}
                  onRemove={() => removeAttachment(a.id)}
                  onRetry={a.status === 'error' ? () => retryAttachment(a.id) : undefined}
                />
              ))}
            </div>
          )}
          {bareAsk && askCardNode}
          {/* Under an ask the bare text line aligns with the question and
              pills above it (18px in). */}
          <div className={bareAsk ? 'flex items-center gap-2 pl-[18px]' : 'flex items-center gap-2'}>
            {/* Spec 074 — the logo-only pin mark fused to the row's left edge
                IS the sub-agent pin (Orbital's mark at rest, the worker's
                while pinned). Renders nothing when no sub-agents are
                installed (PinTargetSelect returns null). A PATCH failure
                keeps the optimistic localPin; handleSend re-persists after
                the first successful pinned send materializes a brand-new
                session. */}
            {!bareAsk && <PinTargetSelect
              agents={pinAgents}
              value={pinnedTarget}
              runningSlugs={runningSubAgents}
              managerReady={managerReady}
              models={pinModels}
              onChange={handlePinChange}
            />}
            <input
              type="file"
              multiple
              ref={fileInputRef}
              className="hidden"
              onChange={handleFilePickerChange}
              data-testid="attachment-file-input"
            />
            {!bareAsk && <button
              type="button"
              // Desktop app: the native dialog attaches by local path.
              onClick={() => void openPicker(() => fileInputRef.current?.click())}
              aria-label={t('chat.attachFiles')}
              className="shrink-0 p-2 text-secondary hover:text-primary rounded max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
            >
              <Plus size={18} />
            </button>}
            <div className="relative flex flex-1 min-w-0" data-testid="composer-input">
            {capsuleRow}
            <textarea
              ref={textareaRef}
              value={inputText}
              onChange={(e) => handleInputChange(e.target.value)}
              // Safety net for a shrink the onChange rule missed (IME
              // composition, programmatic edits): one full fit on blur.
              onBlur={() => adjustTextareaHeight()}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              // No placeholder under a capsule: the capsule already says what
              // the message is about, and the hint would sit behind it.
              placeholder={cardAsk
                ? t(cardAsk.options.length > 0 ? 'chat.ask.placeholderOptions' : 'chat.ask.placeholder')
                : hasCapsule
                ? ''
                : pinnedTarget
                  // The product name, like the keyless hint below; never the slug.
                  ? t('chat.composer.placeholderPinned', { agent: pinnedAgentName ?? pinnedTarget })
                  : t('chat.composer.placeholder')}
              rows={1}
              disabled={isCancelling}
              // First line only: the text starts after the capsule row.
              style={capsuleWidth > 0 ? { textIndent: `${capsuleWidth}px` } : undefined}
              className="flex-1 w-full resize-none text-body max-md:text-base bg-transparent focus:outline-none leading-relaxed disabled:opacity-50"
            />
            </div>
            {/* Send option B (2026-11-24), one slot: while the agent runs an
                empty box shows Stop (Esc also stops it); with text it shows
                Send, which queues. */}
            {(() => {
              const running = !bareAsk && (agentStatus === 'running' || agentStatus === 'waiting');
              const stop = stopCurrentTurn;
              const showStop = running && !canSend;
              const showSend = !showStop;
              return (
                <div className="flex shrink-0 items-center gap-1.5 max-md:gap-0">
                  {showStop && (
                    <SendButton
                      kind="stop"
                      label={isCancelling ? t('chat.cancelling') : t('chat.stop')}
                      title={isCancelling ? t('chat.cancelling') : t('chat.stopEsc')}
                      disabled={isCancelling}
                      pending={isCancelling}
                      testId="composer-stop"
                      onClick={stop}
                      onTouchEnd={(e) => { e.preventDefault(); stop(); }}
                    />
                  )}
                  {showSend && (
                    <SendButton
                      label={t('chat.send')}
                      title={disabledReason || t('chat.send')}
                      disabled={!canSend || isCancelling}
                      testId="composer-send"
                      onClick={handleSend}
                      onTouchEnd={(e) => { e.preventDefault(); if (canSend && !isCancelling) handleSend(); }}
                    />
                  )}
                </div>
              );
            })()}
          </div>
        </div>
        {/* Keyless onboarding: an empty chat on a project whose own agent
            has no key says who the composer is talking to, and why. */}
        {!managerReady && pinnedAgentName && !loading && items.length === 0 && !stream && (
          <p
            data-testid="chat-keyless-hint"
            className="mt-2 px-1 text-caption text-secondary"
          >
            {t('chat.keyless.hint', { agent: pinnedAgentName })}
          </p>
        )}
        </>
        )}
      </div>
      </>
      )}
    </div>
  );
}
