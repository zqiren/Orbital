// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { CheckCheck, Paperclip, User, X, Zap } from 'lucide-react';
import type { QueueAttempt, QueueItem } from '../types';
import { useT } from '../i18n/useT';
import MessageAvatar from './MessageAvatar';
import type { StringKey } from '../i18n/strings';

// Localizable block-reason codes (queue/dispatcher.py). `agent_declared` is
// deliberately absent — that prose is agent content and renders verbatim. So is
// spec 079's `agent_dispatch_failed`: its reason IS the dispatch error ("Error:
// agent 'x' is not installed"), and a translated stand-in would replace the one
// thing that explains why the assigned worker never ran.
const BLOCK_REASON_KEYS: Record<string, StringKey> = {
  daemon_restart: 'queue.blockReason.daemon_restart',
  inject_failed: 'queue.blockReason.inject_failed',
  inject_failed_retry: 'queue.blockReason.inject_failed_retry',
  runtime_cap: 'queue.blockReason.runtime_cap',
  cancelled: 'queue.blockReason.cancelled',
  budget_blocked: 'queue.blockReason.budget_blocked',
  hold_deadline: 'queue.blockReason.hold_deadline',
  contract_violation: 'queue.blockReason.contract_violation',
};

function blockReasonText(latest: QueueAttempt, t: ReturnType<typeof useT>): string {
  const key = latest.block_reason_code
    ? BLOCK_REASON_KEYS[latest.block_reason_code]
    : undefined;
  return key ? t(key) : (latest.block_reason ?? '');
}

interface QueueItemCardProps {
  item: QueueItem;
  /** 1-based position within its section. Used by queued rows for the index column. */
  index?: number;
  onRemove?: (itemId: string) => void;
}

/** 24-hour HH:MM from an ISO timestamp; empty string if unparseable. */
function formatTime(timestamp: string | null | undefined): string {
  if (!timestamp) return '';
  const d = new Date(timestamp);
  if (Number.isNaN(d.getTime())) return '';
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}

/** Returns true when the item was spawned by an automation trigger. */
function isTriggerItem(item: QueueItem): boolean {
  return item.source === 'trigger' || item.trigger_name !== undefined;
}

/**
 * Map the backend's integer priority onto the mockup's low/normal/high label.
 * The composer only ever sets 0 (normal) or 1 (pinned/high); negatives are
 * reserved for a future "low" tier. Purely a display mapping — the stored
 * value is untouched.
 */
function priorityMeta(priority: number): { labelKey: StringKey; cls: string } {
  if (priority > 0) return { labelKey: 'queue.item.priority.high', cls: 'text-warning' };
  if (priority < 0) return { labelKey: 'queue.item.priority.low', cls: 'text-muted' };
  return { labelKey: 'queue.item.priority.normal', cls: 'text-secondary' };
}

function SourceIcon({
  source,
  className = 'w-3 h-3 shrink-0',
}: {
  source: QueueItem['source'];
  className?: string;
}) {
  switch (source) {
    case 'trigger':
      return <Zap className={`${className} text-accent`} aria-hidden />;
    case 'upload':
      return <Paperclip className={`${className} text-secondary`} aria-hidden />;
    default:
      return <User className={`${className} text-secondary`} aria-hidden />;
  }
}

/**
 * One surface for every queue state — the same recipe as WorkbenchCard and the
 * Workbench "Today" strip. Previously each of the four states had its OWN
 * container: running and blocked carried saturated 3px coloured rails plus
 * tinted borders, queued was a dense bordered strip, and done had no surface at
 * all. State is now carried by a small dot and a quiet label, exactly as the
 * sidebar and the Workbench cards do it.
 */
const CARD =
  'rounded-xl border border-border/60 bg-card px-4 py-3 shadow-[0_1px_2px_rgb(0_0_0/0.04)] ' +
  'flex flex-col gap-1.5';

/** Quiet metadata line — matches the Workbench card's meta row. */
const META = 'text-[11px] text-muted';

/** State marker: a 6px dot, optionally pulsing. Replaces the coloured rails. */
function StateDot({ tone, pulse = false }: { tone: string; pulse?: boolean }) {
  return (
    <span className="relative h-1.5 w-1.5 shrink-0" aria-hidden="true">
      {pulse && (
        <span
          className={`absolute inset-0 rounded-full ${tone} opacity-40 motion-safe:animate-ping motion-reduce:hidden`}
        />
      )}
      <span className={`absolute inset-0 rounded-full ${tone}`} />
    </span>
  );
}

function SourceChip({ source }: { source: QueueItem['source'] }) {
  const t = useT();
  const labelKey =
    source === 'trigger'
      ? 'queue.item.source.agent'
      : source === 'upload'
        ? 'queue.item.source.attached'
        : 'queue.item.source.you';
  return (
    <span className={`inline-flex shrink-0 items-center gap-1 ${META}`}>
      <SourceIcon source={source} className="w-2.5 h-2.5 shrink-0" />
      {t(labelKey)}
    </span>
  );
}

/**
 * Spec 079 — the worker the user assigned to run this item. Drawn only when
 * one was chosen: an unassigned item is run by Orbital, which is the norm and
 * needs no mark. The avatar falls back to a monogram badge for a slug whose
 * agent has since been uninstalled, so a stale choice stays visible rather than
 * silently reading as "Orbital runs it".
 */
function AgentChip({ item }: { item: QueueItem }) {
  const t = useT();
  if (!item.agent) return null;
  return (
    <span
      className="inline-flex shrink-0 items-center"
      title={t('queue.item.agent.chip', { name: item.agent })}
      data-testid={`queue-item-agent-${item.id}`}
    >
      <MessageAvatar variant="agent" agentHandle={item.agent} />
    </span>
  );
}

/** Last path segment of a workspace-relative file ref. */
function basename(ref: string): string {
  const parts = ref.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || ref;
}

/**
 * Attached filenames, so an item composed with files shows what it carries.
 * Paths are backend data — rendered verbatim, never translated.
 */
function AttachedFiles({ item }: { item: QueueItem }) {
  const refs = item.file_refs ?? [];
  if (refs.length === 0) return null;
  return (
    <div
      className={`flex flex-wrap items-center gap-x-2 gap-y-0.5 ${META}`}
      data-testid="queue-item-file-refs"
    >
      <Paperclip className="w-2.5 h-2.5 shrink-0" aria-hidden="true" />
      {refs.map((ref) => (
        <span key={ref} className="truncate max-w-[180px]" title={ref}>
          {basename(ref)}
        </span>
      ))}
    </div>
  );
}

function TriggerOrigin({ item }: { item: QueueItem }) {
  const t = useT();
  if (!isTriggerItem(item)) return null;
  return (
    <div className={`flex items-center gap-1 ${META}`} data-testid="queue-item-trigger-origin">
      <Zap className="w-3 h-3 shrink-0 text-muted" aria-hidden="true" />
      <span>{t('queue.item.from', { name: item.trigger_name ?? t('queue.item.automation') })}</span>
    </div>
  );
}

function InterruptedNote({ item }: { item: QueueItem }) {
  const t = useT();
  if (item.interrupted_count <= 0) return null;
  return (
    <p className="text-[11px] text-warning">
      {t('queue.item.interrupted', { n: item.interrupted_count })}
    </p>
  );
}

function RemoveButton({
  item,
  onRemove,
}: {
  item: QueueItem;
  onRemove?: (itemId: string) => void;
}) {
  const t = useT();
  if (!onRemove) return null;
  // LOCKED BEHAVIOR: a running item's delete control stays rendered but
  // DISABLED until the item idles — it is not a stop-first popup, just a
  // disabled button. The user can only remove an item once it is no longer
  // running (queued / blocked / done).
  const running = item.state === 'running';
  return (
    <button
      onClick={() => onRemove(item.id)}
      disabled={running}
      aria-disabled={running}
      aria-label={t('queue.item.removeAria')}
      title={running ? t('queue.item.removeBlockedTitle') : undefined}
      className={
        running
          ? 'text-secondary/40 shrink-0 p-1 -m-1 rounded cursor-not-allowed'
          : 'text-secondary hover:text-error shrink-0 p-1 -m-1 rounded transition-colors'
      }
    >
      <X className="w-3.5 h-3.5" />
    </button>
  );
}

/** Common root: every state must carry the testid + data-state for the test suite. */
function Shell({
  item,
  className,
  children,
}: {
  item: QueueItem;
  className: string;
  children: React.ReactNode;
}) {
  return (
    <div className={className} data-testid={`queue-item-${item.id}`} data-state={item.state}>
      {children}
    </div>
  );
}

/** Running — accent-railed card with a live pulse, started-at, and source chip. */
function RunningCard({
  item,
  onRemove,
}: {
  item: QueueItem;
  onRemove?: (itemId: string) => void;
}) {
  const t = useT();
  const startedAt = item.attempts.length ? item.attempts[item.attempts.length - 1].started_at : null;
  const started = formatTime(startedAt);
  return (
    <Shell item={item} className={CARD}>
      <div className="flex items-center gap-2">
        <StateDot tone="bg-success" pulse />
        <span className="text-[11px] font-medium text-secondary">{t('queue.item.running')}</span>
        {started && (
          <span className={`font-mono tabular-nums ${META}`}>
            {t('queue.item.started', { time: started })}
          </span>
        )}
        <div className="flex-1" />
        <AgentChip item={item} />
        <SourceChip source={item.source} />
        {/* Disabled until the item idles — see RemoveButton's LOCKED BEHAVIOR note. */}
        <RemoveButton item={item} onRemove={onRemove} />
      </div>
      <p className="text-[13px] text-primary whitespace-pre-wrap break-words">{item.content}</p>
      <AttachedFiles item={item} />
      <TriggerOrigin item={item} />
      <InterruptedNote item={item} />
    </Shell>
  );
}

/** Queued — dense grid row on a white panel: #, source, content, priority, time, remove. */
function QueuedRow({
  item,
  index,
  onRemove,
}: {
  item: QueueItem;
  index?: number;
  onRemove?: (itemId: string) => void;
}) {
  const t = useT();
  const pri = priorityMeta(item.priority);
  const added = formatTime(item.created_at);
  return (
    <Shell item={item} className={CARD}>
      <div className="flex items-center gap-2.5">
        {index != null && (
          <span className={`w-4 shrink-0 text-right font-mono tabular-nums ${META}`}>{index}</span>
        )}
        <StateDot tone="bg-idle" />
        <span className="min-w-0 flex-1 truncate text-[13px] text-primary">{item.content}</span>
        <AgentChip item={item} />
        <span className={`shrink-0 text-[11px] ${pri.cls}`}>{t(pri.labelKey)}</span>
        {added && <span className={`shrink-0 font-mono tabular-nums ${META}`}>{added}</span>}
        <RemoveButton item={item} onRemove={onRemove} />
      </div>
      <AttachedFiles item={item} />
      <TriggerOrigin item={item} />
      <InterruptedNote item={item} />
    </Shell>
  );
}

/** Blocked / "Needs Attention" — warning-railed row with a "needs you" pill + reason. */
function BlockedRow({
  item,
  onRemove,
}: {
  item: QueueItem;
  onRemove?: (itemId: string) => void;
}) {
  const t = useT();
  const latest = item.attempts.length ? item.attempts[item.attempts.length - 1] : null;
  return (
    <Shell item={item} className={CARD}>
      <div className="flex items-center gap-2.5">
        <StateDot tone="bg-warning" />
        <span className="min-w-0 flex-1 break-words text-[13px] text-primary">{item.content}</span>
        <AgentChip item={item} />
        <span className="shrink-0 text-[11px] font-medium text-warning">
          {t('queue.item.needsYou')}
        </span>
        <RemoveButton item={item} onRemove={onRemove} />
      </div>
      {latest && (latest.block_reason || latest.block_reason_code) && (
        <p className="text-[11px] text-warning">{blockReasonText(latest, t)}</p>
      )}
      <AttachedFiles item={item} />
      <TriggerOrigin item={item} />
      <InterruptedNote item={item} />
    </Shell>
  );
}

/** Completed — flat row (no card/fill): green check, content, "→ summary", when. */
function DoneRow({
  item,
  onRemove,
}: {
  item: QueueItem;
  onRemove?: (itemId: string) => void;
}) {
  const latest = item.attempts.length ? item.attempts[item.attempts.length - 1] : null;
  const when = formatTime(latest?.ended_at ?? item.created_at);
  return (
    // Same surface, dialled down: a completed item is history, so the border
    // lightens and the content drops to secondary. A long done-list therefore
    // recedes instead of stacking full-weight cards.
    <Shell item={item} className={`${CARD} border-border/40`}>
      <div className="flex items-center gap-2.5">
        <CheckCheck className="w-3.5 h-3.5 shrink-0 text-success/70" aria-hidden />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] text-secondary">{item.content}</div>
          {latest?.summary && (
            <div className={`truncate ${META}`}>
              <span aria-hidden>→ </span>
              <span>{latest.summary}</span>
            </div>
          )}
        </div>
        <AgentChip item={item} />
        {when && <span className={`shrink-0 font-mono tabular-nums ${META}`}>{when}</span>}
        <RemoveButton item={item} onRemove={onRemove} />
      </div>
      <AttachedFiles item={item} />
      <TriggerOrigin item={item} />
    </Shell>
  );
}

export default function QueueItemCard({ item, index, onRemove }: QueueItemCardProps) {
  switch (item.state) {
    case 'running':
      return <RunningCard item={item} onRemove={onRemove} />;
    case 'done':
      return <DoneRow item={item} onRemove={onRemove} />;
    case 'blocked':
      return <BlockedRow item={item} onRemove={onRemove} />;
    case 'queued':
    default:
      return <QueuedRow item={item} index={index} onRemove={onRemove} />;
  }
}
