// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { AlertCircle, CheckCheck, Paperclip, User, X, Zap } from 'lucide-react';
import type { QueueItem } from '../types';
import { useT } from '../i18n/useT';

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
function priorityMeta(priority: number): { labelKey: string; cls: string } {
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

function SourceChip({ source }: { source: QueueItem['source'] }) {
  const t = useT();
  const labelKey =
    source === 'trigger'
      ? 'queue.item.source.agent'
      : source === 'upload'
        ? 'queue.item.source.attached'
        : 'queue.item.source.you';
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border border-border bg-sidebar text-[10px] font-mono font-medium text-secondary shrink-0">
      <SourceIcon source={source} className="w-2.5 h-2.5 shrink-0" />
      {t(labelKey)}
    </span>
  );
}

function TriggerOrigin({ item }: { item: QueueItem }) {
  const t = useT();
  if (!isTriggerItem(item)) return null;
  return (
    <div className="flex items-center gap-1 pl-6" data-testid="queue-item-trigger-origin">
      <Zap className="w-3 h-3 shrink-0 text-accent" aria-hidden="true" />
      <span className="text-xs text-accent">{t('queue.item.from', { name: item.trigger_name ?? t('queue.item.automation') })}</span>
    </div>
  );
}

function InterruptedNote({ item }: { item: QueueItem }) {
  const t = useT();
  if (item.interrupted_count <= 0) return null;
  return <p className="text-xs text-warning pl-6">{t('queue.item.interrupted', { n: item.interrupted_count })}</p>;
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
    <Shell
      item={item}
      className="relative overflow-hidden rounded-[10px] border border-accent/40 border-l-[3px] border-l-accent bg-white p-3 flex flex-col gap-2 max-md:p-2"
    >
      <div className="flex items-center gap-2">
        <span className="relative w-2 h-2 shrink-0" aria-hidden>
          <span className="absolute inset-0 rounded-full bg-success opacity-40 animate-ping" />
          <span className="absolute inset-0 rounded-full bg-success" />
        </span>
        <span className="text-[10.5px] uppercase tracking-wide font-semibold text-success">
          {t('queue.item.running')}
        </span>
        {started && <span className="text-[11px] text-muted font-mono">{t('queue.item.started', { time: started })}</span>}
        <div className="flex-1" />
        <SourceChip source={item.source} />
        {/* Disabled until the item idles — see RemoveButton's LOCKED BEHAVIOR note. */}
        <RemoveButton item={item} onRemove={onRemove} />
      </div>
      <p className="text-[13px] font-medium text-primary whitespace-pre-wrap break-words">
        {item.content}
      </p>
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
    <Shell
      item={item}
      className="bg-white border border-border rounded-lg px-2.5 py-2 flex flex-col gap-1 max-md:px-2"
    >
      <div className="flex items-center gap-2.5">
        {index != null && (
          <span className="w-4 text-right text-[10.5px] font-mono text-muted shrink-0">{index}</span>
        )}
        <SourceIcon source={item.source} />
        <span className="flex-1 min-w-0 text-[12.5px] text-primary truncate">{item.content}</span>
        <span className={`text-[10.5px] font-mono shrink-0 ${pri.cls}`}>{t(pri.labelKey)}</span>
        {added && <span className="text-[10.5px] font-mono text-muted shrink-0">{added}</span>}
        <RemoveButton item={item} onRemove={onRemove} />
      </div>
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
    <Shell
      item={item}
      className="bg-white border border-warning/40 border-l-[3px] border-l-warning rounded-lg px-2.5 py-2 flex flex-col gap-1.5 max-md:px-2"
    >
      <div className="flex items-center gap-2.5">
        <AlertCircle className="w-4 h-4 shrink-0 text-warning" aria-hidden />
        <span className="flex-1 min-w-0 text-[12.5px] text-primary break-words">{item.content}</span>
        <span className="px-2 py-0.5 rounded-full bg-warning/15 text-warning text-[9.5px] font-mono font-semibold tracking-wide shrink-0">
          {t('queue.item.needsYou')}
        </span>
        <RemoveButton item={item} onRemove={onRemove} />
      </div>
      {latest?.block_reason && <p className="text-xs text-warning pl-6">{latest.block_reason}</p>}
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
    <Shell item={item} className="px-2.5 py-1.5 flex flex-col gap-1 max-md:px-2">
      <div className="flex items-center gap-2.5">
        <CheckCheck className="w-3.5 h-3.5 shrink-0 text-success" aria-hidden />
        <SourceIcon source={item.source} className="w-3 h-3 shrink-0 text-muted" />
        <div className="flex-1 min-w-0">
          <div className="text-[12.5px] text-primary truncate">{item.content}</div>
          {latest?.summary && (
            <div className="text-[11px] text-muted font-mono truncate">
              <span aria-hidden>→ </span>
              <span>{latest.summary}</span>
            </div>
          )}
        </div>
        {when && <span className="text-[10.5px] font-mono text-muted shrink-0">{when}</span>}
        <RemoveButton item={item} onRemove={onRemove} />
      </div>
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
