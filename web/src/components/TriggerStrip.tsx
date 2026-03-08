// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useState } from 'react';
import { Clock, Eye, Trash2 } from 'lucide-react';
import type { Trigger } from '../types';

interface TriggerStripProps {
  triggers: Trigger[];
  onToggle: (triggerId: string, enabled: boolean) => void;
  onDelete?: (triggerId: string) => void;
}

function relativeTime(iso: string | null): string {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

/* Issue 1: Replace emoji icons with Lucide React icons */
function TriggerIcon({ type }: { type: string }) {
  if (type === 'file_watch') {
    return <Eye size={16} className="text-secondary shrink-0" aria-label="File watch trigger" />;
  }
  return <Clock size={16} className="text-secondary shrink-0" aria-label="Schedule trigger" />;
}

/* Issue 2: Extracted toggle component with fixed alignment */
function Toggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label
      className="relative inline-flex items-center cursor-pointer shrink-0"
      onClick={(e) => e.stopPropagation()}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="sr-only peer"
      />
      <div className="relative w-9 h-5 bg-border rounded-full peer peer-checked:bg-accent transition-colors duration-150 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:after:translate-x-4" />
    </label>
  );
}

/**
 * Shared detail content rendered in both inline expand and bottom sheet.
 * Issue 4: showName controls whether the name is rendered (false in bottom sheet to avoid duplicate).
 * Issue 5: Cron expression removed — only human-readable schedule shown.
 * Issue 7: Task as section header with muted body text; metadata in labeled grid.
 */
function TriggerDetail({
  trigger,
  showName = true,
  onDelete,
}: {
  trigger: Trigger;
  showName?: boolean;
  onDelete?: (triggerId: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const lastFired = trigger.last_triggered ? relativeTime(trigger.last_triggered) : 'never';

  const handleDelete = async () => {
    if (!onDelete) return;
    setDeleting(true);
    try {
      await onDelete(trigger.id);
    } finally {
      setDeleting(false);
      setConfirmDelete(false);
    }
  };

  return (
    <div className="text-xs space-y-3">
      {showName && <p className="text-primary font-medium">{trigger.name}</p>}

      {/* Task section — header + body for breathing room */}
      <div>
        <p className="font-semibold text-primary">Task</p>
        <p className="text-secondary mt-1 leading-relaxed">{trigger.task}</p>
      </div>

      {/* Metadata grid — semi-bold labels, regular values */}
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 m-0">
        {trigger.type === 'schedule' && trigger.schedule && (
          <>
            <dt className="font-semibold text-primary">Schedule</dt>
            <dd className="text-secondary m-0">{trigger.schedule.human}</dd>
          </>
        )}
        {trigger.type === 'file_watch' && (
          <>
            <dt className="font-semibold text-primary">Watching</dt>
            <dd className="text-secondary m-0">{trigger.watch_path}</dd>
            {trigger.patterns && trigger.patterns.length > 0 && (
              <>
                <dt className="font-semibold text-primary">Patterns</dt>
                <dd className="text-secondary m-0">{trigger.patterns.join(', ')}</dd>
              </>
            )}
            {trigger.recursive && (
              <>
                <dt className="font-semibold text-primary">Recursive</dt>
                <dd className="text-secondary m-0">Yes</dd>
              </>
            )}
          </>
        )}
        <dt className="font-semibold text-primary">Last fired</dt>
        <dd className="text-secondary m-0">{lastFired}</dd>
        {trigger.trigger_count > 0 && (
          <>
            <dt className="font-semibold text-primary">Runs</dt>
            <dd className="text-secondary m-0">{trigger.trigger_count}</dd>
          </>
        )}
      </dl>

      <p className="text-secondary italic">
        To edit, tell the agent below &#8595;
      </p>

      {/* Delete button */}
      {onDelete && (
        <div className="pt-2 border-t border-border mt-3">
          {!confirmDelete ? (
            <button
              onClick={() => setConfirmDelete(true)}
              className="text-xs text-secondary hover:text-error transition-colors"
            >
              Delete trigger
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-xs text-secondary">Delete? Cannot be undone.</span>
              <button
                onClick={() => setConfirmDelete(false)}
                className="text-xs text-secondary hover:text-primary transition-colors"
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                className="text-xs text-error hover:text-error/80 transition-colors"
                disabled={deleting}
              >
                {deleting ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Mobile bottom sheet — slides up from the bottom edge.
 * Issue 3: max-height prevents covering the chat input; bottom offset leaves input tappable.
 * Issue 4: showName=false avoids duplicate trigger name.
 */
function BottomSheet({
  trigger,
  onToggle,
  onDelete,
  onClose,
}: {
  trigger: Trigger;
  onToggle: (triggerId: string, enabled: boolean) => void;
  onDelete?: (triggerId: string) => void;
  onClose: () => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const handleDelete = async () => {
    if (!onDelete) return;
    setDeleting(true);
    try {
      await onDelete(trigger.id);
    } finally {
      setDeleting(false);
      setConfirmDelete(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[70] md:hidden" onClick={onClose}>
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/30" />
      {/* Sheet — offset from bottom to keep chat input visible */}
      <div
        className="absolute bottom-[4.5rem] left-0 right-0 bg-background rounded-t-xl px-4 pb-5 pt-3 animate-slide-up max-h-[60vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Drag handle */}
        <div className="w-10 h-1 bg-border rounded-full mx-auto mb-3" />
        {/* Header with toggle and delete */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 min-w-0">
            <TriggerIcon type={trigger.type} />
            <span className="text-sm font-medium text-primary truncate">{trigger.name}</span>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            {onDelete && (
              <button
                type="button"
                onClick={() => setConfirmDelete(true)}
                className="p-1.5 min-h-[44px] min-w-[44px] flex items-center justify-center text-secondary hover:text-error transition-colors"
                aria-label="Delete trigger"
              >
                <Trash2 size={18} />
              </button>
            )}
            <Toggle
              checked={trigger.enabled}
              onChange={() => onToggle(trigger.id, !trigger.enabled)}
            />
          </div>
        </div>
        {/* Delete confirmation */}
        {confirmDelete && (
          <div className="flex items-center gap-2 mb-3 px-1">
            <span className="text-xs text-secondary">Delete? Cannot be undone.</span>
            <button
              onClick={() => setConfirmDelete(false)}
              className="text-xs text-secondary hover:text-primary transition-colors"
              disabled={deleting}
            >
              Cancel
            </button>
            <button
              onClick={handleDelete}
              className="text-xs text-error hover:text-error/80 transition-colors"
              disabled={deleting}
            >
              {deleting ? 'Deleting...' : 'Delete'}
            </button>
          </div>
        )}
        {/* Separator */}
        <div className="border-t border-border mb-3" />
        {/* Detail — name already shown in header, delete in header */}
        <TriggerDetail trigger={trigger} showName={false} />
      </div>
    </div>
  );
}

function TriggerLine({
  trigger,
  onToggle,
  onDelete,
  expanded,
  onClickLine,
}: {
  trigger: Trigger;
  onToggle: (triggerId: string, enabled: boolean) => void;
  onDelete?: (triggerId: string) => void;
  expanded: boolean;
  onClickLine: () => void;
}) {
  const label =
    trigger.type === 'schedule' && trigger.schedule?.human
      ? trigger.schedule.human
      : trigger.type === 'file_watch' && trigger.watch_path
        ? `Watching ${trigger.watch_path}`
        : trigger.name;

  /* Issue 6: Only show relative time if the trigger has actually fired */
  const lastFired = trigger.last_triggered ? relativeTime(trigger.last_triggered) : null;

  return (
    <div>
      <button
        type="button"
        onClick={onClickLine}
        className="w-full flex items-center gap-2 px-6 py-1.5 max-md:px-4 text-sm hover:bg-sidebar/50 transition-colors duration-100"
      >
        <TriggerIcon type={trigger.type} />
        <span className="text-primary truncate flex-1 text-left">{label}</span>
        {lastFired && (
          <span className="text-xs text-secondary shrink-0">{lastFired}</span>
        )}
        <Toggle
          checked={trigger.enabled}
          onChange={() => onToggle(trigger.id, !trigger.enabled)}
        />
      </button>

      {/* Desktop inline expand */}
      {expanded && (
        <div className="hidden md:block px-6 pb-3 pt-1 text-xs space-y-1 bg-sidebar/30">
          <TriggerDetail trigger={trigger} onDelete={onDelete} />
        </div>
      )}
    </div>
  );
}

export default function TriggerStrip({ triggers, onToggle, onDelete }: TriggerStripProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 767px)');
    setIsMobile(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  const handleClickLine = useCallback(
    (id: string) => {
      setExpandedId((prev) => (prev === id ? null : id));
    },
    [],
  );

  // Reset expansion when triggers change
  useEffect(() => {
    setExpandedId(null);
  }, [triggers.length]);

  if (triggers.length === 0) return null;

  const visible = showAll ? triggers : triggers.slice(0, 2);
  const hiddenCount = triggers.length - 2;
  const expandedTrigger = expandedId
    ? triggers.find((t) => t.id === expandedId) ?? null
    : null;

  return (
    <div className="border-b border-border">
      {visible.map((t) => (
        <TriggerLine
          key={t.id}
          trigger={t}
          onToggle={onToggle}
          onDelete={onDelete}
          expanded={!isMobile && expandedId === t.id}
          onClickLine={() => handleClickLine(t.id)}
        />
      ))}
      {!showAll && hiddenCount > 0 && (
        <button
          type="button"
          onClick={() => setShowAll(true)}
          className="w-full text-xs text-accent px-6 py-1 hover:underline max-md:px-4"
        >
          +{hiddenCount} more
        </button>
      )}
      {showAll && hiddenCount > 0 && (
        <button
          type="button"
          onClick={() => setShowAll(false)}
          className="w-full text-xs text-accent px-6 py-1 hover:underline max-md:px-4"
        >
          Show less
        </button>
      )}

      {/* Mobile bottom sheet */}
      {isMobile && expandedTrigger && (
        <BottomSheet
          trigger={expandedTrigger}
          onToggle={onToggle}
          onDelete={onDelete}
          onClose={() => setExpandedId(null)}
        />
      )}
    </div>
  );
}
