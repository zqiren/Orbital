// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { CheckCircle2, Circle, Loader2, X, AlertCircle } from 'lucide-react';
import type { QueueItem } from '../types';

interface QueueItemCardProps {
  item: QueueItem;
  onRemove?: (itemId: string) => void;
}

const STATE_ACCENT: Record<QueueItem['state'], string> = {
  queued: 'border-border',
  running: 'border-info/40 bg-info/5',
  done: 'border-success/30 bg-success/5',
  blocked: 'border-warning/40 bg-warning/5',
};

function StateIcon({ state }: { state: QueueItem['state'] }) {
  const cls = 'w-4 h-4 shrink-0';
  switch (state) {
    case 'queued':
      return <Circle className={`${cls} text-secondary`} />;
    case 'running':
      return <Loader2 className={`${cls} text-info animate-spin`} />;
    case 'done':
      return <CheckCircle2 className={`${cls} text-success`} />;
    case 'blocked':
      return <AlertCircle className={`${cls} text-warning`} />;
  }
}

export default function QueueItemCard({ item, onRemove }: QueueItemCardProps) {
  const latest = item.attempts.length > 0 ? item.attempts[item.attempts.length - 1] : null;
  const showDetail =
    (item.state === 'done' && latest?.summary) ||
    (item.state === 'blocked' && latest?.block_reason);

  return (
    <div
      className={`rounded-lg border ${STATE_ACCENT[item.state]} p-3 flex flex-col gap-2 max-md:p-2`}
      data-testid={`queue-item-${item.id}`}
      data-state={item.state}
    >
      <div className="flex items-start gap-2">
        <StateIcon state={item.state} />
        <p className="flex-1 text-sm text-primary whitespace-pre-wrap break-words">
          {item.content}
        </p>
        {onRemove && item.state !== 'running' && (
          <button
            onClick={() => onRemove(item.id)}
            aria-label="Remove item"
            className="text-secondary hover:text-error shrink-0 p-1 -m-1 rounded transition-colors"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
      {showDetail && (
        <p className="text-xs text-secondary pl-6">
          {item.state === 'done' ? latest?.summary : latest?.block_reason}
        </p>
      )}
      {item.interrupted_count > 0 && (
        <p className="text-xs text-warning pl-6">
          Interrupted {item.interrupted_count}×
        </p>
      )}
    </div>
  );
}
