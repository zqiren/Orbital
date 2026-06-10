// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useMemo } from 'react';
import { useQueue } from '../hooks/useQueue';
import { useT } from '../i18n/useT';
import type { QueueItem } from '../types';
import AutomationsList from './AutomationsList';
import QueueComposer from './QueueComposer';
import QueueHeader from './QueueHeader';
import QueueItemCard from './QueueItemCard';

interface QueueTabProps {
  projectId: string;
}

function Section({
  title,
  testId,
  items,
  onRemove,
  emptyHint,
}: {
  title: string;
  testId: string;
  items: QueueItem[];
  onRemove?: (itemId: string) => void;
  emptyHint?: string;
}) {
  if (items.length === 0 && !emptyHint) return null;
  return (
    <section className="flex flex-col gap-2" data-testid={`queue-section-${testId}`}>
      <h2 className="text-xs font-semibold text-secondary uppercase tracking-wide px-1">
        {title}
        <span className="ml-2 text-secondary/60 font-normal">{items.length}</span>
      </h2>
      {items.length === 0 && emptyHint ? (
        <p className="text-sm text-secondary px-1 italic">{emptyHint}</p>
      ) : (
        <div className="flex flex-col gap-2">
          {items.map((item, i) => (
            <QueueItemCard key={item.id} item={item} index={i + 1} onRemove={onRemove} />
          ))}
        </div>
      )}
    </section>
  );
}

export default function QueueTab({ projectId }: QueueTabProps) {
  const t = useT();
  const { snapshot, loading, error, addItem, removeItem, stopQueue, resumeQueue } =
    useQueue(projectId);

  const grouped = useMemo(() => {
    const items = snapshot?.items ?? [];
    return {
      running: items.filter((it) => it.state === 'running'),
      blocked: items.filter((it) => it.state === 'blocked'),
      queued: items.filter((it) => it.state === 'queued'),
      done: items.filter((it) => it.state === 'done'),
    };
  }, [snapshot]);

  const isPaused = snapshot?.state === 'paused';

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <QueueHeader
        snapshot={snapshot}
        onStop={stopQueue}
        onResume={resumeQueue}
        disabled={loading}
      />
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 max-md:px-4">
        {error && (
          <div className="text-sm text-error border border-error/40 bg-error/5 rounded px-3 py-2">
            {error}
          </div>
        )}
        <Section
          title={t('queue.section.nowRunning')}
          testId="now-running"
          items={grouped.running}
          onRemove={removeItem}
          emptyHint={
            isPaused
              ? t('queue.empty.paused')
              : grouped.queued.length === 0
                ? t('queue.empty.idle')
                : undefined
          }
        />
        <Section title={t('queue.section.needsAttention')} testId="needs-attention" items={grouped.blocked} onRemove={removeItem} />
        <Section title={t('queue.section.queued')} testId="queued" items={grouped.queued} onRemove={removeItem} />
        <Section title={t('queue.section.completed')} testId="completed" items={grouped.done} onRemove={removeItem} />
        <section className="flex flex-col gap-2" data-testid="queue-section-automations">
          <h2 className="text-xs font-semibold text-secondary uppercase tracking-wide px-1">
            {t('queue.section.automations')}
          </h2>
          <AutomationsList projectId={projectId} />
        </section>
      </div>
      <QueueComposer
        onSubmit={(content, opts) =>
          addItem(content, { priority: opts.priority, review: opts.review })
        }
        hint={
          isPaused
            ? t('queue.composer.hint.paused')
            : t('queue.composer.hint.active')
        }
      />
    </div>
  );
}
