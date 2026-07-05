// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useMemo } from 'react';
import { useQueue } from '../hooks/useQueue';
import { useCost } from '../hooks/useCost';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import type { StringKey } from '../i18n/strings';
import type { Project, QueueItem } from '../types';
import { formatMoney } from '../budget/format';
import { periodToWindow } from '../budget/derive';
import AutomationsList from './AutomationsList';
import QueueComposer from './QueueComposer';
import QueueHeader from './QueueHeader';
import type { QueueBudgetCost } from './QueueHeader';
import QueueItemCard from './QueueItemCard';

interface QueueTabProps {
  projectId: string;
  /**
   * Full project — used to resolve the budget window/limit/currency for the
   * budget-pause banner. Optional so existing tests that mount with just a
   * projectId keep working (banner simply doesn't render without it).
   */
  project?: Project;
}

const WINDOW_WORD: Record<'daily' | 'weekly' | 'monthly' | 'total', StringKey> = {
  daily: 'settings.budget.window.day',
  weekly: 'settings.budget.window.week',
  monthly: 'settings.budget.window.month',
  total: 'settings.budget.window.total',
};

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

export default function QueueTab({ projectId, project }: QueueTabProps) {
  const t = useT();
  const { locale } = useLocale();
  const { snapshot, loading, error, addItem, removeItem, stopQueue, resumeQueue } =
    useQueue(projectId);

  // Budget-pause banner figures: only fetch /cost (and only render the banner)
  // when the queue is paused for budget. The window/limit come from the project.
  const isBudgetPaused =
    snapshot?.state === 'paused' && snapshot?.pause_reason === 'budget';
  const period = project?.budget_period || 'daily';
  const window = periodToWindow(period);
  // useCost subscribes to budget.spend_updated — event-driven, no polling. Pass
  // null when not budget-paused / no project so we don't fetch unnecessarily.
  const { cost: budgetCostResp } = useCost(
    isBudgetPaused && project ? projectId : null,
    window,
  );
  const budgetCost: QueueBudgetCost | null = useMemo(() => {
    if (!isBudgetPaused || !budgetCostResp) return null;
    const ccy = budgetCostResp.converted_total.currency;
    const spent = budgetCostResp.converted_total.amount;
    const limitNum =
      project?.budget_limit_usd != null && project.budget_limit_usd > 0
        ? project.budget_limit_usd
        : null;
    return {
      spent: formatMoney(spent, ccy, locale),
      limit: limitNum != null ? formatMoney(limitNum, ccy, locale) : null,
      window: t(WINDOW_WORD[window]),
    };
    // t is unstable; bind to locale per the i18n convention.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isBudgetPaused, budgetCostResp, project?.budget_limit_usd, window, locale]);

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
    <div className="flex flex-col h-full min-h-0">
      <QueueHeader
        snapshot={snapshot}
        onStop={stopQueue}
        onResume={resumeQueue}
        disabled={loading}
        budgetCost={budgetCost}
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
