// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect } from 'react';
import { Calendar, FolderOpen } from 'lucide-react';
import { useTriggers } from '../hooks/useTriggers';
import type { Trigger } from '../types';
import { useT } from '../i18n/useT';

interface AutomationsListProps {
  projectId: string;
}

function formatLastFired(last: string | null): string {
  if (!last) return '—';
  try {
    const d = new Date(last);
    if (isNaN(d.getTime())) return '—';
    return d.toLocaleString();
  } catch {
    return '—';
  }
}

function TriggerCard({ trigger }: { trigger: Trigger }) {
  const t = useT();
  const isSchedule = trigger.type === 'schedule';
  const condition = isSchedule
    ? (trigger.schedule?.cron ?? '—')
    : (trigger.watch_path ?? '—');
  const conditionLabel = isSchedule ? t('automations.condition.cron') : t('automations.condition.path');

  return (
    // Same card family as the queue items — identical radius, border weight and
    // padding — but deliberately FLAT (no shadow, lighter hairline). An
    // automation is a definition, not work in flight: it produces queue items
    // rather than being one. Keeping it in the family but one step back in the
    // depth hierarchy says "related, different kind of thing" without needing a
    // second visual language. The left slot reinforces it: queue items carry a
    // state dot, automations carry a type icon.
    <div
      className="flex flex-col gap-2 rounded-xl border border-border/50 bg-card px-4 py-3"
      data-testid={`automation-card-${trigger.id}`}
    >
      <div className="flex items-center gap-2">
        {isSchedule ? (
          <Calendar className="w-3.5 h-3.5 shrink-0 text-muted" aria-hidden />
        ) : (
          <FolderOpen className="w-3.5 h-3.5 shrink-0 text-muted" aria-hidden />
        )}
        <span className="flex-1 truncate text-[13px] font-medium text-primary">
          {trigger.name}
        </span>
        {/* Dot + label, matching the queue's state markers, instead of a
            tinted pill. Enabled/disabled stays legible because a paused
            automation is a silent failure mode. */}
        <span
          className={`inline-flex shrink-0 items-center gap-1.5 text-[11px] ${
            trigger.enabled ? 'text-secondary' : 'text-muted'
          }`}
          aria-label={trigger.enabled ? 'enabled' : 'disabled'}
          data-testid={`automation-status-${trigger.id}`}
        >
          <span
            aria-hidden="true"
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
              trigger.enabled ? 'bg-success' : 'bg-idle'
            }`}
          />
          {trigger.enabled ? t('automations.on') : t('automations.off')}
        </span>
      </div>

      <div className="flex flex-col gap-1 pl-[22px]">
        <p className="text-[11px] text-muted">
          <span className="mr-1.5 uppercase tracking-wide">{conditionLabel}</span>
          {/* A cron expression / watch path is an identifier — stays mono. */}
          <span
            className="font-mono text-secondary"
            data-testid={`automation-condition-${trigger.id}`}
          >
            {condition}
          </span>
        </p>
        <p className="text-[11px] text-muted">
          <span className="mr-1.5 uppercase tracking-wide">{t('automations.lastFired')}</span>
          <span
            className="font-mono tabular-nums text-secondary"
            data-testid={`automation-last-fired-${trigger.id}`}
          >
            {formatLastFired(trigger.last_triggered)}
          </span>
        </p>
      </div>
    </div>
  );
}

export default function AutomationsList({ projectId }: AutomationsListProps) {
  const t = useT();
  const { triggers, loading, fetchTriggers } = useTriggers(projectId);

  useEffect(() => {
    void fetchTriggers();
  }, [fetchTriggers]);

  if (loading && triggers.length === 0) {
    return (
      <p className="text-sm text-secondary px-1 italic" data-testid="automations-loading">
        {t('automations.loading')}
      </p>
    );
  }

  if (triggers.length === 0) {
    return (
      <p className="text-sm text-secondary px-1 italic" data-testid="automations-empty">
        {t('automations.empty')}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2" data-testid="automations-list">
      {triggers.map((t) => (
        <TriggerCard key={t.id} trigger={t} />
      ))}
    </div>
  );
}
