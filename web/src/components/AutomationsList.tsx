// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect } from 'react';
import { Calendar, FolderOpen } from 'lucide-react';
import { useTriggers } from '../hooks/useTriggers';
import type { Trigger } from '../types';

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
  const isSchedule = trigger.type === 'schedule';
  const condition = isSchedule
    ? (trigger.schedule?.cron ?? '—')
    : (trigger.watch_path ?? '—');
  const conditionLabel = isSchedule ? 'cron' : 'path';

  return (
    <div
      className="rounded-[6px] border border-border bg-background px-3 py-2.5 flex flex-col gap-1.5"
      data-testid={`automation-card-${trigger.id}`}
    >
      <div className="flex items-center gap-2">
        {isSchedule ? (
          <Calendar className="w-3.5 h-3.5 shrink-0 text-accent" aria-hidden />
        ) : (
          <FolderOpen className="w-3.5 h-3.5 shrink-0 text-accent" aria-hidden />
        )}
        <span className="text-sm font-medium text-primary flex-1 truncate">
          {trigger.name}
        </span>
        <span
          className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full leading-none ${
            trigger.enabled
              ? 'bg-accent/10 text-accent'
              : 'bg-border text-secondary'
          }`}
          aria-label={trigger.enabled ? 'enabled' : 'disabled'}
          data-testid={`automation-status-${trigger.id}`}
        >
          {trigger.enabled ? 'on' : 'off'}
        </span>
      </div>

      <div className="pl-[22px] flex flex-col gap-1">
        <p className="text-xs text-secondary">
          <span className="uppercase tracking-wide text-[10px] text-secondary/60 mr-1">
            {conditionLabel}
          </span>
          <span className="font-mono text-primary/80" data-testid={`automation-condition-${trigger.id}`}>
            {condition}
          </span>
        </p>
        <p className="text-xs text-secondary">
          <span className="uppercase tracking-wide text-[10px] text-secondary/60 mr-1">
            last fired
          </span>
          <span data-testid={`automation-last-fired-${trigger.id}`}>
            {formatLastFired(trigger.last_triggered)}
          </span>
        </p>
      </div>
    </div>
  );
}

export default function AutomationsList({ projectId }: AutomationsListProps) {
  const { triggers, loading, fetchTriggers } = useTriggers(projectId);

  useEffect(() => {
    void fetchTriggers();
  }, [fetchTriggers]);

  if (loading && triggers.length === 0) {
    return (
      <p className="text-sm text-secondary px-1 italic" data-testid="automations-loading">
        Loading…
      </p>
    );
  }

  if (triggers.length === 0) {
    return (
      <p className="text-sm text-secondary px-1 italic" data-testid="automations-empty">
        No automations configured.
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
