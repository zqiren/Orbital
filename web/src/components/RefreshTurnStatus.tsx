// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * RefreshTurnStatus — inline indicator for periodic state checkpoint events.
 *
 * Visually distinct from chat bubbles: centered, colored-border block styled
 * after the agent_notify card (not a left/right bubble). Supports four states:
 *   in_progress — pulsing, neutral color
 *   done        — muted success color
 *   failed      — error color
 *   skipped     — muted, lower opacity
 *
 * The trigger prop shows which mechanism fired: turn_count, agent_decided,
 * or token_pressure.
 */

export type RefreshStatus = 'in_progress' | 'done' | 'failed' | 'skipped';
export type RefreshTrigger = 'turn_count' | 'agent_decided' | 'token_pressure';

interface RefreshTurnStatusProps {
  status: RefreshStatus;
  trigger: RefreshTrigger;
  timestamp: string;
}

const TRIGGER_LABELS: Record<RefreshTrigger, string> = {
  turn_count: 'turn-count',
  agent_decided: 'agent-decided',
  token_pressure: 'token-pressure',
};

const STATUS_CONFIG: Record<RefreshStatus, { label: string; borderClass: string; textClass: string; dotClass: string }> = {
  in_progress: {
    label: 'Checkpointing project state\u2026',
    borderClass: 'border-border/60',
    textClass: 'text-secondary',
    dotClass: 'bg-accent animate-pulse',
  },
  done: {
    label: 'Project state saved',
    borderClass: 'border-border/40',
    textClass: 'text-tertiary',
    dotClass: 'bg-success',
  },
  failed: {
    label: 'Checkpoint failed',
    borderClass: 'border-error/40',
    textClass: 'text-error/80',
    dotClass: 'bg-error',
  },
  skipped: {
    label: 'Checkpoint skipped',
    borderClass: 'border-border/20',
    textClass: 'text-tertiary opacity-60',
    dotClass: 'bg-border',
  },
};

export default function RefreshTurnStatus({ status, trigger, timestamp }: RefreshTurnStatusProps) {
  const config = STATUS_CONFIG[status];
  const triggerLabel = TRIGGER_LABELS[trigger];
  const time = new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className={`my-2 mx-auto max-w-sm rounded border ${config.borderClass} bg-surface-alt/40 px-3 py-2 flex items-center gap-2`}>
      <span className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${config.dotClass}`} />
      <span className={`text-xs ${config.textClass} flex-1`}>
        {config.label}
        <span className="ml-1 opacity-50">({triggerLabel})</span>
      </span>
      <span className="text-xs text-tertiary opacity-40 flex-shrink-0">{time}</span>
    </div>
  );
}
