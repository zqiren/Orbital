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

import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

export type RefreshStatus = 'in_progress' | 'done' | 'failed' | 'skipped';
export type RefreshTrigger = 'turn_count' | 'agent_decided' | 'token_pressure';

interface RefreshTurnStatusProps {
  status: RefreshStatus;
  trigger: RefreshTrigger;
  timestamp: string;
}

const TRIGGER_LABEL_KEYS: Record<RefreshTrigger, StringKey> = {
  turn_count: 'refreshTurn.trigger.turnCount',
  agent_decided: 'refreshTurn.trigger.agentDecided',
  token_pressure: 'refreshTurn.trigger.tokenPressure',
};

const STATUS_CONFIG: Record<RefreshStatus, { labelKey: StringKey; borderClass: string; textClass: string; dotClass: string }> = {
  in_progress: {
    labelKey: 'refreshTurn.inProgress',
    borderClass: 'border-border/60',
    textClass: 'text-secondary',
    dotClass: 'bg-accent animate-pulse',
  },
  done: {
    labelKey: 'refreshTurn.done',
    borderClass: 'border-border/40',
    textClass: 'text-tertiary',
    dotClass: 'bg-success',
  },
  failed: {
    labelKey: 'refreshTurn.failed',
    borderClass: 'border-error/40',
    textClass: 'text-error/80',
    dotClass: 'bg-error',
  },
  skipped: {
    labelKey: 'refreshTurn.skipped',
    borderClass: 'border-border/20',
    textClass: 'text-tertiary opacity-60',
    dotClass: 'bg-border',
  },
};

export default function RefreshTurnStatus({ status, trigger, timestamp }: RefreshTurnStatusProps) {
  const t = useT();
  const config = STATUS_CONFIG[status];
  const triggerLabel = t(TRIGGER_LABEL_KEYS[trigger]);
  const time = new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className={`my-2 mx-auto max-w-sm rounded border ${config.borderClass} bg-surface-alt/40 px-3 py-2 flex items-center gap-2`}>
      <span className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${config.dotClass}`} />
      <span className={`text-xs ${config.textClass} flex-1`}>
        {t(config.labelKey)}
        <span className="ml-1 opacity-50">({triggerLabel})</span>
      </span>
      <span className="text-xs text-tertiary opacity-40 flex-shrink-0">{time}</span>
    </div>
  );
}
