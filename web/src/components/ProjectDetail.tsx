// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { AgentRunStatus, Project, Trigger } from '../types';
import StatusBadge from './StatusBadge';
import TriggerStrip from './TriggerStrip';

export type DetailTab = 'chat' | 'files' | 'settings';

interface ProjectDetailProps {
  project: Project;
  agentStatus: AgentRunStatus;
  statusSummary?: string;
  tab: DetailTab;
  onTabChange: (tab: DetailTab) => void;
  onStopAgent: () => void;
  triggers?: Trigger[];
  onTriggerToggle?: (triggerId: string, enabled: boolean) => void;
  onTriggerDelete?: (triggerId: string) => void;
  children?: React.ReactNode;
}

const TABS: { key: DetailTab; label: string }[] = [
  { key: 'chat', label: 'Chat' },
  { key: 'files', label: 'Files' },
  { key: 'settings', label: 'Settings' },
];

export default function ProjectDetail({
  project,
  agentStatus,
  statusSummary,
  tab,
  onTabChange,
  onStopAgent,
  triggers = [],
  onTriggerToggle,
  onTriggerDelete,
  children,
}: ProjectDetailProps) {
  const isRunning = agentStatus === 'running' || agentStatus === 'waiting';

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-6 pt-5 pb-4 max-md:px-4">
        <div className="flex items-center gap-3 min-w-0">
          <h1 className="text-xl font-semibold text-primary truncate">{project.name}</h1>
          <StatusBadge status={agentStatus} />
        </div>
        {isRunning && (
          <button
            onClick={onStopAgent}
            className="text-sm font-medium rounded-lg px-4 py-2 transition-all duration-150 border border-border text-secondary hover:text-error hover:border-error/40 shrink-0 max-md:min-h-[44px]"
          >
            Stop Agent
          </button>
        )}
      </div>

      {/* Status summary line */}
      {statusSummary && (
        <div className="px-6 pb-2 max-md:px-4">
          <p className="text-xs text-secondary truncate">{statusSummary}</p>
        </div>
      )}

      {/* Trigger strip — between header and tab bar */}
      {triggers.length > 0 && onTriggerToggle && (
        <TriggerStrip triggers={triggers} onToggle={onTriggerToggle} onDelete={onTriggerDelete} />
      )}

      {/* Tab bar */}
      <div className="flex gap-1 px-6 border-b border-border max-md:px-4">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => onTabChange(t.key)}
            className={`text-sm font-medium px-3 py-2 -mb-px transition-all duration-150 max-md:min-h-[44px] max-md:flex max-md:items-center ${
              tab === t.key
                ? 'text-primary border-b-2 border-accent'
                : 'text-secondary hover:text-primary'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-hidden min-h-0">{children}</div>
    </div>
  );
}
