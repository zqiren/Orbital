// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { Dispatch, SetStateAction } from 'react';
import type { AgentRunStatus, Project, Trigger } from '../types';
import type { Route } from '../route';
import StatusBadge from './StatusBadge';
import TriggerStrip from './TriggerStrip';
import SettingsIcon from './SettingsIcon';
import { useQueue } from '../hooks/useQueue';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

interface ProjectDetailProps {
  project: Project;
  agentStatus: AgentRunStatus;
  statusSummary?: string;
  route: Extract<Route, { name: 'project' }>;
  setRoute: Dispatch<SetStateAction<Route>>;
  triggers?: Trigger[];
  onTriggerToggle?: (triggerId: string, enabled: boolean) => void;
  onTriggerDelete?: (triggerId: string) => void;
  /**
   * The global/default LLM model (from app settings). Used as the header's
   * model label only when the project has no per-project model pinned. NOT a
   * substitute for `agent_name` — a model id, not an identity.
   */
  globalDefaultModel?: string;
  children?: React.ReactNode;
}

const TABS: { key: 'queue' | 'chat' | 'files'; labelKey: StringKey }[] = [
  { key: 'queue', labelKey: 'projectDetail.tab.queue' },
  { key: 'chat', labelKey: 'projectDetail.tab.chat' },
  { key: 'files', labelKey: 'projectDetail.tab.files' },
];

export default function ProjectDetail({
  project,
  agentStatus,
  statusSummary,
  route,
  setRoute,
  triggers = [],
  onTriggerToggle,
  onTriggerDelete,
  globalDefaultModel,
  children,
}: ProjectDetailProps) {
  const t = useT();

  // The active tab indicator: when settings overlay is showing, no tab is highlighted
  const activeTab = route.settings ? null : route.tab;

  // Tab count badges (queue only — Chat's session count lives in the sidebar).
  const { snapshot } = useQueue(project.project_id);
  const queueCount = snapshot?.items.filter(
    (item) => item.state === 'queued' || item.state === 'running',
  ).length ?? 0;

  function handleTabChange(tab: 'queue' | 'chat' | 'files') {
    setRoute({ ...route, tab, settings: false });
  }

  function handleSettingsClick() {
    setRoute({ ...route, settings: true });
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-6 pt-5 pb-4 max-md:px-4">
        <div className="flex items-center gap-3 min-w-0">
          <h1 className="font-mono text-[18px] font-semibold text-primary truncate">{project.name}</h1>
          <StatusBadge status={agentStatus} />
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {(() => {
            // Model label: the project's pinned model, else the global default.
            // NEVER agent_name (an identity, not a model). Empty → cost only.
            const modelName = project.model || globalDefaultModel || '';
            const cost = `$${(project.budget_spent_usd ?? 0).toFixed(2)}`;
            const label = modelName ? `${modelName} · ${cost}` : cost;
            return (
              <span className="font-mono text-[11px] text-secondary">{label}</span>
            );
          })()}
          <SettingsIcon onClick={handleSettingsClick} />
        </div>
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
        {TABS.map((tab) => {
          // Chat shows no count badge — the session count lives in the
          // sidebar header. Only the queue badge (pending/running items) shows.
          const count = tab.key === 'queue' ? queueCount : 0;
          return (
            <button
              key={tab.key}
              onClick={() => handleTabChange(tab.key)}
              className={`text-[12.5px] font-medium px-3 py-2 -mb-px transition-all duration-150 max-md:min-h-[44px] max-md:flex max-md:items-center gap-1.5 ${
                activeTab === tab.key
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-secondary hover:text-primary'
              }`}
            >
              {t(tab.labelKey)}
              {count > 0 && (
                <span className="text-[10.5px] font-mono text-secondary">
                  {count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-hidden min-h-0">{children}</div>
    </div>
  );
}
