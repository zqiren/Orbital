// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import {
  FileText,
  FilePlus,
  Pencil,
  FileSearch,
  SearchCode,
  Terminal,
  Search,
  Globe,
  FolderOpen,
  Bot,
  ShieldX,
  Wrench,
  ChevronRight,
  ChevronDown,
  MonitorSmartphone,
  KeyRound,
} from 'lucide-react';
import type { Activity } from '../utils/chatTransform';
import type { GroupedActivity } from '../utils/activityGroup';
import { summarizeActivities, formatDuration } from '../utils/activityGroup';

type AnyActivity = Activity | GroupedActivity;

const ICON_MAP: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  file_read: FileText,
  file_write: FilePlus,
  file_edit: Pencil,
  file_search: FileSearch,
  content_search: SearchCode,
  command_exec: Terminal,
  web_search: Search,
  web_fetch: Globe,
  request_access: FolderOpen,
  agent_message: Bot,
  network_blocked: ShieldX,
  tool_use: Wrench,
  browser_automation: MonitorSmartphone,
  credential_request: KeyRound,
};

interface ActivityBlockProps {
  activities: AnyActivity[];
  startTime: string;
  endTime: string;
}

export default function ActivityBlock({
  activities,
  startTime,
  endTime,
}: ActivityBlockProps) {
  const [expanded, setExpanded] = useState(false);

  if (activities.length === 0) return null;

  const duration = formatDuration(startTime, endTime);
  const summary = summarizeActivities(activities);
  const firstCategory = activities[0].category;
  const PrimaryIcon = ICON_MAP[firstCategory] ?? Wrench;

  return (
    <div className="mb-3">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left px-3 py-2 rounded-lg bg-sidebar hover:bg-card-hover transition-colors duration-150 text-sm font-medium text-secondary cursor-pointer max-md:min-h-[44px]"
      >
        {expanded ? (
          <ChevronDown size={14} className="shrink-0" />
        ) : (
          <ChevronRight size={14} className="shrink-0" />
        )}
        <PrimaryIcon size={14} className="shrink-0" />
        <span>Agent worked</span>
        <span className="text-xs opacity-75">{duration}</span>
        <span className="text-xs opacity-75 truncate">{summary}</span>
      </button>

      {expanded && (
        <div className="ml-5 mt-1 border-l border-border pl-3 space-y-1 overflow-x-auto">
          {activities.map((a) => {
            const Icon = ICON_MAP[a.category] ?? Wrench;
            return (
              <div
                key={a.id}
                className="flex items-center gap-2 text-sm text-secondary py-0.5"
                title={a.timestamp}
              >
                <Icon size={13} className="shrink-0 opacity-70" />
                <span className="truncate">{a.description}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
