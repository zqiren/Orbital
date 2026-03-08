// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { ActivityEvent, ActivityCategory } from '../types';

export interface GroupedActivity {
  id: string;
  category: ActivityCategory;
  description: string;
  toolName: string;
  timestamp: string;
}

export interface ActivityBlock {
  activities: GroupedActivity[];
  startTime: string;
  endTime: string;
}

export const CATEGORY_ICON: Record<string, string> = {
  file_read: 'FileText',
  file_write: 'FilePlus',
  file_edit: 'Pencil',
  command_exec: 'Terminal',
  web_search: 'Search',
  web_fetch: 'Globe',
  request_access: 'FolderOpen',
  agent_message: 'Bot',
  network_blocked: 'ShieldX',
  tool_use: 'Wrench',
};

export function groupActivities(events: ActivityEvent[]): ActivityBlock[] {
  const blocks: ActivityBlock[] = [];
  let current: GroupedActivity[] = [];

  for (const event of events) {
    if (event.category === 'tool_result' || event.category === 'agent_output') {
      continue;
    }

    current.push({
      id: event.id,
      category: event.category,
      description: event.description,
      toolName: event.tool_name,
      timestamp: event.timestamp,
    });
  }

  if (current.length > 0) {
    blocks.push({
      activities: current,
      startTime: current[0].timestamp,
      endTime: current[current.length - 1].timestamp,
    });
  }

  return blocks;
}

export function appendActivityToBlocks(
  blocks: ActivityBlock[],
  event: ActivityEvent,
): ActivityBlock[] {
  if (event.category === 'tool_result' || event.category === 'agent_output') {
    return blocks;
  }

  const activity: GroupedActivity = {
    id: event.id,
    category: event.category,
    description: event.description,
    toolName: event.tool_name,
    timestamp: event.timestamp,
  };

  if (blocks.length === 0) {
    return [
      {
        activities: [activity],
        startTime: event.timestamp,
        endTime: event.timestamp,
      },
    ];
  }

  const last = blocks[blocks.length - 1];
  const updated = [...blocks];
  updated[updated.length - 1] = {
    ...last,
    activities: [...last.activities, activity],
    endTime: event.timestamp,
  };
  return updated;
}

export function summarizeActivities(
  activities: GroupedActivity[],
): string {
  const counts: Record<string, number> = {};
  for (const a of activities) {
    switch (a.category) {
      case 'file_read':
        counts['files read'] = (counts['files read'] ?? 0) + 1;
        break;
      case 'file_write':
        counts['files created'] = (counts['files created'] ?? 0) + 1;
        break;
      case 'file_edit':
        counts['files edited'] = (counts['files edited'] ?? 0) + 1;
        break;
      case 'command_exec':
        counts['commands run'] = (counts['commands run'] ?? 0) + 1;
        break;
      case 'web_search':
        counts['searches'] = (counts['searches'] ?? 0) + 1;
        break;
      case 'web_fetch':
        counts['pages fetched'] = (counts['pages fetched'] ?? 0) + 1;
        break;
      case 'request_access':
        counts['access requests'] = (counts['access requests'] ?? 0) + 1;
        break;
      case 'agent_message':
        counts['messages sent'] = (counts['messages sent'] ?? 0) + 1;
        break;
      case 'network_blocked':
        counts['requests blocked'] = (counts['requests blocked'] ?? 0) + 1;
        break;
      default:
        counts['tool calls'] = (counts['tool calls'] ?? 0) + 1;
    }
  }

  return Object.entries(counts)
    .map(([label, count]) => `${count} ${label}`)
    .join(', ');
}

export function formatDuration(startTime: string, endTime: string): string {
  const start = new Date(startTime).getTime();
  const end = new Date(endTime).getTime();
  const diffMs = Math.max(0, end - start);
  const totalSeconds = Math.round(diffMs / 1000);

  if (totalSeconds < 1) return '<1s';
  if (totalSeconds < 60) return `${totalSeconds}s`;

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (seconds === 0) return `${minutes}m`;
  return `${minutes}m ${seconds}s`;
}
