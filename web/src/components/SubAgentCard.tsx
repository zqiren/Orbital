// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import {
  SubAgentMemoryEditor,
  formatTimestamp,
  formatSize,
  MEMORY_TARGET_BYTES,
  type SubAgentMemoryEntry,
} from './SubAgentMemoryCard';

interface Props {
  projectId: string;
  /** Display name for the agent (from the installed-agents settings list). */
  agentName: string;
  /** Stable slug, e.g. "claude-code". */
  slug: string;
  /** Enabled for this project (i.e. NOT in disabled_sub_agents). */
  enabled: boolean;
  /** Authenticated / ready to dispatch (settings `ready` field). When the
   *  agent is installed but not ready, the header shows a "Sign in" badge. */
  ready: boolean;
  /** Per-project MEMORY.md entry (undefined while the memory list loads). */
  memory?: SubAgentMemoryEntry;
  /** Called with (slug, enabled) when the header toggle changes. */
  onToggle: (slug: string, enabled: boolean) => void;
  /** Called after a successful memory save so the parent can re-fetch. */
  onMemorySaved: () => void | Promise<void>;
}

/**
 * Merged Project-Settings sub-agent card. One card per installed sub-agent:
 *
 *  - Header: the enable toggle (writes project.disabled_sub_agents) + display
 *    name + slug + an auth badge ("⚠ Sign in" when installed-but-not-ready) +
 *    a memory summary (size / "never written" / last edited).
 *  - Body (on expand): the shared MEMORY.md editor (SubAgentMemoryEditor).
 *
 * A disabled agent still renders — dimmed, body collapsed — but stays
 * expandable so its memory remains editable.
 */
export default function SubAgentCard({
  projectId,
  agentName,
  slug,
  enabled,
  ready,
  memory,
  onToggle,
  onMemorySaved,
}: Props) {
  const [expanded, setExpanded] = useState(false);

  const sizeBytes = memory?.size_bytes ?? 0;
  const lastModified = memory?.last_modified ?? null;
  const memorySummary =
    lastModified == null
      ? 'never written'
      : `${formatTimestamp(lastModified)} · ${formatSize(sizeBytes)}`;

  // A fallback entry so the editor can render before the memory list loads
  // (or for an agent that has no memory entry yet).
  const entry: SubAgentMemoryEntry = memory ?? {
    agent_slug: slug,
    agent_name: agentName,
    exists: false,
    content: '',
    last_modified: null,
    size_bytes: 0,
  };

  return (
    <div
      className={`bg-sidebar/30 border border-border rounded-lg transition-opacity duration-150 ${
        enabled ? '' : 'opacity-50'
      }`}
    >
      <div className="flex items-center gap-3 px-4 py-3">
        {/* Enable toggle — clicking the checkbox must not toggle the body. */}
        <label
          className="flex items-center shrink-0 cursor-pointer"
          onClick={(e) => e.stopPropagation()}
        >
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onToggle(slug, e.target.checked)}
            aria-label={agentName}
            className="w-3.5 h-3.5 accent-accent"
          />
        </label>

        {/* Header — clicking expands the memory body. */}
        <button
          type="button"
          onClick={() => setExpanded((x) => !x)}
          className="flex-1 min-w-0 flex items-center justify-between gap-3 text-left max-md:min-h-[44px]"
          aria-expanded={expanded}
          aria-label={`${agentName} memory`}
        >
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium text-primary">
                {agentName}
              </span>
              <code className="text-[11px] text-secondary/80 font-mono">
                {slug}
              </code>
              {!ready && (
                <span className="text-[11px] font-medium text-warning bg-warning/10 border border-warning/30 rounded px-1.5 py-0.5">
                  ⚠ Sign in
                </span>
              )}
            </div>
            <div className="mt-1 flex items-center gap-2 text-xs flex-wrap">
              <span
                className={
                  sizeBytes > MEMORY_TARGET_BYTES
                    ? 'text-warning'
                    : 'text-secondary'
                }
              >
                {memorySummary}
              </span>
            </div>
          </div>
          <span
            className={`shrink-0 text-secondary transition-transform duration-150 ${
              expanded ? 'rotate-90' : ''
            }`}
            aria-hidden="true"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </span>
        </button>
      </div>

      {expanded && (
        <div className="border-t border-border px-4 py-3">
          <SubAgentMemoryEditor
            projectId={projectId}
            entry={entry}
            onSaved={onMemorySaved}
          />
        </div>
      )}
    </div>
  );
}
