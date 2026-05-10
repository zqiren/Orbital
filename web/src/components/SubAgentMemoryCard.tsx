// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import { api, ApiError } from '../config';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SubAgentMemoryEntry {
  agent_slug: string;
  agent_name: string;
  exists: boolean;
  content: string;
  last_modified: number | null;
  size_bytes: number;
}

interface Props {
  projectId: string;
  entry: SubAgentMemoryEntry;
  onSaved: () => void | Promise<void>;
}

// Hard cap matches the backend (10 KB). Soft target shown in the size hint.
const MEMORY_TARGET_BYTES = 2 * 1024;
const MEMORY_MAX_BYTES = 10 * 1024;

function formatTimestamp(epochSeconds: number | null): string {
  if (epochSeconds == null) return 'never written';
  try {
    const d = new Date(epochSeconds * 1000);
    return `last edit ${d.toLocaleString()}`;
  } catch {
    return 'unknown';
  }
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SubAgentMemoryCard({ projectId, entry, onSaved }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [draft, setDraft] = useState(entry.content);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);

  // Keep the draft in sync with the server-side content when the parent
  // re-fetches (e.g. after a disabled-list toggle re-surfaces this agent).
  useEffect(() => {
    setDraft(entry.content);
    setSaved(false);
    setWarning(null);
  }, [entry.content]);

  const draftBytes = new TextEncoder().encode(draft).length;
  const overTarget = draftBytes > MEMORY_TARGET_BYTES;
  const overMax = draftBytes > MEMORY_MAX_BYTES;

  async function persist(content: string) {
    setSaving(true);
    setError(null);
    setWarning(null);
    try {
      const resp = await api<{ warning?: string; size_bytes: number }>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/sub-agent-memory/${encodeURIComponent(entry.agent_slug)}`,
        {
          method: 'PUT',
          body: JSON.stringify({ content }),
        },
      );
      if (resp.warning) {
        setWarning(resp.warning);
      }
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      await onSaved();
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.detail);
      } else {
        setError(e instanceof Error ? e.message : 'Save failed');
      }
    } finally {
      setSaving(false);
    }
  }

  async function handleSave() {
    if (overMax) {
      setError(`Content exceeds the ${MEMORY_MAX_BYTES} byte hard cap.`);
      return;
    }
    await persist(draft);
  }

  async function handleReset() {
    // Reset = save empty content. Does NOT delete the file (per spec).
    setDraft('');
    await persist('');
  }

  return (
    <div className="bg-sidebar/30 border border-border rounded-lg">
      <button
        type="button"
        onClick={() => setExpanded((x) => !x)}
        className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left hover:bg-sidebar/50 transition-all duration-150 max-md:min-h-[44px]"
        aria-expanded={expanded}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-primary">
              {entry.agent_name}
            </span>
            <code className="text-[11px] text-secondary/80 font-mono">
              {entry.agent_slug}
            </code>
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs text-secondary flex-wrap">
            <span>{formatTimestamp(entry.last_modified)}</span>
            <span aria-hidden="true">·</span>
            <span
              className={
                entry.size_bytes > MEMORY_TARGET_BYTES
                  ? 'text-warning'
                  : 'text-secondary'
              }
            >
              {formatSize(entry.size_bytes)} / {formatSize(MEMORY_TARGET_BYTES)} target
            </span>
          </div>
        </div>
        <span
          className={`shrink-0 text-secondary transition-transform duration-150 ${
            expanded ? 'rotate-90' : ''
          }`}
          aria-hidden="true"
        >
          {/* simple chevron */}
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

      {expanded && (
        <div className="border-t border-border px-4 py-3 flex flex-col gap-3">
          {!entry.exists && (
            <p className="text-xs text-secondary italic">
              memory not yet initialized — saving will create it
            </p>
          )}

          <textarea
            rows={8}
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              setSaved(false);
              setWarning(null);
            }}
            placeholder="Free-form markdown. Sub-agents read this every dispatch — keep it concise."
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y font-mono"
          />

          <div className="flex items-center justify-between gap-2 flex-wrap text-xs">
            <span
              className={
                overMax
                  ? 'text-error'
                  : overTarget
                    ? 'text-warning'
                    : 'text-secondary'
              }
            >
              {formatSize(draftBytes)} / {formatSize(MEMORY_TARGET_BYTES)} target
              {overMax && ' — over hard cap'}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleReset}
                disabled={saving}
                className="text-xs text-secondary hover:text-primary border border-border rounded px-3 py-1.5 hover:bg-sidebar/80 transition-all duration-150 disabled:opacity-50 max-md:min-h-[44px]"
              >
                Reset
              </button>
              <button
                type="button"
                onClick={handleSave}
                disabled={saving || overMax}
                className="text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 max-md:min-h-[44px]"
              >
                {saving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>

          {saved && (
            <p className="text-xs text-success">Saved</p>
          )}
          {warning && (
            <p className="text-xs text-warning">{warning}</p>
          )}
          {error && (
            <p className="text-xs text-error">{error}</p>
          )}
        </div>
      )}
    </div>
  );
}
