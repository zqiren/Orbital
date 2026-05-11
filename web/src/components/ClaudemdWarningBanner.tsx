// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Banner shown when the project's workspace CLAUDE.md contains tokens
 * that may conflict with Orbital's project-state inheritance for sub-agents.
 *
 * Triggered by the `workspace_claudemd_warning` WebSocket event. Dismiss
 * is session-scoped (server-side state in daemon memory) — the daemon
 * suppresses re-emission for the same (project_id, content_hash) until
 * restart or content change.
 */

import { AlertTriangle } from 'lucide-react';
import { api } from '../config';

export interface ClaudemdWarning {
  project_id: string;
  claudemd_path: string;
  content_hash: string;
  matched_token: string;
}

interface Props {
  projectId: string;
  warning: ClaudemdWarning;
  onDismiss: () => void;
}

export default function ClaudemdWarningBanner({ projectId, warning, onDismiss }: Props) {
  function handleDismiss() {
    // Optimistically dismiss locally; tell the daemon to suppress
    // re-emission for the rest of the session.
    api(`/api/v2/agents/${encodeURIComponent(projectId)}/claudemd-warning/dismiss`, {
      method: 'POST',
      body: JSON.stringify({ content_hash: warning.content_hash }),
    }).catch(() => {
      // best-effort — local dismissal still hides the banner
    });
    onDismiss();
  }

  return (
    <div
      role="alert"
      className="
        mx-3 my-2 rounded-md border border-amber-500/40 bg-amber-500/10
        px-3 py-2 text-sm text-amber-100
        flex items-start gap-2
        max-md:mx-2 max-md:px-2 max-md:py-2 max-md:flex-col max-md:items-stretch
      "
    >
      <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0 text-amber-400" aria-hidden />
      <div className="flex-1 min-w-0">
        <div className="leading-snug">
          Your workspace CLAUDE.md may conflict with Orbital&apos;s
          project-state inheritance. Sub-agents may behave unexpectedly.
        </div>
        <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-xs">
          <a
            href={`/api/v2/files/${encodeURIComponent(projectId)}?path=${encodeURIComponent('CLAUDE.md')}`}
            target="_blank"
            rel="noreferrer"
            className="underline hover:text-amber-50"
            title={warning.claudemd_path}
          >
            view CLAUDE.md
          </a>
          <button
            type="button"
            onClick={handleDismiss}
            className="underline hover:text-amber-50"
          >
            dismiss
          </button>
          <span className="opacity-70 truncate">
            (matched: &quot;{warning.matched_token}&quot;)
          </span>
        </div>
      </div>
    </div>
  );
}
