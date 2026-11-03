// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * trigger-parsing.ts — the run header an automation stamps on its first
 * message, and the task text without it.
 *
 * agent_os/daemon_v2/trigger_manager.py opens every automation run with
 *   "[Triggered by schedule 'NAME' (every day at 9)]\n\n<task>"
 *   "[Triggered by file_watch 'NAME']\n\nChanged files: a, b\n\n<task>"
 * The session heading and the list row already say which automation ran
 * (sessionLabel.ts reads the same prefix), so the bubble shows the task the
 * user wrote, not the machine line. "Changed files" stays: it is the run's
 * input, not chrome. The stored message is untouched — the daemon's
 * trigger_type_of() and the classifier keep reading the prefix.
 */

export interface TriggerHeader {
  kind: 'schedule' | 'file_watch';
  name: string;
  /** The schedule in words, when the header carries one. */
  schedule: string | null;
}

export interface ParseTriggerResult {
  /** The content after the header, or the content itself when there is none. */
  strippedContent: string;
  trigger: TriggerHeader | null;
}

const HEADER_RE = /^\[Triggered by (schedule|file_watch) '([^'\n]*)'(?: \(([^)\n]*)\))?\]\n*/;

export function parseTriggerHeader(content: string): ParseTriggerResult {
  const m = HEADER_RE.exec(content);
  if (!m) return { strippedContent: content, trigger: null };
  return {
    strippedContent: content.slice(m[0].length),
    trigger: {
      kind: m[1] as TriggerHeader['kind'],
      name: m[2],
      schedule: m[3] ?? null,
    },
  };
}
