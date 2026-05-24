// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { AlertTriangle } from 'lucide-react';
import type { LastTerminalEvent } from '../types';

// SessionStatusGlyph
//
// Renders a per-session status indicator derived from the session's
// `last_terminal_event` field. Currently only the 'error' case lights up
// (warning glyph) per TASK-state-model-alignment-fixes.md §2.4 + §5.
// Other terminal types ('stopped', 'new_session') are recorded by the
// backend for future use but render no indicator today.
//
// Designed as a small atom so Phase 1B's SessionListItem can drop it in
// without ceremony. Returns null for sessions with no recorded terminal
// event or a non-error type.
export interface SessionStatusGlyphProps {
  lastTerminalEvent: LastTerminalEvent | null | undefined;
}

export function SessionStatusGlyph({ lastTerminalEvent }: SessionStatusGlyphProps) {
  if (!lastTerminalEvent || lastTerminalEvent.type !== 'error') {
    return null;
  }
  const details = lastTerminalEvent.details;
  const title = details
    ? `Session errored: ${details}`
    : 'Session errored';
  return (
    <AlertTriangle
      size={14}
      className="shrink-0 text-amber-500"
      aria-label="Session errored"
      role="img"
      data-testid="session-error-glyph"
    >
      <title>{title}</title>
    </AlertTriangle>
  );
}
