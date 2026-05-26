// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SessionListItem — one row in the Chat-tab session sidebar.
 *
 * Renders:
 *   - Status glyph (◐/⟳/⚠/⏸) in the status color, with a subtle
 *     hue variation when origin === 'queue' (desaturated/alternate tint).
 *   - session_id string as the display name (no auto-naming).
 *   - last_activity_at as a short relative time (or "—" if null).
 *   - SessionStatusGlyph error indicator (amber ⚠ when last_terminal_event.type === 'error').
 *   - Hover-revealed three-dot menu (SessionThreeDotMenu — all actions placeholder).
 *
 * Active row: white bg + subtle shadow + font-weight 500.
 */

import { useState } from 'react';
import type { SessionListEntry } from '../types';
import { SessionStatusGlyph } from './SessionStatusGlyph';
import { SessionThreeDotMenu } from './SessionThreeDotMenu';
import { getStatusDisplay } from './sessionStatus';

export interface SessionListItemProps {
  session: SessionListEntry;
  selected: boolean;
  onSelect: () => void;
}

/**
 * Format an ISO timestamp as a short relative time string.
 * Falls back to "—" when the value is null/undefined.
 */
function formatRelativeTime(isoString?: string | null): string {
  if (!isoString) return '—';
  const date = new Date(isoString);
  if (isNaN(date.getTime())) return '—';
  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

export function SessionListItem({ session, selected, onSelect }: SessionListItemProps) {
  const [hovered, setHovered] = useState(false);
  const display = getStatusDisplay(session.status);

  // Queue-origin hue variation: use a slightly desaturated/alternate tint
  // when origin === 'queue'. The base status color is applied normally for
  // manual sessions (undefined → treat as manual).
  const isQueue = session.origin === 'queue';
  // Desaturate the queue-origin dot by blending toward gray (#A1A1AA) at 35%.
  // This is a visual-only capability — backend does not populate origin yet.
  const dotColor = isQueue
    ? mixColorTowardGray(display.color, 0.35)
    : display.color;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-selected={selected}
      aria-label={`Session ${session.session_id}`}
      data-testid={`session-list-item-${session.session_id}`}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect();
        }
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={
        selected
          ? { background: '#fff', boxShadow: '0 1px 2px rgba(0,0,0,0.04)' }
          : undefined
      }
      className={[
        'flex items-center gap-1.5 px-2.5 py-[7px] rounded-md cursor-pointer select-none transition-colors',
        selected
          ? 'font-medium'
          : 'hover:bg-card-hover',
      ].join(' ')}
    >
      {/* Status glyph */}
      <span
        aria-hidden="true"
        data-testid="session-status-glyph"
        data-origin={isQueue ? 'queue' : 'manual'}
        style={{ color: dotColor, fontSize: '13px', lineHeight: 1, flexShrink: 0 }}
      >
        {display.glyph}
      </span>

      {/* Session name + time */}
      <span className="flex-1 min-w-0 flex items-baseline gap-1.5">
        <span
          data-testid="session-name"
          className="font-mono text-[11.5px] text-primary truncate"
          style={{ fontSize: '11.5px' }}
        >
          {session.session_id}
        </span>
        <span
          data-testid="session-time"
          className="text-[10px] text-secondary shrink-0"
        >
          {formatRelativeTime(session.last_activity_at)}
        </span>
      </span>

      {/* Error indicator (amber ⚠ from SessionStatusGlyph) */}
      <SessionStatusGlyph lastTerminalEvent={session.last_terminal_event} />

      {/* Three-dot menu — shown only on hover (or when selected) */}
      <span
        style={{ visibility: hovered || selected ? 'visible' : 'hidden' }}
        aria-hidden={!hovered && !selected}
      >
        <SessionThreeDotMenu ariaLabel={`Options for session ${session.session_id}`} />
      </span>
    </div>
  );
}

/**
 * Mix a hex color toward gray (#A1A1AA) by `amount` (0–1).
 * amount=0 → original color, amount=1 → gray.
 * Used for queue-origin status dot hue variation.
 */
function mixColorTowardGray(hex: string, amount: number): string {
  const gray = { r: 0xa1, g: 0xa1, b: 0xaa };
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const mr = Math.round(r + (gray.r - r) * amount);
  const mg = Math.round(g + (gray.g - g) * amount);
  const mb = Math.round(b + (gray.b - b) * amount);
  return `rgb(${mr}, ${mg}, ${mb})`;
}
