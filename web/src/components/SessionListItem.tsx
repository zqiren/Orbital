// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SessionListItem — one row in the Chat-tab session sidebar.
 *
 * Renders:
 *   - Status glyph (◐/⟳/⚠/⏸) in the status color, with a subtle
 *     hue variation when origin === 'queue' (desaturated/alternate tint).
 *   - Display name (session.name → session_id fallback chain).
 *   - last_activity_at as a short relative time (or "—" if null).
 *   - SessionStatusGlyph error indicator (amber ⚠ when last_terminal_event.type === 'error').
 *   - Hover-revealed three-dot menu with Rename + Delete (real handlers).
 *
 * Inline rename: double-click the name (or pick Rename from the menu) → editable
 * input → Enter or blur saves via onRename; Escape cancels.
 *
 * Delete: pick Delete from the menu → confirmation dialog → Delete confirms via
 * onDelete.
 *
 * Active row: white bg + subtle shadow + font-weight 500.
 */

import { useEffect, useRef, useState } from 'react';
import type { SessionListEntry } from '../types';
import { SessionStatusGlyph } from './SessionStatusGlyph';
import { getStatusDisplay } from './sessionStatus';
import { useT } from '../i18n/useT';

export interface SessionListItemProps {
  session: SessionListEntry;
  selected: boolean;
  onSelect: () => void;
  /** Persist a renamed display label. Receives (sessionId, newName). */
  onRename?: (sessionId: string, name: string) => void | Promise<void>;
  /** Delete this session. Receives the sessionId. */
  onDelete?: (sessionId: string) => void | Promise<void>;
}

/**
 * Format an ISO timestamp as a short relative time string.
 * Falls back to "—" when the value is null/undefined.
 */
function formatRelativeTime(
  isoString: string | null | undefined,
  t: ReturnType<typeof useT>,
): string {
  if (!isoString) return '—';
  const date = new Date(isoString);
  if (isNaN(date.getTime())) return '—';
  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return t('sessionItem.relTime.seconds', { n: diffSec });
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return t('sessionItem.relTime.minutes', { n: diffMin });
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return t('sessionItem.relTime.hours', { n: diffHr });
  const diffDay = Math.floor(diffHr / 24);
  return t('sessionItem.relTime.days', { n: diffDay });
}

/**
 * Resolve the display label. Fallback chain: name → session_id. (A truncated
 * first-message fallback is unnecessary — the backend already derives the name
 * from the first user message, so `name` IS that truncation when available.)
 */
function displayLabel(session: SessionListEntry): string {
  const name = session.name;
  if (name && name.trim()) return name;
  return session.session_id;
}

export function SessionListItem({
  session,
  selected,
  onSelect,
  onRename,
  onDelete,
}: SessionListItemProps) {
  const [hovered, setHovered] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const t = useT();
  const display = getStatusDisplay(session.status);

  const label = displayLabel(session);

  // Queue-origin hue variation: use a slightly desaturated/alternate tint
  // when origin === 'queue'. The base status color is applied normally for
  // manual sessions (undefined → treat as manual).
  const isQueue = session.origin === 'queue';
  // Desaturate the queue-origin dot by blending toward gray (#A1A1AA) at 35%.
  const dotColor = isQueue
    ? mixColorTowardGray(display.color, 0.35)
    : display.color;

  // Focus the rename input when editing starts.
  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  // Close the menu on outside click / Escape.
  useEffect(() => {
    if (!menuOpen) return;
    function onDocMouseDown(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setMenuOpen(false);
    }
    document.addEventListener('mousedown', onDocMouseDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [menuOpen]);

  function startRename() {
    setMenuOpen(false);
    setDraft(label);
    setEditing(true);
  }

  function commitRename() {
    const trimmed = draft.trim();
    setEditing(false);
    if (trimmed && trimmed !== label) {
      void onRename?.(session.session_id, trimmed);
    }
  }

  function cancelRename() {
    setEditing(false);
    setDraft('');
  }

  function confirmDelete() {
    setConfirmingDelete(false);
    void onDelete?.(session.session_id);
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-selected={selected}
      aria-label={t('sessionItem.aria', { label })}
      data-testid={`session-list-item-${session.session_id}`}
      onClick={() => {
        if (editing) return;
        onSelect();
      }}
      onKeyDown={(e) => {
        if (editing) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect();
        }
      }}
      onContextMenu={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setMenuOpen(true);
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={
        selected
          ? { background: '#fff', boxShadow: '0 1px 2px rgba(0,0,0,0.04)' }
          : undefined
      }
      className={[
        'relative flex items-center gap-1.5 px-2.5 py-[7px] rounded-md cursor-pointer select-none transition-colors',
        selected ? 'font-medium' : 'hover:bg-card-hover',
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

      {/* Session name + time (or inline rename input) */}
      <span className="flex-1 min-w-0 flex items-baseline gap-1.5">
        {editing ? (
          <input
            ref={inputRef}
            type="text"
            data-testid="session-rename-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            onBlur={commitRename}
            onKeyDown={(e) => {
              e.stopPropagation();
              // Enter that commits an IME (e.g. Pinyin) candidate must not
              // commit the rename — it belongs to the input method.
              if (e.nativeEvent.isComposing || e.keyCode === 229) return;
              if (e.key === 'Enter') {
                e.preventDefault();
                commitRename();
              } else if (e.key === 'Escape') {
                e.preventDefault();
                cancelRename();
              }
            }}
            className="flex-1 min-w-0 text-xs text-primary bg-card border border-border rounded px-1 py-0.5"
            style={{ fontSize: '11.5px' }}
          />
        ) : (
          <>
            <span
              data-testid="session-name"
              className="text-xs text-primary truncate"
              style={{ fontSize: '11.5px' }}
              onDoubleClick={(e) => {
                e.stopPropagation();
                startRename();
              }}
            >
              {label}
            </span>
            <span
              data-testid="session-time"
              className="text-2xs text-secondary shrink-0"
            >
              {formatRelativeTime(session.last_activity_at, t)}
            </span>
          </>
        )}
      </span>

      {/* Error indicator (amber ⚠ from SessionStatusGlyph) */}
      <SessionStatusGlyph lastTerminalEvent={session.last_terminal_event} />

      {/* Three-dot menu — shown only on hover (or when selected/open) */}
      {!editing && (
        <div
          ref={menuRef}
          className="relative"
          data-testid="session-three-dot-menu"
          style={{ visibility: hovered || selected || menuOpen ? 'visible' : 'hidden' }}
          aria-hidden={!hovered && !selected && !menuOpen}
        >
          <button
            type="button"
            aria-label={t('sessionItem.optionsAria', { label })}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            data-testid="session-three-dot-trigger"
            onClick={(e) => {
              e.stopPropagation();
              setMenuOpen((v) => !v);
            }}
            className="flex items-center justify-center w-5 h-5 rounded text-secondary hover:text-primary hover:bg-card-hover transition-colors"
            style={{ fontSize: '14px', lineHeight: 1 }}
          >
            ⋯
          </button>

          {menuOpen && (
            <div
              role="menu"
              data-testid="session-three-dot-dropdown"
              className="absolute right-0 top-full mt-1 z-50 min-w-[140px] rounded-md border border-border bg-elevated shadow-md py-1"
            >
              <button
                type="button"
                role="menuitem"
                data-testid="session-action-rename"
                onClick={(e) => {
                  e.stopPropagation();
                  startRename();
                }}
                className="w-full text-left px-3 py-1.5 text-sm text-primary hover:bg-card-hover"
              >
                {t('sessionItem.rename')}
              </button>
              <button
                type="button"
                role="menuitem"
                data-testid="session-action-delete"
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpen(false);
                  setConfirmingDelete(true);
                }}
                className="w-full text-left px-3 py-1.5 text-sm text-red-600 hover:bg-red-50"
              >
                {t('sessionItem.delete')}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Delete confirmation dialog */}
      {confirmingDelete && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={t('sessionItem.deleteConfirm.aria')}
          data-testid="session-delete-confirm"
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/30"
          onClick={(e) => {
            e.stopPropagation();
            setConfirmingDelete(false);
          }}
        >
          <div
            className="max-w-sm w-[320px] rounded-lg bg-elevated shadow-xl border border-border p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <p className="text-sm text-primary font-medium mb-1">{t('sessionItem.deleteConfirm.title')}</p>
            <p className="text-xs text-secondary mb-4">
              {t('sessionItem.deleteConfirm.body')}
            </p>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                data-testid="session-delete-cancel"
                onClick={(e) => {
                  e.stopPropagation();
                  setConfirmingDelete(false);
                }}
                className="px-3 py-1.5 text-sm rounded-md border border-border text-secondary hover:bg-card-hover"
              >
                {t('sessionItem.deleteConfirm.cancel')}
              </button>
              <button
                type="button"
                data-testid="session-delete-confirm-button"
                onClick={(e) => {
                  e.stopPropagation();
                  confirmDelete();
                }}
                className="px-3 py-1.5 text-sm rounded-md bg-red-600 text-white hover:bg-red-700"
              >
                {t('sessionItem.delete')}
              </button>
            </div>
          </div>
        </div>
      )}
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
