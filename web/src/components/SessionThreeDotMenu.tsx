// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SessionThreeDotMenu — hover three-dot (⋯) button with placeholder actions.
 *
 * All actions (Rename / Delete) are DISABLED with a "Coming in Batch 4"
 * tooltip. No real handlers — lifecycle actions are deferred per
 * Phase 1B DO-NOT #2.
 */

import { useState, useRef, useEffect } from 'react';
import { useT } from '../i18n/useT';

export interface SessionThreeDotMenuProps {
  /** Accessible label for the trigger button. Defaults to "Session options". */
  ariaLabel?: string;
}

export function SessionThreeDotMenu({ ariaLabel }: SessionThreeDotMenuProps) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const t = useT();
  const batch4Title = t('sessionThreeDot.batch4');
  const resolvedAriaLabel = ariaLabel ?? t('sessionThreeDot.optionsAria');

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    function handleClickOutside(e: MouseEvent) {
      if (
        menuRef.current &&
        !menuRef.current.contains(e.target as Node) &&
        btnRef.current &&
        !btnRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open]);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [open]);

  return (
    <div className="relative" data-testid="session-three-dot-menu">
      <button
        ref={btnRef}
        type="button"
        aria-label={resolvedAriaLabel}
        aria-haspopup="menu"
        aria-expanded={open}
        data-testid="session-three-dot-trigger"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="flex items-center justify-center w-5 h-5 rounded text-secondary hover:text-primary hover:bg-card-hover transition-colors"
        style={{ fontSize: '14px', lineHeight: 1 }}
      >
        ⋯
      </button>

      {open && (
        <div
          ref={menuRef}
          role="menu"
          data-testid="session-three-dot-dropdown"
          className="absolute right-0 top-full mt-1 z-50 min-w-[140px] rounded-md border border-border bg-elevated shadow-md py-1"
        >
          {([
            { id: 'rename', label: t('sessionThreeDot.rename') },
            { id: 'delete', label: t('sessionThreeDot.delete') },
          ] as const).map((action) => (
            <button
              key={action.id}
              type="button"
              role="menuitem"
              disabled
              title={batch4Title}
              data-testid={`session-action-${action.id}`}
              className="w-full text-left px-3 py-1.5 text-sm text-secondary cursor-not-allowed opacity-50"
            >
              {action.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
