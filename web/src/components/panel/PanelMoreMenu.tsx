// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 088 — the workspace panel's More menu (a file's secondary actions). A
 * WAI-ARIA menu button: the trigger opens a named `menu`; arrows, Home and End
 * move between items; Escape or choosing an item closes it and puts focus back
 * on the trigger; a press outside closes it. Escape is stopped here so nothing
 * above it (the panel, an overlay drawer) treats it as its own dismiss.
 */
import { useEffect, useId, useRef, useState } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';
import { Check, Ellipsis } from 'lucide-react';
import { useT } from '../../i18n/useT';

export interface PanelMenuItem {
  id: string;
  label: string;
  icon?: ReactNode;
  /** Present → a radio item (Write / Preview); `true` marks the current one. */
  checked?: boolean;
  onSelect: () => void;
}

export interface PanelMoreMenuProps {
  items: PanelMenuItem[];
  /** Extra wrapper classes, e.g. container-query visibility. */
  className?: string;
}

function itemsOf(menu: HTMLElement | null): HTMLElement[] {
  return Array.from(menu?.querySelectorAll<HTMLElement>('[role^="menuitem"]') ?? []);
}

export default function PanelMoreMenu({ items, className }: PanelMoreMenuProps) {
  const t = useT();
  const menuId = useId();
  const [open, setOpen] = useState(false);
  // The item focused on open: 0 = first, -1 = last (ArrowUp on the trigger).
  const [initialFocus, setInitialFocus] = useState(0);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const all = itemsOf(menuRef.current);
    all[initialFocus < 0 ? all.length - 1 : initialFocus]?.focus({ preventScroll: true });
  }, [open, initialFocus]);

  useEffect(() => {
    if (!open) return;
    const handleMouseDown = (e: MouseEvent) => {
      if (!wrapperRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handleMouseDown);
    return () => document.removeEventListener('mousedown', handleMouseDown);
  }, [open]);

  const openAt = (index: number) => {
    setInitialFocus(index);
    setOpen(true);
  };
  const closeToTrigger = () => {
    setOpen(false);
    triggerRef.current?.focus({ preventScroll: true });
  };

  const handleTriggerKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      openAt(e.key === 'ArrowDown' ? 0 : -1);
    } else if (e.key === 'Escape' && open) {
      e.preventDefault();
      e.stopPropagation();
      closeToTrigger();
    }
  };

  const handleMenuKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const all = itemsOf(menuRef.current);
    const current = all.indexOf(document.activeElement as HTMLElement);
    let next: number;
    switch (e.key) {
      case 'ArrowDown':
        next = (current + 1) % all.length;
        break;
      case 'ArrowUp':
        next = current <= 0 ? all.length - 1 : current - 1;
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = all.length - 1;
        break;
      case 'Escape':
        e.preventDefault();
        e.stopPropagation();
        closeToTrigger();
        return;
      case 'Tab':
        // Focus moves on to wherever Tab takes it; the menu just goes away.
        setOpen(false);
        return;
      default:
        return;
    }
    e.preventDefault();
    all[next]?.focus();
  };

  return (
    <div ref={wrapperRef} className={`relative shrink-0 ${className ?? ''}`}>
      <button
        ref={triggerRef}
        type="button"
        aria-label={t('panel.more')}
        title={t('panel.more')}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => (open ? setOpen(false) : openAt(0))}
        onKeyDown={handleTriggerKeyDown}
        className="flex h-7 w-7 items-center justify-center rounded-md text-secondary hover:text-primary hover:bg-card-hover transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
      >
        <Ellipsis size={16} aria-hidden />
      </button>
      {open && (
        <div
          ref={menuRef}
          id={menuId}
          role="menu"
          aria-label={t('panel.more')}
          onKeyDown={handleMenuKeyDown}
          className="absolute right-0 top-full mt-1 z-50 min-w-[9rem] rounded-md border border-border bg-elevated shadow-md py-1"
        >
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              role={item.checked === undefined ? 'menuitem' : 'menuitemradio'}
              aria-checked={item.checked}
              tabIndex={-1}
              onClick={() => {
                closeToTrigger();
                item.onSelect();
              }}
              className="flex w-full items-center gap-2 whitespace-nowrap px-3 py-1.5 text-left text-xs text-primary hover:bg-card-hover focus:outline-none focus-visible:bg-card-hover"
            >
              {item.checked === undefined ? (
                item.icon
              ) : (
                <Check size={14} aria-hidden className={item.checked ? '' : 'invisible'} />
              )}
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
