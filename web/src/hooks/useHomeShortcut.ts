// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 154 §5.9: ⌘0 / Ctrl+0 goes to the Workbench (home) from anywhere.
 * One window-level keydown listener: Meta or Ctrl with 0, no Shift / Alt,
 * not already handled, and not while focus is inside the live browser view
 * (`[data-browser-live]`), which forwards keys to the agent's page.
 */
import { useEffect, useRef } from 'react';

export function useHomeShortcut(onHome: () => void): void {
  const ref = useRef(onHome);
  ref.current = onHome;
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.defaultPrevented || e.shiftKey || e.altKey) return;
      if (!(e.metaKey || e.ctrlKey) || e.key !== '0') return;
      const target = e.target as Element | null;
      if (target && typeof target.closest === 'function' && target.closest('[data-browser-live]')) return;
      e.preventDefault();
      ref.current();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
}
