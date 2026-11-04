// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/** Spec 154 §5.9: ⌘0 / Ctrl+0 goes to the Workbench from anywhere. */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook, cleanup } from '@testing-library/react';
import { useHomeShortcut } from './useHomeShortcut';

afterEach(() => {
  cleanup();
  document.body.innerHTML = '';
});

function press(init: KeyboardEventInit, target: EventTarget = document.body) {
  const e = new KeyboardEvent('keydown', { key: '0', bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(e);
  return e;
}

describe('useHomeShortcut', () => {
  it('⌘0 and Ctrl+0 go home and prevent the default (zoom reset)', () => {
    const onHome = vi.fn();
    renderHook(() => useHomeShortcut(onHome));
    expect(press({ metaKey: true }).defaultPrevented).toBe(true);
    expect(press({ ctrlKey: true }).defaultPrevented).toBe(true);
    expect(onHome).toHaveBeenCalledTimes(2);
  });

  it('ignores 0 alone, and with Shift or Alt', () => {
    const onHome = vi.fn();
    renderHook(() => useHomeShortcut(onHome));
    press({});
    press({ metaKey: true, shiftKey: true });
    press({ ctrlKey: true, altKey: true });
    press({ metaKey: true, key: '9' });
    expect(onHome).not.toHaveBeenCalled();
  });

  it('skips keys aimed at the live browser view (they go to the agent page)', () => {
    const onHome = vi.fn();
    renderHook(() => useHomeShortcut(onHome));
    const live = document.createElement('div');
    live.setAttribute('data-browser-live', 'true');
    const inner = document.createElement('input');
    live.appendChild(inner);
    document.body.appendChild(live);
    const e = press({ metaKey: true }, inner);
    expect(onHome).not.toHaveBeenCalled();
    expect(e.defaultPrevented).toBe(false);
  });

  it('does nothing when another handler already took the key', () => {
    const onHome = vi.fn();
    renderHook(() => useHomeShortcut(onHome));
    const el = document.createElement('div');
    document.body.appendChild(el);
    el.addEventListener('keydown', (ev) => ev.preventDefault());
    press({ metaKey: true }, el);
    expect(onHome).not.toHaveBeenCalled();
  });

  it('stops listening on unmount', () => {
    const onHome = vi.fn();
    const { unmount } = renderHook(() => useHomeShortcut(onHome));
    unmount();
    press({ metaKey: true });
    expect(onHome).not.toHaveBeenCalled();
  });
});
