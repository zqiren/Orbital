// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 078 §5.1/§5.2/§10 — the panel reducer: expand on the first browser or
 * file event of a run, never on a command-only turn, stay collapsed for the
 * rest of a run the user collapsed, collapse when the turn ends, and remember
 * the view per session across a reload.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import {
  PANEL_DOCK_MIN_WIDTH,
  __resetPanelState,
  usePanelDockable,
  usePanelState,
} from './usePanelState';

beforeEach(() => {
  __resetPanelState();
  localStorage.clear();
});
afterEach(() => {
  vi.restoreAllMocks();
});

function mount(projectId = 'proj-1', sessionId: string | undefined = 'sess-1') {
  return renderHook(() => usePanelState(projectId, sessionId));
}

describe('usePanelState — at rest', () => {
  it('starts collapsed on the Files view with no file selected', () => {
    const { result } = mount();
    expect(result.current).toMatchObject({
      expanded: false,
      userCollapsedThisRun: false,
      view: 'files',
      file: null,
    });
  });

  it('a manual expand keeps the remembered view (default files)', () => {
    const { result } = mount();
    act(() => result.current.expand());
    expect(result.current.expanded).toBe(true);
    expect(result.current.view).toBe('files');
  });

  it('a manual expand on a session that last used Browser reopens on Browser', () => {
    const first = mount();
    act(() => first.result.current.setView('browser'));
    first.unmount();

    // A reload: the module store is gone, only localStorage survives.
    __resetPanelState();
    const { result } = mount();
    expect(result.current.view).toBe('browser');
    act(() => result.current.expand());
    expect(result.current).toMatchObject({ expanded: true, view: 'browser' });
  });
});

describe('usePanelState — runtime lifecycle (D8)', () => {
  it('expands and switches to Browser on a browser event', () => {
    const { result } = mount();
    act(() => result.current.expandForEvent('browser'));
    expect(result.current).toMatchObject({ expanded: true, view: 'browser' });
  });

  it('expands and switches to Files on a file event', () => {
    const { result } = mount();
    act(() => result.current.expandForEvent('browser'));
    act(() => result.current.expandForEvent('files'));
    expect(result.current).toMatchObject({ expanded: true, view: 'files' });
  });

  it('setFile / setSelectFile(null) drives the tree ⇄ preview swap', () => {
    const { result } = mount();
    act(() => result.current.setFile('src/a.ts'));
    expect(result.current.file).toBe('src/a.ts');
    act(() => result.current.setFile(null));
    expect(result.current.file).toBeNull();
  });

  it('collapsing during a run keeps it collapsed for the rest of THAT run', () => {
    const { result } = mount();
    act(() => result.current.onRunStart());
    act(() => result.current.expandForEvent('browser'));
    act(() => result.current.collapse());
    expect(result.current).toMatchObject({ expanded: false, userCollapsedThisRun: true });

    // Further events in the same run must not reopen it.
    act(() => result.current.expandForEvent('files'));
    act(() => result.current.expandForEvent('browser'));
    expect(result.current.expanded).toBe(false);
  });

  it('the next run start clears the user-collapsed veto', () => {
    const { result } = mount();
    act(() => result.current.onRunStart());
    act(() => result.current.collapse());
    act(() => result.current.onRunEnd());
    act(() => result.current.onRunStart());
    expect(result.current.userCollapsedThisRun).toBe(false);
    act(() => result.current.expandForEvent('files'));
    expect(result.current.expanded).toBe(true);
  });

  it('collapses and drops the preview selection when the turn ends', () => {
    const { result } = mount();
    act(() => result.current.onRunStart());
    act(() => result.current.expandForEvent('files'));
    act(() => result.current.setFile('src/a.ts'));
    act(() => result.current.onRunEnd());
    expect(result.current).toMatchObject({ expanded: false, file: null, view: 'files' });
  });

  it('remembers the view across a run so reopening lands where the user left it', () => {
    const { result } = mount();
    act(() => result.current.setView('browser'));
    act(() => result.current.onRunEnd());
    expect(result.current.view).toBe('browser');
  });

  it('a manual expand clears the veto (the user asked for it back)', () => {
    const { result } = mount();
    act(() => result.current.collapse());
    act(() => result.current.expand());
    expect(result.current).toMatchObject({ expanded: true, userCollapsedThisRun: false });
  });
});

describe('usePanelState — per session (D14)', () => {
  it('keeps expansion, view and file separate per session', () => {
    const a = mount('proj-1', 'sess-a');
    act(() => a.result.current.expandForEvent('browser'));
    act(() => a.result.current.setFile('a.md'));

    const b = mount('proj-1', 'sess-b');
    expect(b.result.current).toMatchObject({ expanded: false, view: 'files', file: null });
    expect(a.result.current).toMatchObject({ expanded: true, view: 'browser', file: 'a.md' });
  });

  it('keeps state separate per project for the same session id', () => {
    const a = mount('proj-1', 'sess-1');
    act(() => a.result.current.expand());
    const b = mount('proj-2', 'sess-1');
    expect(b.result.current.expanded).toBe(false);
  });

  it('an undefined session id gets its own slot instead of throwing', () => {
    const { result } = mount('proj-1', undefined);
    act(() => result.current.expand());
    expect(result.current.expanded).toBe(true);
  });
});

describe('usePanelState — localStorage mirror', () => {
  it('writes only the view, under orbital:panel:<projectId>:<sessionId>', () => {
    const { result } = mount('proj-9', 'sess-9');
    act(() => result.current.setView('browser'));
    expect(localStorage.getItem('orbital:panel:proj-9:sess-9')).toBe('browser');
  });

  it('does not persist expansion — at rest the panel is always collapsed after a reload', () => {
    const first = mount();
    act(() => first.result.current.expandForEvent('browser'));
    first.unmount();
    __resetPanelState();
    expect(mount().result.current.expanded).toBe(false);
  });

  it('ignores a corrupt stored value', () => {
    localStorage.setItem('orbital:panel:proj-1:sess-1', 'terminal');
    expect(mount().result.current.view).toBe('files');
  });

  it('survives a localStorage that throws on read and on write', () => {
    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('SecurityError');
    });
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError');
    });

    const { result } = mount('proj-throw', 'sess-throw');
    expect(result.current.view).toBe('files');
    act(() => result.current.setView('browser'));
    expect(result.current.view).toBe('browser');
    expect(getItem).toHaveBeenCalled();
    expect(setItem).toHaveBeenCalled();
  });
});

describe('usePanelDockable', () => {
  function setWidth(width: number) {
    Object.defineProperty(window, 'innerWidth', { value: width, configurable: true, writable: true });
  }

  it(`is false below the ${PANEL_DOCK_MIN_WIDTH}px push threshold and true at or above it`, () => {
    setWidth(PANEL_DOCK_MIN_WIDTH - 1);
    expect(renderHook(() => usePanelDockable()).result.current).toBe(false);
    setWidth(PANEL_DOCK_MIN_WIDTH);
    expect(renderHook(() => usePanelDockable()).result.current).toBe(true);
  });

  it('is false at mobile widths', () => {
    setWidth(390);
    expect(renderHook(() => usePanelDockable()).result.current).toBe(false);
  });

  it('re-reads on window resize', () => {
    setWidth(900);
    const { result } = renderHook(() => usePanelDockable());
    expect(result.current).toBe(false);
    act(() => {
      setWidth(1440);
      window.dispatchEvent(new Event('resize'));
    });
    expect(result.current).toBe(true);
  });
});
