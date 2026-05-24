// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSession } from './useSession';

describe('useSession', () => {
  beforeEach(() => {
    // Clear relevant localStorage keys before each test.
    for (const key of Object.keys(localStorage)) {
      if (key.startsWith('orbital:lastSession:')) {
        localStorage.removeItem(key);
      }
    }
  });

  it('returns null initially when no session has been persisted', () => {
    const { result } = renderHook(() => useSession('proj-new'));
    expect(result.current.activeSessionId).toBeNull();
  });

  it('returns null when projectId is null', () => {
    const { result } = renderHook(() => useSession(null));
    expect(result.current.activeSessionId).toBeNull();
  });

  it('persists the active session to localStorage on setActiveSessionId', () => {
    const { result } = renderHook(() => useSession('proj-a'));

    act(() => {
      result.current.setActiveSessionId('sess-42');
    });

    expect(result.current.activeSessionId).toBe('sess-42');
    expect(localStorage.getItem('orbital:lastSession:proj-a')).toBe('sess-42');
  });

  it('restores persisted session on mount', () => {
    // Pre-populate localStorage as if from a previous session.
    localStorage.setItem('orbital:lastSession:proj-b', 'sess-restored');

    const { result } = renderHook(() => useSession('proj-b'));
    expect(result.current.activeSessionId).toBe('sess-restored');
  });

  it('updates localStorage when setActiveSessionId is called with a new value', () => {
    localStorage.setItem('orbital:lastSession:proj-c', 'sess-old');

    const { result } = renderHook(() => useSession('proj-c'));
    expect(result.current.activeSessionId).toBe('sess-old');

    act(() => {
      result.current.setActiveSessionId('sess-new');
    });

    expect(result.current.activeSessionId).toBe('sess-new');
    expect(localStorage.getItem('orbital:lastSession:proj-c')).toBe('sess-new');
  });

  it('removes the localStorage key when setActiveSessionId is called with null', () => {
    localStorage.setItem('orbital:lastSession:proj-d', 'sess-to-remove');

    const { result } = renderHook(() => useSession('proj-d'));

    act(() => {
      result.current.setActiveSessionId(null);
    });

    expect(result.current.activeSessionId).toBeNull();
    expect(localStorage.getItem('orbital:lastSession:proj-d')).toBeNull();
  });

  it('loads a different stored session when projectId changes', () => {
    localStorage.setItem('orbital:lastSession:proj-x', 'sess-x');
    localStorage.setItem('orbital:lastSession:proj-y', 'sess-y');

    let projectId = 'proj-x';
    const { result, rerender } = renderHook(() => useSession(projectId));
    expect(result.current.activeSessionId).toBe('sess-x');

    // Switch project
    projectId = 'proj-y';
    rerender();

    expect(result.current.activeSessionId).toBe('sess-y');
  });

  it('each project has an independent storage key', () => {
    const { result: r1 } = renderHook(() => useSession('proj-p1'));
    const { result: r2 } = renderHook(() => useSession('proj-p2'));

    act(() => {
      r1.current.setActiveSessionId('sess-for-p1');
    });

    // p2 should still be unaffected
    expect(r2.current.activeSessionId).toBeNull();
    expect(localStorage.getItem('orbital:lastSession:proj-p1')).toBe('sess-for-p1');
    expect(localStorage.getItem('orbital:lastSession:proj-p2')).toBeNull();
  });
});
