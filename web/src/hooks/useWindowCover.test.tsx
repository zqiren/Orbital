// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * 2026-11-04: a maximized document takes the whole window except the 20px
 * edge strip. A project list pinned open collapses to the strip while
 * anything covers the window and comes back when it ends (App reads
 * `useWindowCovered`). Everything that maximizes registers here.
 */

import { act, cleanup, render, renderHook, screen, fireEvent } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { effectivePinned, useCoverWindow, useWindowCovered } from './useWindowCover';
import FilePreviewDrawer from '../components/FilePreviewDrawer';

afterEach(cleanup);

describe('useWindowCover', () => {
  it('nothing covers the window until something maximized registers, and it ends on unmount', () => {
    const covered = renderHook(() => useWindowCovered());
    expect(covered.result.current).toBe(false);
    const a = renderHook(({ on }) => useCoverWindow(on), { initialProps: { on: true } });
    expect(covered.result.current).toBe(true);
    const b = renderHook(() => useCoverWindow(true));
    a.rerender({ on: false });
    expect(covered.result.current).toBe(true); // b still covers
    act(() => b.unmount());
    expect(covered.result.current).toBe(false);
    a.unmount();
  });

  it('a pinned list is shown pinned only while nothing covers the window', () => {
    expect(effectivePinned(true, false)).toBe(true);
    expect(effectivePinned(true, true)).toBe(false);
    expect(effectivePinned(false, true)).toBe(false);
  });

  it('the document panel registers while maximized, docked or as an overlay', () => {
    const covered = renderHook(() => useWindowCovered());
    const view = render(
      <FilePreviewDrawer docked maximized open selectedPath={null} fileContent={null} loading={false} onClose={vi.fn()}>
        <div />
      </FilePreviewDrawer>,
    );
    expect(covered.result.current).toBe(true);
    view.rerender(
      <FilePreviewDrawer docked open selectedPath={null} fileContent={null} loading={false} onClose={vi.fn()}>
        <div />
      </FilePreviewDrawer>,
    );
    expect(covered.result.current).toBe(false);
    view.rerender(
      <FilePreviewDrawer maximized open selectedPath={null} fileContent={null} loading={false} onClose={vi.fn()}>
        <div />
      </FilePreviewDrawer>,
    );
    expect(covered.result.current).toBe(true);
    view.unmount();
    expect(covered.result.current).toBe(false);
  });
});

// Keep the imports used even if a test above is trimmed.
void screen; void fireEvent;
