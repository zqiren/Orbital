// Unit tests for the auto-follow arbiter (backlog #44).
//
// jsdom cannot measure real layout, so every test stubs the scroll
// container's geometry (scrollHeight / clientHeight / scrollTop) on a plain
// object standing in for the DOM element. requestAnimationFrame is stubbed to
// run synchronously so scrollToBottom's write is observable in the same tick.
import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useAutoScroll } from './useAutoScroll';

interface FakeEl {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
}

function makeEl(partial: Partial<FakeEl> = {}): FakeEl {
  return {
    scrollTop: partial.scrollTop ?? 0,
    scrollHeight: partial.scrollHeight ?? 1000,
    clientHeight: partial.clientHeight ?? 500,
  };
}

describe('useAutoScroll', () => {
  let nowSpy: ReturnType<typeof vi.spyOn>;
  let rafSpy: ReturnType<typeof vi.spyOn>;
  let currentNow = 1_000_000;

  beforeEach(() => {
    currentNow = 1_000_000;
    nowSpy = vi.spyOn(Date, 'now').mockImplementation(() => currentNow);
    // Run rAF callbacks immediately so scrollTop writes are synchronous.
    rafSpy = vi
      .spyOn(globalThis, 'requestAnimationFrame')
      .mockImplementation((cb: FrameRequestCallback) => {
        cb(0);
        return 0;
      });
  });

  afterEach(() => {
    nowSpy.mockRestore();
    rafSpy.mockRestore();
  });

  it('detects at-bottom at the 80px boundary (inclusive) and not one pixel above', () => {
    const el = makeEl({ scrollHeight: 1000, clientHeight: 500 });
    const ref = { current: el as unknown as HTMLElement };
    const { result } = renderHook(() => useAutoScroll(ref));

    // distance = scrollHeight - scrollTop - clientHeight = 500 - scrollTop.
    // scrollTop 420 -> distance 80 -> exactly at the threshold (inclusive).
    el.scrollTop = 420;
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(true);

    // scrollTop 419 -> distance 81 -> just past the threshold.
    el.scrollTop = 419;
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(false);

    // Back within the band.
    el.scrollTop = 480;
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(true);
  });

  it('marks new content and suppresses the auto-follow scroll when the user is scrolled up', () => {
    const el = makeEl({ scrollTop: 0, scrollHeight: 1000, clientHeight: 500 });
    const ref = { current: el as unknown as HTMLElement };
    const { result } = renderHook(() => useAutoScroll(ref));

    // User scrolls to the top.
    el.scrollTop = 0;
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(false);

    // A stream delta arrives -> the non-forced scroll must NOT move the view.
    act(() => result.current.scrollToBottom());
    expect(el.scrollTop).toBe(0);
    expect(result.current.showJumpButton).toBe(true);
  });

  it('force override scrolls to the bottom and re-arms auto-follow even when scrolled up', () => {
    const el = makeEl({ scrollTop: 0, scrollHeight: 1000, clientHeight: 500 });
    const ref = { current: el as unknown as HTMLElement };
    const { result } = renderHook(() => useAutoScroll(ref));

    el.scrollTop = 0;
    act(() => result.current.onScroll());
    act(() => result.current.scrollToBottom()); // arrives while scrolled up
    expect(result.current.showJumpButton).toBe(true);

    // Forced jump: pins to bottom, clears the pill, re-enters follow mode.
    act(() => result.current.scrollToBottom({ force: true }));
    expect(el.scrollTop).toBe(1000);
    expect(result.current.isAtBottom).toBe(true);
    expect(result.current.showJumpButton).toBe(false);
  });

  it('ignores the scroll events its own programmatic scroll produces during the guard window', () => {
    const el = makeEl({ scrollTop: 500, scrollHeight: 1000, clientHeight: 500 });
    const ref = { current: el as unknown as HTMLElement };
    const { result } = renderHook(() => useAutoScroll(ref));

    // Force a jump -> opens the ~150ms programmatic guard window.
    act(() => result.current.scrollToBottom({ force: true }));
    expect(result.current.isAtBottom).toBe(true);

    // A scroll event fires WITHIN the guard window reporting not-at-bottom
    // (e.g. the browser settling the programmatic scroll / content growth).
    // It must be attributed to our own scroll and ignored.
    currentNow += 100; // still inside the 150ms window
    el.scrollTop = 0;
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(true);

    // After the window expires, the same geometry is honoured as a real
    // user scroll.
    currentNow += 200; // now past the window
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(false);
  });

  it('reset() re-pins to the bottom and clears the pill (session switch)', () => {
    const el = makeEl({ scrollTop: 0, scrollHeight: 1000, clientHeight: 500 });
    const ref = { current: el as unknown as HTMLElement };
    const { result } = renderHook(() => useAutoScroll(ref));

    el.scrollTop = 0;
    act(() => result.current.onScroll());
    act(() => result.current.scrollToBottom());
    expect(result.current.showJumpButton).toBe(true);

    act(() => result.current.reset());
    expect(el.scrollTop).toBe(1000);
    expect(result.current.isAtBottom).toBe(true);
    expect(result.current.showJumpButton).toBe(false);
  });
});
