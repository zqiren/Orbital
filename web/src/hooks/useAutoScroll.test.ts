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
  // A scrolled-up view records what it is reading; an empty transcript
  // gives it nothing to record.
  children: Element[];
  getBoundingClientRect: () => DOMRect;
}

function makeEl(partial: Partial<Pick<FakeEl, 'scrollTop' | 'scrollHeight' | 'clientHeight'>> = {}): FakeEl {
  return {
    scrollTop: partial.scrollTop ?? 0,
    scrollHeight: partial.scrollHeight ?? 1000,
    clientHeight: partial.clientHeight ?? 500,
    children: [],
    getBoundingClientRect: () => ({ top: 0, bottom: 500, height: 500 }) as DOMRect,
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

// ─── The chat area changes size: at the bottom before means at the bottom after ──
//
// jsdom has no layout and no ResizeObserver, so this models both: a real div
// whose scroll metrics come from a small layout model (scrollTop clamps like a
// browser), children whose rects follow their content offset and the scroll
// position, and a ResizeObserver stub whose callback the test fires where the
// browser would (after layout, before paint).

class MockResizeObserver {
  static instances: MockResizeObserver[] = [];
  observed = new Set<Element>();
  constructor(private cb: ResizeObserverCallback) {
    MockResizeObserver.instances.push(this);
  }
  observe(target: Element) { this.observed.add(target); }
  unobserve(target: Element) { this.observed.delete(target); }
  disconnect() { this.observed.clear(); }
  fire() { this.cb([], this as unknown as ResizeObserver); }
}

interface Block { top: number; height: number }

function makeLayout(opts: { blocks: Block[]; clientHeight?: number; clientWidth?: number }) {
  const el = document.createElement('div');
  document.body.appendChild(el); // a reading anchor must be connected
  const state = {
    scrollTop: 0,
    clientHeight: opts.clientHeight ?? 500,
    clientWidth: opts.clientWidth ?? 800,
    blocks: opts.blocks,
    scrollWrites: 0,
  };
  const scrollHeight = () => {
    const last = state.blocks[state.blocks.length - 1];
    return Math.max(state.clientHeight, last ? last.top + last.height : 0);
  };
  const clamp = (v: number) => Math.max(0, Math.min(v, scrollHeight() - state.clientHeight));
  Object.defineProperty(el, 'scrollTop', {
    configurable: true,
    get: () => state.scrollTop,
    set: (v: number) => { state.scrollWrites++; state.scrollTop = clamp(v); },
  });
  Object.defineProperty(el, 'scrollHeight', { configurable: true, get: scrollHeight });
  Object.defineProperty(el, 'clientHeight', { configurable: true, get: () => state.clientHeight });
  Object.defineProperty(el, 'clientWidth', { configurable: true, get: () => state.clientWidth });
  el.getBoundingClientRect = () => ({ top: 0, bottom: state.clientHeight, height: state.clientHeight }) as DOMRect;
  const kids: HTMLElement[] = state.blocks.map((_, i) => {
    const c = document.createElement('div');
    c.getBoundingClientRect = () => {
      const b = state.blocks[i];
      return { top: b.top - state.scrollTop, bottom: b.top + b.height - state.scrollTop, height: b.height } as DOMRect;
    };
    el.appendChild(c);
    return c;
  });
  /** Re-flow: every block gets `height`, stacked from 0 (a re-wrap). */
  const reflow = (height: number) => {
    state.blocks = state.blocks.map((_, i) => ({ top: i * height, height }));
    state.scrollTop = clamp(state.scrollTop);
  };
  const gap = () => scrollHeight() - state.scrollTop - state.clientHeight;
  return { el, state, kids, reflow, gap, scrollHeight };
}

const stack = (n: number, height: number): Block[] =>
  Array.from({ length: n }, (_, i) => ({ top: i * height, height }));

describe('useAutoScroll: the chat area changes size', () => {
  let rafSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    MockResizeObserver.instances = [];
    vi.stubGlobal('ResizeObserver', MockResizeObserver);
    rafSpy = vi
      .spyOn(globalThis, 'requestAnimationFrame')
      .mockImplementation((cb: FrameRequestCallback) => { cb(0); return 0; });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    rafSpy.mockRestore();
    document.body.innerHTML = '';
  });

  function mount(layout: ReturnType<typeof makeLayout>) {
    const ref = { current: layout.el as HTMLElement };
    const hook = renderHook(() => useAutoScroll(ref));
    const ro = MockResizeObserver.instances[MockResizeObserver.instances.length - 1];
    // The first observation the browser delivers on observe().
    act(() => ro.fire());
    return { ...hook, ro };
  }

  it('observes the container and every child, and children that mount later', async () => {
    const l = makeLayout({ blocks: stack(3, 100) });
    const { ro } = mount(l);
    expect(ro.observed.has(l.el)).toBe(true);
    for (const k of l.kids) expect(ro.observed.has(k)).toBe(true);

    const added = document.createElement('div');
    l.el.appendChild(added);
    await Promise.resolve(); // MutationObserver delivers on a microtask
    expect(ro.observed.has(added)).toBe(true);
    l.el.removeChild(added);
    await Promise.resolve();
    expect(ro.observed.has(added)).toBe(false);
  });

  it('pins on the first observation (a fresh transcript opens at the bottom)', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    mount(l);
    expect(l.gap()).toBe(0);
  });

  it('a growing composer (the container gets shorter) keeps a pinned view at the bottom', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro } = mount(l);
    for (let line = 0; line < 5; line++) {
      l.state.clientHeight -= 21;
      act(() => ro.fire());
      expect(l.gap()).toBe(0);
    }
  });

  it('a narrower column (text re-wraps taller) keeps a pinned view at the bottom', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    l.state.clientWidth = 400;
    l.reflow(180); // 2000px of content becomes 3600px; scrollTop stays put
    expect(l.gap()).toBe(1600);
    act(() => ro.fire());
    expect(l.gap()).toBe(0);
    expect(result.current.isAtBottom).toBe(true);
  });

  it("the pin's own late scroll event does not unpin, even if more content landed since", () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    l.state.clientWidth = 400;
    l.reflow(180);
    act(() => ro.fire());
    // Before the scroll event of that pin is dispatched, a big block lands.
    l.state.blocks.push({ top: 3600, height: 900 });
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(true);
    act(() => ro.fire());
    expect(l.gap()).toBe(0);
  });

  it('engine scroll anchoring that moves a pinned view down (event before the observer) keeps the pin', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    l.state.clientWidth = 400;
    l.reflow(180);
    // Native anchoring moved the view down, but not all the way (360px short),
    // and its scroll event is dispatched before the ResizeObserver callback.
    l.state.scrollTop = l.scrollHeight() - l.state.clientHeight - 360;
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(true);
    act(() => ro.fire());
    expect(l.gap()).toBe(0);
  });

  it('a user who scrolled up is not moved when content or the composer changes size (backlog #44)', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    l.state.scrollTop -= 500; // the user scrolls up
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(false);
    const top = l.state.scrollTop;

    l.state.blocks.push({ top: 2000, height: 300 }); // streaming grows the tail
    act(() => ro.fire());
    l.state.clientHeight -= 63; // the composer grows three lines
    act(() => ro.fire());
    expect(l.state.scrollTop).toBe(top);
    expect(result.current.isAtBottom).toBe(false);
  });

  it('a user scrolling up from a pinned view still unpins (moving up is never ignored)', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { result } = mount(l);
    l.state.scrollTop -= 200;
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(false);
    act(() => result.current.scrollToBottom()); // a stream token
    expect(l.gap()).toBe(200);
    expect(result.current.showJumpButton).toBe(true);
  });

  it('a scrolled-up reader keeps the message at the top of the view when the width changes (no native anchoring)', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    l.state.scrollTop = 730; // block 7 is at the top, 30px of it scrolled off
    act(() => result.current.onScroll());
    expect(result.current.isAtBottom).toBe(false);

    l.state.clientWidth = 400;
    l.reflow(180); // block 7 now starts at 1260; scrollTop stays at 730
    act(() => ro.fire());
    expect(l.kids[7].getBoundingClientRect().top).toBe(-30);

    l.state.clientWidth = 800; // and back
    l.reflow(100);
    act(() => ro.fire());
    expect(l.kids[7].getBoundingClientRect().top).toBe(-30);
    expect(result.current.isAtBottom).toBe(false);
  });

  it('corrects only the residual when the engine already anchored (no double correction)', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    l.state.scrollTop = 730;
    act(() => result.current.onScroll());

    l.state.clientWidth = 400;
    l.reflow(180);
    l.state.scrollTop = 1260 + 30; // native anchoring put block 7 back itself
    const writes = l.state.scrollWrites;
    act(() => ro.fire());
    expect(l.state.scrollWrites).toBe(writes);
    expect(l.kids[7].getBoundingClientRect().top).toBe(-30);
  });

  it('a height-only change while scrolled up does not move the reader (history prepend keeps its own compensation)', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    l.state.scrollTop = 730;
    act(() => result.current.onScroll());
    // Older history prepended above: the caller's layout effect compensates.
    l.state.blocks = [{ top: 0, height: 400 }, ...l.state.blocks.map((b) => ({ ...b, top: b.top + 400 }))];
    l.state.scrollTop += 400;
    const writes = l.state.scrollWrites;
    act(() => ro.fire());
    expect(l.state.scrollWrites).toBe(writes);
    expect(l.state.scrollTop).toBe(1130);
  });

  it('reset() after a scrolled-up width change re-pins and drops the reading anchor', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    l.state.scrollTop = 730;
    act(() => result.current.onScroll());
    act(() => result.current.reset());
    expect(l.gap()).toBe(0);
    l.state.clientWidth = 400;
    l.reflow(180);
    act(() => ro.fire());
    expect(l.gap()).toBe(0);
    expect(result.current.isAtBottom).toBe(true);
  });

  // The user opens something inside the transcript (a fold, a dispatch card,
  // a tool step): it stays under their cursor.
  it('a press in the transcript then a child growing past the threshold: no scroll write, unpinned', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    const top = l.state.scrollTop;
    act(() => { l.kids[19].dispatchEvent(new Event('pointerdown', { bubbles: true })); });
    l.state.blocks[19] = { top: 1900, height: 400 }; // the last item opens
    const writes = l.state.scrollWrites;
    act(() => ro.fire());
    expect(l.state.scrollWrites).toBe(writes);
    expect(l.state.scrollTop).toBe(top);
    expect(result.current.isAtBottom).toBe(false);
    // A later stream token neither yanks the view nor hides the pill.
    act(() => result.current.scrollToBottom());
    expect(l.state.scrollTop).toBe(top);
    expect(result.current.showJumpButton).toBe(true);
  });

  it('a press in the transcript then a small growth within the threshold: stays pinned, not moved', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    const top = l.state.scrollTop;
    act(() => { l.el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })); });
    l.state.blocks[19] = { top: 1900, height: 150 };
    act(() => ro.fire());
    expect(l.state.scrollTop).toBe(top);
    expect(result.current.isAtBottom).toBe(true);
  });

  it('a press in the transcript then the container itself resizes: still re-pins', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    act(() => { l.kids[19].dispatchEvent(new Event('pointerdown', { bubbles: true })); });
    l.state.clientWidth = 400;
    l.reflow(180);
    act(() => ro.fire());
    expect(l.gap()).toBe(0);
    expect(result.current.isAtBottom).toBe(true);
  });

  it('a child growing with no press in the transcript (streaming, a late image) re-pins', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro, result } = mount(l);
    // A press outside the transcript (the composer) does not count.
    act(() => { document.body.dispatchEvent(new Event('pointerdown', { bubbles: true })); });
    l.state.blocks[19] = { top: 1900, height: 400 };
    act(() => ro.fire());
    expect(l.gap()).toBe(0);
    expect(result.current.isAtBottom).toBe(true);
  });

  it('the press window closes: growth after it re-pins again', () => {
    const l = makeLayout({ blocks: stack(20, 100) });
    const { ro } = mount(l);
    let now = 5_000_000;
    const nowSpy = vi.spyOn(Date, 'now').mockImplementation(() => now);
    act(() => { l.kids[19].dispatchEvent(new Event('pointerdown', { bubbles: true })); });
    now += 700;
    l.state.blocks[19] = { top: 1900, height: 400 };
    act(() => ro.fire());
    expect(l.gap()).toBe(0);
    nowSpy.mockRestore();
  });

  it('disconnects its observers on unmount', () => {
    const l = makeLayout({ blocks: stack(3, 100) });
    const { ro, unmount } = mount(l);
    unmount();
    expect(ro.observed.size).toBe(0);
  });
});
