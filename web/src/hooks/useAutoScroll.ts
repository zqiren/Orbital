import { useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';

/**
 * Auto-follow arbiter for a streaming transcript (backlog #44).
 *
 * The chat pane used to force `scrollTop = scrollHeight` on every reasoning /
 * answer token, so a user who scrolled up to read earlier content was yanked
 * back to the bottom on the next frame. This hook makes the follow behaviour
 * conditional: `scrollToBottom()` only moves the view when the user is already
 * pinned to the bottom; when they have scrolled away it no-ops and records that
 * fresh content arrived (so a "jump to latest" pill can surface).
 *
 * Gating lives INSIDE `scrollToBottom`, so the many existing call sites keep
 * calling it unchanged — only the user's own send / the pill / a session switch
 * pass `{ force: true }` to override the gate.
 *
 * One more rule, with no caller involved: if the view was at the bottom before
 * the chat area changed size, it is at the bottom after. A ResizeObserver on
 * the container and on each of its children catches a composer that grows
 * (the container gets shorter), a panel that opens (the column narrows and
 * the text re-wraps) and content that grows on its own, and re-pins in the
 * observer callback — after layout, before paint, so no displaced frame shows.
 * The exception is the user opening something inside the transcript (a press
 * there, then children grow while the container's own box stays put): the
 * thing they clicked stays under the cursor, and a view pushed past the
 * threshold counts as scrolled away.
 * A reader who has scrolled up keeps the message at the top of their view
 * when the column's width changes (engines without native scroll anchoring
 * would leave them at the old pixel offset, far from what they were reading).
 */

/** How close (px) to the bottom still counts as "at the bottom". */
const AT_BOTTOM_THRESHOLD_PX = 80;
/**
 * After the hook scrolls the container itself, the browser emits scroll
 * events for the settle. For this window they are attributed to our own
 * programmatic scroll and not mistaken for the user scrolling away.
 */
const PROGRAMMATIC_SCROLL_GUARD_MS = 150;
/**
 * After a pointer press or Enter/Space inside the transcript, content that
 * grows without the container's own box changing is the user opening
 * something (a fold, a dispatch card, a tool step). It must stay under their
 * cursor, so that growth is measured, not re-pinned. Covers the longest
 * expand animation (AutoHeight, ~180 ms) plus a slow WebKit layout.
 */
const INTERACTION_WINDOW_MS = 600;
/** How deep the reading anchor descends into a message (bubble → paragraph). */
const ANCHOR_MAX_DEPTH = 6;

export interface ScrollToBottomOptions {
  /** Scroll to the bottom AND re-arm auto-follow, regardless of position. */
  force?: boolean;
}

export interface UseAutoScrollResult {
  /** True while the viewport is within the threshold of the bottom. */
  isAtBottom: boolean;
  /** True when content arrived after the user scrolled away from the bottom. */
  hasNewContent: boolean;
  /** Show the jump-to-latest pill: scrolled up AND new content has arrived. */
  showJumpButton: boolean;
  /** Follow the bottom (no-op when scrolled up), or `{ force: true }` to jump. */
  scrollToBottom: (opts?: ScrollToBottomOptions) => void;
  /** Attach to the scroll container's `onScroll`. */
  onScroll: () => void;
  /** Re-pin to the bottom and clear pill state (e.g. on session switch). */
  reset: () => void;
}

interface ReadingAnchor {
  node: Element;
  /** node's top minus the container's top, in px, when last recorded. */
  offset: number;
}

/**
 * The first child of `parent` whose box reaches below `top` (the container's
 * top edge). The transcript's own children are in vertical order, so a long
 * list is binary-searched; deeper levels can be rows, so they are scanned.
 */
function firstReaching(parent: Element, top: number): Element | null {
  const kids = parent.children;
  const n = kids.length;
  if (n === 0) return null;
  if (n > 16) {
    let lo = 0;
    let hi = n - 1;
    let found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (kids[mid].getBoundingClientRect().bottom > top) {
        found = mid;
        hi = mid - 1;
      } else {
        lo = mid + 1;
      }
    }
    return found >= 0 ? kids[found] : null;
  }
  for (let i = 0; i < n; i++) {
    const r = kids[i].getBoundingClientRect();
    if (r.height > 0 && r.bottom > top) return kids[i];
  }
  return null;
}

export function useAutoScroll(
  scrollRef: RefObject<HTMLElement | null>,
): UseAutoScrollResult {
  const [isAtBottom, setIsAtBottom] = useState(true);
  const [hasNewContent, setHasNewContent] = useState(false);

  // Synchronous mirror of `isAtBottom`. scrollToBottom runs from stream-delta
  // handlers that can fire many times per frame; reading the React state there
  // would race its async commit and re-pin the view right after the user
  // scrolled away. The ref is the source of truth for the gate.
  const isAtBottomRef = useRef(true);
  // Timestamp (ms) until which onScroll events are treated as our own.
  const programmaticUntilRef = useRef(0);
  // The scrollTop our last pin produced. Its scroll event arrives a frame
  // later, by which time more content may have landed below; matching the
  // position (not re-measuring the gap) is what says "this was us", so that
  // late event can't unpin the view. A user's scroll lands anywhere else.
  const pinnedTopRef = useRef<number | null>(null);
  // The scrollTop at the last scroll event (or our own write). Leaving the
  // bottom means moving UP; a pinned view that moved down was moved by the
  // engine (scroll anchoring after a re-wrap, which can fire its scroll event
  // before the ResizeObserver runs), never by a user leaving the bottom.
  const lastTopRef = useRef(0);
  // Scrolled up only: what the reader is looking at, kept for a width change.
  const anchorRef = useRef<ReadingAnchor | null>(null);
  // The container's last observed box (-1 until the first observation).
  const widthRef = useRef(-1);
  const heightRef = useRef(-1);
  // Until when (ms) child growth counts as the user opening something.
  const interactionUntilRef = useRef(0);

  const measureAtBottom = useCallback((el: HTMLElement): boolean => {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= AT_BOTTOM_THRESHOLD_PX;
  }, []);

  /** Pin now (layout is current) and remember where the pin landed. */
  const pinNow = useCallback((el: HTMLElement) => {
    const before = el.scrollTop;
    el.scrollTop = el.scrollHeight;
    const after = el.scrollTop;
    // A pin that moved nothing fires no scroll event, so it has none to claim.
    pinnedTopRef.current = after !== before ? after : null;
    lastTopRef.current = after;
  }, []);

  /** Record the deepest element at the top of the view and its offset. */
  const recordAnchor = useCallback((el: HTMLElement) => {
    const top = el.getBoundingClientRect().top;
    let parent: Element = el;
    let node: Element | null = null;
    for (let depth = 0; depth < ANCHOR_MAX_DEPTH; depth++) {
      const next = firstReaching(parent, top);
      if (!next) break;
      node = next;
      // A node that starts inside the view is already a stable anchor.
      if (next.getBoundingClientRect().top >= top) break;
      parent = next;
    }
    anchorRef.current = node ? { node, offset: node.getBoundingClientRect().top - top } : null;
  }, []);

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Swallow the scroll events our own programmatic scroll produced so a
    // forced jump-to-bottom isn't mis-read as the user scrolling away while it
    // settles. Genuine user scroll-ups during streaming are NOT guarded,
    // because auto-follow scrolls never open this window (only force/reset do).
    const top = el.scrollTop;
    const prevTop = lastTopRef.current;
    lastTopRef.current = top;
    if (Date.now() < programmaticUntilRef.current) return;
    if (isAtBottomRef.current) {
      // Pinned: only a move up can leave the bottom. The late event of our
      // own pin, or the engine re-anchoring downward, keeps the pin; the
      // ResizeObserver finishes the job in the same frame.
      const pinnedTop = pinnedTopRef.current;
      if (top > prevTop + 0.5) return;
      if (pinnedTop !== null && Math.abs(top - pinnedTop) < 1) {
        pinnedTopRef.current = null; // one pin, one event
        return;
      }
    }
    pinnedTopRef.current = null;
    const atBottom = measureAtBottom(el);
    isAtBottomRef.current = atBottom;
    setIsAtBottom(atBottom);
    if (atBottom) {
      setHasNewContent(false);
      anchorRef.current = null;
    } else {
      recordAnchor(el);
    }
  }, [scrollRef, measureAtBottom, recordAnchor]);

  const pinToBottom = useCallback((guard: boolean) => {
    const el = scrollRef.current;
    if (!el) return;
    if (guard) {
      programmaticUntilRef.current = Date.now() + PROGRAMMATIC_SCROLL_GUARD_MS;
    }
    isAtBottomRef.current = true;
    anchorRef.current = null;
    setIsAtBottom(true);
    setHasNewContent(false);
    requestAnimationFrame(() => {
      const e = scrollRef.current;
      if (e) pinNow(e);
    });
  }, [scrollRef, pinNow]);

  const scrollToBottom = useCallback((opts?: ScrollToBottomOptions) => {
    const el = scrollRef.current;
    if (!el) return;
    const force = opts?.force ?? false;
    if (!force && !isAtBottomRef.current) {
      // Auto-follow suppressed: the user is reading earlier content. Record
      // that fresh content arrived so the jump-to-latest pill can surface.
      setHasNewContent(true);
      return;
    }
    // Forced jumps re-enter follow mode and guard their settle window;
    // ordinary auto-follow (already at bottom) needs no guard.
    pinToBottom(force);
  }, [scrollRef, pinToBottom]);

  const reset = useCallback(() => {
    pinToBottom(true);
  }, [pinToBottom]);

  /**
   * The container or something in it changed size. Runs as the
   * ResizeObserver callback: layout is done, paint has not happened.
   */
  const onResize = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const width = el.clientWidth;
    const height = el.clientHeight;
    const widthChanged = widthRef.current >= 0 && width !== widthRef.current;
    const boxChanged = widthRef.current < 0 || widthChanged || height !== heightRef.current;
    widthRef.current = width;
    heightRef.current = height;
    if (isAtBottomRef.current) {
      const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
      if (!boxChanged && Date.now() < interactionUntilRef.current) {
        // The user opened something in the transcript: leave it under the
        // cursor. Re-measure honestly; past the threshold the view has left
        // the bottom, so nothing yanks it later and the pill can show.
        if (gap > AT_BOTTOM_THRESHOLD_PX) {
          isAtBottomRef.current = false;
          pinnedTopRef.current = null;
          setIsAtBottom(false);
          recordAnchor(el);
        }
        return;
      }
      if (gap >= 1) pinNow(el);
      return;
    }
    const anchor = anchorRef.current;
    if (widthChanged && anchor && anchor.node.isConnected) {
      // Correct only what is left after the engine's own scroll anchoring,
      // which (where it exists) already ran during this layout.
      const residual =
        anchor.node.getBoundingClientRect().top - el.getBoundingClientRect().top - anchor.offset;
      if (Math.abs(residual) >= 1) {
        el.scrollTop += residual;
        lastTopRef.current = el.scrollTop;
      }
    }
    recordAnchor(el);
  }, [scrollRef, pinNow, recordAnchor]);

  const markInteraction = useCallback(() => {
    interactionUntilRef.current = Date.now() + INTERACTION_WINDOW_MS;
  }, []);
  const markKeyInteraction = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') markInteraction();
  }, [markInteraction]);

  // Observe whichever element the ref holds (the transcript unmounts while a
  // worker is drilled into). Checked every commit; re-wired only on change.
  const observedRef = useRef<{
    el: HTMLElement | null;
    ro: ResizeObserver | null;
    mo: MutationObserver | null;
    off: (() => void) | null;
  }>({ el: null, ro: null, mo: null, off: null });
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const cur = observedRef.current;
    if (el === cur.el) return;
    cur.ro?.disconnect();
    cur.mo?.disconnect();
    cur.off?.();
    observedRef.current = { el, ro: null, mo: null, off: null };
    widthRef.current = -1;
    heightRef.current = -1;
    if (!el || typeof ResizeObserver === 'undefined') return;
    // Only presses inside the transcript: the composer and the panel handle
    // live outside this element.
    el.addEventListener('pointerdown', markInteraction, true);
    el.addEventListener('keydown', markKeyInteraction, true);
    const off = () => {
      el.removeEventListener('pointerdown', markInteraction, true);
      el.removeEventListener('keydown', markKeyInteraction, true);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(el);
    // The container is a flex column whose children ARE the content, so its
    // content height changes exactly when one of them changes size. A child
    // list observer keeps the observed set current as messages mount.
    for (const child of Array.from(el.children)) ro.observe(child);
    let mo: MutationObserver | null = null;
    if (typeof MutationObserver !== 'undefined') {
      mo = new MutationObserver((records) => {
        for (const rec of records) {
          rec.addedNodes.forEach((n) => { if (n instanceof Element) ro.observe(n); });
          rec.removedNodes.forEach((n) => { if (n instanceof Element) ro.unobserve(n); });
        }
      });
      mo.observe(el, { childList: true });
    }
    observedRef.current = { el, ro, mo, off };
  });
  useEffect(() => () => {
    observedRef.current.off?.();
    observedRef.current.ro?.disconnect();
    observedRef.current.mo?.disconnect();
    observedRef.current = { el: null, ro: null, mo: null, off: null };
  }, []);

  const showJumpButton = !isAtBottom && hasNewContent;

  return {
    isAtBottom,
    hasNewContent,
    showJumpButton,
    scrollToBottom,
    onScroll,
    reset,
  };
}

export default useAutoScroll;
