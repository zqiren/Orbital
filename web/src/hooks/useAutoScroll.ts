import { useCallback, useRef, useState, type RefObject } from 'react';

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
 */

/** How close (px) to the bottom still counts as "at the bottom". */
const AT_BOTTOM_THRESHOLD_PX = 80;
/**
 * After the hook scrolls the container itself, the browser emits scroll
 * events for the settle. For this window they are attributed to our own
 * programmatic scroll and not mistaken for the user scrolling away.
 */
const PROGRAMMATIC_SCROLL_GUARD_MS = 150;

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

  const measureAtBottom = useCallback((el: HTMLElement): boolean => {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= AT_BOTTOM_THRESHOLD_PX;
  }, []);

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Swallow the scroll events our own programmatic scroll produced so a
    // forced jump-to-bottom isn't mis-read as the user scrolling away while it
    // settles. Genuine user scroll-ups during streaming are NOT guarded,
    // because auto-follow scrolls never open this window (only force/reset do).
    if (Date.now() < programmaticUntilRef.current) return;
    const atBottom = measureAtBottom(el);
    isAtBottomRef.current = atBottom;
    setIsAtBottom(atBottom);
    if (atBottom) setHasNewContent(false);
  }, [scrollRef, measureAtBottom]);

  const pinToBottom = useCallback((guard: boolean) => {
    const el = scrollRef.current;
    if (!el) return;
    if (guard) {
      programmaticUntilRef.current = Date.now() + PROGRAMMATIC_SCROLL_GUARD_MS;
    }
    isAtBottomRef.current = true;
    setIsAtBottom(true);
    setHasNewContent(false);
    requestAnimationFrame(() => {
      const e = scrollRef.current;
      if (e) e.scrollTop = e.scrollHeight;
    });
  }, [scrollRef]);

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
