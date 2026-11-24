// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useMemo, useState, type RefObject, type UIEvent } from 'react';

/**
 * Design pass D (2026-11-24): a header that belongs to its sheet paints no
 * band and no line at rest; a 1px hairline (index.css `[data-scrolled]`)
 * shows only while the content under it is scrolled.
 *
 * Spread `scrollProps` on the box that holds the header's content: scroll
 * events don't bubble, so it listens in the capture phase and hears every
 * scroller inside it, wherever that scroller is rendered. A box that only
 * scrolls sideways (a wide code block, a grid) is ignored, and so is any
 * scroller inside a nested box that spreads its own `scrollProps` (a file's
 * header inside the panel): that header owns it. `resetKey` clears the line
 * when what the header heads is replaced (another file, another page).
 */
export function useScrollHairline(resetKey?: unknown) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    setScrolled(false);
  }, [resetKey]);

  const onScrollCapture = useCallback((e: UIEvent<HTMLElement>) => {
    const next = scrolledFor(e.target, e.currentTarget);
    if (next !== null) setScrolled((prev) => (prev === next ? prev : next));
  }, []);

  const scrollProps = useMemo(
    () => ({ onScrollCapture, 'data-hairline-scope': '' }),
    [onScrollCapture],
  );

  return { scrolled, scrollProps };
}

/** Whether a scroll on `target` heard by `scope` means "scrolled", or null
 *  when the event is not this header's to judge. */
function scrolledFor(target: EventTarget | null, scope: Element): boolean | null {
  if (!(target instanceof HTMLElement)) return null;
  if (target.scrollHeight <= target.clientHeight) return null;
  const owner = target.closest('[data-hairline-scope]');
  if (owner && owner !== scope) return null;
  return target.scrollTop > 0;
}

/**
 * The same rule for a header that cannot reach the box around its content
 * (FilePreview's file header: one shared row placed by many branches). It
 * listens on its own parent, which is that box, and marks it as a scope so an
 * outer header (the panel bar) leaves those scrollers to it.
 */
export function useParentScrollHairline(ref: RefObject<HTMLElement | null>, resetKey?: unknown) {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    setScrolled(false);
    const host = ref.current?.parentElement;
    if (!host) return;
    host.setAttribute('data-hairline-scope', '');
    const onScroll = (e: Event) => {
      const next = scrolledFor(e.target, host);
      if (next !== null) setScrolled((prev) => (prev === next ? prev : next));
    };
    host.addEventListener('scroll', onScroll, { capture: true, passive: true });
    return () => {
      host.removeEventListener('scroll', onScroll, { capture: true });
      host.removeAttribute('data-hairline-scope');
    };
  }, [ref, resetKey]);
  return scrolled;
}
