// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.3/§5.6 — the Browser view: the agent's page, live and
 * interactive (canvas painted from screencast frames, input forwarded), with
 * a plain browser toolbar (back / forward / reload / address) and the
 * session's last screenshot as fallback when no browser is open.
 *
 * The page fills the view (2026-11-18): the view sends its size when it
 * settles, the route resizes the agent's page to it, and frames arrive at
 * that size and the display's pixel density — drawn 1:1, no bands. While a
 * drag is in progress the last frame is scaled with the panel's edge. A view
 * watching a page another viewer sized letterboxes it.
 *
 * After the agent's browser closed itself when idle (2026-11-18), opening
 * the view brings the last page back live: the route reopens it on its own,
 * and this view keeps the last screenshot up meanwhile (and reports its size
 * from that box, so the page comes back at it). A click on a stale picture
 * asks for the same reopen.
 * CONTRACT FILE: props are final; workstream D implements.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, RotateCw, X } from 'lucide-react';
import { useT } from '../../i18n/useT';
import { useFiles } from '../../hooks/useFiles';
import { useBrowserLive, type NavAction } from '../../hooks/useBrowserLive';
import { createInputCoalescer } from '../../utils/inputCoalescer';
import { resolveAddress, displayAddress } from '../../utils/browserAddress';
import AnnotateOverlay from './AnnotateOverlay';
import type { Annotation, AnnotationBox, AnnotationDraft } from '../../utils/annotations';
import { frameRect } from './browserFrameRect';

/** A size change is sent once the view has held it this long — when a drag
 *  or a window resize settles, not on every step of it. */
export const SIZE_SETTLE_MS = 150;

/** Decoded last screenshots, so a view mounted again (Files and back to
 *  Browser) shows the picture on its first render instead of flashing text
 *  while the file is fetched again. Small: a few recent sessions. */
const fallbackCache = new Map<string, string>();
const FALLBACK_CACHE_MAX = 8;
/** Tests: start every case without pictures from earlier ones. */
export function clearFallbackCache() {
  fallbackCache.clear();
}
function rememberFallback(key: string, dataUrl: string) {
  fallbackCache.delete(key);
  fallbackCache.set(key, dataUrl);
  while (fallbackCache.size > FALLBACK_CACHE_MAX) {
    const oldest = fallbackCache.keys().next().value;
    if (oldest === undefined) break;
    fallbackCache.delete(oldest);
  }
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(Math.max(v, lo), hi);
}

export interface BrowserViewProps {
  projectId: string;
  /** Spec 155: the viewed session — the live view mirrors its own page. */
  sessionId?: string;
  /** Stream only while true (view visible + panel expanded). */
  active: boolean;
  /** Fallback when no live page: the session's last screenshot (workspace-relative path) + title. */
  fallback: { path: string; title?: string } | null;
  annotating: boolean;
  annotations: Annotation[];
  onAddAnnotation: (draft: AnnotationDraft) => void;
}

// CDP modifier bitmask convention (Input.dispatchMouseEvent / dispatchKeyEvent).
const MOD_ALT = 1;
const MOD_CTRL = 2;
const MOD_META = 4;
const MOD_SHIFT = 8;

function modifiersFor(e: {
  altKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
}): number {
  let m = 0;
  if (e.altKey) m |= MOD_ALT;
  if (e.ctrlKey) m |= MOD_CTRL;
  if (e.metaKey) m |= MOD_META;
  if (e.shiftKey) m |= MOD_SHIFT;
  return m;
}

function buttonName(button: number): 'left' | 'right' | 'middle' {
  if (button === 2) return 'right';
  if (button === 1) return 'middle';
  return 'left';
}

// Keys the page should receive rather than the browser chrome (scrolling,
// tabbing focus away, etc. would otherwise steal them).
const PREVENT_DEFAULT_KEYS = new Set([
  'Tab',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  ' ',
  'Backspace',
]);

function isPrintableKey(key: string): boolean {
  return key.length === 1;
}

// A second press within this window and distance is a double-click. Needed
// because PointerEvent.detail is 0 in browsers — MouseEvent's click count
// never made it onto pointer events.
const MULTI_CLICK_MS = 500;
const MULTI_CLICK_PX = 5;

// The page's cursor is painted onto the canvas verbatim, but only keywords —
// a url() cursor from the page must not reach our stylesheet.
const CURSOR_KEYWORDS = new Set([
  'default', 'pointer', 'text', 'move', 'grab', 'grabbing', 'crosshair', 'wait',
  'progress', 'help', 'not-allowed', 'no-drop', 'copy', 'alias', 'cell',
  'context-menu', 'vertical-text', 'all-scroll', 'zoom-in', 'zoom-out',
  'col-resize', 'row-resize', 'n-resize', 's-resize', 'e-resize', 'w-resize',
  'ne-resize', 'nw-resize', 'se-resize', 'sw-resize', 'ew-resize', 'ns-resize',
  'nesw-resize', 'nwse-resize', 'none',
]);
function safeCursor(cursor: string | undefined): string {
  return cursor && CURSOR_KEYWORDS.has(cursor) ? cursor : 'default';
}

function isBrowserAnnotation(a: Annotation): a is Extract<Annotation, { kind: 'browser' }> {
  return a.kind === 'browser';
}

export default function BrowserView({
  projectId,
  sessionId,
  active,
  fallback,
  annotating,
  annotations,
  onAddAnnotation,
}: BrowserViewProps) {
  const t = useT();
  const {
    status, frame, title, url, loading, canGoBack, canGoForward, cursor, owner, requestSize, send, nav,
  } = useBrowserLive(projectId, active, sessionId);
  // Frames are this view's own page (drawn filling it) unless the route
  // says another viewer sized the page.
  const owns = owner === true;
  // A page exists but nothing has been opened in it (about:blank): say so
  // rather than streaming a white rectangle labelled Live.
  const blankPage = status === 'open' && url === 'about:blank';
  const hasFrame = frame !== null;

  // ---- Fallback: session's last screenshot, via the files content route ----
  const { getFileContent } = useFiles();
  const fallbackKey = fallback ? `${projectId}\n${fallback.path}` : null;
  const [fallbackDataUrl, setFallbackDataUrl] = useState<string | null>(
    () => (fallbackKey && !hasFrame ? fallbackCache.get(fallbackKey) ?? null : null),
  );

  useEffect(() => {
    let cancelled = false;
    if (hasFrame || !fallback || !fallbackKey) {
      setFallbackDataUrl(null);
      return;
    }
    const cached = fallbackCache.get(fallbackKey);
    if (cached) setFallbackDataUrl(cached);
    (async () => {
      const content = await getFileContent(projectId, fallback.path);
      if (cancelled || !content || content.type !== 'image') return;
      const dataUrl = `data:${content.mime ?? 'image/png'};base64,${content.content}`;
      rememberFallback(fallbackKey, dataUrl);
      setFallbackDataUrl(dataUrl);
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, fallback, fallbackKey, hasFrame, getFileContent]);
  // The picture stands in for the page (no frame yet, no blank page open):
  // also while connecting, which is when the route may be reopening the
  // page the idle close shut — the picture stays until the live frame.
  const showFallback = !hasFrame && !blankPage && fallbackDataUrl !== null;

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const [canvasWidth, setCanvasWidth] = useState(0);
  const [canvasHeight, setCanvasHeight] = useState(0);
  const frameAreaRef = useRef<HTMLDivElement | null>(null);
  // Paint the pixel buffer at device resolution: frames arrive at the
  // display's density, so a 1:1 draw is pin-sharp on a Retina display.
  const dpr = typeof window !== 'undefined' && window.devicePixelRatio > 0 ? window.devicePixelRatio : 1;

  // Size the canvas to the container's width. ResizeObserver isn't available
  // in every test environment, so fall back to a window resize listener —
  // real layout changes (drawer resize) go through ResizeObserver when it
  // exists.
  useEffect(() => {
    // Only the live frame area is measured: its size is the size the page
    // is given, and any other box (the whole view while connecting or on a
    // blank page) would size the page wrong.
    const el = frameAreaRef.current;
    if (!el) {
      setCanvasWidth(0);
      setCanvasHeight(0);
      return;
    }
    const measure = () => {
      // Whole CSS px: the page is sized to exactly this, so frame pixels
      // and canvas pixels line up 1:1.
      setCanvasWidth(Math.floor(el.clientWidth));
      setCanvasHeight(Math.floor(el.clientHeight));
    };
    measure();
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(measure);
      ro.observe(el);
      return () => ro.disconnect();
    }
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
    // Re-run when the rendered branch changes: the live canvas area only
    // exists once a real page's frame has arrived (not on about:blank), or
    // as the box the last screenshot stands in, and it is that element we
    // must watch.
  }, [hasFrame, status, blankPage, showFallback]);

  // Tell the route this view's size: at once the first time (and the hook
  // re-sends it on every reconnect), then only once a change has settled.
  // Never because a frame disagrees — the route owns the page's size.
  const sizedOnceRef = useRef(false);
  // The picture's box is the same box the live page will fill, so its size
  // is the page's size too: a page the route reopens comes up at it.
  const liveArea = (hasFrame && !blankPage) || showFallback;
  useEffect(() => {
    if (!requestSize || !active || !liveArea || canvasWidth <= 0 || canvasHeight <= 0) return;
    if (!sizedOnceRef.current) {
      sizedOnceRef.current = true;
      requestSize(canvasWidth, canvasHeight, dpr);
      return;
    }
    const timer = setTimeout(() => requestSize(canvasWidth, canvasHeight, dpr), SIZE_SETTLE_MS);
    return () => clearTimeout(timer);
  }, [requestSize, active, liveArea, canvasWidth, canvasHeight, dpr]);

  // Paint each incoming frame onto the canvas, coalesced with rAF so a burst
  // of screencast frames doesn't queue up multiple paints.
  // The last decoded frame, kept so a resize (which resets the canvas bitmap)
  // can repaint at once instead of showing a blank until the next frame.
  const imageRef = useRef<HTMLImageElement | null>(null);
  const paint = useCallback(() => {
    const img = imageRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!img || !frame || !canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Filling the canvas when this view sized the page (1:1 once the size
    // has landed, scaled with the panel's edge mid-drag); letterboxed when
    // another viewer did. Never stretched.
    const rect = frameRect(frame.width, frame.height, canvas.width, canvas.height, owns);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, rect.x, rect.y, rect.w, rect.h);
  }, [frame, owns]);
  useEffect(() => {
    if (!frame) return;
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(paint);
    };
    img.src = frame.jpegDataUrl;
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [frame, paint]);
  // React re-applies width/height on resize, which wipes the bitmap: repaint
  // in the same commit, before the browser paints. A rAF repaint here let
  // WebKit composite the wiped canvas first — measured: 4 blank frames in a
  // 24-step panel drag (WKWebView's stale-layer family of quirks).
  // Keyed on the size only (the latest paint via a ref): a new frame paints
  // itself once decoded, below — repainting the previous image on every
  // frame's commit would double the draws.
  const paintRef = useRef(paint);
  paintRef.current = paint;
  useLayoutEffect(() => {
    if (!imageRef.current) return;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    paintRef.current();
  }, [canvasWidth, canvasHeight, dpr, owns]);

  // Canvas CSS-pixel size used for input scaling and annotation mapping.
  // getBoundingClientRect (mockable even where CSS layout isn't computed)
  // is preferred over clientWidth/Height, which fall back to the measured
  // container width when unset.
  const getCanvasSize = useCallback((): { width: number; height: number } => {
    const canvas = canvasRef.current;
    if (!canvas) return { width: canvasWidth || 1, height: canvasHeight || 1 };
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || canvas.clientWidth || canvasWidth || 1;
    const aspectHeight = frame ? (width * frame.height) / frame.width : 0;
    return {
      width,
      height: rect.height || canvas.clientHeight || canvasHeight || aspectHeight || 1,
    };
  }, [canvasWidth, canvasHeight, frame]);

  const pagePointFromClient = useCallback(
    (clientX: number, clientY: number): { x: number; y: number } | null => {
      const canvas = canvasRef.current;
      if (!canvas || !frame) return null;
      const rect = canvas.getBoundingClientRect();
      const { width, height } = getCanvasSize();
      const fit = frameRect(frame.width, frame.height, width, height, owns);
      return {
        x: clamp(((clientX - rect.left - fit.x) * frame.width) / fit.w, 0, frame.width),
        y: clamp(((clientY - rect.top - fit.y) * frame.height) / fit.h, 0, frame.height),
      };
    },
    [frame, getCanvasSize, owns],
  );

  // Pointer traffic is merged per animation frame (utils/inputCoalescer):
  // the first event goes out at once, the rest of the frame's moves keep the
  // newest position and its wheels add up. Clicks flush it first so they
  // land on the page as scrolled.
  const coalescer = useMemo(() => createInputCoalescer((m) => send(m)), [send]);
  useEffect(() => () => coalescer.cancel(), [coalescer]);

  // Input only reaches a page that is actually there. After the agent's
  // browser closes (idle eviction, stop) the last frame stays on screen as a
  // screenshot; clicks on it would otherwise be errors on a dead page.
  const live = status === 'open';

  // Keyboard goes through a hidden text host rather than the canvas: a
  // canvas cannot receive IME composition, so Chinese (or any composed)
  // input never reached the page. The host is focused on click; its key
  // events bubble to the frame area's handlers like the canvas's do, and
  // composed text arrives as one insert on compositionend.
  const textHostRef = useRef<HTMLTextAreaElement | null>(null);
  const composingRef = useRef(false);
  // Codes whose keydown we forwarded — a keyup for anything else (the Enter
  // that committed an IME composition, say) would reach the page orphaned.
  const downCodesRef = useRef<Set<string>>(new Set());
  const clickRef = useRef({ at: 0, x: 0, y: 0, count: 0 });

  const focusTextHost = useCallback(() => {
    const host = textHostRef.current;
    if (host) host.focus({ preventScroll: true });
    else canvasRef.current?.focus({ preventScroll: true });
  }, []);

  const clickCountFor = useCallback((e: React.PointerEvent<HTMLCanvasElement>): number => {
    if (e.detail > 0) return e.detail;
    const now = Date.now();
    const prev = clickRef.current;
    const near =
      now - prev.at <= MULTI_CLICK_MS &&
      Math.abs(e.clientX - prev.x) <= MULTI_CLICK_PX &&
      Math.abs(e.clientY - prev.y) <= MULTI_CLICK_PX;
    const count = near ? Math.min(prev.count + 1, 3) : 1;
    clickRef.current = { at: now, x: e.clientX, y: e.clientY, count };
    return count;
  }, []);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (annotating || !live) return;
      const p = pagePointFromClient(e.clientX, e.clientY);
      if (!p) return;
      coalescer.move(p.x, p.y, modifiersFor(e));
    },
    [annotating, pagePointFromClient, coalescer, live],
  );

  // A click on a stale picture (the last frame or screenshot, after the
  // agent's browser closed) asks the route to bring that page back; it does
  // nothing when there is no page to bring back.
  const reopenFromPicture = useCallback(
    (e: React.PointerEvent<HTMLElement>) => {
      if (live || annotating || e.button !== 0) return;
      nav('reopen');
    },
    [live, annotating, nav],
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!live) {
        reopenFromPicture(e);
        return;
      }
      if (annotating) return;
      focusTextHost();
      // Keep receiving this pointer even when it leaves the canvas, so the
      // matching release always reaches the page (a drag ending past the
      // edge used to leave the page's button stuck down).
      try {
        e.currentTarget.setPointerCapture?.(e.pointerId);
      } catch {
        // Not every environment supports capture; input still works.
      }
      const p = pagePointFromClient(e.clientX, e.clientY);
      if (!p) return;
      coalescer.flush();
      send({
        type: 'mouse',
        action: 'down',
        x: p.x,
        y: p.y,
        button: buttonName(e.button),
        clickCount: clickCountFor(e),
        modifiers: modifiersFor(e),
      });
    },
    [annotating, pagePointFromClient, send, live, coalescer, focusTextHost, clickCountFor, reopenFromPicture],
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!live) return;
      if (annotating) return;
      try {
        e.currentTarget.releasePointerCapture?.(e.pointerId);
      } catch {
        // ignore
      }
      const p = pagePointFromClient(e.clientX, e.clientY);
      if (!p) return;
      coalescer.flush();
      send({
        type: 'mouse',
        action: 'up',
        x: p.x,
        y: p.y,
        button: buttonName(e.button),
        clickCount: e.detail > 0 ? e.detail : clickRef.current.count || 1,
        modifiers: modifiersFor(e),
      });
    },
    [annotating, pagePointFromClient, send, live, coalescer],
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      if (!live) return;
      if (annotating) return;
      const p = pagePointFromClient(e.clientX, e.clientY);
      if (!p) return;
      coalescer.wheel(p.x, p.y, e.deltaX, e.deltaY, modifiersFor(e));
    },
    [annotating, pagePointFromClient, coalescer, live],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLElement>) => {
      if (!live) return;
      if (annotating) return;
      // Mid-composition keystrokes belong to the IME; the result arrives as
      // text on compositionend.
      if (composingRef.current || e.nativeEvent.isComposing || e.key === 'Process') return;
      if (PREVENT_DEFAULT_KEYS.has(e.key)) e.preventDefault();
      downCodesRef.current.add(e.code || e.key);
      send({
        type: 'key',
        action: 'down',
        key: e.key,
        code: e.code,
        ...(isPrintableKey(e.key) ? { text: e.key } : {}),
        modifiers: modifiersFor(e),
      });
    },
    [annotating, send, live],
  );

  const handleKeyUp = useCallback(
    (e: React.KeyboardEvent<HTMLElement>) => {
      // Whatever a plain keystroke left in the text host is already on the
      // page (sent as keydown text); keep the host empty so nothing piles up.
      const host = textHostRef.current;
      if (host && !composingRef.current) host.value = '';
      if (!live) return;
      if (annotating) return;
      if (composingRef.current || e.nativeEvent.isComposing) return;
      if (!downCodesRef.current.delete(e.code || e.key)) return;
      if (PREVENT_DEFAULT_KEYS.has(e.key)) e.preventDefault();
      send({ type: 'key', action: 'up', key: e.key, code: e.code, modifiers: modifiersFor(e) });
    },
    [annotating, send, live],
  );

  const handleCompositionStart = useCallback(() => {
    composingRef.current = true;
  }, []);

  const handleCompositionEnd = useCallback(
    (e: React.CompositionEvent<HTMLTextAreaElement>) => {
      composingRef.current = false;
      const host = e.currentTarget;
      const text = e.data || host.value;
      host.value = '';
      if (!live || annotating || !text) return;
      send({ type: 'text', text });
    },
    [annotating, send, live],
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLElement>) => {
      if (!live) return;
      if (annotating) return;
      const text = e.clipboardData.getData('text');
      if (!text) return;
      e.preventDefault();
      send({ type: 'text', text });
    },
    [annotating, send, live],
  );

  const handleContextMenu = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
  }, []);

  // ---- Toolbar: back / forward / reload-or-stop / address ----
  // `draft` is what the user is typing; null shows the page's URL.
  const [draft, setDraft] = useState<string | null>(null);
  const addressRef = useRef<HTMLInputElement | null>(null);
  const go = useCallback(
    (action: NavAction, target?: string) => {
      if (target === undefined) nav(action);
      else nav(action, target);
    },
    [nav],
  );
  const handleAddressKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const resolved = resolveAddress(e.currentTarget.value);
        if (!resolved) return;
        go('goto', resolved);
        setDraft(null);
        e.currentTarget.blur();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        setDraft(null);
        e.currentTarget.blur();
      }
    },
    [go],
  );

  // The header is part of what the Browser view IS, so it is there in every
  // state — not only while a page happens to be live (2026-09-05, from the
  // installer: "the browser doesnt have header"). Every project on a fresh
  // daemon reports no_browser until the agent browses something, which is
  // exactly when a user opens the panel wanting to look something up. Back /
  // forward / reload need a page and disable themselves without one; the
  // address field always works, because typing an address is how the user
  // opens the first page (the route creates it — see browser_live.py).
  const toolbar = (
    <div
      role="toolbar"
      aria-label={t('panel.browser.label')}
      className="flex shrink-0 items-center gap-0.5 border-b border-border px-1.5 py-1"
    >
      <button
        type="button"
        aria-label={t('panel.browser.back')}
        title={t('panel.browser.back')}
        disabled={!live || !canGoBack}
        onClick={() => go('back')}
        className="rounded p-1 text-secondary hover:bg-nav-hover hover:text-primary disabled:opacity-40 disabled:hover:bg-transparent"
      >
        <ArrowLeft size={14} aria-hidden="true" />
      </button>
      <button
        type="button"
        aria-label={t('panel.browser.forward')}
        title={t('panel.browser.forward')}
        disabled={!live || !canGoForward}
        onClick={() => go('forward')}
        className="rounded p-1 text-secondary hover:bg-nav-hover hover:text-primary disabled:opacity-40 disabled:hover:bg-transparent"
      >
        <ArrowRight size={14} aria-hidden="true" />
      </button>
      <button
        type="button"
        aria-label={loading ? t('panel.browser.stop') : t('panel.browser.reload')}
        title={loading ? t('panel.browser.stop') : t('panel.browser.reload')}
        aria-busy={loading || undefined}
        disabled={!live}
        onClick={() => go(loading ? 'stop' : 'reload')}
        className="rounded p-1 text-secondary hover:bg-nav-hover hover:text-primary disabled:opacity-40 disabled:hover:bg-transparent"
      >
        {loading ? <X size={14} aria-hidden="true" /> : <RotateCw size={14} aria-hidden="true" />}
      </button>
      <input
        ref={addressRef}
        type="text"
        aria-label={t('panel.browser.address')}
        placeholder={t('panel.browser.addressPlaceholder')}
        value={draft ?? displayAddress(url)}
        onChange={(e) => setDraft(e.target.value)}
        onFocus={(e) => e.currentTarget.select()}
        onBlur={() => setDraft(null)}
        onKeyDown={handleAddressKeyDown}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        className="ml-1 h-6 min-w-0 flex-1 rounded-md border border-border bg-card px-2 text-caption text-primary outline-none placeholder:text-muted focus:border-accent"
      />
    </div>
  );

  // Existing 'browser' annotations, mapped from page CSS px to the canvas's
  // current on-screen (overlay) px so pins line up with what's drawn now.
  const overlayBoxes = useMemo(() => {
    if (!frame) return [];
    const { width, height } = getCanvasSize();
    const fit = frameRect(frame.width, frame.height, width, height, owns);
    const sx = fit.w / frame.width;
    const sy = fit.h / frame.height;
    return annotations.filter(isBrowserAnnotation).map((a) => ({
      n: a.n,
      box: { x: a.box.x * sx + fit.x, y: a.box.y * sy + fit.y, w: a.box.w * sx, h: a.box.h * sy },
    }));
  }, [annotations, frame, getCanvasSize, owns]);

  const handleOverlayAdd = useCallback(
    (box: AnnotationBox, note: string) => {
      if (!frame) return;
      const { width, height } = getCanvasSize();
      const fit = frameRect(frame.width, frame.height, width, height, owns);
      const sx = frame.width / fit.w;
      const sy = frame.height / fit.h;
      const pageBox: AnnotationBox = {
        x: (box.x - fit.x) * sx, y: (box.y - fit.y) * sy, w: box.w * sx, h: box.h * sy,
      };
      const canvas = canvasRef.current;
      const imageDataUrl = canvas ? canvas.toDataURL('image/png') : frame.jpegDataUrl;
      onAddAnnotation({ kind: 'browser', pageTitle: title, box: pageBox, note, imageDataUrl });
    },
    [frame, getCanvasSize, owns, title, onAddAnnotation],
  );

  if (hasFrame && !blankPage) {
    return (
      <div ref={wrapperRef} className="relative flex h-full w-full flex-col">
        {toolbar}
        <div
          ref={frameAreaRef}
          // Spec 154: App's ⌘0 (go home) skips keys aimed at the agent's page.
          data-browser-live="true"
          // The viewport is a Page card inset in the panel (index.css
          // .browser-sheet). The canvas is measured from this box's own
          // client size, and that is the size the page is given.
          className="browser-sheet relative flex-1 min-h-0 overflow-hidden"
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          onPaste={handlePaste}
        >
          <canvas
            ref={canvasRef}
            role="img"
            aria-label={title}
            tabIndex={0}
            width={canvasWidth ? Math.round(canvasWidth * dpr) : undefined}
            height={canvasHeight ? Math.round(canvasHeight * dpr) : undefined}
            style={{
              width: canvasWidth ? `${canvasWidth}px` : '100%',
              height: canvasHeight ? `${canvasHeight}px` : '100%',
              display: 'block',
              // Not live: a click asks for the page back (see above).
              cursor: live ? safeCursor(cursor) : 'default',
            }}
            className="outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
            onPointerMove={handlePointerMove}
            onPointerDown={handlePointerDown}
            onPointerUp={handlePointerUp}
            onWheel={handleWheel}
            onContextMenu={handleContextMenu}
          />
          <textarea
            ref={textHostRef}
            data-testid="browser-text-host"
            aria-hidden="true"
            tabIndex={-1}
            autoComplete="off"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            onCompositionStart={handleCompositionStart}
            onCompositionEnd={handleCompositionEnd}
            className="pointer-events-none absolute left-0 top-0 h-px w-px resize-none opacity-0"
          />
          {annotating && (
            <AnnotateOverlay active boxes={overlayBoxes} onAdd={handleOverlayAdd} />
          )}
        </div>
        <div className="flex min-w-0 items-center gap-1.5 px-2 py-1 text-caption text-secondary">
          <span
            aria-hidden="true"
            className="inline-block h-1.5 w-1.5 shrink-0 rounded-full"
            style={{ backgroundColor: live ? '#22C55E' : '#9CA3AF' }}
          />
          <span className="shrink-0">{live ? t('panel.browser.live') : t('panel.browser.lastScreenshot')}</span>
          {title && <span className="truncate">{title}</span>}
        </div>
      </div>
    );
  }

  if (blankPage) {
    // A page with nothing on it still gets the toolbar: typing an address
    // here is how the user opens something themselves.
    return (
      <div ref={wrapperRef} className="flex h-full w-full flex-col">
        {toolbar}
        <div className="p-3 text-caption text-secondary">{t('panel.browser.empty')}</div>
      </div>
    );
  }

  if (showFallback) {
    return (
      <div ref={wrapperRef} className="flex h-full w-full flex-col">
        {toolbar}
        <div
          ref={frameAreaRef}
          className="browser-sheet flex flex-1 items-center justify-center overflow-auto bg-sidebar p-2"
          onPointerDown={reopenFromPicture}
        >
          <img
            src={fallbackDataUrl ?? undefined}
            alt={fallback?.title ?? ''}
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
          />
        </div>
        <div className="truncate px-2 py-1 text-caption text-secondary">
          {t('panel.browser.lastScreenshot')}
          {fallback?.title ? ` · ${fallback.title}` : ''}
        </div>
      </div>
    );
  }

  if (status === 'connecting') {
    return (
      <div ref={wrapperRef} className="flex h-full w-full flex-col">
        {toolbar}
        <div className="p-3 text-caption text-secondary">{t('panel.browser.connecting')}</div>
      </div>
    );
  }

  return (
    <div ref={wrapperRef} className="flex h-full w-full flex-col">
      {toolbar}
      <div className="p-3 text-caption text-secondary">{t('panel.browser.empty')}</div>
    </div>
  );
}
