// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.3/§5.6 — the Browser view: the agent's page, live and
 * interactive (canvas painted from screencast frames, input forwarded), with
 * the session's last screenshot as fallback when no browser is open.
 * CONTRACT FILE: props are final; workstream D implements.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useT } from '../../i18n/useT';
import { useFiles } from '../../hooks/useFiles';
import { useBrowserLive } from '../../hooks/useBrowserLive';
import AnnotateOverlay from './AnnotateOverlay';
import type { Annotation, AnnotationBox, AnnotationDraft } from '../../utils/annotations';

/** Largest rect with the source aspect that fits in the box, centred. */
function fitRect(srcW: number, srcH: number, boxW: number, boxH: number) {
  if (srcW <= 0 || srcH <= 0 || boxW <= 0 || boxH <= 0) return { x: 0, y: 0, w: boxW || 1, h: boxH || 1 };
  const scale = Math.min(boxW / srcW, boxH / srcH);
  const w = srcW * scale;
  const h = srcH * scale;
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}
function clamp(v: number, lo: number, hi: number) {
  return Math.min(Math.max(v, lo), hi);
}

export interface BrowserViewProps {
  projectId: string;
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

function isBrowserAnnotation(a: Annotation): a is Extract<Annotation, { kind: 'browser' }> {
  return a.kind === 'browser';
}

export default function BrowserView({
  projectId,
  active,
  fallback,
  annotating,
  annotations,
  onAddAnnotation,
}: BrowserViewProps) {
  const t = useT();
  const { status, frame, title, send } = useBrowserLive(projectId, active);
  const hasFrame = frame !== null;

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const [canvasWidth, setCanvasWidth] = useState(0);
  const [canvasHeight, setCanvasHeight] = useState(0);
  const frameAreaRef = useRef<HTMLDivElement | null>(null);
  // Paint the pixel buffer at device resolution: a 1280px page scaled into a
  // ~600px column is unreadable at 1× on a Retina display.
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // Size the canvas to the container's width. ResizeObserver isn't available
  // in every test environment, so fall back to a window resize listener —
  // real layout changes (drawer resize) go through ResizeObserver when it
  // exists.
  useEffect(() => {
    const el = frameAreaRef.current ?? wrapperRef.current;
    if (!el) return;
    const measure = () => {
      setCanvasWidth(el.clientWidth);
      setCanvasHeight(el.clientHeight);
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
    // exists once a frame has arrived, and it is that element we must watch.
  }, [hasFrame, status]);

  // Fit the page to the panel (spec 078 D9 amendment): tell the route the
  // canvas's CSS size whenever the stream is open and the size is known, and
  // again when a frame arrives in a different size (a fresh page opened at the
  // browser's default viewport). Debounced; never resent for the same size.
  const lastViewportRef = useRef<string>('');
  useEffect(() => {
    if (status !== 'open' || canvasWidth < 200 || canvasHeight < 200) return;
    const w = Math.round(canvasWidth);
    const h = Math.round(canvasHeight);
    const key = `${w}x${h}@${dpr}`;
    const frameMatches = frame && Math.abs(frame.width - w) <= 2 && Math.abs(frame.height - h) <= 2;
    if (lastViewportRef.current === key && (frameMatches || !frame)) return;
    const t = setTimeout(() => {
      lastViewportRef.current = key;
      send({ type: 'viewport', width: w, height: h, dpr });
    }, 150);
    return () => clearTimeout(t);
  }, [status, canvasWidth, canvasHeight, dpr, frame, send]);

  // Paint each incoming frame onto the canvas, coalesced with rAF so a burst
  // of screencast frames doesn't queue up multiple paints.
  useEffect(() => {
    if (!frame) return;
    const img = new Image();
    img.onload = () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (!canvas || !ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Contain-fit: once the page has taken the panel's size the frame
        // fills the canvas exactly; in the moment before, it is letterboxed
        // rather than stretched.
        const fit = fitRect(frame.width, frame.height, canvas.width, canvas.height);
        ctx.drawImage(img, fit.x, fit.y, fit.w, fit.h);
      });
    };
    img.src = frame.jpegDataUrl;
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [frame]);

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
      const fit = fitRect(frame.width, frame.height, width, height);
      return {
        x: clamp(((clientX - rect.left - fit.x) * frame.width) / fit.w, 0, frame.width),
        y: clamp(((clientY - rect.top - fit.y) * frame.height) / fit.h, 0, frame.height),
      };
    },
    [frame, getCanvasSize],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (annotating) return;
      const p = pagePointFromClient(e.clientX, e.clientY);
      if (!p) return;
      send({ type: 'mouse', action: 'move', x: p.x, y: p.y, modifiers: modifiersFor(e) });
    },
    [annotating, pagePointFromClient, send],
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (annotating) return;
      canvasRef.current?.focus({ preventScroll: true });
      const p = pagePointFromClient(e.clientX, e.clientY);
      if (!p) return;
      send({
        type: 'mouse',
        action: 'down',
        x: p.x,
        y: p.y,
        button: buttonName(e.button),
        clickCount: e.detail || 1,
        modifiers: modifiersFor(e),
      });
    },
    [annotating, pagePointFromClient, send],
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (annotating) return;
      const p = pagePointFromClient(e.clientX, e.clientY);
      if (!p) return;
      send({
        type: 'mouse',
        action: 'up',
        x: p.x,
        y: p.y,
        button: buttonName(e.button),
        clickCount: e.detail || 1,
        modifiers: modifiersFor(e),
      });
    },
    [annotating, pagePointFromClient, send],
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      if (annotating) return;
      const p = pagePointFromClient(e.clientX, e.clientY);
      if (!p) return;
      send({
        type: 'mouse',
        action: 'wheel',
        x: p.x,
        y: p.y,
        deltaX: e.deltaX,
        deltaY: e.deltaY,
        modifiers: modifiersFor(e),
      });
    },
    [annotating, pagePointFromClient, send],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLCanvasElement>) => {
      if (annotating) return;
      if (PREVENT_DEFAULT_KEYS.has(e.key)) e.preventDefault();
      send({
        type: 'key',
        action: 'down',
        key: e.key,
        code: e.code,
        ...(isPrintableKey(e.key) ? { text: e.key } : {}),
        modifiers: modifiersFor(e),
      });
    },
    [annotating, send],
  );

  const handleKeyUp = useCallback(
    (e: React.KeyboardEvent<HTMLCanvasElement>) => {
      if (annotating) return;
      if (PREVENT_DEFAULT_KEYS.has(e.key)) e.preventDefault();
      send({ type: 'key', action: 'up', key: e.key, code: e.code, modifiers: modifiersFor(e) });
    },
    [annotating, send],
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLCanvasElement>) => {
      if (annotating) return;
      const text = e.clipboardData.getData('text');
      if (!text) return;
      e.preventDefault();
      send({ type: 'text', text });
    },
    [annotating, send],
  );

  const handleContextMenu = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
  }, []);

  // Existing 'browser' annotations, mapped from page CSS px to the canvas's
  // current on-screen (overlay) px so pins line up with what's drawn now.
  const overlayBoxes = useMemo(() => {
    if (!frame) return [];
    const { width, height } = getCanvasSize();
    const fit = fitRect(frame.width, frame.height, width, height);
    const sx = fit.w / frame.width;
    const sy = fit.h / frame.height;
    return annotations.filter(isBrowserAnnotation).map((a) => ({
      n: a.n,
      box: { x: a.box.x * sx + fit.x, y: a.box.y * sy + fit.y, w: a.box.w * sx, h: a.box.h * sy },
    }));
  }, [annotations, frame, getCanvasSize]);

  const handleOverlayAdd = useCallback(
    (box: AnnotationBox, note: string) => {
      if (!frame) return;
      const { width, height } = getCanvasSize();
      const fit = fitRect(frame.width, frame.height, width, height);
      const sx = frame.width / fit.w;
      const sy = frame.height / fit.h;
      const pageBox: AnnotationBox = {
        x: (box.x - fit.x) * sx, y: (box.y - fit.y) * sy, w: box.w * sx, h: box.h * sy,
      };
      const canvas = canvasRef.current;
      const imageDataUrl = canvas ? canvas.toDataURL('image/png') : frame.jpegDataUrl;
      onAddAnnotation({ kind: 'browser', pageTitle: title, box: pageBox, note, imageDataUrl });
    },
    [frame, getCanvasSize, title, onAddAnnotation],
  );

  // ---- Fallback: session's last screenshot, via the files content route ----
  const { getFileContent } = useFiles();
  const [fallbackDataUrl, setFallbackDataUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (hasFrame || !fallback) {
      setFallbackDataUrl(null);
      return;
    }
    (async () => {
      const content = await getFileContent(projectId, fallback.path);
      if (cancelled || !content || content.type !== 'image') return;
      setFallbackDataUrl(`data:${content.mime ?? 'image/png'};base64,${content.content}`);
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, fallback, hasFrame, getFileContent]);

  if (hasFrame) {
    return (
      <div ref={wrapperRef} className="relative flex h-full w-full flex-col">
        <div ref={frameAreaRef} className="relative flex-1 min-h-0 overflow-hidden">
          <canvas
            ref={canvasRef}
            role="img"
            aria-label={title}
            tabIndex={0}
            width={canvasWidth ? Math.round(canvasWidth * dpr) : undefined}
            height={canvasHeight ? Math.round(canvasHeight * dpr) : undefined}
            style={{ width: '100%', height: '100%', display: 'block' }}
            className="cursor-default outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
            onPointerMove={handlePointerMove}
            onPointerDown={handlePointerDown}
            onPointerUp={handlePointerUp}
            onWheel={handleWheel}
            onKeyDown={handleKeyDown}
            onKeyUp={handleKeyUp}
            onPaste={handlePaste}
            onContextMenu={handleContextMenu}
          />
          {annotating && (
            <AnnotateOverlay active boxes={overlayBoxes} onAdd={handleOverlayAdd} />
          )}
        </div>
        <div className="flex min-w-0 items-center gap-1.5 px-2 py-1 text-xs text-secondary">
          <span
            aria-hidden="true"
            className="inline-block h-1.5 w-1.5 shrink-0 rounded-full"
            style={{ backgroundColor: '#22C55E' }}
          />
          <span className="shrink-0">{t('panel.browser.live')}</span>
          {title && <span className="truncate">{title}</span>}
        </div>
      </div>
    );
  }

  if (status === 'connecting') {
    return (
      <div ref={wrapperRef} className="p-3 text-xs text-secondary">
        {t('panel.browser.connecting')}
      </div>
    );
  }

  if (fallbackDataUrl) {
    return (
      <div ref={wrapperRef} className="flex h-full w-full flex-col">
        <div className="flex flex-1 items-center justify-center overflow-auto bg-sidebar p-2">
          <img
            src={fallbackDataUrl}
            alt={fallback?.title ?? ''}
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
          />
        </div>
        <div className="truncate px-2 py-1 text-xs text-secondary">
          {t('panel.browser.lastScreenshot')}
          {fallback?.title ? ` · ${fallback.title}` : ''}
        </div>
      </div>
    );
  }

  return (
    <div ref={wrapperRef} className="p-3 text-xs text-secondary">
      {t('panel.browser.empty')}
    </div>
  );
}
