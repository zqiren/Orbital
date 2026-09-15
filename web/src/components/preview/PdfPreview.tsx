// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — PDF engine: pdf.js with a thin Orbital viewer.
 *
 * The LEGACY build, deliberately: the modern 6.x build calls
 * `Map.prototype.getOrInsertComputed` and `Math.sumPrecise` unguarded, and the
 * desktop app's WKWebView (system WebKit 18.x on macOS 15) has neither. The
 * legacy build polyfills them.
 *
 * Virtualized: only pages within RENDER_MARGIN of the viewport are mounted.
 * Each draws a DPR-scaled canvas (capped at MAX_CANVAS_PIXELS, WebKit's canvas
 * budget) plus a text layer, so selection — and the panel's Quote pill — work.
 * Fit-width is the default zoom. Page x / y and zoom are published as
 * PdfNavState for DocumentNav to render. The worker, CMaps, standard fonts,
 * wasm decoders and ICC profile are all served from the app bundle (see the
 * `pdfjsAssets` plugin in vite.config.ts), never a CDN.
 */
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from 'react';
import * as pdfjs from 'pdfjs-dist/legacy/build/pdf.mjs';
import PdfWorker from 'pdfjs-dist/legacy/build/pdf.worker.min.mjs?worker';
import { useT } from '../../i18n/useT';
import { pdfDocumentOptions } from './pdfDocumentOptions';
import type { EngineProps } from './types';
import { clampZoom, zoomIn, zoomOut, type ZoomState } from './zoom';
import './pdfTextLayer.css';

type PdfDocument = Awaited<ReturnType<typeof pdfjs.getDocument>['promise']>;
type PdfPage = Awaited<ReturnType<PdfDocument['getPage']>>;
type RenderTask = ReturnType<PdfPage['render']>;

/** 100% zoom = the PDF's physical size in CSS pixels. */
const CSS_UNITS = pdfjs.PixelsPerInch.PDF_TO_CSS_UNITS;
const PAD = 12;
const GAP = 12;
const RENDER_MARGIN = 2;
const MAX_CANVAS_PIXELS = 16_777_216;

/** Page size in PDF points at scale 1. */
interface PageSize {
  w: number;
  h: number;
}

function assetBase(): string {
  return new URL(`${import.meta.env.BASE_URL}pdfjs-${pdfjs.version}/`, window.location.href).href;
}

export default function PdfPreview({ bytes, fileName, onNav, onError }: EngineProps) {
  const t = useT();
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const [doc, setDoc] = useState<PdfDocument | null>(null);
  const [password, setPassword] = useState<{
    submit: (value: string) => void;
    incorrect: boolean;
    attempt: number;
  } | null>(null);
  const [sizes, setSizes] = useState<PageSize[]>([]);
  const [zoom, setZoom] = useState<ZoomState>({ mode: 'fit' });
  const [view, setView] = useState({ top: 0, width: 0, height: 0 });
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const scrollRatioRef = useRef(0);
  const frameRef = useRef(0);

  // ── load ────────────────────────────────────────────────────────────────
  useEffect(() => {
    const port = new PdfWorker();
    // pdf.js's generated class typings infer `port` as `null`; the runtime (and
    // its own PDFWorkerParameters type) take a Worker.
    const worker = new pdfjs.PDFWorker({ port } as unknown as ConstructorParameters<typeof pdfjs.PDFWorker>[0]);
    const task = pdfjs.getDocument({
      // pdf.js transfers the buffer to its worker; keep the host's copy intact
      // for Download.
      ...pdfDocumentOptions(new Uint8Array(bytes.slice(0)), assetBase()),
      worker,
    });
    let cancelled = false;
    let attempt = 0;
    task.onPassword = (submit: (value: string) => void, reason: number) => {
      if (cancelled) return;
      attempt += 1;
      setPassword({
        submit,
        incorrect: reason === pdfjs.PasswordResponses.INCORRECT_PASSWORD,
        attempt,
      });
    };
    task.promise
      .then(async (pdf) => {
        const first = await pdf.getPage(1);
        if (cancelled) return;
        const vp = first.getViewport({ scale: 1 });
        setPassword(null);
        setSizes(Array.from({ length: pdf.numPages }, () => ({ w: vp.width, h: vp.height })));
        setDoc(pdf);
      })
      .catch(() => {
        if (!cancelled) onErrorRef.current('unreadable');
      });
    return () => {
      cancelled = true;
      void task.destroy();
      worker.destroy();
      // pdf.js does not terminate a port it was handed.
      port.terminate();
    };
  }, [bytes]);

  // ── viewport tracking ───────────────────────────────────────────────────
  useLayoutEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const measure = () => setView({ top: el.scrollTop, width: el.clientWidth, height: el.clientHeight });
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => () => cancelAnimationFrame(frameRef.current), []);

  const handleScroll = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    scrollRatioRef.current = el.scrollHeight > 0 ? el.scrollTop / el.scrollHeight : 0;
    cancelAnimationFrame(frameRef.current);
    frameRef.current = requestAnimationFrame(() =>
      setView({ top: el.scrollTop, width: el.clientWidth, height: el.clientHeight }),
    );
  }, []);

  // ── layout ──────────────────────────────────────────────────────────────
  const widestPage = useMemo(() => sizes.reduce((max, s) => Math.max(max, s.w), 0), [sizes]);
  const fitScale =
    view.width > 0 && widestPage > 0
      ? clampZoom((view.width - 2 * PAD) / (widestPage * CSS_UNITS))
      : 1;
  const scale = zoom.mode === 'fit' ? fitScale : zoom.scale;

  const layout = useMemo(() => {
    const px = scale * CSS_UNITS;
    const tops: number[] = new Array(sizes.length);
    let y = PAD;
    let widest = 0;
    for (let i = 0; i < sizes.length; i++) {
      tops[i] = y;
      y += sizes[i].h * px + GAP;
      widest = Math.max(widest, sizes[i].w * px);
    }
    return { px, tops, height: sizes.length ? y - GAP + PAD : 0, width: widest + 2 * PAD };
  }, [sizes, scale]);

  // Keep the reading position when the zoom (or the fit width) changes.
  useLayoutEffect(() => {
    const el = scrollerRef.current;
    if (el) el.scrollTop = scrollRatioRef.current * el.scrollHeight;
  }, [layout.px]);

  const pageAt = (y: number) => {
    const { tops } = layout;
    let lo = 0;
    let hi = tops.length - 1;
    let found = 0;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (tops[mid] <= y) {
        found = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return found;
  };
  const from = Math.max(0, pageAt(view.top) - RENDER_MARGIN);
  const to = Math.min(sizes.length - 1, pageAt(view.top + view.height) + RENDER_MARGIN);
  const current = sizes.length ? pageAt(view.top + view.height * 0.35) + 1 : 0;
  const contentWidth = Math.max(layout.width, view.width);

  const reportSize = useCallback((index: number, size: PageSize) => {
    setSizes((prev) => {
      const old = prev[index];
      if (!old || (Math.abs(old.w - size.w) < 0.5 && Math.abs(old.h - size.h) < 0.5)) return prev;
      const next = prev.slice();
      next[index] = size;
      return next;
    });
  }, []);

  const goTo = (page: number) => {
    const el = scrollerRef.current;
    if (!el || sizes.length === 0) return;
    const index = Math.min(sizes.length, Math.max(1, page)) - 1;
    el.scrollTo({ top: layout.tops[index] - PAD });
  };

  // ── navigation state (rendered by DocumentNav, wherever the host puts it) ──
  // Actions are stable and reach the latest layout/scale through a ref, so
  // publishing re-runs only when the visible values change.
  const latestRef = useRef({ goTo, scale });
  useEffect(() => {
    latestRef.current = { goTo, scale };
  });
  const actions = useMemo(
    () => ({
      goTo: (page: number) => latestRef.current.goTo(page),
      zoomIn: () => setZoom(zoomIn(latestRef.current.scale)),
      zoomOut: () => setZoom(zoomOut(latestRef.current.scale)),
      fitWidth: () => setZoom({ mode: 'fit' }),
    }),
    [],
  );
  const pageCount = doc?.numPages ?? 0;
  const zoomPercent = Math.round(scale * 100);
  const fit = zoom.mode === 'fit';
  useEffect(() => {
    if (pageCount > 0) onNav({ kind: 'pdf', page: current, pageCount, zoomPercent, fit, ...actions });
  }, [onNav, pageCount, current, zoomPercent, fit, actions]);
  useEffect(() => () => onNav(null), [onNav]);

  return (
    <div
      ref={scrollerRef}
      onScroll={handleScroll}
      aria-label={fileName}
      className="absolute inset-0 overflow-auto bg-sidebar"
      data-testid="pdf-scroller"
    >
      {password ? (
        <PasswordPrompt key={password.attempt} incorrect={password.incorrect} onSubmit={password.submit} />
      ) : !doc ? (
        <p className="p-4 text-xs text-secondary">{t('filePreview.doc.loading')}</p>
      ) : (
        <div className="relative" style={{ height: layout.height, width: contentWidth }}>
          {view.width > 0 &&
            to >= from &&
            Array.from({ length: to - from + 1 }, (_, k) => from + k).map((index) => (
              <PdfPageView
                key={index}
                doc={doc}
                index={index}
                px={layout.px}
                top={layout.tops[index]}
                left={Math.max(PAD, (contentWidth - sizes[index].w * layout.px) / 2)}
                size={sizes[index]}
                label={t('filePreview.doc.pageLabel', { n: index + 1 })}
                onSize={reportSize}
              />
            ))}
        </div>
      )}
    </div>
  );
}

function PdfPageView({
  doc,
  index,
  px,
  top,
  left,
  size,
  label,
  onSize,
}: {
  doc: PdfDocument;
  index: number;
  px: number;
  top: number;
  left: number;
  size: PageSize;
  label: string;
  onSize: (index: number, size: PageSize) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const textRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const text = textRef.current;
    if (!canvas || !text) return;
    let cancelled = false;
    let renderTask: RenderTask | null = null;
    let textLayer: pdfjs.TextLayer | null = null;
    void (async () => {
      try {
        const page = await doc.getPage(index + 1);
        if (cancelled) return;
        const natural = page.getViewport({ scale: 1 });
        onSize(index, { w: natural.width, h: natural.height });
        const viewport = page.getViewport({ scale: px });
        const outputScale = Math.min(
          window.devicePixelRatio || 1,
          Math.sqrt(MAX_CANVAS_PIXELS / Math.max(1, viewport.width * viewport.height)),
        );
        canvas.width = Math.max(1, Math.floor(viewport.width * outputScale));
        canvas.height = Math.max(1, Math.floor(viewport.height * outputScale));
        renderTask = page.render({
          canvas,
          viewport,
          transform: outputScale === 1 ? undefined : [outputScale, 0, 0, outputScale, 0, 0],
        });
        await renderTask.promise;
        if (cancelled) return;
        text.replaceChildren();
        textLayer = new pdfjs.TextLayer({
          textContentSource: page.streamTextContent({ includeMarkedContent: true, disableNormalization: true }),
          container: text,
          viewport,
        });
        await textLayer.render();
      } catch (err) {
        if (!cancelled && !(err instanceof pdfjs.RenderingCancelledException)) {
          console.warn('[PdfPreview] page render failed', err);
        }
      }
    })();
    return () => {
      cancelled = true;
      renderTask?.cancel();
      textLayer?.cancel();
    };
  }, [doc, index, px, onSize]);

  // Release the bitmap as soon as the page scrolls out of the render window;
  // WebKit reclaims canvas memory lazily otherwise.
  useEffect(
    () => () => {
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = 0;
        canvas.height = 0;
      }
    },
    [],
  );

  const style = {
    top,
    left,
    width: size.w * px,
    height: size.h * px,
    '--scale-factor': px,
    '--user-unit': 1,
    '--total-scale-factor': px,
    '--scale-round-x': '1px',
    '--scale-round-y': '1px',
  } as CSSProperties;

  return (
    <div className="absolute bg-white shadow-sm" style={style} role="group" aria-label={label} data-page={index + 1}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" aria-hidden />
      <div ref={textRef} className="textLayer" />
    </div>
  );
}

function PasswordPrompt({
  incorrect,
  onSubmit,
}: {
  incorrect: boolean;
  onSubmit: (value: string) => void;
}) {
  const t = useT();
  const [value, setValue] = useState('');
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(value);
      }}
      className="mx-auto mt-10 max-w-xs flex flex-col gap-2 rounded-lg border border-border bg-card p-4"
    >
      <p className="text-sm text-primary">{t('filePreview.doc.password.prompt')}</p>
      {incorrect && <p className="text-xs text-error">{t('filePreview.doc.password.incorrect')}</p>}
      <input
        type="password"
        autoFocus
        aria-label={t('filePreview.doc.password.label')}
        placeholder={t('filePreview.doc.password.label')}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="h-8 rounded-md border border-border bg-transparent px-2 text-sm text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
      />
      <button
        type="submit"
        className="bg-accent text-white text-sm font-medium rounded-lg px-4 py-1.5 hover:bg-accent/90 transition-colors"
      >
        {t('filePreview.doc.password.submit')}
      </button>
    </form>
  );
}
