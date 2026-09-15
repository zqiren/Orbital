// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — the Word engines' shared display: sanitized markup in an
 * `<iframe sandbox srcdoc>` with no allow-scripts (opaque origin, a CSP with no
 * remote loads), plus zoom navigation.
 *
 * The frame cannot be measured or scripted from here, so zoom is a CSS `zoom`
 * baked into the srcdoc. Paged output (DOCX) fits its declared page width;
 * reflowing output (DOC, `pageWidth` null) starts at 100%, which is also what
 * "fit" returns to.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useT } from '../../i18n/useT';
import { buildDocumentSrcdoc } from './documentSanitize';
import type { DocumentNavState } from './types';
import { clampZoom, useZoomNav, type ZoomState } from './zoom';

/** Horizontal wrapper padding + scrollbar allowance inside the frame. */
const FRAME_CHROME_PX = 48;
const RESIZE_DEBOUNCE_MS = 150;

export default function SandboxedDocumentFrame({
  title,
  html,
  pageWidth,
  frameCss,
  onNav,
}: {
  title: string;
  /** Sanitized markup, or null while the engine is still parsing. */
  html: string | null;
  /** Declared page width in CSS px (the fit-width target); null for reflowing output. */
  pageWidth: number | null;
  /** Engine chrome for the frame document: a trusted constant, never document data. */
  frameCss: string;
  onNav: (nav: DocumentNavState | null) => void;
}) {
  const t = useT();
  const [zoom, setZoom] = useState<ZoomState>({ mode: 'fit' });
  const [frameWidth, setFrameWidth] = useState(0);
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    setFrameWidth(el.clientWidth);
    if (typeof ResizeObserver === 'undefined') return;
    let timer = 0;
    const observer = new ResizeObserver(() => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => setFrameWidth(el.clientWidth), RESIZE_DEBOUNCE_MS);
    });
    observer.observe(el);
    return () => {
      window.clearTimeout(timer);
      observer.disconnect();
    };
  }, []);

  const fitScale =
    pageWidth && frameWidth > 0 ? clampZoom((frameWidth - FRAME_CHROME_PX) / pageWidth) : 1;
  // Rounded so a sub-pixel resize does not rebuild (and reload) the frame.
  const scale = Math.round((zoom.mode === 'fit' ? fitScale : zoom.scale) * 100) / 100;
  useZoomNav(onNav, html !== null, scale, zoom, setZoom);

  const srcDoc = useMemo(
    () => (html === null ? null : buildDocumentSrcdoc(html, scale, frameCss)),
    [html, scale, frameCss],
  );

  return (
    <div ref={hostRef} className="absolute inset-0 bg-sidebar">
      {srcDoc ? (
        <iframe
          title={title}
          srcDoc={srcDoc}
          sandbox=""
          referrerPolicy="no-referrer"
          className="block w-full h-full border-0"
        />
      ) : (
        <div className="flex items-center justify-center h-full" role="status">
          <p className="text-xs text-secondary animate-pulse">{t('filePreview.doc.loading')}</p>
        </div>
      )}
    </div>
  );
}
