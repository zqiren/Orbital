// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — legacy Word (.doc, Word 97–2003 binary) engine: @file-viewer/doc
 * parses the Compound File and renders reflowing HTML.
 *
 * - Always handed BYTES, never a URL string (the library only fetches when
 *   given one), with external links and linked resources blocked.
 * - Parses in a module worker, terminated on unmount or after
 *   PARSE_TIMEOUT_MS; without Worker support (tests) it parses in-thread.
 * - The library's own DOMPurify pass is kept (into an inert document, so
 *   nothing loads), then the shared sanitizer and SandboxedDocumentFrame apply
 *   exactly the DOCX isolation.
 */
import { useEffect, useRef, useState } from 'react';
import {
  MsDocWorkerClient,
  parseMsDocToHtml,
  sanitizeMsDocHtml,
  type MsDocRenderOptions,
  type MsDocRenderResult,
} from '@file-viewer/doc';
import { sanitizeDocumentHtml } from './documentSanitize';
import SandboxedDocumentFrame from './SandboxedDocumentFrame';
import type { EngineProps } from './types';

const PARSE_TIMEOUT_MS = 60_000;
const RENDER_OPTIONS: MsDocRenderOptions = {
  externalLinkPolicy: 'block',
  externalResourcePolicy: 'block',
};
const DOC_FRAME_CSS = 'body{background:#fff}';

export default function DocPreview({ bytes, fileName, onNav, onError }: EngineProps) {
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let client: MsDocWorkerClient | null = null;
    const stop = () => {
      cancelled = true;
      window.clearTimeout(timer);
      client?.destroy();
      client = null;
    };
    const timer = window.setTimeout(() => {
      stop();
      onErrorRef.current('unreadable');
    }, PARSE_TIMEOUT_MS);

    void (async () => {
      try {
        // A copy: a worker transfer must never detach the host's bytes (Download).
        const input = new Uint8Array(bytes.slice(0));
        let rendered: MsDocRenderResult;
        if (typeof Worker === 'undefined') {
          rendered = await parseMsDocToHtml(input, { renderOptions: RENDER_OPTIONS });
        } else {
          const { default: DocWorker } = await import('./docWorker?worker');
          if (cancelled) return;
          client = new MsDocWorkerClient(new DocWorker());
          rendered = await parseMsDocToHtml(input, { workerClient: client, renderOptions: RENDER_OPTIONS });
        }
        if (cancelled) return;
        // The library's sanitizer returns a fragment; serialize it inside an
        // inert document so no image or style in it ever loads in the app.
        const inert = document.implementation.createHTMLDocument('');
        const holder = inert.createElement('div');
        holder.append(sanitizeMsDocHtml(rendered.html, window));
        setHtml(
          sanitizeDocumentHtml(
            `<style>${rendered.css}</style><div class="msdoc-root">${holder.innerHTML}</div>`,
          ),
        );
        stop();
      } catch {
        if (!cancelled) {
          stop();
          onErrorRef.current('unreadable');
        }
      }
    })();

    return stop;
  }, [bytes]);

  return (
    <SandboxedDocumentFrame
      title={fileName}
      html={html}
      pageWidth={null}
      frameCss={DOC_FRAME_CSS}
      onNav={onNav}
    />
  );
}
