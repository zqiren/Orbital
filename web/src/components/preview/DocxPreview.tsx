// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — DOCX engine. docx-preview renders into DETACHED containers (its
 * output never touches the app document); the result is sanitized and shown
 * through SandboxedDocumentFrame, fitted to the page width docx-preview
 * declares on its first section.
 */
import { useEffect, useRef, useState } from 'react';
import { renderAsync } from 'docx-preview';
import { sanitizeDocumentHtml } from './documentSanitize';
import SandboxedDocumentFrame from './SandboxedDocumentFrame';
import type { EngineProps } from './types';
import { checkZip } from './zipGuard';

/** Letter width at 96 dpi, for documents that declare no section width. */
const DEFAULT_PAGE_PX = 816;

const DOCX_FRAME_CSS =
  'body{background:#e5e7eb}' +
  '.docx-wrapper{background:transparent!important;padding:16px!important}' +
  '.docx-wrapper>section.docx{margin-bottom:16px!important}';

const UNIT_PX: Record<string, number> = { px: 1, pt: 96 / 72, in: 96, cm: 96 / 2.54, mm: 96 / 25.4 };

function cssLengthPx(value: string | undefined): number | null {
  const match = /^([\d.]+)(px|pt|in|cm|mm)$/.exec((value ?? '').trim());
  return match ? parseFloat(match[1]) * UNIT_PX[match[2]] : null;
}

export default function DocxPreview({ bytes, fileName, onNav, onError }: EngineProps) {
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const [rendered, setRendered] = useState<{ html: string; pageWidth: number } | null>(null);

  useEffect(() => {
    const guard = checkZip(new Uint8Array(bytes));
    if (guard !== 'ok') {
      onErrorRef.current(guard === 'too_large' ? 'too_large' : 'unreadable');
      return;
    }
    let cancelled = false;
    const body = document.createElement('div');
    const styles = document.createElement('div');
    renderAsync(bytes, body, styles, {
      inWrapper: true,
      breakPages: true,
      ignoreLastRenderedPageBreak: true,
      renderAltChunks: false,
      renderComments: false,
      renderChanges: false,
      // Images and fonts as data: URLs — the sandboxed frame loads nothing else.
      useBase64URL: true,
      experimental: false,
      trimXmlDeclaration: true,
      debug: false,
    })
      .then(() => {
        if (cancelled) return;
        const section = body.querySelector<HTMLElement>('section.docx');
        setRendered({
          html: sanitizeDocumentHtml(styles.innerHTML + body.innerHTML),
          pageWidth: cssLengthPx(section?.style.width) ?? DEFAULT_PAGE_PX,
        });
      })
      .catch(() => {
        if (!cancelled) onErrorRef.current('unreadable');
      });
    return () => {
      cancelled = true;
    };
  }, [bytes]);

  return (
    <SandboxedDocumentFrame
      title={fileName}
      html={rendered?.html ?? null}
      pageWidth={rendered?.pageWidth ?? null}
      frameCss={DOCX_FRAME_CSS}
      onNav={onNav}
    />
  );
}
