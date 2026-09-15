// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — isolation for the Word engines' generated HTML (docx-preview for
 * DOCX, @file-viewer/doc for DOC).
 *
 * That HTML is untrusted: docx-preview copies relationship targets straight
 * into `href`, and both pass document styling through. Three layers, each
 * enough on its own for the threat it names:
 *   1. DOMPurify: no scripts / frames / forms / event handlers; links only
 *      http(s), mailto and in-document anchors; media only `data:` URLs.
 *   2. CSS scrub: no `@import`, no non-`data:` `url()` (a remote fetch is an
 *      exfiltration beacon even without script).
 *   3. The host renders the result in `<iframe sandbox srcdoc>` with NO
 *      `allow-scripts` (opaque origin), and the document carries a CSP that
 *      permits only inline styles and `data:` images/fonts.
 */
import DOMPurify from 'dompurify';

export const DOCUMENT_SRCDOC_CSP =
  "default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:";

const SAFE_LINK = /^(?:(?:https?|mailto):|#)/i;
const EXTERNAL_LINK = /^(?:https?|mailto):/i;
const IMPORT_RULE = /@import[^;]*;?/gi;
const REMOTE_URL = /url\(\s*(?!['"]?\s*data:)[^)]*\)/gi;

/** Remove `@import` rules and every `url()` that is not a `data:` URL. */
export function scrubCss(css: string): string {
  return css.replace(IMPORT_RULE, '').replace(REMOTE_URL, 'none');
}

const FORBID_TAGS = [
  'script', 'iframe', 'frame', 'frameset', 'object', 'embed', 'applet', 'portal',
  'form', 'input', 'button', 'textarea', 'select', 'option',
  'link', 'meta', 'base', 'audio', 'video', 'source', 'track',
];
const FORBID_ATTR = ['srcset', 'ping', 'action', 'formaction', 'background', 'poster'];

type Purifier = ReturnType<typeof DOMPurify>;
let purifier: Purifier | null = null;

function getPurifier(): Purifier {
  if (purifier) return purifier;
  // A private instance, so these hooks never leak into any other sanitizer.
  const p = DOMPurify(window);
  p.addHook('uponSanitizeElement', (node, data) => {
    if (data.tagName === 'style' && node.textContent) {
      node.textContent = scrubCss(node.textContent);
    }
  });
  p.addHook('afterSanitizeAttributes', (node) => {
    if (node.nodeType !== 1) return;
    const el = node as Element;
    const style = el.getAttribute('style');
    if (style !== null) el.setAttribute('style', scrubCss(style));

    const isLink = el.localName === 'a';
    for (const attr of ['src', 'href', 'xlink:href']) {
      const value = el.getAttribute(attr);
      if (value === null) continue;
      const allowed =
        attr === 'src' ? /^data:/i.test(value) : isLink ? SAFE_LINK.test(value) : /^#/.test(value);
      if (!allowed) el.removeAttribute(attr);
    }
    if (isLink) {
      const href = el.getAttribute('href');
      if (href !== null && EXTERNAL_LINK.test(href)) {
        // The sandbox has no allow-popups, so this makes the click inert
        // instead of navigating the preview frame to a remote page.
        el.setAttribute('target', '_blank');
        el.setAttribute('rel', 'noopener noreferrer');
      } else {
        el.removeAttribute('target');
      }
    }
  });
  purifier = p;
  return p;
}

/** Sanitize generated document styles + body markup. */
export function sanitizeDocumentHtml(html: string): string {
  return getPurifier().sanitize(html, {
    // Keep the leading <style> blocks the engines generate.
    FORCE_BODY: true,
    // No custom ALLOWED_URI_REGEXP: DOMPurify tests every non-URI-safe
    // attribute value against it, so a links-only pattern also strips
    // `colspan="2"`, `width`, `align` and collapses merged table cells. The
    // afterSanitizeAttributes hook enforces the link and media schemes.
    FORBID_TAGS,
    FORBID_ATTR,
    ALLOW_DATA_ATTR: false,
  }) as string;
}

/**
 * The iframe document: CSP first, then the sanitized markup at `zoom`.
 * `frameCss` is engine chrome (a trusted constant), never document data.
 */
export function buildDocumentSrcdoc(sanitizedHtml: string, zoom: number, frameCss = ''): string {
  return (
    '<!doctype html><html><head><meta charset="utf-8">' +
    `<meta http-equiv="Content-Security-Policy" content="${DOCUMENT_SRCDOC_CSP}">` +
    '<meta name="referrer" content="no-referrer">' +
    `<style>html,body{margin:0}body{zoom: ${zoom}}${frameCss}</style>` +
    `</head><body>${sanitizedHtml}</body></html>`
  );
}
