// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { buildDocumentSrcdoc, DOCUMENT_SRCDOC_CSP, sanitizeDocumentHtml } from './documentSanitize';

describe('sanitizeDocumentHtml', () => {
  it('strips javascript: links and event-handler attributes', () => {
    const out = sanitizeDocumentHtml(
      '<p><a href="javascript:alert(1)">x</a>' +
        '<img src="data:image/png;base64,iVBORw0KGgo=" onerror="alert(2)"></p>',
    );
    expect(out).not.toMatch(/javascript:/i);
    expect(out).not.toMatch(/onerror/i);
    expect(out).not.toMatch(/alert/);
    // Embedded images (the engines' data: URL output) survive.
    expect(out).toContain('src="data:image/png;base64,iVBORw0KGgo="');
  });

  it('keeps http(s) and mailto links, opened in a new (sandbox-blocked) context', () => {
    const out = sanitizeDocumentHtml(
      '<a href="https://example.com/a">a</a><a href="mailto:x@example.com">m</a>',
    );
    const host = document.createElement('div');
    host.innerHTML = out;
    const links = [...host.querySelectorAll('a')];
    expect(links.map((a) => a.getAttribute('href'))).toEqual([
      'https://example.com/a',
      'mailto:x@example.com',
    ]);
    for (const a of links) {
      expect(a.getAttribute('target')).toBe('_blank');
      expect(a.getAttribute('rel')).toBe('noopener noreferrer');
    }
  });

  it('drops other schemes and relative targets but keeps in-document anchors', () => {
    const out = sanitizeDocumentHtml(
      '<a href="file:///etc/passwd">f</a><a href="vbscript:x">v</a>' +
        '<a href="data:text/html,<script>1</script>">d</a><a href="other.docx">r</a>' +
        '<a href="#_Toc123">toc</a>',
    );
    const host = document.createElement('div');
    host.innerHTML = out;
    expect([...host.querySelectorAll('a')].map((a) => a.getAttribute('href'))).toEqual([
      null,
      null,
      null,
      null,
      '#_Toc123',
    ]);
  });

  it('removes scripts, frames, objects, forms and remote media', () => {
    const out = sanitizeDocumentHtml(
      '<script>alert(1)</script><iframe src="https://x.example"></iframe>' +
        '<object data="https://x.example/o"></object><form action="https://x.example"><input></form>' +
        '<img src="https://tracker.example/p.png"><svg><image href="https://tracker.example/s.png"/></svg>' +
        '<link rel="stylesheet" href="https://x.example/a.css"><p>body</p>',
    );
    expect(out).not.toMatch(/<script|<iframe|<object|<form|<input|<link/i);
    expect(out).not.toContain('tracker.example');
    expect(out).not.toContain('x.example');
    expect(out).toContain('<p>body</p>');
  });

  it('keeps generated <style> but scrubs remote url() and @import', () => {
    const out = sanitizeDocumentHtml(
      '<style>@import url(https://evil.example/a.css); .docx{background:url(https://evil.example/b.png)} .ok{color:red}</style>' +
        '<p style="background-image:url(\'http://evil.example/c.png\'); color: blue">t</p>',
    );
    expect(out).toContain('<style>');
    expect(out).toContain('.ok{color:red}');
    expect(out).toContain('color: blue');
    expect(out).not.toContain('evil.example');
    expect(out).not.toMatch(/@import/i);
  });

  it('keeps table structure attributes (merged cells must not collapse)', () => {
    const out = sanitizeDocumentHtml(
      '<table width="600" border="1"><tr>' +
        '<td colspan="2" rowspan="3" width="120" align="center" valign="top">merged</td>' +
        '<td style="width:159px">x</td></tr></table><ol start="3"><li>c</li></ol>',
    );
    const host = document.createElement('div');
    host.innerHTML = out;
    const cell = host.querySelector('td')!;
    expect(cell.getAttribute('colspan')).toBe('2');
    expect(cell.getAttribute('rowspan')).toBe('3');
    expect(cell.getAttribute('width')).toBe('120');
    expect(cell.getAttribute('align')).toBe('center');
    expect(cell.getAttribute('valign')).toBe('top');
    expect(host.querySelector('table')!.getAttribute('width')).toBe('600');
    expect(host.querySelector('ol')!.getAttribute('start')).toBe('3');
  });

  it('keeps data: url() in styles (embedded fonts)', () => {
    const out = sanitizeDocumentHtml(
      '<style>@font-face{font-family:"E";src:url(data:font/ttf;base64,AAAA)}</style><p>x</p>',
    );
    expect(out).toContain('url(data:font/ttf;base64,AAAA)');
  });
});

describe('buildDocumentSrcdoc', () => {
  it('declares the CSP before any document content', () => {
    const doc = buildDocumentSrcdoc('<p>hi</p>', 1);
    expect(doc).toContain(DOCUMENT_SRCDOC_CSP);
    expect(doc.indexOf('Content-Security-Policy')).toBeLessThan(doc.indexOf('<p>hi</p>'));
  });

  it('allows no scripts and no remote loads', () => {
    expect(DOCUMENT_SRCDOC_CSP).toContain("default-src 'none'");
    expect(DOCUMENT_SRCDOC_CSP).not.toMatch(/script-src/);
    expect(DOCUMENT_SRCDOC_CSP).not.toMatch(/https?:/);
  });

  it('applies the zoom factor and the engine frame CSS', () => {
    const doc = buildDocumentSrcdoc('<p>x</p>', 0.5, 'body{background:#fff}');
    expect(doc).toContain('zoom: 0.5');
    expect(doc).toContain('body{background:#fff}');
  });
});
