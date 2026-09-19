// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { pdfDocumentOptions } from './pdfDocumentOptions';

describe('pdfDocumentOptions (spec 090)', () => {
  const base = 'http://127.0.0.1:8000/pdfjs-6.3.289/';

  it('never allows eval-compiled fonts or XFA forms', () => {
    const options = pdfDocumentOptions(new Uint8Array([1]), base);
    expect(options.isEvalSupported).toBe(false);
    expect(options.enableXfa).toBe(false);
  });

  it('loads every binary asset from the app bundle, never a CDN', () => {
    const options = pdfDocumentOptions(new Uint8Array([1]), base);
    expect(options).toMatchObject({
      cMapUrl: `${base}cmaps/`,
      cMapPacked: true,
      standardFontDataUrl: `${base}standard_fonts/`,
      wasmUrl: `${base}wasm/`,
      iccUrl: `${base}iccs/`,
    });
  });

  it('passes the bytes through untouched', () => {
    const data = new Uint8Array([37, 80, 68, 70]);
    expect(pdfDocumentOptions(data, base).data).toBe(data);
  });
});
