// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — `getDocument` options for a read-only preview.
 *
 * - `isEvalSupported: false`: defense in depth for the CVE-2024-4367 class
 *   (font code compiled with `new Function`). pdfjs-dist 6.3.289 already
 *   removed that path and no longer reads this option (its builds contain no
 *   `new Function(`), so today it is a no-op kept deliberately, in case a
 *   future build brings the option back.
 * - `enableXfa: false`: XFA forms are interactive content a preview never needs.
 * - Every binary asset (CMaps, standard fonts, wasm decoders, the CMYK ICC
 *   profile) comes from the app bundle under `assetBase`, never a CDN.
 */
export function pdfDocumentOptions(data: Uint8Array, assetBase: string) {
  return {
    data,
    cMapUrl: `${assetBase}cmaps/`,
    cMapPacked: true,
    standardFontDataUrl: `${assetBase}standard_fonts/`,
    wasmUrl: `${assetBase}wasm/`,
    iccUrl: `${assetBase}iccs/`,
    isEvalSupported: false,
    enableXfa: false,
  };
}
