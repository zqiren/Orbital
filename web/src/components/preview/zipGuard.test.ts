// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment node
import { describe, expect, it } from 'vitest';
import * as XLSX from 'xlsx';
import { checkZip, inspectZip } from './zipGuard';

/** A central directory + end record declaring the given [compressed, uncompressed] sizes. */
function fakeZip(sizes: Array<[number, number]>): Uint8Array {
  const name = new TextEncoder().encode('a');
  const cdLen = sizes.length * (46 + name.length);
  const out = new Uint8Array(cdLen + 22);
  const dv = new DataView(out.buffer);
  let o = 0;
  for (const [compressed, uncompressed] of sizes) {
    dv.setUint32(o, 0x02014b50, true);
    dv.setUint32(o + 20, compressed, true);
    dv.setUint32(o + 24, uncompressed, true);
    dv.setUint16(o + 28, name.length, true);
    out.set(name, o + 46);
    o += 46 + name.length;
  }
  dv.setUint32(o, 0x06054b50, true);
  dv.setUint16(o + 8, sizes.length, true);
  dv.setUint16(o + 10, sizes.length, true);
  dv.setUint32(o + 12, cdLen, true);
  dv.setUint32(o + 16, 0, true);
  return out;
}

describe('zipGuard', () => {
  it('reads the entry count and declared sizes of a real OOXML package', () => {
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['a', 1]]), 'S');
    const bytes = new Uint8Array(XLSX.write(wb, { type: 'array', bookType: 'xlsx' }) as ArrayBuffer);
    const stats = inspectZip(bytes);
    expect(stats).not.toBeNull();
    expect(stats!.entries).toBeGreaterThan(3);
    expect(stats!.uncompressed).toBeGreaterThan(0);
    expect(checkZip(bytes)).toBe('ok');
  });

  it('reports non-zip bytes', () => {
    const bytes = new TextEncoder().encode('%PDF-1.4 nope');
    expect(inspectZip(bytes)).toBeNull();
    expect(checkZip(bytes)).toBe('not_zip');
  });

  it('sums declared sizes across entries', () => {
    expect(inspectZip(fakeZip([[10, 100], [20, 300]]))).toEqual({
      entries: 2,
      compressed: 30,
      uncompressed: 400,
    });
  });

  it('flags a package whose declared expansion exceeds the limit', () => {
    expect(checkZip(fakeZip([[1000, 0xf0000000]]))).toBe('too_large');
  });

  it('flags a package with too many entries', () => {
    const bytes = fakeZip([[1, 1], [1, 1], [1, 1]]);
    expect(checkZip(bytes, { maxEntries: 2, maxUncompressed: 1_000 })).toBe('too_large');
    expect(checkZip(bytes, { maxEntries: 3, maxUncompressed: 1_000 })).toBe('ok');
  });

  it('treats a truncated central directory as not a zip', () => {
    const bytes = fakeZip([[1, 1]]);
    // Point the directory offset past the end of the buffer.
    new DataView(bytes.buffer).setUint32(bytes.length - 22 + 16, 9999, true);
    expect(checkZip(bytes)).toBe('not_zip');
  });
});
