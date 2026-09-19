// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 §9 — bound OOXML (ZIP) packages before a parser inflates them.
 *
 * Reads only the central directory, never decompresses anything, and sums the
 * DECLARED entry sizes. A bomb can lie about sizes, so this is a cheap first
 * gate, not the only one: the sheet parser also runs in a terminable worker.
 */

export interface ZipStats {
  entries: number;
  compressed: number;
  uncompressed: number;
}

export interface ZipLimits {
  maxEntries: number;
  maxUncompressed: number;
}

export const ZIP_LIMITS: ZipLimits = {
  maxEntries: 10_000,
  maxUncompressed: 512 * 1024 * 1024,
};

const EOCD_SIG = 0x06054b50;
const CENTRAL_SIG = 0x02014b50;
const ZIP64_LOCATOR_SIG = 0x07064b50;
const ZIP64_EOCD_SIG = 0x06064b50;
const U16_MAX = 0xffff;
const U32_MAX = 0xffffffff;

/**
 * Entry count and summed declared sizes, or null when `bytes` is not a
 * readable ZIP. Stops before walking the directory once `maxEntries` is
 * exceeded (the count alone decides).
 */
export function inspectZip(bytes: Uint8Array, maxEntries = Infinity): ZipStats | null {
  const len = bytes.byteLength;
  if (len < 22) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, len);

  // End of central directory: the last 22 bytes plus up to a 65535-byte comment.
  let eocd = -1;
  for (let i = len - 22; i >= Math.max(0, len - 22 - U16_MAX); i--) {
    if (view.getUint32(i, true) === EOCD_SIG) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) return null;

  let entries = view.getUint16(eocd + 10, true);
  let offset = view.getUint32(eocd + 16, true);
  if (entries === U16_MAX || offset === U32_MAX) {
    const locator = eocd - 20;
    if (locator < 0 || view.getUint32(locator, true) !== ZIP64_LOCATOR_SIG) return null;
    const zip64 = Number(view.getBigUint64(locator + 8, true));
    if (zip64 + 56 > len || view.getUint32(zip64, true) !== ZIP64_EOCD_SIG) return null;
    entries = Number(view.getBigUint64(zip64 + 32, true));
    offset = Number(view.getBigUint64(zip64 + 48, true));
  }

  const stats: ZipStats = { entries, compressed: 0, uncompressed: 0 };
  if (entries > maxEntries) return stats;

  for (let i = 0; i < entries; i++) {
    if (offset + 46 > len || view.getUint32(offset, true) !== CENTRAL_SIG) return null;
    let compressed = view.getUint32(offset + 20, true);
    let uncompressed = view.getUint32(offset + 24, true);
    const nameLen = view.getUint16(offset + 28, true);
    const extraLen = view.getUint16(offset + 30, true);
    const commentLen = view.getUint16(offset + 32, true);

    if (compressed === U32_MAX || uncompressed === U32_MAX) {
      // ZIP64 extended information (0x0001): uncompressed, then compressed,
      // each present only when its 32-bit field is saturated.
      let field = offset + 46 + nameLen;
      const end = Math.min(field + extraLen, len);
      while (field + 4 <= end) {
        const id = view.getUint16(field, true);
        const size = view.getUint16(field + 2, true);
        if (id === 0x0001) {
          let p = field + 4;
          const stop = Math.min(p + size, end);
          if (uncompressed === U32_MAX && p + 8 <= stop) {
            uncompressed = Number(view.getBigUint64(p, true));
            p += 8;
          }
          if (compressed === U32_MAX && p + 8 <= stop) {
            compressed = Number(view.getBigUint64(p, true));
          }
          break;
        }
        field += 4 + size;
      }
    }

    stats.compressed += compressed;
    stats.uncompressed += uncompressed;
    offset += 46 + nameLen + extraLen + commentLen;
  }
  return stats;
}

export function checkZip(
  bytes: Uint8Array,
  limits: ZipLimits = ZIP_LIMITS,
): 'ok' | 'not_zip' | 'too_large' {
  const stats = inspectZip(bytes, limits.maxEntries);
  if (!stats) return 'not_zip';
  if (stats.entries > limits.maxEntries || stats.uncompressed > limits.maxUncompressed) {
    return 'too_large';
  }
  return 'ok';
}
