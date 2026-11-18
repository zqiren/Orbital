// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Where a live-browser frame is drawn inside the view (BrowserView). Drawing
 * and input mapping both go through `frameRect`, so a pointer always lands
 * on the page point under it.
 */

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Largest rect with the source aspect that fits in the box, centred. */
export function fitRect(srcW: number, srcH: number, boxW: number, boxH: number): Rect {
  if (srcW <= 0 || srcH <= 0 || boxW <= 0 || boxH <= 0) return { x: 0, y: 0, w: boxW || 1, h: boxH || 1 };
  const scale = Math.min(boxW / srcW, boxH / srcH);
  const w = srcW * scale;
  const h = srcH * scale;
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}

/**
 * The view that sized the page (`owner`) shows it filling the box from the
 * top-left corner — exactly 1:1 once the page has the box's size. While a
 * new size is on its way (a drag, a window resize) the last frame covers the
 * box: scaled just enough to reach both edges (1:1 when only one side
 * shrank), the overflow cropped at the right and bottom. So the page stays
 * attached to the panel's edges and no blank band ever shows (a width-fit
 * left one growing at the bottom while narrowing). Any other view watches a
 * page sized by someone else and letterboxes it.
 */
export function frameRect(srcW: number, srcH: number, boxW: number, boxH: number, owner: boolean): Rect {
  if (!owner) return fitRect(srcW, srcH, boxW, boxH);
  if (srcW <= 0 || srcH <= 0 || boxW <= 0 || boxH <= 0) return { x: 0, y: 0, w: boxW || 1, h: boxH || 1 };
  const scale = Math.max(boxW / srcW, boxH / srcH);
  return { x: 0, y: 0, w: srcW * scale, h: srcH * scale };
}
