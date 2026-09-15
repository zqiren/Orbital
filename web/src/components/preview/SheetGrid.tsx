// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — owned, read-only, virtualized grid.
 *
 * Owned rather than @glideapps/glide-data-grid because 6.0.3 declares React
 * ^16 || 17 || 18 peers (plus lodash / marked / react-responsive-carousel)
 * and this app runs React 19. Only the cells inside the scroll window (plus
 * OVERSCAN) are mounted; the column-letter header and the row-number gutter
 * are sticky. Merged ranges render as one spanning cell over the cells they
 * cover. Cells are plain text, so selection quoting works.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { columnLabel, type SheetData, type SheetMerge } from './sheetModel';

const ROW_H = 24;
const HEADER_H = 24;
const MIN_COL_W = 48;
const MAX_COL_W = 320;
const OVERSCAN = 4;
const SAMPLE_ROWS = 200;
const NUMERIC = /^[-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?$/;

/** Rough rendered width of a 12px label; wide (CJK) glyphs are ~em-wide. */
function textWidth(text: string): number {
  let width = 0;
  for (const ch of text) width += (ch.codePointAt(0) ?? 0) >= 0x2e80 ? 13 : 8;
  return width;
}

function columnWidths(sheet: SheetData): number[] {
  const widths = new Array<number>(sheet.colCount).fill(MIN_COL_W);
  const sample = Math.min(sheet.rows.length, SAMPLE_ROWS);
  for (let r = 0; r < sample; r++) {
    const row = sheet.rows[r];
    const n = Math.min(row.length, sheet.colCount);
    for (let c = 0; c < n; c++) {
      const firstLine = row[c].split('\n', 1)[0];
      widths[c] = Math.max(widths[c], Math.min(MAX_COL_W, textWidth(firstLine) + 14));
    }
  }
  return widths;
}

/** Index of the last offset <= x (offsets ascending, offsets[0] = 0). */
function indexAt(offsets: number[], x: number): number {
  let lo = 0;
  let hi = offsets.length - 1;
  let found = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (offsets[mid] <= x) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

export default function SheetGrid({ sheet }: { sheet: SheetData }) {
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef(0);
  const [view, setView] = useState({ top: 0, left: 0, width: 0, height: 0 });

  const widths = useMemo(() => columnWidths(sheet), [sheet]);
  const lefts = useMemo(() => {
    const out = new Array<number>(widths.length + 1);
    out[0] = 0;
    for (let c = 0; c < widths.length; c++) out[c + 1] = out[c] + widths[c];
    return out;
  }, [widths]);

  const rowCount = sheet.rows.length;
  const gutter = Math.max(40, String(rowCount).length * 8 + 16);
  const bodyW = lefts[lefts.length - 1];
  const bodyH = rowCount * ROW_H;

  const measure = useCallback(() => {
    const el = scrollerRef.current;
    if (el) setView({ top: el.scrollTop, left: el.scrollLeft, width: el.clientWidth, height: el.clientHeight });
  }, []);

  useLayoutEffect(() => {
    measure();
    const el = scrollerRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [measure]);

  useEffect(() => () => cancelAnimationFrame(frameRef.current), []);

  const handleScroll = () => {
    cancelAnimationFrame(frameRef.current);
    frameRef.current = requestAnimationFrame(measure);
  };

  // Unmeasured (first paint, jsdom): assume a modest viewport.
  const viewW = view.width || 800;
  const viewH = view.height || 480;
  const r0 = Math.max(0, Math.floor(view.top / ROW_H) - OVERSCAN);
  const r1 = Math.min(rowCount - 1, Math.ceil((view.top + viewH - HEADER_H) / ROW_H) + OVERSCAN);
  const c0 = Math.max(0, indexAt(lefts, view.left) - 1);
  const c1 = Math.min(sheet.colCount - 1, indexAt(lefts, view.left + viewW - gutter) + 1);

  const merges = useMemo(
    () =>
      sheet.merges.filter(
        (m) => m.r <= r1 && m.r + m.rs - 1 >= r0 && m.c <= c1 && m.c + m.cs - 1 >= c0,
      ),
    [sheet.merges, r0, r1, c0, c1],
  );
  const hidden = useMemo(() => {
    // Every in-window cell a visible merge covers, anchors included (anchors
    // render once, from the merge list, so a merge whose anchor has scrolled
    // out of the window still shows).
    const set = new Set<number>();
    for (const m of merges) {
      for (let r = Math.max(m.r, r0); r <= Math.min(m.r + m.rs - 1, r1); r++) {
        for (let c = Math.max(m.c, c0); c <= Math.min(m.c + m.cs - 1, c1); c++) {
          set.add(r * sheet.colCount + c);
        }
      }
    }
    return set;
  }, [merges, r0, r1, c0, c1, sheet.colCount]);

  const cell = (key: string, value: string, top: number, left: number, width: number, height: number) => (
    <div
      key={key}
      role="gridcell"
      title={value.length > 24 ? value : undefined}
      className="absolute px-1.5 text-xs text-primary whitespace-nowrap overflow-hidden text-ellipsis border-r border-b border-border bg-card select-text"
      style={{
        top,
        left,
        width,
        height,
        lineHeight: `${ROW_H - 1}px`,
        textAlign: NUMERIC.test(value) ? 'right' : 'left',
      }}
    >
      {value.replace(/\s*\n\s*/g, ' ')}
    </div>
  );

  const cells = [];
  for (let r = r0; r <= r1; r++) {
    const row = sheet.rows[r];
    for (let c = c0; c <= c1; c++) {
      if (hidden.has(r * sheet.colCount + c)) continue;
      cells.push(cell(`${r}:${c}`, row[c] ?? '', r * ROW_H, lefts[c], widths[c], ROW_H));
    }
  }
  const mergedCells = merges.map((m: SheetMerge) =>
    cell(
      `m${m.r}:${m.c}`,
      sheet.rows[m.r]?.[m.c] ?? '',
      m.r * ROW_H,
      lefts[m.c],
      lefts[Math.min(m.c + m.cs, lefts.length - 1)] - lefts[m.c],
      m.rs * ROW_H,
    ),
  );

  const rowNumbers = [];
  for (let r = r0; r <= r1; r++) {
    rowNumbers.push(
      <div
        key={r}
        className="absolute left-0 pr-1.5 text-right text-2xs text-secondary tabular-nums bg-sidebar border-r border-b border-border"
        style={{ top: r * ROW_H, width: gutter, height: ROW_H, lineHeight: `${ROW_H - 1}px` }}
      >
        {r + 1}
      </div>,
    );
  }

  const columnHeaders = [];
  for (let c = c0; c <= c1; c++) {
    columnHeaders.push(
      <div
        key={c}
        className="absolute top-0 text-center text-2xs font-medium text-secondary bg-sidebar border-r border-b border-border"
        style={{ left: gutter + lefts[c], width: widths[c], height: HEADER_H, lineHeight: `${HEADER_H - 1}px` }}
      >
        {columnLabel(c)}
      </div>,
    );
  }

  return (
    <div
      ref={scrollerRef}
      onScroll={handleScroll}
      role="grid"
      aria-rowcount={rowCount}
      aria-colcount={sheet.colCount}
      tabIndex={0}
      className="absolute inset-0 overflow-auto bg-card focus:outline-none"
      data-testid="sheet-grid"
    >
      <div className="relative" style={{ width: gutter + bodyW, height: HEADER_H + bodyH }}>
        <div className="sticky top-0 z-20" style={{ width: gutter + bodyW, height: HEADER_H }}>
          {columnHeaders}
          <div
            className="sticky left-0 z-10 bg-sidebar border-r border-b border-border"
            style={{ width: gutter, height: HEADER_H }}
          />
        </div>
        <div className="sticky left-0 z-10" style={{ width: gutter, height: bodyH }}>
          {rowNumbers}
        </div>
        <div className="absolute" style={{ top: HEADER_H, left: gutter, width: bodyW, height: bodyH }}>
          {cells}
          {mergedCells}
        </div>
      </div>
    </div>
  );
}
