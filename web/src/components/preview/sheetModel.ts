// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — the read-only sheet model behind the CSV / XLSX / XLS preview.
 *
 * Pure (no DOM), so it runs in the sheet worker and in node tests alike. Cells
 * become display strings: a formula shows its CACHED value (SheetJS is told not
 * to read formulas at all, and nothing here ever evaluates one). Large sheets
 * are capped by rows, columns and total cells, with `truncated` set so the UI
 * can say so.
 */
import Papa from 'papaparse';
import * as XLSX from 'xlsx';

export type SheetFormat = 'csv' | 'xlsx' | 'xls';

/** A merged range: top-left cell plus its row/column span. */
export interface SheetMerge {
  r: number;
  c: number;
  rs: number;
  cs: number;
}

export interface SheetData {
  name: string;
  /** Shown rows. Rows may be ragged; a missing cell is empty. */
  rows: string[][];
  /** Shown columns. */
  colCount: number;
  /** Merges clipped to the shown window. */
  merges: SheetMerge[];
  /** Rows or columns beyond the caps were dropped. */
  truncated: boolean;
}

export interface WorkbookModel {
  sheets: SheetData[];
}

export interface SheetLimits {
  maxRows: number;
  maxCols: number;
  maxCells: number;
}

export const SHEET_LIMITS: SheetLimits = {
  maxRows: 100_000,
  maxCols: 1_000,
  maxCells: 2_000_000,
};

const MAX_MERGES = 10_000;

export type CsvEncoding = 'utf-8' | 'utf-16le' | 'utf-16be' | 'gb18030';

/**
 * Decode CSV bytes. BOMs win; otherwise strict UTF-8, falling back to GB18030
 * (a superset of GBK) because Chinese Excel exports CSV in the ANSI code page.
 */
export function decodeCsvBytes(bytes: ArrayBuffer | Uint8Array): {
  text: string;
  encoding: CsvEncoding;
} {
  const u8 = ArrayBuffer.isView(bytes)
    ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    : new Uint8Array(bytes);
  if (u8[0] === 0xef && u8[1] === 0xbb && u8[2] === 0xbf) {
    return { text: new TextDecoder('utf-8', { ignoreBOM: true }).decode(u8.subarray(3)), encoding: 'utf-8' };
  }
  if (u8[0] === 0xff && u8[1] === 0xfe) {
    return { text: new TextDecoder('utf-16le', { ignoreBOM: true }).decode(u8.subarray(2)), encoding: 'utf-16le' };
  }
  if (u8[0] === 0xfe && u8[1] === 0xff) {
    return { text: new TextDecoder('utf-16be', { ignoreBOM: true }).decode(u8.subarray(2)), encoding: 'utf-16be' };
  }
  try {
    return { text: new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(u8), encoding: 'utf-8' };
  } catch {
    return { text: new TextDecoder('gb18030').decode(u8), encoding: 'gb18030' };
  }
}

/** The shown window for a sheet of `rowCount` × `colCount` under `limits`. */
function windowFor(rowCount: number, colCount: number, limits: SheetLimits) {
  const cols = Math.min(colCount, limits.maxCols);
  const rows = Math.min(rowCount, limits.maxRows, Math.floor(limits.maxCells / Math.max(1, cols)));
  return { rows, cols, truncated: rows < rowCount || cols < colCount };
}

export function csvToModel(
  text: string,
  name: string,
  limits: SheetLimits = SHEET_LIMITS,
): WorkbookModel {
  // `preview` bounds parsing; one extra record tells us rows were dropped.
  const data = Papa.parse<string[]>(text, {
    preview: limits.maxRows + 1,
    dynamicTyping: false,
    skipEmptyLines: false,
  }).data;
  // A final newline yields one empty trailing record.
  const last = data[data.length - 1];
  if (last && last.length === 1 && last[0] === '') data.pop();

  let colCount = 0;
  for (const row of data) if (row.length > colCount) colCount = row.length;
  const win = windowFor(data.length, colCount, limits);
  const rows = data
    .slice(0, win.rows)
    .map((row) => (row.length > win.cols ? row.slice(0, win.cols) : row));
  return { sheets: [{ name, rows, colCount: win.cols, merges: [], truncated: win.truncated }] };
}

function cellText(cell: XLSX.CellObject | undefined): string {
  if (!cell) return '';
  if (cell.w != null) return cell.w;
  if (cell.v == null) return '';
  return String(cell.v);
}

function worksheetToSheet(name: string, ws: XLSX.WorkSheet, limits: SheetLimits): SheetData {
  // With `sheetRows`, `!ref` is the truncated range and `!fullref` the original.
  const ref = (ws['!fullref'] as string | undefined) ?? ws['!ref'];
  if (!ref) return { name, rows: [], colCount: 0, merges: [], truncated: false };
  const full = XLSX.utils.decode_range(ref);
  const win = windowFor(full.e.r + 1, full.e.c + 1, limits);

  const dense = (ws as XLSX.DenseWorkSheet)['!data'] ?? [];
  const rows: string[][] = [];
  for (let r = 0; r < win.rows; r++) {
    const src = dense[r];
    const out: string[] = [];
    if (src) {
      const n = Math.min(src.length, win.cols);
      for (let c = 0; c < n; c++) out.push(cellText(src[c]));
    }
    rows.push(out);
  }

  const merges: SheetMerge[] = [];
  for (const m of ws['!merges'] ?? []) {
    if (merges.length >= MAX_MERGES) break;
    if (m.s.r >= win.rows || m.s.c >= win.cols) continue;
    const rs = Math.min(m.e.r, win.rows - 1) - m.s.r + 1;
    const cs = Math.min(m.e.c, win.cols - 1) - m.s.c + 1;
    if (rs * cs > 1) merges.push({ r: m.s.r, c: m.s.c, rs, cs });
  }

  return { name, rows, colCount: win.cols, merges, truncated: win.truncated };
}

export function workbookToModel(bytes: ArrayBuffer, limits: SheetLimits = SHEET_LIMITS): WorkbookModel {
  const wb = XLSX.read(new Uint8Array(bytes), {
    type: 'array',
    dense: true,
    // Bounds parse memory; the extra row lets `!fullref` report truncation.
    sheetRows: limits.maxRows + 1,
    // Formulas are never read, let alone evaluated: cells carry cached values.
    cellFormula: false,
    cellHTML: false,
    cellStyles: false,
    cellDates: false,
    cellText: true,
    bookVBA: false,
  });
  return {
    sheets: wb.SheetNames.map((name) => {
      const ws = wb.Sheets[name];
      return ws
        ? worksheetToSheet(name, ws, limits)
        : { name, rows: [], colCount: 0, merges: [], truncated: false };
    }),
  };
}

export function buildWorkbookModel(
  bytes: ArrayBuffer,
  format: SheetFormat,
  name: string,
  limits: SheetLimits = SHEET_LIMITS,
): WorkbookModel {
  if (format === 'csv') return csvToModel(decodeCsvBytes(bytes).text, name, limits);
  // An .xlsx is always a ZIP package. Without this check SheetJS would happily
  // read arbitrary bytes as delimited text and show garbage. (.xls stays open:
  // many ".xls" exports are really HTML or XML tables, which SheetJS reads.)
  const head = new Uint8Array(bytes, 0, Math.min(2, bytes.byteLength));
  if (format === 'xlsx' && !(head[0] === 0x50 && head[1] === 0x4b)) {
    throw new Error('Not an XLSX package');
  }
  return workbookToModel(bytes, limits);
}

/** 0 → A, 25 → Z, 26 → AA. */
export function columnLabel(index: number): string {
  let n = index + 1;
  let label = '';
  while (n > 0) {
    const rem = (n - 1) % 26;
    label = String.fromCharCode(65 + rem) + label;
    n = Math.floor((n - 1) / 26);
  }
  return label;
}
