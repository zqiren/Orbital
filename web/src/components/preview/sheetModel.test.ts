// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment node
import { describe, expect, it } from 'vitest';
import * as XLSX from 'xlsx';
import { buildWorkbookModel, columnLabel, csvToModel, decodeCsvBytes } from './sheetModel';

const utf8 = (s: string) => new TextEncoder().encode(s);

describe('decodeCsvBytes', () => {
  it('decodes plain UTF-8', () => {
    expect(decodeCsvBytes(utf8('名称,数量\n苹果,3\n'))).toEqual({
      text: '名称,数量\n苹果,3\n',
      encoding: 'utf-8',
    });
  });

  it('strips a UTF-8 BOM', () => {
    const bytes = new Uint8Array([0xef, 0xbb, 0xbf, ...utf8('a,b\n')]);
    expect(decodeCsvBytes(bytes)).toEqual({ text: 'a,b\n', encoding: 'utf-8' });
  });

  it('falls back to GB18030 when the bytes are not valid UTF-8 (Chinese Excel export)', () => {
    // "名称,数量\n苹果,3\n" encoded as GBK/GB18030.
    const gbk = new Uint8Array([
      0xc3, 0xfb, 0xb3, 0xc6, 0x2c, 0xca, 0xfd, 0xc1, 0xbf, 0x0a, 0xc6, 0xbb, 0xb9, 0xfb, 0x2c,
      0x33, 0x0a,
    ]);
    expect(decodeCsvBytes(gbk)).toEqual({ text: '名称,数量\n苹果,3\n', encoding: 'gb18030' });
  });

  it('decodes UTF-16LE with a BOM (Excel "Unicode text")', () => {
    const body = 'a\tb\n';
    const bytes = new Uint8Array(2 + body.length * 2);
    bytes[0] = 0xff;
    bytes[1] = 0xfe;
    for (let i = 0; i < body.length; i++) bytes[2 + i * 2] = body.charCodeAt(i);
    expect(decodeCsvBytes(bytes)).toEqual({ text: body, encoding: 'utf-16le' });
  });
});

describe('csvToModel', () => {
  it('parses quoted fields with embedded delimiters and newlines, without a trailing empty row', () => {
    const sheet = csvToModel('a,"b,1"\n"line\nbreak",2\n', 'data.csv').sheets[0];
    expect(sheet).toEqual({
      name: 'data.csv',
      rows: [
        ['a', 'b,1'],
        ['line\nbreak', '2'],
      ],
      colCount: 2,
      merges: [],
      truncated: false,
    });
  });

  it('caps rows and flags the sheet as truncated', () => {
    const text = Array.from({ length: 30 }, (_, i) => `${i},x`).join('\n');
    const sheet = csvToModel(text, 'big.csv', { maxRows: 10, maxCols: 50, maxCells: 1000 }).sheets[0];
    expect(sheet.rows).toHaveLength(10);
    expect(sheet.rows[9]).toEqual(['9', 'x']);
    expect(sheet.truncated).toBe(true);
  });

  it('caps columns and total cells', () => {
    const row = Array.from({ length: 8 }, (_, i) => `c${i}`).join(',');
    const text = Array.from({ length: 10 }, () => row).join('\n');
    const sheet = csvToModel(text, 'wide.csv', { maxRows: 100, maxCols: 4, maxCells: 12 }).sheets[0];
    expect(sheet.colCount).toBe(4);
    expect(sheet.rows).toHaveLength(3);
    expect(sheet.rows[0]).toEqual(['c0', 'c1', 'c2', 'c3']);
    expect(sheet.truncated).toBe(true);
  });

  it('never interprets formula-looking text', () => {
    const sheet = csvToModel('=1+1,=HYPERLINK("http://x")\n', 'f.csv').sheets[0];
    expect(sheet.rows[0]).toEqual(['=1+1', '=HYPERLINK("http://x")']);
  });
});

function makeWorkbook(bookType: 'xlsx' | 'xls'): ArrayBuffer {
  const wb = XLSX.utils.book_new();
  const ws = XLSX.utils.aoa_to_sheet([
    ['地区', '销量', '合计'],
    ['华东', 12, 0],
    ['华北', 18, null],
  ]);
  // A formula cell whose cached value (30) differs from anything we could compute
  // from its text; the model must show the cached value and never the formula.
  ws['C2'] = { t: 'n', v: 30, f: 'B2+B3' };
  ws['!merges'] = [{ s: { r: 2, c: 1 }, e: { r: 2, c: 2 } }];
  XLSX.utils.book_append_sheet(wb, ws, '汇总');
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['only']]), 'Notes');
  return XLSX.write(wb, { type: 'array', bookType }) as ArrayBuffer;
}

describe('buildWorkbookModel', () => {
  it.each(['xlsx', 'xls'] as const)(
    'extracts sheet tabs, cached formula values and merges from %s',
    (bookType) => {
      const model = buildWorkbookModel(makeWorkbook(bookType), bookType, `book.${bookType}`);
      expect(model.sheets.map((s) => s.name)).toEqual(['汇总', 'Notes']);
      const sheet = model.sheets[0];
      expect(sheet.colCount).toBe(3);
      expect(sheet.rows[0]).toEqual(['地区', '销量', '合计']);
      expect(sheet.rows[1]).toEqual(['华东', '12', '30']);
      expect(sheet.merges).toEqual([{ r: 2, c: 1, rs: 1, cs: 2 }]);
      expect(sheet.truncated).toBe(false);
      expect(model.sheets[1].rows).toEqual([['only']]);
    },
  );

  it('caps a large worksheet and clips merges to the shown window', () => {
    const wb = XLSX.utils.book_new();
    const aoa = Array.from({ length: 50 }, (_, r) => [`r${r}`, r]);
    const ws = XLSX.utils.aoa_to_sheet(aoa);
    ws['!merges'] = [
      { s: { r: 1, c: 0 }, e: { r: 3, c: 1 } },
      { s: { r: 40, c: 0 }, e: { r: 41, c: 1 } },
    ];
    XLSX.utils.book_append_sheet(wb, ws, 'Big');
    const bytes = XLSX.write(wb, { type: 'array', bookType: 'xlsx' }) as ArrayBuffer;
    const sheet = buildWorkbookModel(bytes, 'xlsx', 'big.xlsx', {
      maxRows: 3,
      maxCols: 10,
      maxCells: 100,
    }).sheets[0];
    expect(sheet.rows).toHaveLength(3);
    expect(sheet.truncated).toBe(true);
    // The first merge is clipped to rows 1..2; the second lies beyond the window.
    expect(sheet.merges).toEqual([{ r: 1, c: 0, rs: 2, cs: 2 }]);
  });

  it('routes CSV bytes through the decoder', () => {
    const model = buildWorkbookModel(utf8('x,y\n1,2\n').buffer as ArrayBuffer, 'csv', 'data.csv');
    expect(model.sheets[0].rows).toEqual([
      ['x', 'y'],
      ['1', '2'],
    ]);
  });

  it('throws on bytes that are not a workbook', () => {
    expect(() =>
      buildWorkbookModel(utf8('%PDF-1.4 not a workbook').buffer as ArrayBuffer, 'xlsx', 'bad.xlsx'),
    ).toThrow();
  });
});

describe('columnLabel', () => {
  it('uses spreadsheet letters', () => {
    expect([0, 1, 25, 26, 27, 51, 52, 701, 702].map(columnLabel)).toEqual([
      'A', 'B', 'Z', 'AA', 'AB', 'AZ', 'BA', 'ZZ', 'AAA',
    ]);
  });
});
