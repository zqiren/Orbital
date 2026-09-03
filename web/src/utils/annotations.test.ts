// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

// Spec 078 §5.4 — the quotes block is what the agent actually reads, so it is
// asserted BYTE-EXACT here: a drift in spacing or the `> ` prefix silently
// changes the prompt every annotated message carries.

import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  formatQuotes,
  splitQuotesBlock,
  renderAnnotatedPng,
  annotationFilename,
  type Annotation,
} from './annotations';

const browserAnn: Annotation = {
  n: 1,
  kind: 'browser',
  pageTitle: 'Queue',
  box: { x: 120, y: 64, w: 240, h: 48 },
  note: 'click this one, not the ad',
  imageDataUrl: 'data:image/png;base64,AAAA',
};

const imageAnn: Annotation = {
  n: 2,
  kind: 'image',
  path: 'orbital/output/screenshots/12.png',
  box: { x: 4, y: 5, w: 6, h: 7 },
  note: 'this button',
  imageDataUrl: 'data:image/png;base64,BBBB',
};

const textAnn: Annotation = {
  n: 3,
  kind: 'text',
  path: 'web/src/components/QueueHeader.tsx',
  text: '<span className="text-xs">\n  {t(\'queue.header.pending\')}',
  lines: [14, 15],
  note: 'use the short label here',
};

const fileAnn: Annotation = {
  n: 4,
  kind: 'file',
  path: 'docs/handbook.pdf',
  note: 'what does section 3 say?',
};

describe('formatQuotes', () => {
  it('returns an empty string for no annotations', () => {
    expect(formatQuotes([])).toBe('');
  });

  it('formats a browser box quote byte-exactly', () => {
    expect(formatQuotes([browserAnn])).toBe(
      [
        'Annotations:',
        '[1] Browser · "Queue" · box 120,64 240×48 (page pixels) — see attached annotation-1.png',
        '    note: click this one, not the ad',
      ].join('\n'),
    );
  });

  it('formats an image-region quote byte-exactly', () => {
    expect(formatQuotes([imageAnn])).toBe(
      [
        'Annotations:',
        '[2] orbital/output/screenshots/12.png · box 4,5 6×7 — see attached annotation-2.png',
        '    note: this button',
      ].join('\n'),
    );
  });

  it('formats a text-span quote with the line range and a verbatim > block', () => {
    expect(formatQuotes([textAnn])).toBe(
      [
        'Annotations:',
        '[3] web/src/components/QueueHeader.tsx lines 14–15',
        '    > <span className="text-xs">',
        "    >   {t('queue.header.pending')}",
        '    note: use the short label here',
      ].join('\n'),
    );
  });

  it('omits the line range when the selection could not be located', () => {
    const { lines: _lines, ...rest } = textAnn;
    void _lines;
    const out = formatQuotes([rest as Annotation]);
    expect(out).toContain('[3] web/src/components/QueueHeader.tsx\n');
    expect(out).not.toContain('lines');
  });

  it('formats a whole-file quote byte-exactly', () => {
    expect(formatQuotes([fileAnn])).toBe(
      ['Annotations:', '[4] docs/handbook.pdf (whole file)', '    note: what does section 3 say?'].join('\n'),
    );
  });

  it('omits the note line entirely when the note is empty or blank', () => {
    expect(formatQuotes([{ ...fileAnn, note: '' }])).toBe(
      ['Annotations:', '[4] docs/handbook.pdf (whole file)'].join('\n'),
    );
    expect(formatQuotes([{ ...fileAnn, note: '   ' }])).toBe(
      ['Annotations:', '[4] docs/handbook.pdf (whole file)'].join('\n'),
    );
  });

  it('emits a bare > for empty lines inside a quoted span', () => {
    const out = formatQuotes([{ ...textAnn, text: 'a\n\nb', lines: undefined, note: '' }]);
    expect(out.split('\n')).toEqual([
      'Annotations:',
      '[3] web/src/components/QueueHeader.tsx',
      '    > a',
      '    >',
      '    > b',
    ]);
  });

  it('keeps every annotation, in order, under one heading', () => {
    const out = formatQuotes([browserAnn, textAnn, fileAnn]);
    expect(out.match(/^Annotations:$/gm)).toHaveLength(1);
    expect(out.match(/^\[\d+\] /gm)).toHaveLength(3);
    expect(out.indexOf('[1]')).toBeLessThan(out.indexOf('[3]'));
    expect(out.indexOf('[3]')).toBeLessThan(out.indexOf('[4]'));
  });

  it('rounds fractional box coordinates', () => {
    const out = formatQuotes([{ ...imageAnn, box: { x: 4.4, y: 5.6, w: 6.5, h: 7.2 }, note: '' }]);
    expect(out).toContain('box 4,6 7×7');
  });
});

describe('annotationFilename', () => {
  it('follows the annotation-<n>.png convention', () => {
    expect(annotationFilename(2)).toBe('annotation-2.png');
  });
});

describe('splitQuotesBlock', () => {
  it('leaves a message with no block untouched', () => {
    expect(splitQuotesBlock('just a message')).toEqual({
      text: 'just a message',
      quotes: null,
      count: 0,
    });
  });

  it('splits the typed text from the appended block and counts the items', () => {
    const content = `look at this\n\n${formatQuotes([browserAnn, fileAnn])}`;
    const out = splitQuotesBlock(content);
    expect(out.text).toBe('look at this');
    expect(out.count).toBe(2);
    expect(out.quotes).toBe(formatQuotes([browserAnn, fileAnn]));
  });

  it('handles a message that is nothing but the block', () => {
    const content = formatQuotes([fileAnn]);
    const out = splitQuotesBlock(content);
    expect(out.text).toBe('');
    expect(out.count).toBe(1);
  });

  it('ignores a bare "Annotations:" heading with no [n] item under it', () => {
    const content = 'Annotations:\nare a new feature';
    expect(splitQuotesBlock(content).quotes).toBeNull();
  });

  it('splits on the LAST block when the text quotes an earlier one', () => {
    const inner = formatQuotes([fileAnn]);
    const content = `here is what you sent:\n\n${inner}\n\n${formatQuotes([browserAnn])}`;
    const out = splitQuotesBlock(content);
    expect(out.count).toBe(1);
    expect(out.text).toContain(inner);
  });
});

// --- renderAnnotatedPng -----------------------------------------------------
// jsdom has no canvas backend (getContext throws "Not implemented"), so the
// 2D context and toBlob are faked and the DRAW CALLS are what we assert.

interface FakeCtx {
  calls: string[];
  drawImage: (...a: unknown[]) => void;
  strokeRect: (...a: unknown[]) => void;
  fillRect: (...a: unknown[]) => void;
  fillText: (...a: unknown[]) => void;
  lineWidth: number;
  strokeStyle: string;
  fillStyle: string;
  font: string;
  textBaseline: string;
}

function installCanvasMocks(): { ctx: FakeCtx; canvases: HTMLCanvasElement[] } {
  const calls: string[] = [];
  const ctx: FakeCtx = {
    calls,
    drawImage: (...a) => calls.push(`drawImage(${a.slice(1).join(',')})`),
    strokeRect: (...a) => calls.push(`strokeRect(${a.join(',')})`),
    fillRect: (...a) => calls.push(`fillRect(${a.join(',')})`),
    fillText: (...a) => calls.push(`fillText(${a.join(',')})`),
    lineWidth: 0,
    strokeStyle: '',
    fillStyle: '',
    font: '',
    textBaseline: '',
  };
  const canvases: HTMLCanvasElement[] = [];
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
    this: HTMLCanvasElement,
  ) {
    canvases.push(this);
    return ctx as unknown as CanvasRenderingContext2D;
  } as unknown as typeof HTMLCanvasElement.prototype.getContext);
  vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb) => {
    (cb as BlobCallback)(new Blob(['png-bytes'], { type: 'image/png' }));
  });
  return { ctx, canvases };
}

/** Make `img.src = …` resolve immediately with a known intrinsic size. */
function installImageMock(width: number, height: number, fail = false) {
  Object.defineProperty(HTMLImageElement.prototype, 'src', {
    configurable: true,
    set(this: HTMLImageElement) {
      queueMicrotask(() => {
        if (fail) this.onerror?.(new Event('error'));
        else this.onload?.(new Event('load'));
      });
    },
    get() {
      return 'data:image/png;base64,AAAA';
    },
  });
  Object.defineProperty(HTMLImageElement.prototype, 'naturalWidth', {
    configurable: true,
    get: () => width,
  });
  Object.defineProperty(HTMLImageElement.prototype, 'naturalHeight', {
    configurable: true,
    get: () => height,
  });
}

describe('renderAnnotatedPng', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    for (const prop of ['src', 'naturalWidth', 'naturalHeight']) {
      delete (HTMLImageElement.prototype as unknown as Record<string, unknown>)[prop];
    }
  });

  it('sizes the canvas to the image and draws every box with a numbered pin', async () => {
    const { ctx, canvases } = installCanvasMocks();
    installImageMock(800, 600);

    const blob = await renderAnnotatedPng('data:image/png;base64,AAAA', [
      { n: 1, box: { x: 10, y: 40, w: 100, h: 50 } },
      { n: 2, box: { x: 200, y: 0, w: 60, h: 20 } },
    ]);

    expect(blob.type).toBe('image/png');
    expect(canvases[0].width).toBe(800);
    expect(canvases[0].height).toBe(600);
    // The full image is drawn (§13 Q2: never a crop), then one stroked box per
    // annotation at its own coordinates.
    expect(ctx.calls[0]).toBe('drawImage(0,0,800,600)');
    expect(ctx.calls).toContain('strokeRect(10,40,100,50)');
    expect(ctx.calls).toContain('strokeRect(200,0,60,20)');
    expect(ctx.calls.filter((c) => c.startsWith('strokeRect'))).toHaveLength(2);
    // A numbered pin per box: box 1 sits above its box, box 2 (at y=0) has its
    // pin clamped into the frame rather than off the top edge.
    expect(ctx.calls).toContain('fillText(1,15,25)');
    expect(ctx.calls).toContain('fillText(2,205,3)');
    expect(ctx.lineWidth).toBe(2);
  });

  it('throws a clear error when there is no 2D context', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    installImageMock(10, 10);
    await expect(renderAnnotatedPng('data:image/png;base64,AAAA', [])).rejects.toThrow(
      /canvas 2D context unavailable/,
    );
  });

  it('throws a clear error when the image cannot be decoded', async () => {
    installCanvasMocks();
    installImageMock(10, 10, true);
    await expect(renderAnnotatedPng('not-an-image', [])).rejects.toThrow(
      /could not decode the image/,
    );
  });

  it('throws when the image has no intrinsic size', async () => {
    installCanvasMocks();
    installImageMock(0, 0);
    await expect(renderAnnotatedPng('data:image/png;base64,AAAA', [])).rejects.toThrow(
      /no intrinsic size/,
    );
  });
});
