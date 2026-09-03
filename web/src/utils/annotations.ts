// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.4 — annotations ("quote an element to ask about it").
 * CONTRACT FILE: types are final; function bodies are implemented by the
 * annotations workstream. Consumers import the types from here.
 */
export interface AnnotationBox { x: number; y: number; w: number; h: number }

export type Annotation =
  | { n: number; kind: 'browser'; pageTitle: string; box: AnnotationBox; note: string; imageDataUrl: string }
  | { n: number; kind: 'image'; path: string; box: AnnotationBox; note: string; imageDataUrl?: string }
  | { n: number; kind: 'text'; path: string; text: string; lines?: [number, number]; note: string }
  | { n: number; kind: 'file'; path: string; note: string };

export type AnnotationDraft =
  | Omit<Extract<Annotation, { kind: 'browser' }>, 'n'>
  | Omit<Extract<Annotation, { kind: 'image' }>, 'n'>
  | Omit<Extract<Annotation, { kind: 'text' }>, 'n'>
  | Omit<Extract<Annotation, { kind: 'file' }>, 'n'>;

/**
 * The heading that opens the quotes block. It is also the marker the chat
 * renderer looks for when collapsing the block behind a chip on a sent
 * message, so the two must stay in step — hence one constant.
 */
export const QUOTES_HEADING = 'Annotations:';

/** Annotation filename convention (§5.4 step 2): `annotation-<n>.png`. */
export function annotationFilename(n: number): string {
  return `annotation-${n}.png`;
}

function boxText(box: AnnotationBox): string {
  const r = (v: number) => Math.round(v);
  return `${r(box.x)},${r(box.y)} ${r(box.w)}×${r(box.h)}`;
}

/**
 * The quotes block appended to the user's message (spec 078 §5.4 step 1).
 *
 * Deterministic and human-readable: the agent reads this as text, so the
 * verbatim quoted span is the payload that matters (its `edit` tool matches on
 * exact `old_text`) and the coordinates/line numbers are context. Emitted in
 * array order, numbered by each annotation's own `n`.
 *
 * Empty input yields an empty string so callers can append unconditionally.
 */
export function formatQuotes(annotations: Annotation[]): string {
  if (annotations.length === 0) return '';
  const lines: string[] = [QUOTES_HEADING];
  for (const a of annotations) {
    switch (a.kind) {
      case 'browser':
        lines.push(
          `[${a.n}] Browser · "${a.pageTitle}" · box ${boxText(a.box)} (page pixels)` +
            ` — see attached ${annotationFilename(a.n)}`,
        );
        break;
      case 'image':
        lines.push(
          `[${a.n}] ${a.path} · box ${boxText(a.box)} — see attached ${annotationFilename(a.n)}`,
        );
        break;
      case 'text': {
        const range = a.lines ? ` lines ${a.lines[0]}–${a.lines[1]}` : '';
        lines.push(`[${a.n}] ${a.path}${range}`);
        // Verbatim, one `> ` line each. An empty source line becomes a bare
        // `>` rather than a line with trailing whitespace.
        for (const line of a.text.split('\n')) {
          lines.push(line ? `    > ${line}` : '    >');
        }
        break;
      }
      case 'file':
        lines.push(`[${a.n}] ${a.path} (whole file)`);
        break;
    }
    const note = a.note.trim();
    if (note) lines.push(`    note: ${note}`);
  }
  return lines.join('\n');
}

/**
 * Split a sent user message back into its typed text and the quotes block the
 * composer appended, so the chat can collapse the block behind a chip.
 *
 * Recognized only when the heading sits at the start of a line and is followed
 * immediately by a `[n] ` item — the same "machine markup, not something the
 * user typed" heuristic `parseAttachmentsBlock` uses for `<attached_files>`.
 * The LAST such heading wins: the block is always appended last.
 */
export function splitQuotesBlock(content: string): {
  text: string;
  quotes: string | null;
  count: number;
} {
  const re = new RegExp(`(^|\\n)${QUOTES_HEADING}\\n(?=\\[\\d+\\] )`, 'g');
  let match: RegExpExecArray | null;
  let last: RegExpExecArray | null = null;
  while ((match = re.exec(content)) !== null) last = match;
  if (!last) return { text: content, quotes: null, count: 0 };
  const start = last.index + (last[1] ? 1 : 0);
  const quotes = content.slice(start);
  const text = content.slice(0, start).replace(/\n+$/, '');
  const count = (quotes.match(/^\[\d+\] /gm) ?? []).length;
  return { text, quotes, count };
}

/** Stroke colour for the drawn boxes: the app accent, with a visible fallback. */
function accentColor(): string {
  try {
    const v = getComputedStyle(document.documentElement)
      .getPropertyValue('--color-accent')
      .trim();
    if (v) return v;
  } catch {
    // getComputedStyle can throw in exotic hosts — fall through.
  }
  return '#ff3b30';
}

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = document.createElement('img');
    img.onload = () => resolve(img);
    img.onerror = () =>
      reject(new Error('renderAnnotatedPng: could not decode the image'));
    img.src = dataUrl;
  });
}

function toPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    if (typeof canvas.toBlob !== 'function') {
      reject(new Error('renderAnnotatedPng: canvas.toBlob is unavailable'));
      return;
    }
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('renderAnnotatedPng: canvas.toBlob produced no data'));
    }, 'image/png');
  });
}

/**
 * Draw numbered boxes onto a copy of the image; returns a PNG blob for upload.
 *
 * Full image with the box drawn on it, never a crop (§13 Q2): the crop would
 * lose the page the box refers to, which is most of what a vision model needs.
 * Throws with a clear message wherever the environment cannot do this (no DOM,
 * no 2D context, undecodable image) — the caller falls back to sending the
 * coordinates and the note as text, which carry the meaning on their own.
 */
export async function renderAnnotatedPng(
  imageDataUrl: string,
  boxes: { n: number; box: AnnotationBox }[],
): Promise<Blob> {
  if (typeof document === 'undefined') {
    throw new Error('renderAnnotatedPng: no DOM available');
  }
  const img = await loadImage(imageDataUrl);
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;
  if (!width || !height) {
    throw new Error('renderAnnotatedPng: the image has no intrinsic size');
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  let ctx: CanvasRenderingContext2D | null = null;
  try {
    ctx = canvas.getContext('2d');
  } catch {
    ctx = null;
  }
  if (!ctx) {
    throw new Error('renderAnnotatedPng: canvas 2D context unavailable');
  }

  ctx.drawImage(img, 0, 0, width, height);

  const color = accentColor();
  const PIN = 18;
  ctx.lineWidth = 2;
  ctx.font = 'bold 13px system-ui, sans-serif';
  ctx.textBaseline = 'top';
  for (const { n, box } of boxes) {
    ctx.strokeStyle = color;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
    // Numbered pin, tucked just above the box's top-left corner (clamped into
    // the frame so a box at y=0 keeps its number visible).
    const label = String(n);
    const pinX = box.x;
    const pinY = box.y >= PIN ? box.y - PIN : box.y;
    ctx.fillStyle = color;
    ctx.fillRect(pinX, pinY, PIN + (label.length - 1) * 7, PIN);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, pinX + 5, pinY + 3);
  }

  return toPngBlob(canvas);
}
