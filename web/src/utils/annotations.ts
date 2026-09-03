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

/** The quotes block appended to the user's message (spec 078 §5.4 step 1). */
export function formatQuotes(annotations: Annotation[]): string {
  // TODO(spec 078 workstream E): implement per §5.4. Stub keeps consumers compiling.
  return annotations.length ? '' : '';
}

/** Draw numbered boxes onto a copy of the image; returns a PNG blob for upload. */
export async function renderAnnotatedPng(
  _imageDataUrl: string,
  _boxes: { n: number; box: AnnotationBox }[],
): Promise<Blob> {
  // TODO(spec 078 workstream E)
  throw new Error('renderAnnotatedPng: not implemented');
}
