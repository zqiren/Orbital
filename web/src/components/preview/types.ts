// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { DocumentFormat } from '../../types';

/** Why a document fell back to the download card. */
export type PreviewFailure = 'too_large' | 'unreadable';

/** PDF: page x / y plus zoom. */
export interface PdfNavState {
  kind: 'pdf';
  page: number;
  pageCount: number;
  zoomPercent: number;
  fit: boolean;
  goTo: (page: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  fitWidth: () => void;
}

/** Workbook: the sheet list and the active sheet. */
export interface SheetNavState {
  kind: 'sheet';
  sheets: string[];
  active: number;
  select: (index: number) => void;
}

/** Word engines (DOCX, DOC): zoom only. */
export interface ZoomNavState {
  kind: 'zoom';
  zoomPercent: number;
  fit: boolean;
  zoomIn: () => void;
  zoomOut: () => void;
  fitWidth: () => void;
}

/**
 * An engine's navigation as plain state + stable actions. Engines publish it;
 * `DocumentNav` renders it wherever the host wants (DocumentPreview's own
 * toolbar, or a header row).
 */
export type DocumentNavState = PdfNavState | SheetNavState | ZoomNavState;

/**
 * What every format engine receives. An engine owns only its body and its
 * navigation state; loading, failure and the download fallback belong to
 * DocumentPreview. Adding a format (PPTX) = one engine file + one entry in
 * ENGINE_BY_FORMAT and DocumentPreview's lazy map.
 */
export interface EngineProps {
  bytes: ArrayBuffer;
  format: DocumentFormat;
  fileName: string;
  /** Publish (or clear, with null) this engine's navigation. Stable identity. */
  onNav: (nav: DocumentNavState | null) => void;
  /** The document cannot be shown; the host swaps in the download card. */
  onError: (failure: PreviewFailure) => void;
}

export type EngineKind = 'pdf' | 'sheet' | 'docx' | 'doc';

export const ENGINE_BY_FORMAT: Record<DocumentFormat, EngineKind> = {
  pdf: 'pdf',
  csv: 'sheet',
  xlsx: 'sheet',
  xls: 'sheet',
  docx: 'docx',
  doc: 'doc',
};
