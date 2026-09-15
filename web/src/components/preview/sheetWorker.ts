// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — parses a CSV / XLSX / XLS off the main thread. The host
 * terminates this worker on unmount or timeout, which is the cancellation
 * story for a pathological workbook.
 */
import { buildWorkbookModel, type SheetFormat, type WorkbookModel } from './sheetModel';
import { checkZip } from './zipGuard';

export interface SheetWorkerRequest {
  bytes: ArrayBuffer;
  format: SheetFormat;
  name: string;
}

export type SheetWorkerResponse =
  | { ok: true; model: WorkbookModel }
  | { ok: false; reason: 'too_large' | 'unreadable' };

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<SheetWorkerRequest>) => void) | null;
  postMessage: (message: SheetWorkerResponse) => void;
};

scope.onmessage = (event) => {
  const { bytes, format, name } = event.data;
  try {
    if (format === 'xlsx' && checkZip(new Uint8Array(bytes)) === 'too_large') {
      scope.postMessage({ ok: false, reason: 'too_large' });
      return;
    }
    scope.postMessage({ ok: true, model: buildWorkbookModel(bytes, format, name) });
  } catch {
    scope.postMessage({ ok: false, reason: 'unreadable' });
  }
};
