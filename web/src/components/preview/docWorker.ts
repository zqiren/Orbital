// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — module worker for the DOC engine: the Compound File parse and
 * HTML render run here, and die with `worker.terminate()`.
 *
 * It speaks MsDocWorkerClient's message protocol ({id, type, buffer, options}
 * → {id, ok, result | error}) with an explicit handler instead of importing
 * `@file-viewer/doc/worker` for its side effect: the package declares itself
 * side-effect-free, so a production build tree-shakes a bare import away and
 * leaves a worker that never answers.
 */
import { parseMsDoc, renderMsDoc, type MsDocParseToHtmlOptions } from '@file-viewer/doc';

interface DocWorkerRequest {
  id: number;
  type: string;
  buffer: ArrayBuffer;
  options?: MsDocParseToHtmlOptions;
}

const scope = self as unknown as {
  addEventListener: (type: 'message', listener: (event: MessageEvent<DocWorkerRequest>) => void) => void;
  postMessage: (message: unknown) => void;
};

scope.addEventListener('message', (event) => {
  const { id, type, buffer, options } = event.data ?? ({} as DocWorkerRequest);
  try {
    if (type !== 'parseToHtml') throw new Error(`Unsupported request: ${String(type)}`);
    const parsed = parseMsDoc(buffer, options?.parseOptions ?? {});
    scope.postMessage({ id, ok: true, result: renderMsDoc(parsed, options?.renderOptions ?? {}) });
  } catch (error) {
    scope.postMessage({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
});
