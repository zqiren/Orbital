// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — lifts a document preview's navigation and download action to
 * whoever renders the header.
 *
 *   const controller = useDocumentController();
 *   <DocumentNav nav={controller.nav} compact />            // anywhere
 *   <DocumentPreview controller={controller} showToolbar={false} … />
 *   controller.download                                      // e.g. a More item
 *
 * The engine publishes through `controller.sink` (stable), so the header and
 * the body stay in sync without either owning the other.
 */
import { useMemo, useState } from 'react';
import type { DocumentNavState } from './types';

export interface DocumentControllerSink {
  setNav: (nav: DocumentNavState | null) => void;
  setDownload: (download: (() => void) | null) => void;
}

export interface DocumentController {
  /** The active engine's navigation; null while loading, on fallback, or when it has none (CSV). */
  nav: DocumentNavState | null;
  /** Saves the fetched bytes; null until they arrive, and when the file is over the ceiling. */
  download: (() => void) | null;
  /** Pass the whole controller to DocumentPreview; it publishes through this. */
  sink: DocumentControllerSink;
}

export function useDocumentController(): DocumentController {
  const [nav, setNav] = useState<DocumentNavState | null>(null);
  const [download, setDownloadState] = useState<(() => void) | null>(null);
  const sink = useMemo<DocumentControllerSink>(
    () => ({
      setNav,
      // A function value must be wrapped, or React would call it as an updater.
      setDownload: (fn) => setDownloadState(() => fn),
    }),
    [],
  );
  return { nav, download, sink };
}
