// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — read-only preview for PDF, DOC, DOCX, XLSX/XLS and CSV.
 *
 * Dispatches a `type: 'document'` envelope to one lazily imported engine per
 * format (so pdf.js, SheetJS, docx-preview and the DOC parser never enter the
 * main bundle) and owns everything the engines share: the raw-bytes fetch,
 * loading, the "may differ" note, Download / Quote this file, and the
 * download-card fallback for over-ceiling, unreadable or crashing documents.
 *
 * Navigation: engines publish plain state through a DocumentController. By
 * default this component keeps its own controller and renders DocumentNav in
 * its toolbar row. A host that shows navigation elsewhere (a header) creates
 * the controller with `useDocumentController()`, passes it in, and sets
 * `showToolbar={false}`.
 */
import {
  Component,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useState,
  type ComponentType,
  type ErrorInfo,
  type LazyExoticComponent,
  type ReactNode,
} from 'react';
import { Download, File } from 'lucide-react';
import { ApiError } from '../../config';
import { useFileBytes } from '../../hooks/useFiles';
import { useT } from '../../i18n/useT';
import type { FileContent } from '../../types';
import { useDocumentController, type DocumentController } from './documentController';
import DocumentNav from './DocumentNav';
import { ENGINE_BY_FORMAT, type EngineKind, type EngineProps, type PreviewFailure } from './types';

const ENGINES: Record<EngineKind, LazyExoticComponent<ComponentType<EngineProps>>> = {
  pdf: lazy(() => import('./PdfPreview')),
  sheet: lazy(() => import('./SheetPreview')),
  docx: lazy(() => import('./DocxPreview')),
  doc: lazy(() => import('./DocPreview')),
};

export interface DocumentPreviewProps {
  /** A `type: 'document'` content envelope. */
  fileContent: FileContent;
  /**
   * Host-owned controller from `useDocumentController()`. The engine publishes
   * its navigation (`controller.nav`) and this component the download action
   * (`controller.download`) there. Omit it and an internal one is used.
   */
  controller?: DocumentController;
  /** Render the built-in navigation + actions row (default true). */
  showToolbar?: boolean;
  /**
   * Render the "may differ from the original app" note row (default true). A
   * host that states it elsewhere (the panel header's info icon) passes false.
   */
  showNote?: boolean;
  /** Whole-file quote (the right panel passes it; the Files tab does not). */
  onQuoteFile?: () => void;
}

/** Turns an engine crash (render throw, failed chunk load) into the fallback. */
class EngineBoundary extends Component<
  { onError: (failure: PreviewFailure) => void; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.warn('[DocumentPreview] engine failed', error, info.componentStack);
    this.props.onError('unreadable');
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function PreviewLoading() {
  const t = useT();
  return (
    <div className="flex items-center justify-center h-full min-h-[160px]" role="status">
      <p className="text-xs text-secondary animate-pulse">{t('filePreview.doc.loading')}</p>
    </div>
  );
}

export default function DocumentPreview({
  fileContent,
  controller,
  showToolbar = true,
  showNote = true,
  onQuoteFile,
}: DocumentPreviewProps) {
  const t = useT();
  const internalController = useDocumentController();
  const { nav, sink } = controller ?? internalController;

  const fileName = fileContent.path.split('/').pop() ?? fileContent.path;
  const format = fileContent.format;
  const engineKind = format ? ENGINE_BY_FORMAT[format] : undefined;
  const tooLarge = fileContent.preview_unavailable === 'too_large';
  const previewUrl = fileContent.preview_url;

  const bytesState = useFileBytes(!tooLarge && engineKind && previewUrl ? previewUrl : null);
  const [engineFailure, setEngineFailure] = useState<PreviewFailure | null>(null);

  const onError = useCallback((failure: PreviewFailure) => {
    setEngineFailure((prev) => prev ?? failure);
  }, []);

  const saveBytes = useCallback(
    (bytes: ArrayBuffer) => {
      // A download anchor writes a file and never renders it in the app origin.
      const blob = new Blob([bytes], { type: fileContent.mime || 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    },
    [fileContent.mime, fileName],
  );

  const readyBytes = bytesState.status === 'ready' ? bytesState.bytes : null;

  useEffect(() => {
    sink.setDownload(readyBytes ? () => saveBytes(readyBytes) : null);
  }, [sink, readyBytes, saveBytes]);
  useEffect(
    () => () => {
      sink.setDownload(null);
      sink.setNav(null);
    },
    [sink],
  );

  let failure: PreviewFailure | null = null;
  if (tooLarge) failure = 'too_large';
  else if (!engineKind || !format || !previewUrl) failure = 'unreadable';
  else if (bytesState.status === 'error') {
    failure =
      bytesState.error instanceof ApiError && bytesState.error.status === 413
        ? 'too_large'
        : 'unreadable';
  } else failure = engineFailure;

  if (failure || !engineKind || !format) {
    return (
      <div className="flex items-center justify-center h-full p-8" data-testid="document-fallback">
        <div className="bg-sidebar rounded-lg p-6 max-w-sm w-full text-center">
          <File size={48} className="mx-auto text-secondary mb-4" aria-hidden />
          <p className="font-semibold text-sm text-primary mb-1 break-all">{fileName}</p>
          <p className="text-xs text-secondary mb-4">
            {failure === 'too_large'
              ? t('filePreview.doc.tooLarge', { size: formatSize(fileContent.size) })
              : t('filePreview.doc.failed')}
          </p>
          {readyBytes && (
            <button
              type="button"
              onClick={() => saveBytes(readyBytes)}
              className="inline-flex items-center gap-1.5 bg-accent text-white text-sm font-medium rounded-lg px-4 py-2 hover:bg-accent/90 transition-all duration-150"
            >
              <Download size={14} aria-hidden />
              {t('fileExplorer.download')}
            </button>
          )}
          {onQuoteFile && (
            <div>
              <button
                type="button"
                onClick={onQuoteFile}
                className="mt-3 inline-flex items-center gap-1.5 border border-border text-primary text-sm font-medium rounded-lg px-4 py-2 hover:bg-card transition-all duration-150"
              >
                {t('panel.files.quoteFile')}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  const Engine = ENGINES[engineKind];
  return (
    <div className="flex flex-col h-full min-h-0" data-testid="document-preview">
      {showToolbar && (
        // A size container so DocumentNav's `compact` pieces hide when narrow.
        // Wraps rather than squeezes: in the narrow panel the actions drop to
        // a second line and the navigation keeps its width.
        <div className="@container flex flex-wrap items-center gap-x-2 gap-y-1 px-2 py-1 border-b border-border shrink-0 min-h-[34px]">
          <div className="flex items-center gap-2 max-w-full overflow-x-auto" data-testid="document-nav">
            <DocumentNav nav={nav} compact />
          </div>
          <div className="flex items-center gap-3 shrink-0 ml-auto">
            {readyBytes && (
              <button
                type="button"
                onClick={() => saveBytes(readyBytes)}
                className="flex items-center gap-1 text-xs text-secondary hover:text-primary transition-colors"
              >
                <Download size={14} aria-hidden />
                {t('fileExplorer.download')}
              </button>
            )}
            {onQuoteFile && (
              <button
                type="button"
                onClick={onQuoteFile}
                className="text-xs text-secondary hover:text-primary transition-colors"
              >
                {t('panel.files.quoteFile')}
              </button>
            )}
          </div>
        </div>
      )}
      {showNote && (
        <p className="px-3 py-1 text-2xs text-secondary bg-sidebar border-b border-border shrink-0">
          {t('filePreview.doc.mayDiffer')}
        </p>
      )}
      <div className="relative flex-1 min-h-0">
        {readyBytes ? (
          <EngineBoundary onError={onError}>
            <Suspense fallback={<PreviewLoading />}>
              <Engine
                bytes={readyBytes}
                format={format}
                fileName={fileName}
                onNav={sink.setNav}
                onError={onError}
              />
            </Suspense>
          </EngineBoundary>
        ) : (
          <PreviewLoading />
        )}
      </div>
    </div>
  );
}
