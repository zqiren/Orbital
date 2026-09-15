// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '../../config';
import type { FileBytesState } from '../../hooks/useFiles';
import type { DocumentFormat, FileContent } from '../../types';
import { useDocumentController } from './documentController';
import DocumentNav from './DocumentNav';
import DocumentPreview from './DocumentPreview';
import type { EngineProps } from './types';

// ---------------------------------------------------------------------------
// Mocks — the bytes hook, and one stub per engine so routing, navigation
// publishing and failure handling are observable without the real libraries.
// ---------------------------------------------------------------------------

const bytesFor = vi.fn<(url: string | null) => FileBytesState>();
vi.mock('../../hooks/useFiles', () => ({
  useFileBytes: (url: string | null) => bytesFor(url),
}));

const goTo = vi.fn();
vi.mock('./PdfPreview', async () => {
  const { useEffect } = await import('react');
  function MockPdfPreview({ onNav }: EngineProps) {
    useEffect(() => {
      onNav({
        kind: 'pdf',
        page: 1,
        pageCount: 3,
        zoomPercent: 100,
        fit: true,
        goTo,
        zoomIn: () => {},
        zoomOut: () => {},
        fitWidth: () => {},
      });
      return () => onNav(null);
    }, [onNav]);
    return <div data-testid="engine-pdf" />;
  }
  return { default: MockPdfPreview };
});
vi.mock('./SheetPreview', () => ({
  default: (props: EngineProps) => {
    if (props.fileName === 'explode.xlsx') throw new Error('engine crashed');
    return <div data-testid="engine-sheet">{props.format}</div>;
  },
}));
vi.mock('./DocxPreview', () => ({
  default: (props: EngineProps) => (
    <button type="button" data-testid="engine-docx" onClick={() => props.onError('unreadable')}>
      fail
    </button>
  ),
}));
vi.mock('./DocPreview', async () => {
  const { useEffect } = await import('react');
  function MockDocPreview({ fileName, onError }: EngineProps) {
    // The real engine rejects a corrupt Compound File as soon as it parses.
    useEffect(() => {
      if (fileName === 'corrupt.doc') onError('unreadable');
    }, [fileName, onError]);
    return <div data-testid="engine-doc" />;
  }
  return { default: MockDocPreview };
});

const READY: FileBytesState = { status: 'ready', bytes: new Uint8Array([1, 2, 3]).buffer, error: null };

function doc(format: DocumentFormat, overrides: Partial<FileContent> = {}): FileContent {
  return {
    path: `docs/file.${format}`,
    content: '',
    size: 2048,
    truncated: false,
    type: 'document',
    format,
    mime: 'application/octet-stream',
    preview_url: `/api/v2/projects/p1/files/preview?path=docs/file.${format}`,
    download_url: `/api/v2/projects/p1/files/download?path=docs/file.${format}`,
    ...overrides,
  };
}

function spyDownloads() {
  const created = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  const clicked: HTMLAnchorElement[] = [];
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    clicked.push(this);
  });
  return { created, clicked };
}

beforeEach(() => {
  bytesFor.mockReset();
  bytesFor.mockReturnValue(READY);
  goTo.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('DocumentPreview (spec 090)', () => {
  it.each([
    ['pdf', 'engine-pdf'],
    ['xlsx', 'engine-sheet'],
    ['xls', 'engine-sheet'],
    ['csv', 'engine-sheet'],
    ['docx', 'engine-docx'],
    ['doc', 'engine-doc'],
  ] as const)('routes %s to its engine', async (format, testId) => {
    render(<DocumentPreview fileContent={doc(format)} />);
    expect(await screen.findByTestId(testId)).toBeInTheDocument();
    expect(bytesFor).toHaveBeenCalledWith(`/api/v2/projects/p1/files/preview?path=docs/file.${format}`);
    expect(screen.getByText('Preview may differ from the original app')).toBeInTheDocument();
  });

  it('shows loading until the bytes arrive, without mounting an engine', () => {
    bytesFor.mockReturnValue({ status: 'loading', bytes: null, error: null });
    render(<DocumentPreview fileContent={doc('pdf')} />);
    expect(screen.getByText('Loading preview…')).toBeInTheDocument();
    expect(screen.queryByTestId('engine-pdf')).toBeNull();
  });

  it('renders the engine’s published navigation in its own toolbar by default', async () => {
    render(<DocumentPreview fileContent={doc('pdf')} />);
    const nav = screen.getByTestId('document-nav');
    expect(await within(nav).findByText('of 3')).toBeInTheDocument();
    fireEvent.click(within(nav).getByRole('button', { name: 'Next page' }));
    expect(goTo).toHaveBeenCalledWith(2);
  });

  it('a host controller receives navigation + download, and the built-in toolbar can be hidden', async () => {
    const { created } = spyDownloads();
    function Host({ fileContent }: { fileContent: FileContent }) {
      const controller = useDocumentController();
      return (
        <>
          <div data-testid="host-header">
            <DocumentNav nav={controller.nav} compact />
            {controller.download && (
              <button type="button" onClick={controller.download}>
                Host download
              </button>
            )}
          </div>
          <DocumentPreview fileContent={fileContent} controller={controller} showToolbar={false} />
        </>
      );
    }
    const { unmount } = render(<Host fileContent={doc('pdf')} />);
    const header = screen.getByTestId('host-header');
    expect(await within(header).findByText('of 3')).toBeInTheDocument();
    expect(screen.queryByTestId('document-nav')).toBeNull();
    fireEvent.click(within(header).getByRole('button', { name: 'Host download' }));
    expect(created).toHaveBeenCalledTimes(1);
    unmount();
  });

  it.each(['pdf', 'doc'] as const)(
    'over the ceiling (%s): download card, and the bytes are never fetched',
    (format) => {
      render(<DocumentPreview fileContent={doc(format, { preview_unavailable: 'too_large' })} />);
      expect(screen.getByTestId('document-fallback')).toBeInTheDocument();
      expect(screen.getByText('Too large to preview (2.0 KB). The limit is 50 MB.')).toBeInTheDocument();
      expect(bytesFor).toHaveBeenCalledWith(null);
      expect(screen.queryByTestId(`engine-${format}`)).toBeNull();
    },
  );

  it('a corrupt .doc falls back to the download card with the fetched bytes', async () => {
    const { clicked } = spyDownloads();
    render(<DocumentPreview fileContent={doc('doc', { path: 'forms/corrupt.doc' })} />);
    expect(await screen.findByTestId('document-fallback')).toBeInTheDocument();
    expect(screen.getByText("This file couldn't be previewed.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download' }));
    expect(clicked[0].getAttribute('download')).toBe('corrupt.doc');
  });

  it('a 413 from the bytes route (file grew) is also the too-large card', () => {
    bytesFor.mockReturnValue({
      status: 'error',
      bytes: null,
      error: new ApiError(413, 'preview_unavailable: too_large'),
    });
    render(<DocumentPreview fileContent={doc('pdf')} />);
    expect(screen.getByText('Too large to preview (2.0 KB). The limit is 50 MB.')).toBeInTheDocument();
  });

  it('a failed bytes fetch falls back to the card', () => {
    bytesFor.mockReturnValue({ status: 'error', bytes: null, error: new Error('network down') });
    render(<DocumentPreview fileContent={doc('docx')} />);
    expect(screen.getByText("This file couldn't be previewed.")).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Download' })).toBeNull();
  });

  it('an engine error swaps in the download card, and Download saves the fetched bytes', async () => {
    const { created, clicked } = spyDownloads();
    render(<DocumentPreview fileContent={doc('docx')} />);
    fireEvent.click(await screen.findByTestId('engine-docx'));

    expect(screen.getByTestId('document-fallback')).toBeInTheDocument();
    expect(screen.getByText("This file couldn't be previewed.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download' }));
    expect(created).toHaveBeenCalledTimes(1);
    expect(clicked).toHaveLength(1);
    expect(clicked[0].getAttribute('download')).toBe('file.docx');
  });

  it('an engine that throws while rendering falls back instead of breaking the pane', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    render(<DocumentPreview fileContent={doc('xlsx', { path: 'explode.xlsx' })} />);
    expect(await screen.findByTestId('document-fallback')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Download' })).toBeInTheDocument();
  });

  it('a document without a known format never fetches', () => {
    render(<DocumentPreview fileContent={doc('pdf', { format: undefined })} />);
    expect(screen.getByTestId('document-fallback')).toBeInTheDocument();
    expect(bytesFor).toHaveBeenCalledWith(null);
  });

  it('Quote this file quotes the whole file, in the preview and in the fallback', async () => {
    const onQuoteFile = vi.fn();
    const { unmount } = render(<DocumentPreview fileContent={doc('pdf')} onQuoteFile={onQuoteFile} />);
    await screen.findByTestId('engine-pdf');
    fireEvent.click(screen.getByRole('button', { name: 'Quote this file' }));
    expect(onQuoteFile).toHaveBeenCalledTimes(1);
    unmount();

    render(
      <DocumentPreview
        fileContent={doc('pdf', { preview_unavailable: 'too_large' })}
        onQuoteFile={onQuoteFile}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Quote this file' }));
    expect(onQuoteFile).toHaveBeenCalledTimes(2);
  });
});
