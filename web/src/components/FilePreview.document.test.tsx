// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import FilePreview from './FilePreview';
import type { FileContent } from '../types';

// The bytes stay loading: what's under test is how the header, toolbar and
// note compose around a document, not the engines themselves.
vi.mock('../hooks/useFiles', () => ({
  useFileBytes: () => ({ status: 'loading', bytes: null, error: null }),
}));

const MAY_DIFFER = 'Preview may differ from the original app';

const pdf: FileContent = {
  path: 'docs/report.pdf',
  content: '',
  size: 2048,
  truncated: false,
  type: 'document',
  format: 'pdf',
  mime: 'application/pdf',
  preview_url: '/api/v2/projects/p/files/preview?path=docs/report.pdf',
  download_url: '/api/v2/projects/p/files/download?path=docs/report.pdf',
};

describe('FilePreview — documents and the panel header (spec 088 × 090)', () => {
  it('Files tab keeps the document toolbar and the may-differ note row', () => {
    render(<FilePreview fileContent={pdf} loading={false} selectedPath="docs/report.pdf" />);

    expect(screen.getByTestId('document-preview')).not.toBeNull();
    expect(screen.getByTestId('document-nav')).not.toBeNull();
    expect(screen.getByText(MAY_DIFFER)).not.toBeNull();
    expect(screen.queryByTestId('file-preview-panel-header')).toBeNull();
  });

  it('panel mode folds navigation, the note and the file actions into one header row', () => {
    const onBack = vi.fn();
    const onOpenInFiles = vi.fn();
    const onQuote = vi.fn();
    render(
      <FilePreview
        fileContent={pdf}
        loading={false}
        selectedPath="docs/report.pdf"
        quoting
        onQuote={onQuote}
        panelHeader={{ onBack, onOpenInFiles }}
      />,
    );

    expect(screen.getByTestId('file-preview-panel-header')).not.toBeNull();
    // No second toolbar row and no note row above the document.
    expect(screen.queryByTestId('document-nav')).toBeNull();
    expect(screen.queryByText(MAY_DIFFER)).toBeNull();
    // The note survives as the header's labelled info icon.
    expect(screen.getByRole('img', { name: MAY_DIFFER })).not.toBeNull();

    // Download is absent while the bytes are still loading; the others are in More.
    fireEvent.click(screen.getByRole('button', { name: 'More actions' }));
    expect(screen.queryByRole('menuitem', { name: 'Download' })).toBeNull();
    fireEvent.click(screen.getByRole('menuitem', { name: 'Quote this file' }));
    expect(onQuote).toHaveBeenCalledWith({ path: 'docs/report.pdf' });

    fireEvent.click(screen.getByRole('button', { name: 'More actions' }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Open in Files' }));
    expect(onOpenInFiles).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('button', { name: 'Back to files' }));
    expect(onBack).toHaveBeenCalledTimes(1);
  });
});

describe('FilePreview — documents offer the reveal (spec 093)', () => {
  it('Files tab header button and panel More item both reveal the document', () => {
    Object.defineProperty(window.navigator, 'platform', { value: 'MacIntel', configurable: true });
    try {
      const onReveal = vi.fn();
      const { unmount } = render(
        <FilePreview fileContent={pdf} loading={false} selectedPath="docs/report.pdf" onReveal={onReveal} />,
      );
      fireEvent.click(screen.getByRole('button', { name: 'Reveal in Finder' }));
      expect(onReveal).toHaveBeenCalledWith('docs/report.pdf');
      unmount();

      const onRevealPanel = vi.fn();
      render(
        <FilePreview
          fileContent={pdf}
          loading={false}
          selectedPath="docs/report.pdf"
          panelHeader={{ onBack: vi.fn(), onOpenInFiles: vi.fn() }}
          onReveal={onRevealPanel}
        />,
      );
      fireEvent.click(screen.getByRole('button', { name: 'More actions' }));
      const items = screen.getAllByRole('menuitem').map((i) => i.textContent);
      expect(items.slice(-2)).toEqual(['Open in Files', 'Reveal in Finder']);
      fireEvent.click(screen.getByRole('menuitem', { name: 'Reveal in Finder' }));
      expect(onRevealPanel).toHaveBeenCalledWith('docs/report.pdf');
    } finally {
      delete (window.navigator as unknown as { platform?: string }).platform;
    }
  });
});
