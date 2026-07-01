// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import FilePreview from './FilePreview';
import type { FileContent } from '../types';

const HTML_BODY =
  '<!doctype html><html><body><h1>Dashboard</h1><script>1+1</script></body></html>';

function htmlContent(overrides: Partial<FileContent> = {}): FileContent {
  return {
    path: 'report.html',
    content: HTML_BODY,
    size: HTML_BODY.length,
    truncated: false,
    type: 'html',
    mime: 'text/html',
    ...overrides,
  };
}

describe('FilePreview — html branch (spec 003)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders a sandboxed iframe with scripts but NO same-origin', () => {
    const { container } = render(
      <FilePreview fileContent={htmlContent()} loading={false} selectedPath="report.html" />,
    );
    const iframe = container.querySelector('iframe');
    expect(iframe).not.toBeNull();
    // Scripts enabled (Option A), but opaque origin: same-origin must be absent.
    const sandbox = iframe!.getAttribute('sandbox') ?? '';
    expect(sandbox.split(/\s+/)).toContain('allow-scripts');
    expect(sandbox).not.toContain('allow-same-origin');
    // srcDoc carries the HTML; we never use dangerouslySetInnerHTML — so the
    // markup lives in the attribute string, NOT as live nodes in the parent DOM.
    expect(iframe!.getAttribute('srcdoc')).toBe(HTML_BODY);
    expect(container.querySelector('h1')).toBeNull();
    expect(container.querySelector('script')).toBeNull();
  });

  it('defaults to Rendered and switches to a <pre> source view on toggle', () => {
    const { container } = render(
      <FilePreview fileContent={htmlContent()} loading={false} selectedPath="report.html" />,
    );
    // Default = Rendered: iframe present, no <pre>.
    expect(container.querySelector('iframe')).not.toBeNull();
    expect(container.querySelector('pre')).toBeNull();

    // Toggle to Source.
    fireEvent.click(screen.getByRole('button', { name: 'Source' }));
    const pre = container.querySelector('pre');
    expect(pre).not.toBeNull();
    expect(pre!.textContent).toBe(HTML_BODY);
    expect(container.querySelector('iframe')).toBeNull();
  });

  it('Download saves the file via a download anchor — NOT a same-origin navigation', () => {
    // blob: URLs are SAME-ORIGIN with the app, so window.open of one would run
    // agent <script> with access to the app's localStorage relay JWT. The
    // escape hatch must be a download (writes a file, never executes), never a
    // window.open. Guard both: the anchor carries `download`, and window.open
    // is never called.
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    const createUrlSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:mock-url');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    render(
      <FilePreview fileContent={htmlContent()} loading={false} selectedPath="report.html" />,
    );
    expect(screen.queryByRole('button', { name: 'Open in new tab' })).toBeNull();

    // Capture the anchor the download handler creates (spy set up AFTER render
    // so only the handler's createElement('a') is intercepted).
    const realCreate = document.createElement.bind(document);
    let anchor: HTMLAnchorElement | undefined;
    vi.spyOn(document, 'createElement').mockImplementation(((tag: string) => {
      const el = realCreate(tag);
      if (tag === 'a') {
        anchor = el as HTMLAnchorElement;
        anchor.click = vi.fn(); // don't let jsdom attempt a navigation
      }
      return el;
    }) as typeof document.createElement);

    fireEvent.click(screen.getByRole('button', { name: 'Download' }));

    expect(anchor).toBeDefined();
    expect(anchor!.getAttribute('download')).toBe('report.html');
    expect(anchor!.click).toHaveBeenCalledTimes(1);
    // The downloaded Blob is typed text/html, and we never window.open it.
    const blobArg = createUrlSpy.mock.calls[0][0] as Blob;
    expect(blobArg.type).toBe('text/html');
    expect(openSpy).not.toHaveBeenCalled();
  });

  it('does NOT render an iframe for non-html types (svg stays an image)', () => {
    const svg: FileContent = {
      path: 'logo.svg',
      content: 'PHN2Zz48L3N2Zz4=', // base64, image branch
      size: 10,
      truncated: false,
      type: 'image',
      mime: 'image/svg+xml',
    };
    const { container } = render(
      <FilePreview fileContent={svg} loading={false} selectedPath="logo.svg" />,
    );
    expect(container.querySelector('iframe')).toBeNull();
    expect(container.querySelector('img')).not.toBeNull();
  });

  it('shows the truncation banner for oversized html', () => {
    render(
      <FilePreview
        fileContent={htmlContent({ truncated: true })}
        loading={false}
        selectedPath="report.html"
      />,
    );
    expect(screen.getByText(/truncated/i)).toBeInTheDocument();
  });
});
