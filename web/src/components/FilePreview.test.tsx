// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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

const MD_BODY = '# Notes\n\nsome original body\n';

function mdContent(overrides: Partial<FileContent> = {}): FileContent {
  return {
    path: 'notes.md',
    content: MD_BODY,
    size: MD_BODY.length,
    truncated: false,
    type: 'text',
    ...overrides,
  };
}

describe('FilePreview — markdown editing (spec: editable .md)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows an Edit button for a .md text file when onSave is provided', () => {
    render(
      <FilePreview
        fileContent={mdContent()}
        loading={false}
        selectedPath="notes.md"
        onSave={vi.fn()}
      />,
    );
    expect(screen.getByRole('button', { name: 'Edit' })).toBeInTheDocument();
  });

  it('shows NO Edit button when onSave is omitted', () => {
    render(
      <FilePreview fileContent={mdContent()} loading={false} selectedPath="notes.md" />,
    );
    expect(screen.queryByRole('button', { name: 'Edit' })).toBeNull();
  });

  it('hides Edit and shows a too-large hint for a truncated markdown file', () => {
    render(
      <FilePreview
        fileContent={mdContent({ truncated: true })}
        loading={false}
        selectedPath="notes.md"
        onSave={vi.fn()}
      />,
    );
    expect(screen.queryByRole('button', { name: 'Edit' })).toBeNull();
    expect(screen.getByText(/too large to edit/i)).toBeInTheDocument();
  });

  it('shows no Edit button for a non-.md text file even with onSave', () => {
    render(
      <FilePreview
        fileContent={mdContent({ path: 'notes.txt' })}
        loading={false}
        selectedPath="notes.txt"
        onSave={vi.fn()}
      />,
    );
    expect(screen.queryByRole('button', { name: 'Edit' })).toBeNull();
  });

  it('clicking Edit reveals a textarea bound to the file content', () => {
    render(
      <FilePreview
        fileContent={mdContent()}
        loading={false}
        selectedPath="notes.md"
        onSave={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    const textarea = screen.getByRole('textbox');
    expect(textarea).toBeInTheDocument();
    expect((textarea as HTMLTextAreaElement).value).toBe(MD_BODY);
  });

  it('Save calls onSave with the edited text and the pane then shows the new content', async () => {
    const onSave = vi.fn().mockResolvedValue(true);
    render(
      <FilePreview
        fileContent={mdContent()}
        loading={false}
        selectedPath="notes.md"
        onSave={onSave}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'edited-body-marker' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith('notes.md', 'edited-body-marker'),
    );
    // Back in view mode: the just-saved draft renders through MarkdownContent,
    // even though `fileContent` (a prop) still carries the old body.
    await screen.findByText('edited-body-marker');
    expect(screen.queryByRole('textbox')).toBeNull();
  });

  it('keeps the draft and surfaces a failure message when onSave returns false', async () => {
    const onSave = vi.fn().mockResolvedValue(false);
    render(
      <FilePreview
        fileContent={mdContent()}
        loading={false}
        selectedPath="notes.md"
        onSave={onSave}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: 'unsaved-draft-marker' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));

    await screen.findByText(/couldn't save/i);
    // Still editing, draft intact — edits are not lost on failure.
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
      'unsaved-draft-marker',
    );
  });
});

// ---------------------------------------------------------------------------
// Spec 078 §5.4 — quoting. `quoting` is opt-in: the panel's Files view passes
// it, the full-width Files tab does not, and with it off this pane must behave
// exactly as the tests above already assert.
// ---------------------------------------------------------------------------

/** The first text node under `root` whose content includes `needle`. */
function findTextNode(root: Node, needle: string): Text {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let cur: Node | null;
  while ((cur = walker.nextNode()) !== null) {
    if ((cur.textContent ?? '').includes(needle)) return cur as Text;
  }
  throw new Error(`no text node containing ${JSON.stringify(needle)}`);
}

/** Select `[start, end)` of a text node, the way a drag-select would. */
function selectRange(node: Text, start: number, end: number) {
  const range = document.createRange();
  range.setStart(node, start);
  range.setEnd(node, end);
  const sel = window.getSelection()!;
  sel.removeAllRanges();
  sel.addRange(range);
}

const SOURCE = 'line one\nline two\nline three\nline four\n';

function txtContent(overrides: Partial<FileContent> = {}): FileContent {
  return {
    path: 'notes.txt',
    content: SOURCE,
    size: SOURCE.length,
    truncated: false,
    type: 'text',
    ...overrides,
  };
}

describe('FilePreview — quoting off (spec 078 §5.4)', () => {
  it('renders no quote affordances and no wrapper when `quoting` is absent', () => {
    const { container } = render(
      <FilePreview fileContent={txtContent()} loading={false} selectedPath="notes.txt" />,
    );
    expect(screen.queryByTestId('quote-region')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Quote' })).toBeNull();
    expect(container.querySelector('pre')!.textContent).toBe(SOURCE);
  });

  it('shows no Quote pill on selection when `quoting` is false', () => {
    render(
      <FilePreview
        fileContent={txtContent()}
        loading={false}
        selectedPath="notes.txt"
        quoting={false}
        onQuote={vi.fn()}
      />,
    );
    expect(screen.queryByTestId('quote-region')).toBeNull();
  });

  it('offers no whole-file quote on a binary preview when quoting is off', () => {
    render(
      <FilePreview
        fileContent={{
          path: 'handbook.pdf',
          content: '',
          size: 4096,
          truncated: false,
          type: 'binary',
          mime: 'application/pdf',
        }}
        loading={false}
        selectedPath="handbook.pdf"
      />,
    );
    expect(screen.queryByRole('button', { name: 'Quote this file' })).toBeNull();
  });
});

describe('FilePreview — text-span quoting (source view)', () => {
  afterEach(() => {
    window.getSelection()?.removeAllRanges();
  });

  it('shows a Quote pill on a selection and quotes the verbatim text with its line range', () => {
    const onQuote = vi.fn();
    render(
      <FilePreview
        fileContent={txtContent()}
        loading={false}
        selectedPath="notes.txt"
        quoting
        onQuote={onQuote}
      />,
    );
    const region = screen.getByTestId('quote-region');
    expect(screen.queryByRole('button', { name: 'Quote' })).toBeNull();

    // "line two\nline three" — offsets 9..28 of the source.
    selectRange(findTextNode(region, 'line one'), 9, 28);
    fireEvent.mouseUp(region);

    fireEvent.click(screen.getByRole('button', { name: 'Quote' }));
    expect(onQuote).toHaveBeenCalledWith({
      path: 'notes.txt',
      text: 'line two\nline three',
      lines: [2, 3],
    });
  });

  it('does not count the following line when the selection stops at a newline', () => {
    const onQuote = vi.fn();
    render(
      <FilePreview
        fileContent={txtContent()}
        loading={false}
        selectedPath="notes.txt"
        quoting
        onQuote={onQuote}
      />,
    );
    const region = screen.getByTestId('quote-region');
    // "line two\n" — the trailing newline must not claim line 3.
    selectRange(findTextNode(region, 'line one'), 9, 18);
    fireEvent.mouseUp(region);
    fireEvent.click(screen.getByRole('button', { name: 'Quote' }));
    expect(onQuote.mock.calls[0][0].lines).toEqual([2, 2]);
  });

  it('hides the pill again when the selection is emptied', () => {
    render(
      <FilePreview
        fileContent={txtContent()}
        loading={false}
        selectedPath="notes.txt"
        quoting
        onQuote={vi.fn()}
      />,
    );
    const region = screen.getByTestId('quote-region');
    selectRange(findTextNode(region, 'line one'), 0, 8);
    fireEvent.mouseUp(region);
    expect(screen.getByRole('button', { name: 'Quote' })).toBeInTheDocument();

    window.getSelection()!.removeAllRanges();
    fireEvent.mouseUp(region);
    expect(screen.queryByRole('button', { name: 'Quote' })).toBeNull();
  });

  it('ignores a whitespace-only selection', () => {
    render(
      <FilePreview
        fileContent={txtContent()}
        loading={false}
        selectedPath="notes.txt"
        quoting
        onQuote={vi.fn()}
      />,
    );
    const region = screen.getByTestId('quote-region');
    selectRange(findTextNode(region, 'line one'), 8, 9); // just the newline
    fireEvent.mouseUp(region);
    expect(screen.queryByRole('button', { name: 'Quote' })).toBeNull();
  });

  it('quotes from the html Source view too', () => {
    const onQuote = vi.fn();
    render(
      <FilePreview
        fileContent={htmlContent()}
        loading={false}
        selectedPath="report.html"
        quoting
        onQuote={onQuote}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Source' }));
    const region = screen.getByTestId('quote-region');
    const idx = HTML_BODY.indexOf('<h1>Dashboard</h1>');
    selectRange(findTextNode(region, '<h1>'), idx, idx + '<h1>Dashboard</h1>'.length);
    fireEvent.mouseUp(region);
    fireEvent.click(screen.getByRole('button', { name: 'Quote' }));
    expect(onQuote).toHaveBeenCalledWith({
      path: 'report.html',
      text: '<h1>Dashboard</h1>',
      lines: [1, 1],
    });
  });
});

const MD_QUOTE_SOURCE = '# Notes\n\nalpha unique line\n\nbeta and beta again\n';

describe('FilePreview — text-span quoting (rendered markdown)', () => {
  afterEach(() => {
    window.getSelection()?.removeAllRanges();
  });

  it('resolves the line range when the selected text occurs exactly once', () => {
    const onQuote = vi.fn();
    render(
      <FilePreview
        fileContent={mdContent({ content: MD_QUOTE_SOURCE })}
        loading={false}
        selectedPath="notes.md"
        quoting
        onQuote={onQuote}
      />,
    );
    const region = screen.getByTestId('quote-region');
    const node = findTextNode(region, 'alpha unique line');
    const at = node.textContent!.indexOf('unique line');
    selectRange(node, at, at + 'unique line'.length);
    fireEvent.mouseUp(region);
    fireEvent.click(screen.getByRole('button', { name: 'Quote' }));
    expect(onQuote).toHaveBeenCalledWith({
      path: 'notes.md',
      text: 'unique line',
      lines: [3, 3],
    });
  });

  it('omits the line range when the selected text is ambiguous in the source', () => {
    const onQuote = vi.fn();
    render(
      <FilePreview
        fileContent={mdContent({ content: MD_QUOTE_SOURCE })}
        loading={false}
        selectedPath="notes.md"
        quoting
        onQuote={onQuote}
      />,
    );
    const region = screen.getByTestId('quote-region');
    const node = findTextNode(region, 'beta and beta again');
    selectRange(node, 0, 4); // "beta" — twice in the source
    fireEvent.mouseUp(region);
    fireEvent.click(screen.getByRole('button', { name: 'Quote' }));
    // Verbatim text always; lines only when unique (spec §13 Q4).
    expect(onQuote).toHaveBeenCalledWith({ path: 'notes.md', text: 'beta' });
  });
});

describe('FilePreview — image and binary quoting', () => {
  const image: FileContent = {
    path: 'shots/queue.png',
    content: 'AAAA',
    size: 4,
    truncated: false,
    type: 'image',
    mime: 'image/png',
  };

  it('mounts the annotate overlay over the image and quotes a dragged box', () => {
    const onQuote = vi.fn();
    render(
      <FilePreview
        fileContent={image}
        loading={false}
        selectedPath="shots/queue.png"
        quoting
        onQuote={onQuote}
      />,
    );
    const overlay = screen.getByTestId('annotate-overlay');
    expect(overlay).toHaveAttribute('data-active', 'true');

    fireEvent.pointerDown(overlay, { clientX: 10, clientY: 20, pointerId: 1 });
    fireEvent.pointerMove(overlay, { clientX: 70, clientY: 80, pointerId: 1 });
    fireEvent.pointerUp(overlay, { clientX: 70, clientY: 80, pointerId: 1 });
    fireEvent.change(screen.getByTestId('annotate-note'), { target: { value: 'this button' } });
    fireEvent.keyDown(screen.getByTestId('annotate-note'), { key: 'Enter' });

    expect(onQuote).toHaveBeenCalledWith({
      path: 'shots/queue.png',
      box: { x: 10, y: 20, w: 60, h: 60 },
      imageDataUrl: 'data:image/png;base64,AAAA',
    });
  });

  it('mounts no overlay when quoting is off', () => {
    render(
      <FilePreview fileContent={image} loading={false} selectedPath="shots/queue.png" />,
    );
    expect(screen.queryByTestId('annotate-overlay')).toBeNull();
  });

  it('offers a whole-file quote for a preview with no renderer', () => {
    const onQuote = vi.fn();
    render(
      <FilePreview
        fileContent={{
          path: 'docs/handbook.pdf',
          content: '',
          size: 4096,
          truncated: false,
          type: 'binary',
          mime: 'application/pdf',
        }}
        loading={false}
        selectedPath="docs/handbook.pdf"
        quoting
        onQuote={onQuote}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Quote this file' }));
    expect(onQuote).toHaveBeenCalledWith({ path: 'docs/handbook.pdf' });
  });
});
