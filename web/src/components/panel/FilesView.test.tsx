// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 078 §5.3 (D10) / §10 — the Files view: tree with touched badges and a
 * pinned "Touched this session" group, swapping to the preview on click, and
 * the three annotation forms a quote can take.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { DirectoryListing, FileContent } from '../../types';
import type { AnnotationDraft } from '../../utils/annotations';
import type { TouchedFile } from '../../utils/panelSelectors';

// ---------------------------------------------------------------------------
// Mocks — the workspace tree / content routes, and FilePreview (a heavy
// renderer whose only role here is to hand a quote back up).
// ---------------------------------------------------------------------------

const listDirectory = vi.fn<(projectId: string, path?: string) => Promise<DirectoryListing | null>>();
const getFileContent = vi.fn<(projectId: string, path: string) => Promise<FileContent | null>>();
const resolvePath = vi.fn<(projectId: string, path: string) => Promise<string[] | null>>();

vi.mock('../../hooks/useFiles', () => ({
  useFiles: () => ({
    directory: null,
    fileContent: null,
    loading: false,
    error: null,
    listDirectory,
    getFileContent,
    resolvePath,
    saveFileContent: vi.fn(),
  }),
}));

let lastFilePreviewProps: Record<string, unknown> = {};
vi.mock('../FilePreview', () => ({
  default: (props: Record<string, unknown>) => {
    lastFilePreviewProps = props;
    return <div data-testid="file-preview">{String(props.selectedPath ?? '')}</div>;
  },
}));

import FilesView from './FilesView';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TREE: Record<string, DirectoryListing> = {
  '': {
    path: '',
    entries: [
      { name: 'src', type: 'directory' },
      { name: 'README.md', type: 'file', size: 10 },
    ],
  },
  src: {
    path: 'src',
    entries: [
      { name: 'deep', type: 'directory' },
      { name: 'app.ts', type: 'file', size: 20 },
    ],
  },
  'src/deep': { path: 'src/deep', entries: [{ name: 'x.ts', type: 'file', size: 5 }] },
};

function renderView(overrides: Partial<React.ComponentProps<typeof FilesView>> = {}) {
  const props: React.ComponentProps<typeof FilesView> = {
    projectId: 'proj-1',
    touched: [],
    file: null,
    onSelectFile: vi.fn(),
    onOpenInFiles: vi.fn(),
    onAddAnnotation: vi.fn(),
    ...overrides,
  };
  const utils = render(<FilesView {...props} />);
  return { ...utils, props };
}

/** Flush the fetch promises the tree/preview effects kick off. */
async function settle() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(() => {
  listDirectory.mockReset();
  getFileContent.mockReset();
  resolvePath.mockReset();
  lastFilePreviewProps = {};
  listDirectory.mockImplementation(async (_projectId, path) => TREE[path ?? ''] ?? null);
  getFileContent.mockImplementation(async (_projectId, path) => ({
    path,
    content: 'hello',
    size: 5,
    truncated: false,
    type: 'text',
  }));
  resolvePath.mockResolvedValue([]);
});
afterEach(() => cleanup());

// ---------------------------------------------------------------------------
// Tree state
// ---------------------------------------------------------------------------

describe('FilesView — tree state', () => {
  it('lists the workspace root', async () => {
    renderView();
    await settle();
    expect(listDirectory).toHaveBeenCalledWith('proj-1', undefined);
    expect(screen.getByText('README.md')).toBeInTheDocument();
    expect(screen.getByText('src')).toBeInTheDocument();
  });

  it('lazily expands a directory on click and collapses it again', async () => {
    renderView();
    await settle();
    expect(screen.queryByText('app.ts')).toBeNull();

    fireEvent.click(screen.getByText('src'));
    await settle();
    expect(listDirectory).toHaveBeenCalledWith('proj-1', 'src');
    expect(screen.getByText('app.ts')).toBeInTheDocument();

    fireEvent.click(screen.getByText('src'));
    await settle();
    expect(screen.queryByText('app.ts')).toBeNull();
  });

  it('badges touched files in the tree with read / edited / written', async () => {
    const touched: TouchedFile[] = [
      { path: 'README.md', op: 'written' },
      { path: 'src/app.ts', op: 'edited' },
      { path: 'src/deep/x.ts', op: 'read' },
    ];
    renderView({ touched });
    await settle();
    // Every touched file's folders are auto-expanded, so all three are visible.
    expect(screen.getByText('x.ts')).toBeInTheDocument();
    expect(screen.getAllByText('written')).not.toHaveLength(0);
    expect(screen.getAllByText('edited')).not.toHaveLength(0);
    expect(screen.getAllByText('read')).not.toHaveLength(0);
  });

  it('auto-expands every ancestor of a touched file', async () => {
    renderView({ touched: [{ path: 'src/deep/x.ts', op: 'read' }] });
    await settle();
    expect(listDirectory).toHaveBeenCalledWith('proj-1', 'src');
    expect(listDirectory).toHaveBeenCalledWith('proj-1', 'src/deep');
    expect(screen.getByText('x.ts')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Touched group
// ---------------------------------------------------------------------------

describe('FilesView — "Touched this session" group', () => {
  it('is absent when nothing was touched', async () => {
    renderView({ touched: [] });
    await settle();
    expect(screen.queryByTestId('panel-touched-group')).toBeNull();
  });

  it('uses the singular string for one file and the counted string for more', async () => {
    const one = renderView({ touched: [{ path: 'README.md', op: 'read' }] });
    await settle();
    expect(screen.getByTestId('panel-touched-group')).toHaveTextContent('Touched this session (1)');
    one.unmount();

    renderView({
      touched: [
        { path: 'README.md', op: 'read' },
        { path: 'src/app.ts', op: 'edited' },
      ],
    });
    await settle();
    expect(screen.getByTestId('panel-touched-group')).toHaveTextContent('Touched this session (2)');
  });

  it('collapses and re-expands on click', async () => {
    renderView({ touched: [{ path: 'src/app.ts', op: 'edited' }] });
    await settle();
    const group = screen.getByTestId('panel-touched-group');
    expect(screen.getByText('src/app.ts')).toBeInTheDocument();

    fireEvent.click(group);
    expect(screen.queryByText('src/app.ts')).toBeNull();
    expect(group).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(group);
    expect(screen.getByText('src/app.ts')).toBeInTheDocument();
  });

  it('selecting a row in the group opens that file', async () => {
    const { props } = renderView({ touched: [{ path: 'src/app.ts', op: 'edited' }] });
    await settle();
    fireEvent.click(screen.getByText('src/app.ts'));
    expect(props.onSelectFile).toHaveBeenCalledWith('src/app.ts');
  });
});

// ---------------------------------------------------------------------------
// Preview state
// ---------------------------------------------------------------------------

describe('FilesView — preview state', () => {
  it('clicking a file in the tree asks the parent to select it', async () => {
    const { props } = renderView();
    await settle();
    fireEvent.click(screen.getByText('README.md'));
    expect(props.onSelectFile).toHaveBeenCalledWith('README.md');
  });

  it('replaces the tree with the preview — never both at once (D10)', async () => {
    renderView({ file: 'README.md' });
    await settle();
    expect(screen.getByTestId('files-view-preview')).toBeInTheDocument();
    expect(screen.queryByTestId('files-view-tree')).toBeNull();
    expect(screen.getByTestId('file-preview')).toHaveTextContent('README.md');
    expect(getFileContent).toHaveBeenCalledWith('proj-1', 'README.md');
  });

  it('falls back to the resolve endpoint for an abbreviated path', async () => {
    getFileContent.mockImplementation(async (_projectId, path) =>
      path === 'src/app.ts'
        ? { path, content: 'x', size: 1, truncated: false, type: 'text' }
        : null,
    );
    resolvePath.mockResolvedValue(['src/app.ts']);

    renderView({ file: 'app.ts' });
    await settle();
    expect(resolvePath).toHaveBeenCalledWith('proj-1', 'app.ts');
    expect(screen.getByTestId('file-preview')).toHaveTextContent('src/app.ts');
  });

  it('"‹ Files" returns to the tree by clearing the selection', async () => {
    const { props } = renderView({ file: 'README.md' });
    await settle();
    fireEvent.click(screen.getByRole('button', { name: 'Files' }));
    expect(props.onSelectFile).toHaveBeenCalledWith(null);
  });

  it('"Open in Files" hands the resolved path to the parent', async () => {
    const { props } = renderView({ file: 'README.md' });
    await settle();
    fireEvent.click(screen.getByRole('button', { name: 'Open in Files' }));
    expect(props.onOpenInFiles).toHaveBeenCalledWith('README.md');
  });

  it('always enables quoting in the preview — Files has no Annotate mode', async () => {
    renderView({ file: 'README.md' });
    await settle();
    expect(lastFilePreviewProps.quoting).toBe(true);
    expect(typeof lastFilePreviewProps.onQuote).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Quote → annotation mapping (§5.4)
// ---------------------------------------------------------------------------

describe('FilesView — onQuote maps to the three annotation forms', () => {
  async function quote(payload: Record<string, unknown>): Promise<AnnotationDraft> {
    const onAddAnnotation = vi.fn();
    renderView({ file: 'README.md', onAddAnnotation });
    await settle();
    act(() => {
      (lastFilePreviewProps.onQuote as (q: Record<string, unknown>) => void)(payload);
    });
    expect(onAddAnnotation).toHaveBeenCalledTimes(1);
    return onAddAnnotation.mock.calls[0][0] as AnnotationDraft;
  }

  it('text present → a text-span quote carrying the verbatim text and line range', async () => {
    expect(
      await quote({ path: 'README.md', text: 'const x = 1', lines: [14, 17] }),
    ).toEqual({ kind: 'text', path: 'README.md', text: 'const x = 1', lines: [14, 17], note: '' });
  });

  it('text present without lines → still a text quote (lines are secondary)', async () => {
    expect(await quote({ path: 'README.md', text: 'hello' })).toEqual({
      kind: 'text',
      path: 'README.md',
      text: 'hello',
      lines: undefined,
      note: '',
    });
  });

  it('box present → an image-region quote with the captured image', async () => {
    expect(
      await quote({
        path: 'shot.png',
        box: { x: 1, y: 2, w: 3, h: 4 },
        imageDataUrl: 'data:image/png;base64,AAA',
      }),
    ).toEqual({
      kind: 'image',
      path: 'shot.png',
      box: { x: 1, y: 2, w: 3, h: 4 },
      note: '',
      imageDataUrl: 'data:image/png;base64,AAA',
    });
  });

  it('neither → the whole-file fallback every file type can use', async () => {
    expect(await quote({ path: 'report.pdf' })).toEqual({
      kind: 'file',
      path: 'report.pdf',
      note: '',
    });
  });
});
