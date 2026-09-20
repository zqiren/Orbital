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
const revealPath = vi.fn<(projectId: string, path: string) => Promise<boolean>>();

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
    revealPath,
  }),
}));

// The live event stream that drives the panel's refresh. One handler per type
// is enough here; `emit` plays a daemon broadcast.
const wsHandlers = vi.hoisted(() => new Map<string, (event: unknown) => void>());
vi.mock('../../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    connectionState: 'connected',
    subscribe: vi.fn(),
    on: (type: string, fn: (event: unknown) => void) => wsHandlers.set(type, fn),
    off: (type: string) => wsHandlers.delete(type),
  }),
}));

// Spec 093: whether this client can reveal in Finder / File Explorer.
const reveal = vi.hoisted(() => ({ can: true }));
vi.mock('../../utils/clientPlatform', () => ({
  canRevealInFileManager: () => reveal.can,
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
  revealPath.mockReset();
  revealPath.mockResolvedValue(true);
  reveal.can = true;
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

  it('paints no navigation row of its own — Back and More live in the preview header (spec 088)', async () => {
    renderView({ file: 'README.md' });
    await settle();
    // FilePreview is mocked to a bare div, so any button here would be a
    // FilesView-owned bar stacked above the file header.
    expect(screen.queryAllByRole('button')).toHaveLength(0);
    expect(lastFilePreviewProps.panelHeader).toBeDefined();
  });

  it('the header’s Back returns to the tree by clearing the selection', async () => {
    const { props } = renderView({ file: 'README.md' });
    await settle();
    act(() => {
      (lastFilePreviewProps.panelHeader as { onBack: () => void }).onBack();
    });
    expect(props.onSelectFile).toHaveBeenCalledWith(null);
  });

  it('the header’s "Open in Files" hands the resolved path to the parent', async () => {
    getFileContent.mockImplementation(async (_projectId, path) =>
      path === 'src/app.ts'
        ? { path, content: 'x', size: 1, truncated: false, type: 'text' }
        : null,
    );
    resolvePath.mockResolvedValue(['src/app.ts']);
    const { props } = renderView({ file: 'app.ts' });
    await settle();
    act(() => {
      (lastFilePreviewProps.panelHeader as { onOpenInFiles: () => void }).onOpenInFiles();
    });
    expect(props.onOpenInFiles).toHaveBeenCalledWith('src/app.ts');
  });

  it('keeps the header (and so Back) while the file is still loading', async () => {
    getFileContent.mockImplementation(() => new Promise(() => {}));
    renderView({ file: 'README.md' });
    await settle();
    expect(lastFilePreviewProps.loading).toBe(true);
    expect(lastFilePreviewProps.selectedPath).toBe('README.md');
    expect(lastFilePreviewProps.panelHeader).toBeDefined();
  });

  it('always enables quoting in the preview — Files has no Annotate mode', async () => {
    renderView({ file: 'README.md' });
    await settle();
    expect(lastFilePreviewProps.quoting).toBe(true);
    expect(typeof lastFilePreviewProps.onQuote).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Reveal in Finder / File Explorer (spec 093)
// ---------------------------------------------------------------------------

describe('FilesView — reveal in the file manager (spec 093)', () => {
  it('hands the preview an onReveal that reveals the resolved path', async () => {
    getFileContent.mockImplementation(async (_projectId, path) =>
      path === 'src/app.ts'
        ? { path, content: 'x', size: 1, truncated: false, type: 'text' }
        : null,
    );
    resolvePath.mockResolvedValue(['src/app.ts']);
    renderView({ file: 'app.ts' });
    await settle();
    const onReveal = lastFilePreviewProps.onReveal as (path: string) => void;
    expect(typeof onReveal).toBe('function');
    act(() => onReveal('src/app.ts'));
    expect(revealPath).toHaveBeenCalledWith('proj-1', 'src/app.ts');
  });

  it('offers nothing where the reveal cannot work (relay, a phone)', async () => {
    reveal.can = false;
    renderView({ file: 'README.md' });
    await settle();
    expect(lastFilePreviewProps.onReveal).toBeUndefined();
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

// ---------------------------------------------------------------------------
// Live refresh — the panel is how a user WATCHES the agent work. It used to
// read each directory once and each file once per selection, and it opens a
// file on the tool CALL event (before the write has run), so what it showed
// was routinely the pre-edit snapshot, for good.
// ---------------------------------------------------------------------------

describe('FilesView — live refresh', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  async function emit(type: string, event: Record<string, unknown>) {
    await act(async () => {
      wsHandlers.get(type)?.({ type, project_id: 'proj-1', ...event });
      await vi.advanceTimersByTimeAsync(300);
    });
  }

  it('re-reads the open file when a tool result lands, and shows the new content', async () => {
    renderView({ file: 'plan.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect((lastFilePreviewProps.fileContent as FileContent).content).toBe('hello');

    getFileContent.mockImplementation(async (_projectId, path) => ({
      path, content: 'hello, edited', size: 13, truncated: false, type: 'text', revision: 'r2',
    }));
    await emit('agent.activity', { category: 'tool_result' });

    expect((lastFilePreviewProps.fileContent as FileContent).content).toBe('hello, edited');
    // Silent: a refresh never drops the view back to a loading state.
    expect(lastFilePreviewProps.loading).toBe(false);
  });

  it('keeps the same content object when the file did not change', async () => {
    getFileContent.mockImplementation(async (_projectId, path) => ({
      path, content: 'hello', size: 5, truncated: false, type: 'text', revision: 'r1',
    }));
    renderView({ file: 'plan.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    const before = lastFilePreviewProps.fileContent;

    await emit('agent.activity', { category: 'tool_result' });

    expect(getFileContent).toHaveBeenCalledTimes(2);
    expect(lastFilePreviewProps.fileContent).toBe(before);
  });

  it('shows a file the agent created after the tree was first listed', async () => {
    renderView();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.queryByText('plan.md')).not.toBeInTheDocument();

    listDirectory.mockImplementation(async (_projectId, path) =>
      (path ?? '') === ''
        ? { path: '', entries: [...TREE[''].entries, { name: 'plan.md', type: 'file', size: 3 }] }
        : TREE[path ?? ''] ?? null,
    );
    await emit('agent.activity', { category: 'tool_result' });

    expect(screen.getByText('plan.md')).toBeInTheDocument();
  });

  it('refreshes on sub-agent output and on a status change too', async () => {
    renderView({ file: 'plan.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    await emit('chat.sub_agent_message', {});
    await emit('agent.status', { status: 'idle' });
    expect(getFileContent).toHaveBeenCalledTimes(3);
  });

  it('ignores tool CALL events and other projects', async () => {
    renderView({ file: 'plan.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    await emit('agent.activity', { category: 'file_write' });
    await emit('agent.activity', { category: 'tool_result', project_id: 'someone-else' });
    expect(getFileContent).toHaveBeenCalledTimes(1);
  });

  it('collapses a burst of results into one refresh', async () => {
    renderView({ file: 'plan.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    await act(async () => {
      for (let i = 0; i < 5; i += 1) {
        wsHandlers.get('agent.activity')?.({
          type: 'agent.activity', project_id: 'proj-1', category: 'tool_result',
        });
      }
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(getFileContent).toHaveBeenCalledTimes(2);
  });
});
