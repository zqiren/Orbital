// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 078 §5.3 (D10) / §10 — the Files view: tree with touched badges, the
 * spec 187 最近修改 / Recent group above it (which replaced "Touched this
 * session"), swapping to the preview on click, and the three annotation forms
 * a quote can take.
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
const getFileContent = vi.fn<(projectId: string, path: string, ifRevision?: string) => Promise<FileContent | null>>();
const resolvePath = vi.fn<(projectId: string, path: string) => Promise<string[] | null>>();
const revealPath = vi.fn<(projectId: string, path: string) => Promise<boolean>>();

vi.mock('../../hooks/useFiles', async (importOriginal) => ({
  // The real listRecent / useRecentFiles (spec 187): their request goes
  // through the `api` mock below, so 404 / 503 take the real failure path.
  ...(await importOriginal<typeof import('../../hooks/useFiles')>()),
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

// Spec 187: GET /files/recent. Anything else keeps the real `api`.
const recentApi = vi.fn<(path: string) => Promise<unknown>>();
vi.mock('../../config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../config')>();
  return {
    ...actual,
    api: (path: string, options?: RequestInit) =>
      path.includes('/files/recent') ? recentApi(path) : actual.api(path, options),
  };
});

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
import TouchedGroup from './TouchedGroup';
import { ApiError } from '../../config';
import { LocaleProvider } from '../../i18n/LocaleContext';
import { LOCALE_STORAGE_KEY } from '../../i18n/locales';
import type { RecentFile } from '../../types';

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
  recentApi.mockReset();
  recentApi.mockResolvedValue({ files: [] });
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
// Spec 187 — 最近修改 / Recent group. It replaced "Touched this session" at
// the top of the tree (owner decision §10.1): the project's latest-modified
// files from GET /files/recent, five shown, 更多 / More expanding to all.
// ---------------------------------------------------------------------------

const nowS = () => Date.now() / 1000;

/** `n` files, newest first, a minute apart, the newest two minutes old. */
function recentFiles(n: number): RecentFile[] {
  return Array.from({ length: n }, (_, i) => ({
    path: `out/f${i}.md`,
    modified_at: nowS() - 120 - i * 60,
  }));
}

function recentRows(): HTMLElement[] {
  return screen.queryAllByTestId('panel-recent-row');
}

describe('FilesView — 最近修改 / Recent group (spec 187)', () => {
  it('asks the recent route for this project on mount', async () => {
    renderView();
    await settle();
    expect(recentApi).toHaveBeenCalledWith('/api/v2/projects/proj-1/files/recent');
  });

  it('is hidden when the project has no recent files', async () => {
    renderView();
    await settle();
    expect(screen.queryByTestId('panel-recent-group')).toBeNull();
    expect(screen.getByText('README.md')).toBeInTheDocument();
  });

  it.each([
    ['404 from an older daemon', 404, 'Not Found'],
    ['503 when ripgrep is missing', 503, 'ripgrep not found'],
  ])('is hidden when the first request fails: %s', async (_label, status, detail) => {
    recentApi.mockRejectedValue(new ApiError(status, detail));
    renderView();
    await settle();
    expect(screen.queryByTestId('panel-recent-group')).toBeNull();
    // The tree is unaffected.
    expect(screen.getByText('README.md')).toBeInTheDocument();
  });

  it('replaces the touched group: none even when the manager touched files', async () => {
    recentApi.mockResolvedValue({ files: recentFiles(2) });
    renderView({ touched: [{ path: 'README.md', op: 'read' }] });
    await settle();
    expect(screen.queryByTestId('panel-touched-group')).toBeNull();
    // A capped list carries no count in its header.
    expect(screen.getByTestId('panel-recent-group')).toHaveTextContent(/^Recent$/);
    // The tree badges from `touched` stay (spec 078 D10).
    expect(screen.getAllByText('read')).not.toHaveLength(0);
  });

  it('shows the newest five; More expands to all, Less collapses back', async () => {
    recentApi.mockResolvedValue({ files: recentFiles(7) });
    renderView();
    await settle();
    expect(recentRows().map((row) => row.title)).toEqual([
      'out/f0.md', 'out/f1.md', 'out/f2.md', 'out/f3.md', 'out/f4.md',
    ]);
    const more = screen.getByTestId('panel-recent-more');
    expect(more).toHaveTextContent('More');

    fireEvent.click(more);
    expect(recentRows()).toHaveLength(7);
    expect(more).toHaveTextContent('Less');

    fireEvent.click(more);
    expect(recentRows()).toHaveLength(5);
  });

  it('has no More row with five files or fewer', async () => {
    recentApi.mockResolvedValue({ files: recentFiles(5) });
    renderView();
    await settle();
    expect(recentRows()).toHaveLength(5);
    expect(screen.queryByTestId('panel-recent-more')).toBeNull();
  });

  it('shows the basename with its folder, and a click opens the full path', async () => {
    recentApi.mockResolvedValue({
      files: [{ path: 'docs/reports/q3.pdf', modified_at: nowS() - 600 }],
    });
    const { props } = renderView();
    await settle();
    const [row] = recentRows();
    expect(row).toHaveTextContent('q3.pdf');
    expect(row).toHaveTextContent('docs/reports');
    fireEvent.click(row);
    expect(props.onSelectFile).toHaveBeenCalledWith('docs/reports/q3.pdf');
  });

  it('collapses and re-expands on its header', async () => {
    recentApi.mockResolvedValue({ files: recentFiles(2) });
    renderView();
    await settle();
    const group = screen.getByTestId('panel-recent-group');
    expect(recentRows()).toHaveLength(2);

    fireEvent.click(group);
    expect(recentRows()).toHaveLength(0);
    expect(group).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(group);
    expect(recentRows()).toHaveLength(2);
  });
});

describe('FilesView — Recent refresh (spec 187)', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  async function emit(type: string, event: Record<string, unknown> = {}) {
    await act(async () => {
      wsHandlers.get(type)?.({ type, project_id: 'proj-1', ...event });
      await vi.advanceTimersByTimeAsync(300);
    });
  }

  it('refetches on sub-agent output and shows what the worker wrote', async () => {
    renderView();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(recentApi).toHaveBeenCalledTimes(1);

    recentApi.mockResolvedValue({ files: [{ path: 'chart.png', modified_at: nowS() }] });
    await emit('chat.sub_agent_message');

    expect(recentApi).toHaveBeenCalledTimes(2);
    expect(screen.getByText('chart.png')).toBeInTheDocument();
  });

  it('never fetches in preview state', async () => {
    renderView({ file: 'README.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    await emit('agent.activity', { category: 'tool_result' });
    await emit('chat.sub_agent_message');
    await emit('agent.status', { status: 'idle' });
    expect(recentApi).not.toHaveBeenCalled();
  });

  it('a failed refresh keeps the list already shown', async () => {
    recentApi.mockResolvedValue({ files: [{ path: 'kept.md', modified_at: nowS() }] });
    renderView();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByText('kept.md')).toBeInTheDocument();

    recentApi.mockRejectedValue(new ApiError(503, 'ripgrep not found'));
    await emit('agent.activity', { category: 'tool_result' });

    expect(recentApi).toHaveBeenCalledTimes(2);
    expect(screen.getByText('kept.md')).toBeInTheDocument();
  });

  it('a slow request: ticks landing mid-flight bring exactly one follow-up', async () => {
    let finish: (value: unknown) => void = () => {};
    recentApi.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    renderView();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(recentApi).toHaveBeenCalledTimes(1);

    // Three separate ticks while the first scan is still running.
    await emit('agent.activity', { category: 'tool_result' });
    await emit('chat.sub_agent_message');
    await emit('agent.status', { status: 'running' });
    expect(recentApi).toHaveBeenCalledTimes(1);

    recentApi.mockResolvedValue({ files: [{ path: 'after.md', modified_at: nowS() }] });
    await act(async () => {
      finish({ files: [{ path: 'first.md', modified_at: nowS() }] });
      await vi.advanceTimersByTimeAsync(1000);
    });

    expect(recentApi).toHaveBeenCalledTimes(2);
    expect(screen.getByText('after.md')).toBeInTheDocument();
  });

  it('a project switch mid-flight drops the late answer and its follow-up', async () => {
    let finish: (value: unknown) => void = () => {};
    recentApi.mockImplementation((path: string) =>
      path.includes('/proj-1/')
        ? new Promise((resolve) => { finish = resolve; })
        : Promise.resolve({ files: [{ path: 'two.md', modified_at: nowS() }] }),
    );
    const { rerender, props } = renderView();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    // A tick during the request: the follow-up is now pending.
    await emit('agent.activity', { category: 'tool_result' });

    rerender(<FilesView {...props} projectId="proj-2" />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByText('two.md')).toBeInTheDocument();

    await act(async () => {
      finish({ files: [{ path: 'stale.md', modified_at: nowS() }] });
      await vi.advanceTimersByTimeAsync(1000);
    });

    expect(screen.queryByText('stale.md')).toBeNull();
    expect(screen.getByText('two.md')).toBeInTheDocument();
    const proj1Calls = recentApi.mock.calls.filter(([path]) => path.includes('/proj-1/'));
    expect(proj1Calls).toHaveLength(1);
  });
});

describe('FilesView — Recent ages, en and zh (spec 187)', () => {
  const AGES: [number, string, string][] = [
    [30, 'just now', '刚刚'],
    [5 * 60, '5m ago', '5 分钟前'],
    [3 * 3600, '3h ago', '3 小时前'],
    [2 * 86400, '2d ago', '2 天前'],
  ];
  const FORTY_FIVE_DAYS = 45 * 86400;

  function agedFiles(): RecentFile[] {
    const now = nowS();
    return [
      ...AGES.map(([age], i) => ({ path: `a${i}.md`, modified_at: now - age })),
      { path: 'old.md', modified_at: now - FORTY_FIVE_DAYS },
    ];
  }

  function rowText(name: string): string {
    const row = recentRows().find((r) => r.title === name);
    return row?.textContent ?? '';
  }

  it('in English', async () => {
    const files = agedFiles();
    recentApi.mockResolvedValue({ files });
    renderView();
    await settle();
    AGES.forEach(([, en], i) => expect(rowText(`a${i}.md`)).toContain(en));
    const date = new Date(files[4].modified_at * 1000).toLocaleDateString();
    expect(rowText('old.md')).toContain(date);
  });

  it('in Chinese, with the zh header and More / Less', async () => {
    localStorage.setItem(LOCALE_STORAGE_KEY, 'zh');
    try {
      const files = [...agedFiles(), ...recentFiles(2)];
      recentApi.mockResolvedValue({ files });
      render(
        <LocaleProvider>
          <FilesView
            projectId="proj-1"
            touched={[]}
            file={null}
            onSelectFile={vi.fn()}
            onAddAnnotation={vi.fn()}
          />
        </LocaleProvider>,
      );
      await settle();
      expect(screen.getByTestId('panel-recent-group')).toHaveTextContent('最近修改');
      const more = screen.getByTestId('panel-recent-more');
      expect(more).toHaveTextContent('更多');
      fireEvent.click(more);
      expect(more).toHaveTextContent('收起');
      AGES.forEach(([, , zh], i) => expect(rowText(`a${i}.md`)).toContain(zh));
      const date = new Date(files[4].modified_at * 1000).toLocaleDateString();
      expect(rowText('old.md')).toContain(date);
    } finally {
      localStorage.removeItem(LOCALE_STORAGE_KEY);
    }
  });
});

// The old group is kept, unrendered, so restoring it is one line (§10.1).
describe('TouchedGroup — kept for a one-line restore', () => {
  it('still renders its counted header and opens a row', () => {
    const onSelectFile = vi.fn();
    render(
      <TouchedGroup
        touched={[
          { path: 'README.md', op: 'read' },
          { path: 'src/app.ts', op: 'edited' },
        ]}
        onSelectFile={onSelectFile}
      />,
    );
    expect(screen.getByTestId('panel-touched-group')).toHaveTextContent('Touched this session (2)');
    fireEvent.click(screen.getByText('src/app.ts'));
    expect(onSelectFile).toHaveBeenCalledWith('src/app.ts');
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

  it('the header’s "Open full width" hands the resolved path to the parent', async () => {
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

  // Spec 103 (P2): the re-read is conditional. The panel sends the revision
  // it holds; a matching daemon answers ~100 bytes instead of the file.
  it('sends the held revision on a refresh, never on the first open', async () => {
    getFileContent.mockImplementation(async (_projectId, path) => ({
      path, content: 'hello', size: 5, truncated: false, type: 'text', revision: 'r1',
    }));
    renderView({ file: 'plan.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(getFileContent).toHaveBeenCalledTimes(1);
    expect(getFileContent.mock.calls[0][2]).toBeUndefined();

    await emit('agent.activity', { category: 'tool_result' });
    expect(getFileContent).toHaveBeenCalledTimes(2);
    expect(getFileContent.mock.calls[1]).toEqual(['proj-1', 'plan.md', 'r1']);
  });

  it('an unchanged answer keeps the content already shown', async () => {
    getFileContent.mockImplementation(async (_projectId, path) => ({
      path, content: 'hello', size: 5, truncated: false, type: 'text', revision: 'r1',
    }));
    renderView({ file: 'plan.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    const before = lastFilePreviewProps.fileContent;

    getFileContent.mockImplementation(async (_projectId, path) =>
      ({ path, revision: 'r1', unchanged: true }) as unknown as FileContent,
    );
    await emit('agent.activity', { category: 'tool_result' });

    expect(lastFilePreviewProps.fileContent).toBe(before);
    expect((lastFilePreviewProps.fileContent as FileContent).content).toBe('hello');
    expect(lastFilePreviewProps.loading).toBe(false);
  });

  it('a new selection reads in full even though the previous file had a revision', async () => {
    getFileContent.mockImplementation(async (_projectId, path) => ({
      path, content: 'hello', size: 5, truncated: false, type: 'text', revision: 'r1',
    }));
    const { rerender, props } = renderView({ file: 'plan.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    rerender(<FilesView {...props} file="other.md" />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    const last = getFileContent.mock.calls.at(-1);
    expect(last?.[1]).toBe('other.md');
    expect(last?.[2]).toBeUndefined();
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

// ---------------------------------------------------------------------------
// Spec 103 (R3) — the tree refresh re-lists what is on screen, nothing more:
// the root plus the expanded folders, and nothing at all while the view is
// in preview state (the tree is not even mounted). Coming back to the tree
// refreshes it once.
// ---------------------------------------------------------------------------

describe('FilesView — refresh scope (spec 103)', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  async function emit(type: string, event: Record<string, unknown>) {
    await act(async () => {
      wsHandlers.get(type)?.({ type, project_id: 'proj-1', ...event });
      await vi.advanceTimersByTimeAsync(300);
    });
  }

  function listedPaths(): (string | undefined)[] {
    return listDirectory.mock.calls.map((call) => call[1]);
  }

  it('lists no directory while in preview state', async () => {
    renderView({ file: 'README.md', touched: [{ path: 'src/deep/x.ts', op: 'read' }] });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    listDirectory.mockClear();

    await emit('agent.activity', { category: 'tool_result' });
    await emit('agent.status', { status: 'idle' });

    expect(listDirectory).not.toHaveBeenCalled();
    // The open file still refreshes.
    expect(getFileContent.mock.calls.length).toBeGreaterThan(1);
  });

  it('refreshes the tree once on returning from the preview', async () => {
    const { rerender, props } = renderView({ file: 'README.md' });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    listDirectory.mockClear();

    rerender(<FilesView {...props} file={null} />);
    await act(async () => { await vi.advanceTimersByTimeAsync(300); });

    expect(listedPaths()).toEqual([undefined]);
  });

  it('a tick re-lists only the root and the expanded folders', async () => {
    renderView();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    fireEvent.click(screen.getByText('src'));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    fireEvent.click(screen.getByText('deep'));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByText('x.ts')).toBeInTheDocument();
    // Collapse src/deep again: it has been listed, but it is no longer open.
    fireEvent.click(screen.getByText('deep'));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    listDirectory.mockClear();

    await emit('agent.activity', { category: 'tool_result' });

    expect(listedPaths().sort()).toEqual([undefined, 'src'].sort());
  });

  it('a failed listing keeps what is shown', async () => {
    renderView();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByText('README.md')).toBeInTheDocument();
    listDirectory.mockResolvedValue(null);

    await emit('agent.activity', { category: 'tool_result' });

    expect(screen.getByText('README.md')).toBeInTheDocument();
    expect(screen.getByText('src')).toBeInTheDocument();
  });

  // Spec 137 gap 2: the bar's Upload lands a file; the parent bumps
  // refreshKey and the tree shows it without waiting for a daemon event.
  it('re-lists the tree when refreshKey changes', async () => {
    const { rerender, props } = renderView();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    listDirectory.mockImplementation(async (_projectId, path) =>
      (path ?? '') === ''
        ? { path: '', entries: [...TREE[''].entries, { name: 'upload.csv', type: 'file', size: 3 }] }
        : TREE[path ?? ''] ?? null,
    );
    rerender(<FilesView {...props} refreshKey={1} />);
    await act(async () => { await vi.advanceTimersByTimeAsync(300); });
    expect(screen.getByText('upload.csv')).toBeInTheDocument();
  });
});

describe('FilesView — no full-width target (phone sheet)', () => {
  it('omits "Open full width" when the parent gives no onOpenInFiles', async () => {
    renderView({ file: 'README.md', onOpenInFiles: undefined });
    await settle();
    const header = lastFilePreviewProps.panelHeader as { onOpenInFiles?: () => void };
    expect(header).toBeDefined();
    expect(header.onOpenInFiles).toBeUndefined();
  });
});
