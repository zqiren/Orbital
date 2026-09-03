// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { LocaleProvider } from '../../i18n/LocaleContext';
import type { BrowserLive } from '../../hooks/useBrowserLive';
import type { BrowserViewProps } from './BrowserView';

// ---------------------------------------------------------------------------
// Mocks — declared before the module under test is imported.
// ---------------------------------------------------------------------------

const liveMock = vi.fn<() => BrowserLive>();
vi.mock('../../hooks/useBrowserLive', () => ({
  useBrowserLive: () => liveMock(),
}));

const getFileContentMock = vi.fn();
vi.mock('../../hooks/useFiles', () => ({
  useFiles: () => ({
    directory: null,
    fileContent: null,
    loading: false,
    error: null,
    listDirectory: vi.fn(),
    getFileContent: getFileContentMock,
    resolvePath: vi.fn(),
    saveFileContent: vi.fn(),
  }),
}));

// The real overlay is workstream E's; stand in with a minimal control surface
// so BrowserView's integration with it (mount when annotating, onAdd wiring)
// is exercised without depending on its implementation.
vi.mock('./AnnotateOverlay', () => ({
  default: (props: { active: boolean; onAdd: (box: unknown, note: string) => void }) => (
    <div data-testid="annotate-overlay" data-active={String(props.active)}>
      <button onClick={() => props.onAdd({ x: 1, y: 2, w: 3, h: 4 }, 'note text')}>
        add-annotation
      </button>
    </div>
  ),
}));

import BrowserView from './BrowserView';

function renderView(overrides: Partial<BrowserViewProps> = {}) {
  const onAddAnnotation = vi.fn();
  const props: BrowserViewProps = {
    projectId: 'p1',
    active: true,
    fallback: null,
    annotating: false,
    annotations: [],
    onAddAnnotation,
    ...overrides,
  };
  const utils = render(
    <LocaleProvider>
      <BrowserView {...props} />
    </LocaleProvider>,
  );
  return { ...utils, onAddAnnotation };
}

function mockRect(canvas: HTMLCanvasElement, width: number, height: number) {
  canvas.getBoundingClientRect = () =>
    ({
      left: 0,
      top: 0,
      right: width,
      bottom: height,
      width,
      height,
      x: 0,
      y: 0,
      toJSON() {},
    }) as DOMRect;
}

beforeAll(() => {
  // jsdom doesn't implement canvas 2D — stub what the paint effect and the
  // annotation capture (canvas.toDataURL) touch.
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    clearRect: vi.fn(),
    drawImage: vi.fn(),
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.toDataURL = vi.fn(() => 'data:image/png;base64,ANNOTATED');
});

beforeEach(() => {
  liveMock.mockReset();
  getFileContentMock.mockReset();
  liveMock.mockReturnValue({ status: 'idle', frame: null, title: '', send: vi.fn() });
  getFileContentMock.mockResolvedValue(null);
});

afterEach(() => {
  cleanup();
});

describe('BrowserView — empty / connecting / fallback states', () => {
  it('shows the empty message when idle with no fallback', () => {
    renderView({ fallback: null });
    expect(screen.getByText('No page open yet.')).toBeInTheDocument();
  });

  it('shows the connecting message while connecting with no frame yet', () => {
    liveMock.mockReturnValue({ status: 'connecting', frame: null, title: '', send: vi.fn() });
    renderView({ fallback: { path: 'orbital/output/screenshots/x.png', title: 'Last page' } });
    expect(screen.getByText('Connecting to the browser…')).toBeInTheDocument();
  });

  it('renders the fallback screenshot via the files content route when no live frame exists', async () => {
    liveMock.mockReturnValue({ status: 'closed', frame: null, title: '', send: vi.fn() });
    getFileContentMock.mockResolvedValue({
      path: 'orbital/output/screenshots/x.png',
      content: 'ZmFrZQ==',
      type: 'image',
      mime: 'image/png',
      size: 4,
    });

    renderView({ fallback: { path: 'orbital/output/screenshots/x.png', title: 'Last page' } });

    const img = await screen.findByAltText('Last page');
    expect(img).toHaveAttribute('src', 'data:image/png;base64,ZmFrZQ==');
    expect(getFileContentMock).toHaveBeenCalledWith('p1', 'orbital/output/screenshots/x.png');
    expect(screen.getByText('Last screenshot · Last page')).toBeInTheDocument();
  });

  it('falls back to the empty message when no live frame and no fallback prop', () => {
    liveMock.mockReturnValue({ status: 'no_browser', frame: null, title: '', send: vi.fn() });
    renderView({ fallback: null });
    expect(screen.getByText('No page open yet.')).toBeInTheDocument();
  });
});

describe('BrowserView — live canvas', () => {
  it('renders a canvas with the page title as caption and aria-label', () => {
    liveMock.mockReturnValue({
      status: 'open',
      frame: { jpegDataUrl: 'data:image/jpeg;base64,xx', width: 1000, height: 500 },
      title: 'Example Domain',
      send: vi.fn(),
    });
    const { container } = renderView();
    const canvas = container.querySelector('canvas');
    expect(canvas).not.toBeNull();
    expect(canvas).toHaveAttribute('role', 'img');
    expect(canvas).toHaveAttribute('aria-label', 'Example Domain');
    expect(canvas).toHaveAttribute('tabIndex', '0');
    expect(screen.getByText('Example Domain')).toBeInTheDocument();
    expect(screen.getByText('Live')).toBeInTheDocument();
    // No URL, no dimensions, no status text in the caption (spec D9).
    expect(screen.queryByText(/open|closed|connecting/i)).not.toBeInTheDocument();
  });

  it('scales a click at canvas (10,10) with canvas width 500 to page x=20 for a 1000px-wide frame', () => {
    const sendMock = vi.fn();
    liveMock.mockReturnValue({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'T',
      send: sendMock,
    });
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    mockRect(canvas, 500, 250);

    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, button: 0, detail: 1 });
    fireEvent.pointerUp(canvas, { clientX: 10, clientY: 10, button: 0, detail: 1 });

    expect(sendMock).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'mouse', action: 'down', x: 20, button: 'left', clickCount: 1 }),
    );
    expect(sendMock).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'mouse', action: 'up', x: 20, button: 'left', clickCount: 1 }),
    );
  });

  it('forwards pointer move', () => {
    const sendMock = vi.fn();
    liveMock.mockReturnValue({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'T',
      send: sendMock,
    });
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    mockRect(canvas, 500, 250);

    fireEvent.pointerMove(canvas, { clientX: 50, clientY: 25 });

    expect(sendMock).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'mouse', action: 'move', x: 100, y: 50 }),
    );
  });

  it('sends key messages on keydown, with text only for printable single characters', () => {
    const sendMock = vi.fn();
    liveMock.mockReturnValue({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 100, height: 100 },
      title: 'T',
      send: sendMock,
    });
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;

    fireEvent.keyDown(canvas, { key: 'Enter', code: 'Enter' });
    fireEvent.keyDown(canvas, { key: 'a', code: 'KeyA' });

    expect(sendMock).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'key', action: 'down', key: 'Enter', code: 'Enter' }),
    );
    const enterCall = sendMock.mock.calls.find((c) => c[0].key === 'Enter')?.[0];
    expect(enterCall.text).toBeUndefined();

    const aCall = sendMock.mock.calls.find((c) => c[0].key === 'a')?.[0];
    expect(aCall).toMatchObject({ type: 'key', action: 'down', key: 'a', code: 'KeyA', text: 'a' });
  });

  it('forwards wheel deltas', () => {
    const sendMock = vi.fn();
    liveMock.mockReturnValue({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'T',
      send: sendMock,
    });
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    mockRect(canvas, 500, 250);

    fireEvent.wheel(canvas, { clientX: 10, clientY: 10, deltaX: 3, deltaY: 42 });

    expect(sendMock).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'mouse', action: 'wheel', deltaX: 3, deltaY: 42 }),
    );
  });
});

describe('BrowserView — annotating', () => {
  it('does not forward pointer input while annotating, and mounts the overlay', () => {
    const sendMock = vi.fn();
    liveMock.mockReturnValue({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'T',
      send: sendMock,
    });
    const { container } = renderView({ annotating: true });
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    mockRect(canvas, 500, 250);

    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10 });
    fireEvent.pointerUp(canvas, { clientX: 10, clientY: 10 });
    fireEvent.wheel(canvas, { clientX: 10, clientY: 10, deltaY: 10 });

    expect(sendMock).not.toHaveBeenCalled();
    const overlay = screen.getByTestId('annotate-overlay');
    expect(overlay).toBeInTheDocument();
    expect(overlay).toHaveAttribute('data-active', 'true');
  });

  it('does not mount the overlay when not annotating', () => {
    liveMock.mockReturnValue({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 100, height: 100 },
      title: 'T',
      send: vi.fn(),
    });
    renderView({ annotating: false });
    expect(screen.queryByTestId('annotate-overlay')).not.toBeInTheDocument();
  });

  it('onAddAnnotation receives a browser-kind draft with pageTitle and an image data URL', () => {
    liveMock.mockReturnValue({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'Example Domain',
      send: vi.fn(),
    });
    const { onAddAnnotation } = renderView({ annotating: true });

    fireEvent.click(screen.getByText('add-annotation'));

    expect(onAddAnnotation).toHaveBeenCalledTimes(1);
    const draft = onAddAnnotation.mock.calls[0][0];
    expect(draft).toMatchObject({
      kind: 'browser',
      pageTitle: 'Example Domain',
      note: 'note text',
      imageDataUrl: 'data:image/png;base64,ANNOTATED',
    });
    expect(draft.box).toBeDefined();
  });
});
