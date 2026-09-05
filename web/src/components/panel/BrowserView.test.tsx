// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from 'vitest';
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react';
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

/** A BrowserLive value with every field the view reads, overridable. */
function live(partial: Partial<BrowserLive>): BrowserLive {
  return {
    status: 'idle',
    frame: null,
    title: '',
    loading: false,
    canGoBack: false,
    canGoForward: false,
    cursor: 'default',
    send: vi.fn(),
    nav: vi.fn(),
    ...partial,
  };
}

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
  liveMock.mockReturnValue(live({ status: 'idle', frame: null, title: '', send: vi.fn() }));
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
    liveMock.mockReturnValue(live({ status: 'connecting', frame: null, title: '', send: vi.fn() }));
    renderView({ fallback: { path: 'orbital/output/screenshots/x.png', title: 'Last page' } });
    expect(screen.getByText('Connecting to the browser…')).toBeInTheDocument();
  });

  it('renders the fallback screenshot via the files content route when no live frame exists', async () => {
    liveMock.mockReturnValue(live({ status: 'closed', frame: null, title: '', send: vi.fn() }));
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
    liveMock.mockReturnValue(live({ status: 'no_browser', frame: null, title: '', send: vi.fn() }));
    renderView({ fallback: null });
    expect(screen.getByText('No page open yet.')).toBeInTheDocument();
  });
});

describe('BrowserView — live canvas', () => {
  it('renders a canvas with the page title as caption and aria-label', () => {
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'data:image/jpeg;base64,xx', width: 1000, height: 500 },
      title: 'Example Domain',
      send: vi.fn(),
    }));
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
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'T',
      send: sendMock,
    }));
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
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'T',
      send: sendMock,
    }));
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
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 100, height: 100 },
      title: 'T',
      send: sendMock,
    }));
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
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'T',
      send: sendMock,
    }));
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
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'T',
      send: sendMock,
    }));
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
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 100, height: 100 },
      title: 'T',
      send: vi.fn(),
    }));
    renderView({ annotating: false });
    expect(screen.queryByTestId('annotate-overlay')).not.toBeInTheDocument();
  });

  it('onAddAnnotation receives a browser-kind draft with pageTitle and an image data URL', () => {
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
      title: 'Example Domain',
      send: vi.fn(),
    }));
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


describe('BrowserView — after the page closes', () => {
  it('keeps the last frame on screen labelled "Last screenshot" and forwards no input', () => {
    const send = vi.fn();
    liveMock.mockReturnValue(live({
      status: 'closed',
      frame: { jpegDataUrl: 'data:image/jpeg;base64,AAAA', width: 1000, height: 500 },
      title: 'Example Domain',
      send,
    }));
    renderView();
    expect(screen.getByText('Last screenshot')).toBeInTheDocument();
    expect(screen.queryByText('Live')).toBeNull();
    const canvas = screen.getByRole('img') as HTMLCanvasElement;
    mockRect(canvas, 500, 250);
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, button: 0, pointerId: 1 });
    fireEvent.pointerUp(canvas, { clientX: 10, clientY: 10, button: 0, pointerId: 1 });
    fireEvent.keyDown(canvas, { key: 'Enter', code: 'Enter' });
    expect(send).not.toHaveBeenCalled();
  });
});


describe('BrowserView — a blank page is not a page', () => {
  it('shows the empty line instead of a white live canvas when the page is about:blank', () => {
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'data:image/jpeg;base64,AAAA', width: 800, height: 600 },
      title: '',
      url: 'about:blank',
      send: vi.fn(),
    }));
    renderView();
    expect(screen.getByText('No page open yet.')).toBeInTheDocument();
    expect(screen.queryByRole('img')).toBeNull();
  });
});


describe('BrowserView — toolbar', () => {
  function openWith(partial: Partial<BrowserLive> = {}) {
    const nav = vi.fn();
    liveMock.mockReturnValue(
      live({
        status: 'open',
        frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
        title: 'T',
        url: 'https://example.com/path',
        nav,
        ...partial,
      }),
    );
    const utils = renderView();
    return { ...utils, nav };
  }

  it('shows back, forward, reload and the page address', () => {
    openWith({ canGoBack: true, canGoForward: false });
    expect(screen.getByRole('button', { name: 'Back' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Forward' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Reload' })).toBeEnabled();
    expect(screen.getByLabelText('Address')).toHaveValue('https://example.com/path');
  });

  it('back / forward / reload send nav actions', () => {
    const { nav } = openWith({ canGoBack: true, canGoForward: true });
    fireEvent.click(screen.getByRole('button', { name: 'Back' }));
    fireEvent.click(screen.getByRole('button', { name: 'Forward' }));
    fireEvent.click(screen.getByRole('button', { name: 'Reload' }));
    expect(nav.mock.calls.map((c) => c[0])).toEqual(['back', 'forward', 'reload']);
  });

  it('reload becomes stop while the page is loading', () => {
    const { nav } = openWith({ loading: true });
    expect(screen.queryByRole('button', { name: 'Reload' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Stop loading' }));
    expect(nav).toHaveBeenCalledWith('stop');
  });

  it('Enter in the address field navigates to the resolved address; Escape restores the page URL', () => {
    const { nav } = openWith();
    const field = screen.getByLabelText('Address') as HTMLInputElement;
    fireEvent.change(field, { target: { value: 'news.ycombinator.com' } });
    expect(field).toHaveValue('news.ycombinator.com');
    fireEvent.keyDown(field, { key: 'Enter' });
    expect(nav).toHaveBeenCalledWith('goto', 'https://news.ycombinator.com');
    expect(field).toHaveValue('https://example.com/path');

    fireEvent.change(field, { target: { value: 'typo' } });
    fireEvent.keyDown(field, { key: 'Escape' });
    expect(field).toHaveValue('https://example.com/path');
    expect(nav).toHaveBeenCalledTimes(1);
  });

  it('a search is sent as a search URL and a non-web scheme is refused', () => {
    const { nav } = openWith();
    const field = screen.getByLabelText('Address');
    fireEvent.change(field, { target: { value: 'orbital agents' } });
    fireEvent.keyDown(field, { key: 'Enter' });
    expect(nav).toHaveBeenLastCalledWith('goto', 'https://www.google.com/search?q=orbital%20agents');
    fireEvent.change(field, { target: { value: 'file:///etc/passwd' } });
    fireEvent.keyDown(field, { key: 'Enter' });
    expect(nav).toHaveBeenCalledTimes(1);
  });

  it('a blank page keeps the toolbar so the user can open something', () => {
    openWith({ url: 'about:blank', frame: { jpegDataUrl: 'd', width: 8, height: 8 } });
    expect(screen.getByText('No page open yet.')).toBeInTheDocument();
    expect(screen.getByLabelText('Address')).toHaveValue('');
  });

  it('no toolbar without a page', () => {
    liveMock.mockReturnValue(live({ status: 'closed', frame: { jpegDataUrl: 'd', width: 8, height: 8 }, title: 'T' }));
    renderView();
    expect(screen.queryByLabelText('Address')).toBeNull();
  });
});

describe('BrowserView — cursor, capture, click count, coalescing', () => {
  function openCanvas(partial: Partial<BrowserLive> = {}) {
    const send = vi.fn();
    liveMock.mockReturnValue(
      live({
        status: 'open',
        frame: { jpegDataUrl: 'd', width: 1000, height: 500 },
        title: 'T',
        send,
        ...partial,
      }),
    );
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    mockRect(canvas, 500, 250);
    return { canvas, send };
  }

  it('paints the page cursor onto the canvas, keywords only', () => {
    const { canvas } = openCanvas({ cursor: 'pointer' });
    expect(canvas.style.cursor).toBe('pointer');
    cleanup();
    const { canvas: c2 } = openCanvas({ cursor: 'url(evil.png), auto' });
    expect(c2.style.cursor).toBe('default');
    cleanup();
    const { canvas: c3 } = openCanvas({ status: 'closed', cursor: 'pointer' });
    expect(c3.style.cursor).toBe('not-allowed');
  });

  it('captures the pointer on press and releases it on release', () => {
    const { canvas } = openCanvas();
    const setCapture = vi.fn();
    const releaseCapture = vi.fn();
    canvas.setPointerCapture = setCapture;
    canvas.releasePointerCapture = releaseCapture;
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, button: 0, pointerId: 7 });
    fireEvent.pointerUp(canvas, { clientX: 10, clientY: 10, button: 0, pointerId: 7 });
    expect(setCapture).toHaveBeenCalledWith(7);
    expect(releaseCapture).toHaveBeenCalledWith(7);
  });

  it('counts a quick second press at the same spot as a double-click when detail is 0', () => {
    const { canvas, send } = openCanvas();
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, button: 0, detail: 0 });
    fireEvent.pointerUp(canvas, { clientX: 10, clientY: 10, button: 0, detail: 0 });
    fireEvent.pointerDown(canvas, { clientX: 11, clientY: 10, button: 0, detail: 0 });
    fireEvent.pointerUp(canvas, { clientX: 11, clientY: 10, button: 0, detail: 0 });
    const counts = send.mock.calls.filter((c) => c[0].type === 'mouse').map((c) => c[0].clickCount);
    expect(counts).toEqual([1, 1, 2, 2]);
  });

  it('a burst of wheel events within one frame sends only the first at once', () => {
    const { canvas, send } = openCanvas();
    for (let i = 0; i < 10; i++) fireEvent.wheel(canvas, { clientX: 10, clientY: 10, deltaX: 0, deltaY: 8 });
    expect(send.mock.calls.filter((c) => c[0].action === 'wheel')).toHaveLength(1);
  });

  it('a press flushes pending pointer traffic before the click', () => {
    const { canvas, send } = openCanvas();
    fireEvent.wheel(canvas, { clientX: 10, clientY: 10, deltaY: 8 });
    fireEvent.wheel(canvas, { clientX: 10, clientY: 10, deltaY: 8 });
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, button: 0, detail: 1 });
    const actions = send.mock.calls.map((c) => c[0].action);
    expect(actions).toEqual(['wheel', 'wheel', 'down']);
  });
});

describe('BrowserView — keyboard through the text host (IME)', () => {
  function openCanvas() {
    const send = vi.fn();
    liveMock.mockReturnValue(
      live({ status: 'open', frame: { jpegDataUrl: 'd', width: 100, height: 100 }, title: 'T', send }),
    );
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    const host = screen.getByTestId('browser-text-host') as HTMLTextAreaElement;
    return { canvas, host, send };
  }

  it('a press focuses the hidden text host', () => {
    const { canvas, host } = openCanvas();
    mockRect(canvas, 100, 100);
    fireEvent.pointerDown(canvas, { clientX: 5, clientY: 5, button: 0, detail: 1 });
    expect(document.activeElement).toBe(host);
  });

  it('composed text arrives as one text message; keystrokes during composition are not forwarded', () => {
    const { host, send } = openCanvas();
    fireEvent.compositionStart(host);
    fireEvent.keyDown(host, { key: 'Process', code: 'KeyN' });
    fireEvent.keyDown(host, { key: 'n', code: 'KeyN', isComposing: true });
    fireEvent.compositionEnd(host, { data: '你好' });
    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith({ type: 'text', text: '你好' });
  });

  it('a keyup whose keydown was never sent is dropped', () => {
    const { host, send } = openCanvas();
    fireEvent.keyUp(host, { key: 'Enter', code: 'Enter' });
    expect(send).not.toHaveBeenCalled();
    fireEvent.keyDown(host, { key: 'Enter', code: 'Enter' });
    fireEvent.keyUp(host, { key: 'Enter', code: 'Enter' });
    expect(send.mock.calls.map((c) => c[0].action)).toEqual(['down', 'up']);
  });
});

describe('BrowserView — a mirror of the agent\'s page', () => {
  // jsdom lays nothing out: give the frame area a size, as a real panel has.
  const proto = HTMLElement.prototype;
  const original = {
    width: Object.getOwnPropertyDescriptor(proto, 'clientWidth'),
    height: Object.getOwnPropertyDescriptor(proto, 'clientHeight'),
  };
  beforeEach(() => {
    vi.useFakeTimers();
    Object.defineProperty(proto, 'clientWidth', { configurable: true, get: () => 690 });
    Object.defineProperty(proto, 'clientHeight', { configurable: true, get: () => 820 });
  });
  afterEach(() => {
    vi.useRealTimers();
    if (original.width) Object.defineProperty(proto, 'clientWidth', original.width);
    if (original.height) Object.defineProperty(proto, 'clientHeight', original.height);
  });

  it('never asks the page to take the panel\'s size (2026-09-05: resizing it changed what the agent browsed)', () => {
    const sendMock = vi.fn();
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1280, height: 720 },
      title: 'T',
      send: sendMock,
    }));
    renderView();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(sendMock.mock.calls.filter(([m]) => (m as { type: string }).type === 'viewport')).toHaveLength(0);
  });

  it('maps input through the letterbox: a 1280×720 page in a 690×820 panel', () => {
    const sendMock = vi.fn();
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1280, height: 720 },
      title: 'T',
      send: sendMock,
    }));
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    mockRect(canvas, 690, 820);
    // The page fits the width (690 wide, 388 tall) centred: the panel's
    // centre is the page's centre, and the top band is off the page.
    fireEvent.pointerDown(canvas, { clientX: 345, clientY: 410, button: 0, detail: 1 });
    fireEvent.pointerDown(canvas, { clientX: 345, clientY: 10, button: 0, detail: 1 });
    const downs = sendMock.mock.calls
      .map(([m]) => m as { type: string; action: string; x: number; y: number })
      .filter((m) => m.type === 'mouse' && m.action === 'down');
    expect(downs).toHaveLength(2);
    expect(downs[0].x).toBeCloseTo(640, 5);
    expect(downs[0].y).toBeCloseTo(360, 5);
    expect(downs[1].y).toBe(0);
  });
});
