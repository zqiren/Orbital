// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from 'vitest';
import { render, screen, fireEvent, cleanup, act, waitFor } from '@testing-library/react';
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

import BrowserView, { SIZE_SETTLE_MS, clearFallbackCache } from './BrowserView';

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
  clearFallbackCache();
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
    const nav = vi.fn();
    liveMock.mockReturnValue(live({
      status: 'closed',
      frame: { jpegDataUrl: 'data:image/jpeg;base64,AAAA', width: 1000, height: 500 },
      title: 'Example Domain',
      send,
      nav,
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
    // The click is not page input: it asks for the closed page back.
    expect(nav.mock.calls).toEqual([['reopen']]);
  });

  it('a right-click or an annotating click on the stale frame does not reopen', () => {
    const nav = vi.fn();
    liveMock.mockReturnValue(live({
      status: 'closed',
      frame: { jpegDataUrl: 'data:image/jpeg;base64,AAAA', width: 1000, height: 500 },
      title: 'Example Domain',
      nav,
    }));
    renderView({ annotating: true });
    const canvas = screen.getByRole('img') as HTMLCanvasElement;
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, button: 0 });
    cleanup();
    renderView();
    fireEvent.pointerDown(screen.getByRole('img'), { clientX: 10, clientY: 10, button: 2 });
    expect(nav).not.toHaveBeenCalled();
  });
});

describe('BrowserView — reopening after the idle close (2026-11-18)', () => {
  const shot = { type: 'image', mime: 'image/png', content: 'U0hPVA==' };

  it('shows the last screenshot, not the connecting line, while the route reopens the page', async () => {
    getFileContentMock.mockResolvedValue(shot);
    liveMock.mockReturnValue(live({ status: 'connecting' }));
    renderView({ fallback: { path: 'orbital/shot.png', title: 'Shot' } });
    const img = await screen.findByAltText('Shot');
    expect(img).toHaveAttribute('src', 'data:image/png;base64,U0hPVA==');
    expect(screen.queryByText('Connecting to the browser…')).toBeNull();
    // No new words on screen: the same caption as any last screenshot.
    expect(screen.getByText('Last screenshot · Shot')).toBeInTheDocument();
  });

  it('keeps the screenshot when the page is attached but its first frame has not landed', async () => {
    getFileContentMock.mockResolvedValue(shot);
    liveMock.mockReturnValue(live({ status: 'open', url: 'https://example.com/', frame: null }));
    renderView({ fallback: { path: 'orbital/shot.png', title: 'Shot' } });
    expect(await screen.findByAltText('Shot')).toBeInTheDocument();
  });

  it('a click on the screenshot asks for the page back', async () => {
    getFileContentMock.mockResolvedValue(shot);
    const nav = vi.fn();
    liveMock.mockReturnValue(live({ status: 'no_browser', nav }));
    renderView({ fallback: { path: 'orbital/shot.png', title: 'Shot' } });
    const img = await screen.findByAltText('Shot');
    fireEvent.pointerDown(img, { button: 0 });
    expect(nav.mock.calls).toEqual([['reopen']]);
  });

  it('opened again (Files and back), the picture is there on the first render', async () => {
    getFileContentMock.mockResolvedValue(shot);
    liveMock.mockReturnValue(live({ status: 'connecting' }));
    renderView({ fallback: { path: 'orbital/shot.png', title: 'Shot' } });
    await screen.findByAltText('Shot');
    cleanup();
    getFileContentMock.mockReturnValue(new Promise(() => {})); // the refetch never lands
    renderView({ fallback: { path: 'orbital/shot.png', title: 'Shot' } });
    expect(screen.getByAltText('Shot')).toHaveAttribute('src', 'data:image/png;base64,U0hPVA==');
    expect(screen.queryByText('Connecting to the browser…')).toBeNull();
  });

  it('the screenshot is replaced by the live frame when it arrives', async () => {
    getFileContentMock.mockResolvedValue(shot);
    liveMock.mockReturnValue(live({ status: 'connecting' }));
    const { rerender, container } = renderView({ fallback: { path: 'orbital/shot.png', title: 'Shot' } });
    await screen.findByAltText('Shot');
    liveMock.mockReturnValue(live({
      status: 'open',
      url: 'https://example.com/',
      title: 'Example Domain',
      frame: { jpegDataUrl: 'd', width: 690, height: 820 },
    }));
    rerender(
      <LocaleProvider>
        <BrowserView projectId="p1" active fallback={{ path: 'orbital/shot.png', title: 'Shot' }} annotating={false} annotations={[]} onAddAnnotation={vi.fn()} />
      </LocaleProvider>,
    );
    expect(container.querySelector('canvas')).not.toBeNull();
    expect(screen.queryByAltText('Shot')).toBeNull();
    expect(screen.getByText('Live')).toBeInTheDocument();
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

  // 2026-09-05, from the installer: "the browser doesnt have header". Every
  // project reports no_browser until the agent browses something, and that is
  // exactly when a user opens the panel wanting to look something up — so the
  // header is there in every state, and the address field is the way in.
  it('keeps the header with no page open, and the address field opens one', () => {
    const nav = vi.fn();
    liveMock.mockReturnValue(live({ status: 'no_browser', nav }));
    renderView();

    expect(screen.getByText('No page open yet.')).toBeInTheDocument();
    const field = screen.getByLabelText('Address') as HTMLInputElement;
    expect(field).toHaveValue('');
    expect(screen.getByRole('button', { name: 'Back' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Forward' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Reload' })).toBeDisabled();

    fireEvent.change(field, { target: { value: 'example.com' } });
    fireEvent.keyDown(field, { key: 'Enter' });
    expect(nav).toHaveBeenCalledWith('goto', 'https://example.com');
  });

  it('keeps the header over a page that has gone away, with navigation disabled', () => {
    liveMock.mockReturnValue(
      live({
        status: 'closed',
        frame: { jpegDataUrl: 'd', width: 8, height: 8 },
        title: 'T',
        url: 'https://example.com/path',
        canGoBack: true,
      }),
    );
    renderView();
    expect(screen.getByLabelText('Address')).toHaveValue('https://example.com/path');
    // canGoBack is stale history from a page that is no longer there.
    expect(screen.getByRole('button', { name: 'Back' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Reload' })).toBeDisabled();
  });

  it('keeps the header over the last screenshot', async () => {
    getFileContentMock.mockResolvedValue({
      type: 'image',
      mime: 'image/png',
      content: 'AAAA',
    });
    liveMock.mockReturnValue(live({ status: 'no_browser' }));
    renderView({ fallback: { path: 'orbital/shot.png', title: 'Shot' } });
    expect(await screen.findByRole('img')).toBeInTheDocument();
    expect(screen.getByLabelText('Address')).toBeInTheDocument();
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
    // Not live: no page cursor, and not "forbidden" — a click asks for the
    // page back.
    expect(c3.style.cursor).toBe('default');
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

describe('BrowserView — the page fills the view (2026-11-18)', () => {
  // jsdom lays nothing out: give the frame area a size, as a real panel has.
  const proto = HTMLElement.prototype;
  const original = {
    width: Object.getOwnPropertyDescriptor(proto, 'clientWidth'),
    height: Object.getOwnPropertyDescriptor(proto, 'clientHeight'),
  };
  let boxW = 690;
  let boxH = 820;
  beforeEach(() => {
    vi.useFakeTimers();
    boxW = 690;
    boxH = 820;
    Object.defineProperty(proto, 'clientWidth', { configurable: true, get: () => boxW });
    Object.defineProperty(proto, 'clientHeight', { configurable: true, get: () => boxH });
  });
  afterEach(() => {
    vi.useRealTimers();
    if (original.width) Object.defineProperty(proto, 'clientWidth', original.width);
    if (original.height) Object.defineProperty(proto, 'clientHeight', original.height);
  });

  function resizeTo(w: number, h: number) {
    boxW = w;
    boxH = h;
    act(() => {
      window.dispatchEvent(new Event('resize'));
    });
  }

  const downsOf = (send: ReturnType<typeof vi.fn>) =>
    send.mock.calls
      .map(([m]) => m as { type: string; action: string; x: number; y: number })
      .filter((m) => m.type === 'mouse' && m.action === 'down');

  it('asks for its own size at once, then once per settled change — never per step', () => {
    const requestSize = vi.fn();
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1280, height: 720 },
      title: 'T',
      requestSize,
    }));
    renderView();
    expect(requestSize.mock.calls).toEqual([[690, 820, 1]]);
    // A drag: many steps inside the settle window.
    for (const w of [680, 660, 640, 620, 600]) {
      resizeTo(w, 820);
      act(() => {
        vi.advanceTimersByTime(40);
      });
    }
    expect(requestSize).toHaveBeenCalledTimes(1);
    act(() => {
      vi.advanceTimersByTime(SIZE_SETTLE_MS);
    });
    expect(requestSize.mock.calls).toEqual([[690, 820, 1], [600, 820, 1]]);
  });

  it('asks for the frame area’s size, also when the view started on a blank page', () => {
    // The frame area is smaller than the whole view (toolbar, status line, inset).
    Object.defineProperty(proto, 'clientWidth', {
      configurable: true,
      get(this: HTMLElement) { return this.dataset?.browserLive ? 662 : 690; },
    });
    Object.defineProperty(proto, 'clientHeight', {
      configurable: true,
      get(this: HTMLElement) { return this.dataset?.browserLive ? 760 : 820; },
    });
    const requestSize = vi.fn();
    const blank = live({
      status: 'open',
      url: 'about:blank',
      frame: { jpegDataUrl: 'd', width: 1280, height: 720 },
      title: '',
      requestSize,
    });
    liveMock.mockReturnValue(blank);
    const { rerender } = renderView();
    expect(requestSize).not.toHaveBeenCalled();
    liveMock.mockReturnValue({ ...blank, url: 'https://example.com/' });
    rerender(
      <LocaleProvider>
        <BrowserView projectId="p1" active fallback={null} annotating={false} annotations={[]} onAddAnnotation={vi.fn()} />
      </LocaleProvider>,
    );
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(requestSize.mock.calls).toEqual([[662, 760, 1]]);
  });

  it('reports its size from the last screenshot’s box, so a reopened page comes up at it', async () => {
    vi.useRealTimers();
    getFileContentMock.mockResolvedValue({ type: 'image', mime: 'image/png', content: 'U0hPVA==' });
    const requestSize = vi.fn();
    liveMock.mockReturnValue(live({ status: 'connecting', requestSize }));
    renderView({ fallback: { path: 'orbital/shot.png', title: 'Shot' } });
    await screen.findByAltText('Shot');
    await waitFor(() => expect(requestSize.mock.calls).toEqual([[690, 820, 1]]));
  });

  it('a new frame alone never asks again', () => {
    const requestSize = vi.fn();
    const value = live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1280, height: 720 },
      title: 'T',
      requestSize,
      owner: true,
    });
    liveMock.mockReturnValue(value);
    const { rerender } = renderView();
    for (const [w, h] of [[1280, 633], [690, 733], [690, 820]]) {
      liveMock.mockReturnValue({ ...value, frame: { jpegDataUrl: `d${w}${h}`, width: w, height: h } });
      rerender(
        <LocaleProvider>
          <BrowserView projectId="p1" active fallback={null} annotating={false} annotations={[]} onAddAnnotation={vi.fn()} />
        </LocaleProvider>,
      );
      act(() => {
        vi.advanceTimersByTime(500);
      });
    }
    expect(requestSize).toHaveBeenCalledTimes(1);
  });

  it('owner: its page is drawn 1:1, so input maps 1:1', () => {
    const send = vi.fn();
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 690, height: 820 },
      title: 'T',
      send,
      owner: true,
    }));
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    expect(canvas.style.width).toBe('690px');
    expect(canvas.style.height).toBe('820px');
    mockRect(canvas, 690, 820);
    fireEvent.pointerDown(canvas, { clientX: 12, clientY: 800, button: 0, detail: 1 });
    expect(downsOf(send)[0]).toMatchObject({ x: 12, y: 800 });
  });

  it('owner mid-drag: the last frame follows the panel edge, and input still lands on it', () => {
    const send = vi.fn();
    // The page is still 690 wide; the panel was just dragged to 600.
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 690, height: 820 },
      title: 'T',
      send,
      owner: true,
    }));
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    mockRect(canvas, 600, 820);
    // Narrowed: the frame stays 1:1 from the top-left, its right edge
    // cropped — no band anywhere — and a point maps to the same page point.
    fireEvent.pointerDown(canvas, { clientX: 300, clientY: 100, button: 0, detail: 1 });
    expect(downsOf(send)[0].x).toBeCloseTo(300, 5);
    expect(downsOf(send)[0].y).toBeCloseTo(100, 5);
    // Widened past the page (and taller): scaled up just enough to cover.
    mockRect(canvas, 1380, 1000);
    fireEvent.pointerDown(canvas, { clientX: 690, clientY: 410, button: 0, detail: 1 });
    expect(downsOf(send)[1].x).toBeCloseTo(345, 5);
    expect(downsOf(send)[1].y).toBeCloseTo(205, 5);
  });

  it('not the owner: another viewer sized the page, so it is letterboxed', () => {
    const send = vi.fn();
    liveMock.mockReturnValue(live({
      status: 'open',
      frame: { jpegDataUrl: 'd', width: 1280, height: 720 },
      title: 'T',
      send,
      owner: false,
    }));
    const { container } = renderView();
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    mockRect(canvas, 690, 820);
    // The page fits the width (690 wide, 388 tall) centred: the panel's
    // centre is the page's centre, and the top band is off the page.
    fireEvent.pointerDown(canvas, { clientX: 345, clientY: 410, button: 0, detail: 1 });
    fireEvent.pointerDown(canvas, { clientX: 345, clientY: 10, button: 0, detail: 1 });
    const downs = downsOf(send);
    expect(downs).toHaveLength(2);
    expect(downs[0].x).toBeCloseTo(640, 5);
    expect(downs[0].y).toBeCloseTo(360, 5);
    expect(downs[1].y).toBe(0);
  });
});
