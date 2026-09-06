// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';

vi.mock('../config', () => ({
  BASE_URL: 'http://localhost:8000',
  isRelayMode: false,
}));

import { useBrowserLive } from './useBrowserLive';

// ---------------------------------------------------------------------------
// Fake WebSocket — assigned to globalThis.WebSocket. Mirrors the browser API
// surface the hook touches: constructor(url), send, close, readyState, and
// the on* callback properties (the hook assigns functions directly rather
// than addEventListener, so tests drive it the same way).
// ---------------------------------------------------------------------------
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState = FakeWebSocket.CONNECTING;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private closedByClient = false;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.closedByClient = true;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }

  /** Test helper: server accepts the connection. */
  simulateOpen() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  /** Test helper: server sends a JSON message. */
  simulateMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }

  /** Test helper: the connection drops unexpectedly (not via close()). */
  simulateDrop() {
    if (this.closedByClient) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  (globalThis as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;
  try {
    localStorage.clear();
  } catch {
    // ignore
  }
});

afterEach(() => {
  vi.useRealTimers();
});

describe('useBrowserLive connection lifecycle', () => {
  it('does not connect while inactive', () => {
    renderHook(({ active }) => useBrowserLive('p1', active), {
      initialProps: { active: false },
    });
    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it('connects only once active becomes true, to the live route', () => {
    const { rerender, result } = renderHook(({ active }) => useBrowserLive('p1', active), {
      initialProps: { active: false },
    });
    expect(FakeWebSocket.instances).toHaveLength(0);

    act(() => {
      rerender({ active: true });
    });

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0].url).toBe(
      'ws://localhost:8000/api/v2/agents/p1/browser/live',
    );
    expect(result.current.status).toBe('connecting');
  });

  it('closes the socket when it becomes inactive', () => {
    const { result, rerender } = renderHook(({ active }) => useBrowserLive('p1', active), {
      initialProps: { active: true },
    });
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());

    act(() => {
      rerender({ active: false });
    });

    expect(ws.readyState).toBe(FakeWebSocket.CLOSED);
    expect(result.current.status).toBe('idle');
  });

  it('closes the socket on unmount', () => {
    const { unmount } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());

    unmount();

    expect(ws.readyState).toBe(FakeWebSocket.CLOSED);
  });
});

describe('useBrowserLive message parsing', () => {
  it('parses a frame message into a data URL, dimensions, and title', () => {
    const { result } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());

    act(() => {
      ws.simulateMessage({ type: 'frame', jpeg: 'abcd1234', width: 1024, height: 768, title: 'Example' });
    });

    expect(result.current.frame).toEqual({
      jpegDataUrl: 'data:image/jpeg;base64,abcd1234',
      width: 1024,
      height: 768,
    });
    expect(result.current.title).toBe('Example');
    expect(result.current.status).toBe('open');
  });

  it('updates status and title from a state message', () => {
    const { result } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());

    act(() => {
      ws.simulateMessage({ type: 'state', status: 'no_browser', title: 'Idle tab' });
    });

    expect(result.current.status).toBe('no_browser');
    expect(result.current.title).toBe('Idle tab');
  });

  it('sets status to error on an error message', () => {
    const { result } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());

    act(() => {
      ws.simulateMessage({ type: 'error', message: 'boom' });
    });

    expect(result.current.status).toBe('error');
  });

  it('keeps only the latest frame (no queue)', () => {
    const { result } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());

    act(() => {
      ws.simulateMessage({ type: 'frame', jpeg: 'first', width: 10, height: 10 });
      ws.simulateMessage({ type: 'frame', jpeg: 'second', width: 20, height: 20 });
    });

    expect(result.current.frame?.jpegDataUrl).toBe('data:image/jpeg;base64,second');
  });
});

describe('useBrowserLive send()', () => {
  it('drops the message when the socket is not open', () => {
    const { result } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    // Still CONNECTING — never opened.
    act(() => {
      result.current.send({ type: 'mouse', action: 'move', x: 1, y: 2 });
    });
    expect(ws.sent).toHaveLength(0);
  });

  it('sends JSON-serialized messages once open', () => {
    const { result } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());

    act(() => {
      result.current.send({ type: 'mouse', action: 'down', x: 5, y: 6 });
    });

    expect(ws.sent).toEqual([JSON.stringify({ type: 'mouse', action: 'down', x: 5, y: 6 })]);
  });
});

describe('useBrowserLive reconnection backoff', () => {
  it('reconnects with 1s, then 2s, then 4s backoff while active', () => {
    vi.useFakeTimers();
    renderHook(() => useBrowserLive('p1', true));
    expect(FakeWebSocket.instances).toHaveLength(1);

    // Unexpected drop — not an explicit close() from the hook.
    act(() => {
      FakeWebSocket.instances[0].simulateDrop();
    });
    act(() => {
      vi.advanceTimersByTime(999);
    });
    expect(FakeWebSocket.instances).toHaveLength(1);
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(FakeWebSocket.instances).toHaveLength(2);

    act(() => {
      FakeWebSocket.instances[1].simulateDrop();
    });
    act(() => {
      vi.advanceTimersByTime(1999);
    });
    expect(FakeWebSocket.instances).toHaveLength(2);
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(FakeWebSocket.instances).toHaveLength(3);

    act(() => {
      FakeWebSocket.instances[2].simulateDrop();
    });
    act(() => {
      vi.advanceTimersByTime(3999);
    });
    expect(FakeWebSocket.instances).toHaveLength(3);
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(FakeWebSocket.instances).toHaveLength(4);
  });

  it('does not reconnect once inactive', () => {
    vi.useFakeTimers();
    const { rerender } = renderHook(({ active }) => useBrowserLive('p1', active), {
      initialProps: { active: true },
    });
    act(() => {
      FakeWebSocket.instances[0].simulateDrop();
    });
    act(() => {
      rerender({ active: false });
    });
    act(() => {
      vi.advanceTimersByTime(20000);
    });
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});

describe('useBrowserLive relay token', () => {
  it('appends the relay JWT as a query param when in relay mode', async () => {
    vi.resetModules();
    vi.doMock('../config', () => ({ BASE_URL: 'http://localhost:8000', isRelayMode: true }));
    localStorage.setItem('relay_jwt', 'tok-123');
    const { useBrowserLive: useBrowserLiveRelay } = await import('./useBrowserLive');

    renderHook(() => useBrowserLiveRelay('p1', true));

    expect(FakeWebSocket.instances[0].url).toBe(
      'ws://localhost:8000/api/v2/agents/p1/browser/live?token=tok-123',
    );
    vi.doUnmock('../config');
  });
});

describe('useBrowserLive — navigation state, cursor, nav()', () => {
  it('parses loading and history flags from a state message', () => {
    const { result } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());
    expect(result.current.loading).toBe(false);
    expect(result.current.canGoBack).toBe(false);
    act(() =>
      ws.simulateMessage({
        type: 'state', status: 'open', title: 'T', url: 'https://a.example/',
        loading: true, canGoBack: true, canGoForward: false,
      }),
    );
    expect(result.current.loading).toBe(true);
    expect(result.current.canGoBack).toBe(true);
    expect(result.current.canGoForward).toBe(false);
    expect(result.current.url).toBe('https://a.example/');
  });

  it('mirrors the cursor message and resets it when the page goes away', () => {
    const { result } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());
    expect(result.current.cursor).toBe('default');
    act(() => ws.simulateMessage({ type: 'cursor', cursor: 'pointer' }));
    expect(result.current.cursor).toBe('pointer');
    act(() => ws.simulateMessage({ type: 'state', status: 'closed' }));
    expect(result.current.cursor).toBe('default');
  });

  it('nav() sends a nav message, with the url only for goto', () => {
    const { result } = renderHook(() => useBrowserLive('p1', true));
    const ws = FakeWebSocket.instances[0];
    act(() => ws.simulateOpen());
    act(() => {
      result.current.nav('back');
      result.current.nav('goto', 'https://b.example/');
    });
    expect(ws.sent.map((s) => JSON.parse(s))).toEqual([
      { type: 'nav', action: 'back' },
      { type: 'nav', action: 'goto', url: 'https://b.example/' },
    ]);
  });
});
