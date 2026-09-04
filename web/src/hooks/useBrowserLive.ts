// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.6 — live view of the agent's browser over one WS route.
 * CONTRACT FILE (protocol below is shared with the backend route):
 *   WS  /api/v2/agents/{project_id}/browser/live[?token=<relay jwt>]
 *   server → client: {type:"frame", jpeg:<base64>, width, height, title, url}
 *                    {type:"state", status:"no_browser"|"open"|"closed", title?, url?,
 *                                   loading?, canGoBack?, canGoForward?}
 *                    {type:"cursor", cursor:<css cursor keyword under the pointer>}
 *                    {type:"error", message}
 *   client → server: {type:"mouse", action:"move"|"down"|"up"|"wheel", x, y, button?, clickCount?, deltaX?, deltaY?, modifiers?}
 *                    {type:"key", action:"down"|"up", key, code, text?, modifiers?}
 *                    {type:"text", text}
 *                    {type:"nav", action:"back"|"forward"|"reload"|"stop"|"goto", url?}
 *                    {type:"viewport", width, height, dpr}  — size the page to the panel (CSS px)
 *   x/y are CSS pixels of the page viewport (same space as the frame width/height).
 * Workstream D implements.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { BASE_URL, isRelayMode } from '../config';

export type LiveStatus = 'idle' | 'connecting' | 'open' | 'no_browser' | 'closed' | 'error';
export type NavAction = 'back' | 'forward' | 'reload' | 'stop' | 'goto';
export interface LiveFrame { jpegDataUrl: string; width: number; height: number }
export interface BrowserLive {
  status: LiveStatus;
  frame: LiveFrame | null;
  title: string;
  /** The page's URL; 'about:blank' (or '') means the agent has nothing open yet. */
  url?: string;
  /** Main frame is loading (reload shows as stop while true). */
  loading?: boolean;
  canGoBack?: boolean;
  canGoForward?: boolean;
  /** The page's CSS cursor under the pointer, mirrored onto the canvas. */
  cursor?: string;
  send: (msg: Record<string, unknown>) => void;
  /** Plain browser navigation on the agent's page. */
  nav: (action: NavAction, url?: string) => void;
}

const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 10000;

/** Mirrors useWebSocket.tsx's WS_URL derivation (~line 32-40) for the live-browser route. */
function buildWsUrl(projectId: string): string {
  let url =
    BASE_URL.replace(/^http/, 'ws') +
    `/api/v2/agents/${encodeURIComponent(projectId)}/browser/live`;
  if (isRelayMode) {
    try {
      const token = localStorage.getItem('relay_jwt');
      if (token) {
        url += (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token);
      }
    } catch {
      // localStorage unavailable — proceed without token
    }
  }
  return url;
}

export function useBrowserLive(projectId: string, active: boolean): BrowserLive {
  const [status, setStatus] = useState<LiveStatus>('idle');
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const [title, setTitle] = useState('');
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [canGoBack, setCanGoBack] = useState(false);
  const [canGoForward, setCanGoForward] = useState(false);
  const [cursor, setCursor] = useState('default');

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef(INITIAL_BACKOFF_MS);

  // Read the latest `active` from inside stable callbacks (e.g. the onclose
  // reconnect handler) without pulling it into their dependency arrays —
  // frame/title/status update ~10x/sec and must never retrigger the connect
  // effect.
  const activeRef = useRef(active);
  activeRef.current = active;

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const closeSocket = useCallback(() => {
    const ws = wsRef.current;
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onclose = null;
      ws.onerror = null;
      ws.close();
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (!activeRef.current) return;
    closeSocket();
    clearReconnectTimer();
    setStatus('connecting');

    const ws = new WebSocket(buildWsUrl(projectId));
    wsRef.current = ws;

    ws.onopen = () => {
      // A fresh connection succeeded — reset the backoff for the next drop.
      backoffRef.current = INITIAL_BACKOFF_MS;
    };

    ws.onmessage = (event: MessageEvent) => {
      let payload: unknown;
      try {
        payload = JSON.parse(typeof event.data === 'string' ? event.data : '');
      } catch {
        return;
      }
      if (!payload || typeof payload !== 'object') return;
      const msg = payload as Record<string, unknown>;

      if (msg.type === 'frame') {
        const jpeg = typeof msg.jpeg === 'string' ? msg.jpeg : '';
        const width = typeof msg.width === 'number' ? msg.width : 0;
        const height = typeof msg.height === 'number' ? msg.height : 0;
        // Keep the latest frame only — replacing state, never queuing.
        setFrame({ jpegDataUrl: 'data:image/jpeg;base64,' + jpeg, width, height });
        if (typeof msg.title === 'string') setTitle(msg.title);
        if (typeof msg.url === 'string') setUrl(msg.url);
        setStatus('open');
      } else if (msg.type === 'state') {
        if (typeof msg.status === 'string') setStatus(msg.status as LiveStatus);
        if (typeof msg.title === 'string') setTitle(msg.title);
        if (typeof msg.url === 'string') setUrl(msg.url);
        if (typeof msg.loading === 'boolean') setLoading(msg.loading);
        if (typeof msg.canGoBack === 'boolean') setCanGoBack(msg.canGoBack);
        if (typeof msg.canGoForward === 'boolean') setCanGoForward(msg.canGoForward);
        if (msg.status !== 'open') setCursor('default');
      } else if (msg.type === 'cursor') {
        if (typeof msg.cursor === 'string') setCursor(msg.cursor);
      } else if (msg.type === 'error') {
        setStatus('error');
      }
    };

    ws.onerror = () => {
      setStatus('error');
    };

    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
      if (!activeRef.current) {
        setStatus('idle');
        return;
      }
      setStatus('connecting');
      const delay = backoffRef.current;
      backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
      reconnectTimerRef.current = setTimeout(() => {
        connectRef.current();
      }, delay);
    };
  }, [projectId, closeSocket, clearReconnectTimer]);

  // `ws.onclose` above calls `connectRef.current()` rather than `connect`
  // directly — `connect` and the reconnect scheduling are mutually
  // recursive, and a ref sidesteps the useCallback dependency cycle while
  // always invoking the latest `connect`.
  const connectRef = useRef(connect);
  connectRef.current = connect;

  useEffect(() => {
    if (active) {
      backoffRef.current = INITIAL_BACKOFF_MS;
      connect();
    } else {
      clearReconnectTimer();
      closeSocket();
      setStatus('idle');
    }
    return () => {
      clearReconnectTimer();
      closeSocket();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, projectId, connect, closeSocket, clearReconnectTimer]);

  const send = useCallback((msg: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
    // Deliberately no queue: stale input is worse than lost input.
  }, []);

  const nav = useCallback(
    (action: NavAction, target?: string) => {
      send(action === 'goto' ? { type: 'nav', action, url: target } : { type: 'nav', action });
    },
    [send],
  );

  return { status, frame, title, url, loading, canGoBack, canGoForward, cursor, send, nav };
}
