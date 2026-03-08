// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { WS_URL, isRelayMode } from '../config';
import type { WebSocketEvent } from '../types';

export type ConnectionState = 'connected' | 'connecting' | 'disconnected';

type EventType = WebSocketEvent['type'];
type ListenerFn = (event: WebSocketEvent) => void;

interface WebSocketContextValue {
  connectionState: ConnectionState;
  subscribe: (projectIds: string[]) => void;
  on: (type: EventType, fn: ListenerFn) => void;
  off: (type: EventType, fn: ListenerFn) => void;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const wsRef = useRef<WebSocket | null>(null);
  const listenersRef = useRef<Map<EventType, Set<ListenerFn>>>(new Map());
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef(1000);
  const subscribedIdsRef = useRef<string[]>([]);
  const mountedRef = useRef(true);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const dispatch = useCallback((event: WebSocketEvent) => {
    const handlers = listenersRef.current.get(event.type);
    if (handlers) {
      handlers.forEach((fn) => fn(event));
    }
  }, []);

  const sendSubscribe = useCallback((ids: string[]) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'subscribe', project_ids: ids }));
    }
  }, []);

  const connect = useCallback(() => {
    // Close any existing connection before creating a new one
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionState('connecting');
    let wsUrl = WS_URL;
    if (isRelayMode) {
      try {
        const token = localStorage.getItem('relay_jwt');
        if (token) {
          wsUrl += (wsUrl.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token);
        }
      } catch {
        // localStorage unavailable — proceed without token
      }
    }
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setConnectionState('connected');
      backoffRef.current = 1000;
      if (subscribedIdsRef.current.length > 0) {
        sendSubscribe(subscribedIdsRef.current);
      }
    };

    ws.onmessage = (e) => {
      if (!mountedRef.current) return;
      try {
        const parsed = JSON.parse(e.data);
        // Respond to server heartbeat
        if (parsed.type === 'ping') {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'pong' }));
          }
          return;
        }
        // Unwrap legacy relay event envelope: {type: "event", event: {...}}
        const event: WebSocketEvent = (parsed.type === 'event' && parsed.event?.type)
          ? parsed.event
          : parsed;
        if (event.type) {
          const seq = (event as unknown as Record<string, unknown>).seq;
          if (event.type === 'chat.stream_delta' || event.type === 'agent.status') {
            console.log(`[ws] received type=${event.type} seq=${seq ?? ''}`);
          }
          dispatch(event);
        }
      } catch {
        // non-JSON or malformed message — ignore
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      wsRef.current = null;
      setConnectionState('disconnected');
      const delay = backoffRef.current;
      backoffRef.current = Math.min(delay * 2, 30000);
      reconnectTimerRef.current = setTimeout(() => {
        if (mountedRef.current) connect();
      }, delay);
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, [dispatch, sendSubscribe]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearReconnectTimer();
      if (wsRef.current) {
        // Null out handlers before closing to prevent any late async
        // callbacks from firing (e.g. onclose scheduling a reconnect)
        wsRef.current.onopen = null;
        wsRef.current.onmessage = null;
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect, clearReconnectTimer]);

  const subscribe = useCallback(
    (projectIds: string[]) => {
      subscribedIdsRef.current = projectIds;
      sendSubscribe(projectIds);
    },
    [sendSubscribe],
  );

  const on = useCallback((type: EventType, fn: ListenerFn) => {
    let set = listenersRef.current.get(type);
    if (!set) {
      set = new Set();
      listenersRef.current.set(type, set);
    }
    set.add(fn);
  }, []);

  const off = useCallback((type: EventType, fn: ListenerFn) => {
    const set = listenersRef.current.get(type);
    if (set) {
      set.delete(fn);
      if (set.size === 0) listenersRef.current.delete(type);
    }
  }, []);

  const value: WebSocketContextValue = {
    connectionState,
    subscribe,
    on,
    off,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket(): WebSocketContextValue {
  const ctx = useContext(WebSocketContext);
  if (!ctx) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return ctx;
}
