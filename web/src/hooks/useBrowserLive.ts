// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.6 — live view of the agent's browser over one WS route.
 * CONTRACT FILE (protocol below is shared with the backend route):
 *   WS  /api/v2/agents/{project_id}/browser/live[?token=<relay jwt>]
 *   server → client: {type:"frame", jpeg:<base64>, width, height, title}
 *                    {type:"state", status:"no_browser"|"open"|"closed", title?}
 *                    {type:"error", message}
 *   client → server: {type:"mouse", action:"move"|"down"|"up"|"wheel", x, y, button?, clickCount?, deltaX?, deltaY?, modifiers?}
 *                    {type:"key", action:"down"|"up", key, code, text?, modifiers?}
 *                    {type:"text", text}
 *   x/y are CSS pixels of the page viewport (same space as the frame width/height).
 * Workstream D implements.
 */
export type LiveStatus = 'idle' | 'connecting' | 'open' | 'no_browser' | 'closed' | 'error';
export interface LiveFrame { jpegDataUrl: string; width: number; height: number }
export interface BrowserLive {
  status: LiveStatus;
  frame: LiveFrame | null;
  title: string;
  send: (msg: Record<string, unknown>) => void;
}

export function useBrowserLive(_projectId: string, _active: boolean): BrowserLive {
  // TODO(spec 078 workstream D)
  return { status: 'idle', frame: null, title: '', send() {} };
}
