// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §6 — pure selectors over the session's ChatMessage[] the chat
 * already holds. CONTRACT FILE: signatures are final; workstream B implements.
 */
import type { ActivityEvent, ChatMessage } from '../types';

export type TouchedOp = 'read' | 'edited' | 'written';
export interface TouchedFile { path: string; op: TouchedOp; lastAt?: string }
export type PanelView = 'files' | 'browser';

/** Files this session touched, newest first, one row per path (strongest op wins: written > edited > read). */
export function touchedFiles(_messages: ChatMessage[]): TouchedFile[] {
  return []; // TODO(spec 078 workstream B)
}

/** Latest browser screenshot for the session (from tool-result `_meta`), or null. */
export function latestScreenshot(_messages: ChatMessage[]): { path: string; url?: string; title?: string } | null {
  return null; // TODO(spec 078 workstream B)
}

/** Which panel view a live activity event belongs to; null when it should not move the panel. */
export function viewForEvent(_event: ActivityEvent): PanelView | null {
  return null; // TODO(spec 078 workstream B)
}
