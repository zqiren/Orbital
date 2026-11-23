// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The Welcome page's import, run in the background (batch 1123 follow-up).
 *
 * Start lands on the Workbench at once and fires `POST /onboarding/import`
 * without waiting for it (15–22 s for 114–280 folders). This module holds
 * the one request's state outside React, because the Welcome page unmounts
 * the moment Start is pressed:
 *
 *  - `idle`    — nothing running (also every fresh page load: the state is
 *                in memory only, so a reload never shows a stale spinner;
 *                the request itself lives in the daemon).
 *  - `pending` — the request is out. The sidebar's 项目 header shows a quiet
 *                spinner and App polls the project list so rows appear.
 *  - `failed`  — it answered with an error or the daemon went away. The
 *                sidebar shows a dismissible notice whose retry re-fires the
 *                SAME request (the daemon skips folders already imported).
 *
 * `finished` counts every answer (success or failure, partial creates are
 * possible either way) so App can refetch the projects and the feed once per
 * answer from an effect, without closures over updater-mutated variables.
 */

import { useSyncExternalStore } from 'react';
import { api } from '../config';
import type { ImportRequest, ImportResult } from './onboarding';

export type BackgroundImportStatus = 'idle' | 'pending' | 'failed';

export interface BackgroundImportState {
  status: BackgroundImportStatus;
  /** The last request fired: what Retry sends again. */
  request: ImportRequest | null;
  /** Bumped on every answer (settled or failed). */
  finished: number;
}

export type BackgroundImportEvent =
  | { type: 'start'; request: ImportRequest }
  | { type: 'settled' }
  | { type: 'failed' }
  | { type: 'dismiss' };

export const IDLE_IMPORT: BackgroundImportState = { status: 'idle', request: null, finished: 0 };

/** Pure transitions. A start while one is pending is ignored (one request at
 *  a time: two concurrent imports could create the same folder twice). */
export function importReducer(s: BackgroundImportState, e: BackgroundImportEvent): BackgroundImportState {
  switch (e.type) {
    case 'start':
      if (s.status === 'pending') return s;
      return { status: 'pending', request: e.request, finished: s.finished };
    case 'settled':
      if (s.status !== 'pending') return s;
      return { status: 'idle', request: s.request, finished: s.finished + 1 };
    case 'failed':
      if (s.status !== 'pending') return s;
      return { status: 'failed', request: s.request, finished: s.finished + 1 };
    case 'dismiss':
      if (s.status !== 'failed') return s;
      return { status: 'idle', request: s.request, finished: s.finished };
  }
}

/** What the sidebar's 项目 header and notice show for a state. */
export function importIndicator(s: BackgroundImportState): { spinner: boolean; failed: boolean } {
  return { spinner: s.status === 'pending', failed: s.status === 'failed' };
}

export function isEmptyImport(r: ImportRequest | null | undefined): boolean {
  return !r || r.folders.length + r.groups.length === 0;
}

// ── the store ──

let state: BackgroundImportState = IDLE_IMPORT;
const listeners = new Set<() => void>();

function dispatch(e: BackgroundImportEvent): void {
  const next = importReducer(state, e);
  if (next === state) return;
  state = next;
  listeners.forEach((l) => l());
}

export function getBackgroundImport(): BackgroundImportState {
  return state;
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => { listeners.delete(l); };
}

/** Fire the import without waiting for it. Returns the request's promise for
 *  callers (tests) that want to; the UI never awaits it. Ignored (resolves at
 *  once) while one is pending or when there is nothing to import. */
export function startBackgroundImport(request: ImportRequest): Promise<void> {
  if (isEmptyImport(request) || state.status === 'pending') return Promise.resolve();
  dispatch({ type: 'start', request });
  return api<ImportResult>('/api/v2/onboarding/import', { method: 'POST', body: JSON.stringify(request) })
    .then(() => dispatch({ type: 'settled' }), () => dispatch({ type: 'failed' }));
}

/** Retry: the same selection again. */
export function retryBackgroundImport(): Promise<void> {
  if (state.status !== 'failed' || !state.request) return Promise.resolve();
  return startBackgroundImport(state.request);
}

export function dismissBackgroundImport(): void {
  dispatch({ type: 'dismiss' });
}

export function useBackgroundImport(): BackgroundImportState {
  return useSyncExternalStore(subscribe, getBackgroundImport, getBackgroundImport);
}

/** Tests only: back to a fresh page load. */
export function resetBackgroundImportForTests(): void {
  state = IDLE_IMPORT;
  listeners.forEach((l) => l());
}
