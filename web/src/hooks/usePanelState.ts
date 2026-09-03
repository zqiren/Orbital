// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.1/§5.2 — per-session panel state + lifecycle (expand at runtime,
 * collapse at rest). CONTRACT FILE: the shape below is final; the lifecycle
 * helpers (`expandForEvent` / `onRunStart` / `onRunEnd`) are additive.
 *
 * State lives in a module-level Map keyed by `<projectId>:<sessionId>` and is
 * read through `useSyncExternalStore` (the same pattern `useAnnotations` uses),
 * so switching sessions swaps state without a remount dance and a tab
 * round-trip does not reset the panel. `view` — and only `view` — is mirrored
 * to `localStorage['orbital:panel:<projectId>:<sessionId>']`; expansion is
 * runtime state and deliberately does NOT survive a reload (D8: at rest the
 * panel is collapsed).
 */
import { useCallback, useSyncExternalStore } from 'react';
import type { PanelView } from '../utils/panelSelectors';

export interface PanelState {
  expanded: boolean;
  userCollapsedThisRun: boolean;
  view: PanelView;
  file: string | null;
}
export interface PanelStateApi extends PanelState {
  expand: () => void;
  collapse: () => void; // user intent: stays collapsed for this run
  setView: (v: PanelView) => void;
  setFile: (path: string | null) => void;
  /** A run touched the browser / a file: expand and switch, unless the user collapsed this run. */
  expandForEvent: (v: PanelView) => void;
  /** A turn started: the user's "stay collapsed" veto only lasts one run. */
  onRunStart: () => void;
  /** A turn ended: collapse back to the handle and drop the preview. */
  onRunEnd: () => void;
}

/** Below this content width the panel falls back to the overlay drawer (§5.1, §7). */
export const PANEL_DOCK_MIN_WIDTH = 1200;
/** Mirrors App.tsx's `matchMedia('(max-width: 767px)')` mobile split. */
export const PANEL_MOBILE_MAX_WIDTH = 767;

const DEFAULT_STATE: PanelState = {
  expanded: false,
  userCollapsedThisRun: false,
  view: 'files',
  file: null,
};

const store = new Map<string, PanelState>();
const listeners = new Set<() => void>();

function storageKey(projectId: string, sessionId: string | undefined) {
  return `orbital:panel:${projectId}:${sessionId ?? '__none__'}`;
}

function readSavedView(projectId: string, sessionId: string | undefined): PanelView {
  try {
    const saved = localStorage.getItem(storageKey(projectId, sessionId));
    return saved === 'browser' || saved === 'files' ? saved : DEFAULT_STATE.view;
  } catch {
    // localStorage may be unavailable in private / locked-down webviews.
    return DEFAULT_STATE.view;
  }
}

function persistView(projectId: string, sessionId: string | undefined, view: PanelView) {
  try {
    localStorage.setItem(storageKey(projectId, sessionId), view);
  } catch {
    // Same: a throwing localStorage must never break the panel.
  }
}

/**
 * Read the entry for a key, seeding it from localStorage on first touch. The
 * seeded entry is cached in the map so `getSnapshot` keeps returning the same
 * object reference (a fresh object every call would re-render forever).
 */
function get(key: string, projectId: string, sessionId: string | undefined): PanelState {
  const existing = store.get(key);
  if (existing !== undefined) return existing;
  const seeded: PanelState = { ...DEFAULT_STATE, view: readSavedView(projectId, sessionId) };
  store.set(key, seeded);
  return seeded;
}

function set(key: string, next: PanelState) {
  store.set(key, next);
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

export function usePanelState(
  projectId: string,
  sessionId: string | undefined,
): PanelStateApi {
  const key = `${projectId}:${sessionId ?? '__none__'}`;
  const read = useCallback(() => get(key, projectId, sessionId), [key, projectId, sessionId]);
  const state = useSyncExternalStore(subscribe, read, read);

  const update = useCallback(
    (patch: Partial<PanelState>) => set(key, { ...get(key, projectId, sessionId), ...patch }),
    [key, projectId, sessionId],
  );

  const expand = useCallback(() => {
    update({ expanded: true, userCollapsedThisRun: false });
  }, [update]);

  const collapse = useCallback(() => {
    update({ expanded: false, userCollapsedThisRun: true });
  }, [update]);

  const setView = useCallback(
    (v: PanelView) => {
      persistView(projectId, sessionId, v);
      update({ view: v });
    },
    [projectId, sessionId, update],
  );

  const setFile = useCallback((path: string | null) => update({ file: path }), [update]);

  const expandForEvent = useCallback(
    (v: PanelView) => {
      const current = get(key, projectId, sessionId);
      // D8: collapsing during a run keeps it collapsed for THAT run. The view
      // is left alone too — reopening should land on what the user chose.
      if (current.userCollapsedThisRun) return;
      persistView(projectId, sessionId, v);
      update({ expanded: true, view: v });
    },
    [key, projectId, sessionId, update],
  );

  const onRunStart = useCallback(() => {
    update({ userCollapsedThisRun: false });
  }, [update]);

  const onRunEnd = useCallback(() => {
    // Back to the handle, and back to the tree: "expanding at rest shows
    // Files (tree) by default" — the remembered `view` survives, the preview
    // selection does not.
    update({ expanded: false, userCollapsedThisRun: false, file: null });
  }, [update]);

  return {
    ...state,
    expand,
    collapse,
    setView,
    setFile,
    expandForEvent,
    onRunStart,
    onRunEnd,
  };
}

function readDockable(): boolean {
  if (typeof window === 'undefined') return false;
  return (
    window.innerWidth > PANEL_MOBILE_MAX_WIDTH &&
    window.innerWidth >= PANEL_DOCK_MIN_WIDTH
  );
}

/**
 * Whether the docked panel fits (§5.1: push above ~1200 px, overlay below,
 * never on mobile). Shared by ChatTab (renders the third column) and
 * ProjectDetail (suppresses its overlay drawer) so the two cannot disagree
 * and show both surfaces — or neither — for one `previewPath`.
 */
export function usePanelDockable(): boolean {
  return useSyncExternalStore(subscribeToWindowWidth, readDockable, readNotDockable);
}

/** Server render: no window, so never docked. */
function readNotDockable(): boolean {
  return false;
}

/** Current window width (0 on the server); re-renders on resize. */
export function useWindowWidth(): number {
  return useSyncExternalStore(subscribeToWindowWidth, readWindowWidth, readZeroWidth);
}
function readWindowWidth(): number {
  return typeof window === 'undefined' ? 0 : window.innerWidth;
}
function readZeroWidth(): number {
  return 0;
}

function subscribeToWindowWidth(onChange: () => void) {
  if (typeof window === 'undefined') return () => {};
  window.addEventListener('resize', onChange);
  return () => window.removeEventListener('resize', onChange);
}

/** Test helper: wipe every session's panel state. */
export function __resetPanelState() {
  store.clear();
  listeners.forEach((l) => l());
}
