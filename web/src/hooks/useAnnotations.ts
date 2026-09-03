// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.2/§5.4 — per-session annotation drafts + the "annotating" mode.
 * A tiny module-level store (useSyncExternalStore) so the panel (ChatTab) and
 * the composer (ChatView) share one state per session without prop drilling.
 * CONTRACT: the API shape is final; workstream E may extend, never rename.
 */
import { useCallback, useSyncExternalStore } from 'react';
import type { Annotation, AnnotationDraft } from '../utils/annotations';

export interface AnnotationsApi {
  annotating: boolean;
  setAnnotating: (on: boolean) => void;
  annotations: Annotation[];
  add: (draft: AnnotationDraft) => number;
  remove: (n: number) => void;
  clear: () => void;
  /**
   * Edit a draft's note after the fact — the composer chip expands into an
   * editable note per annotation, so the note the overlay captured is not the
   * last word on it.
   */
  updateNote: (n: number, note: string) => void;
}

interface SessionState { annotating: boolean; annotations: Annotation[]; next: number }
const EMPTY: SessionState = { annotating: false, annotations: [], next: 1 };
const store = new Map<string, SessionState>();
const listeners = new Set<() => void>();
function key(sessionId: string | undefined) { return sessionId ?? '__none__'; }
function get(sessionId: string | undefined): SessionState { return store.get(key(sessionId)) ?? EMPTY; }
function set(sessionId: string | undefined, next: SessionState) {
  store.set(key(sessionId), next);
  listeners.forEach((l) => l());
}
function subscribe(l: () => void) { listeners.add(l); return () => { listeners.delete(l); }; }

export function useAnnotations(sessionId: string | undefined): AnnotationsApi {
  const state = useSyncExternalStore(subscribe, () => get(sessionId), () => get(sessionId));
  const setAnnotating = useCallback((on: boolean) => set(sessionId, { ...get(sessionId), annotating: on }), [sessionId]);
  const add = useCallback((draft: AnnotationDraft) => {
    const cur = get(sessionId);
    const n = cur.next;
    set(sessionId, { ...cur, next: n + 1, annotations: [...cur.annotations, { ...draft, n } as Annotation] });
    return n;
  }, [sessionId]);
  const remove = useCallback((n: number) => {
    const cur = get(sessionId);
    set(sessionId, { ...cur, annotations: cur.annotations.filter((a) => a.n !== n) });
  }, [sessionId]);
  const clear = useCallback(() => set(sessionId, { ...get(sessionId), annotations: [], next: 1, annotating: false }), [sessionId]);
  const updateNote = useCallback((n: number, note: string) => {
    const cur = get(sessionId);
    if (!cur.annotations.some((a) => a.n === n)) return;
    set(sessionId, {
      ...cur,
      annotations: cur.annotations.map((a) => (a.n === n ? { ...a, note } : a)),
    });
  }, [sessionId]);
  return { annotating: state.annotating, setAnnotating, annotations: state.annotations, add, remove, clear, updateNote };
}

/** Test helper: wipe all sessions. */
export function __resetAnnotationsStore() { store.clear(); listeners.forEach((l) => l()); }

/**
 * Test helper: seed a session's drafts without driving the hook. Lets a
 * component test (the composer) start from "the panel already staged these"
 * without rendering the panel.
 */
export function __seedAnnotations(sessionId: string | undefined, annotations: Annotation[]) {
  const next = annotations.reduce((m, a) => Math.max(m, a.n + 1), 1);
  set(sessionId, { annotating: false, annotations, next });
}
