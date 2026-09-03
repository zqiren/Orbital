// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

// Spec 078 §5.2/§5.4 — the annotation drafts are PER SESSION (D14) and shared
// between the panel that stages them and the composer that sends them, so the
// isolation between sessions is the load-bearing property here.

import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { useAnnotations, __resetAnnotationsStore } from './useAnnotations';
import type { AnnotationDraft } from '../utils/annotations';

const textDraft: AnnotationDraft = {
  kind: 'text',
  path: 'a.ts',
  text: 'const x = 1',
  lines: [3, 3],
  note: '',
};
const fileDraft: AnnotationDraft = { kind: 'file', path: 'b.pdf', note: 'later' };

beforeEach(() => {
  __resetAnnotationsStore();
});

describe('useAnnotations', () => {
  it('numbers annotations from 1 and returns the number it assigned', () => {
    const { result } = renderHook(() => useAnnotations('s1'));
    let first = 0;
    let second = 0;
    act(() => {
      first = result.current.add(textDraft);
      second = result.current.add(fileDraft);
    });
    expect([first, second]).toEqual([1, 2]);
    expect(result.current.annotations.map((a) => a.n)).toEqual([1, 2]);
    expect(result.current.annotations[0]).toMatchObject({ kind: 'text', path: 'a.ts', n: 1 });
  });

  it('removes by number and keeps the numbers of the survivors', () => {
    const { result } = renderHook(() => useAnnotations('s1'));
    act(() => {
      result.current.add(textDraft);
      result.current.add(fileDraft);
    });
    act(() => result.current.remove(1));
    expect(result.current.annotations.map((a) => a.n)).toEqual([2]);
    // The next one keeps counting up — numbers are identities, not positions.
    act(() => {
      result.current.add(fileDraft);
    });
    expect(result.current.annotations.map((a) => a.n)).toEqual([2, 3]);
  });

  it('updateNote edits one annotation in place and ignores unknown numbers', () => {
    const { result } = renderHook(() => useAnnotations('s1'));
    act(() => {
      result.current.add(textDraft);
      result.current.add(fileDraft);
    });
    act(() => result.current.updateNote(2, 'use the short label'));
    expect(result.current.annotations.map((a) => a.note)).toEqual(['', 'use the short label']);
    act(() => result.current.updateNote(99, 'nowhere'));
    expect(result.current.annotations.map((a) => a.note)).toEqual(['', 'use the short label']);
  });

  it('clear empties the list, resets the numbering and leaves the mode off', () => {
    const { result } = renderHook(() => useAnnotations('s1'));
    act(() => {
      result.current.setAnnotating(true);
      result.current.add(textDraft);
    });
    expect(result.current.annotating).toBe(true);
    act(() => result.current.clear());
    expect(result.current.annotations).toEqual([]);
    expect(result.current.annotating).toBe(false);
    act(() => {
      result.current.add(fileDraft);
    });
    expect(result.current.annotations[0].n).toBe(1);
  });

  it('tracks the annotating mode', () => {
    const { result } = renderHook(() => useAnnotations('s1'));
    expect(result.current.annotating).toBe(false);
    act(() => result.current.setAnnotating(true));
    expect(result.current.annotating).toBe(true);
    act(() => result.current.setAnnotating(false));
    expect(result.current.annotating).toBe(false);
  });

  it('keeps sessions isolated — drafts never leak between them', () => {
    const a = renderHook(() => useAnnotations('s1'));
    const b = renderHook(() => useAnnotations('s2'));
    act(() => {
      a.result.current.add(textDraft);
      a.result.current.setAnnotating(true);
    });
    expect(a.result.current.annotations).toHaveLength(1);
    expect(b.result.current.annotations).toHaveLength(0);
    expect(b.result.current.annotating).toBe(false);

    // Both sessions number from 1 independently.
    act(() => {
      b.result.current.add(fileDraft);
    });
    expect(b.result.current.annotations[0].n).toBe(1);
    act(() => a.result.current.clear());
    expect(b.result.current.annotations).toHaveLength(1);
  });

  it('shares one list between two consumers of the same session', () => {
    const panel = renderHook(() => useAnnotations('s1'));
    const composer = renderHook(() => useAnnotations('s1'));
    act(() => {
      panel.result.current.add(textDraft);
    });
    expect(composer.result.current.annotations).toHaveLength(1);
    act(() => composer.result.current.remove(1));
    expect(panel.result.current.annotations).toHaveLength(0);
  });

  it('treats an unresolved session id as its own bucket without throwing', () => {
    const { result } = renderHook(() => useAnnotations(undefined));
    act(() => {
      result.current.add(fileDraft);
    });
    expect(result.current.annotations).toHaveLength(1);
    const named = renderHook(() => useAnnotations('s1'));
    expect(named.result.current.annotations).toHaveLength(0);
  });
});
