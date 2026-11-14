// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 180 §5.3 — one project picker, three anchors. The composer's listbox,
 * extracted unchanged in look (`wb-pop` / `wb-opt`, the Sparkles 自动 row,
 * Folder rows, the check mark):
 *
 *   | anchor                          | 自动 row          | current row |
 *   | composer chip                   | yes (keyless: off) | the chip    |
 *   | a card's project while picking  | yes, ticked, inert | none        |
 *   | a card's project after routing  | no                 | ticked      |
 *   | a session row's ⋯ → 移动到项目… | no                 | ticked      |
 *
 * The composer renders `ProjectPicker` in place, as before. The card and the
 * session row open `AnchoredProjectPicker`: the same list in a portal, placed
 * below its anchor when there is room, else above (the list scrolls); the
 * session row's opens beside the row on a wide screen. It closes on a press
 * outside, Esc, a scroll or a resize. Picking the current project only
 * closes it (the caller decides; `onPick` is not called for it).
 */

import { useCallback, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { Check, Folder, Sparkles } from 'lucide-react';
import type { Project } from '../../types';
import { useT } from '../../i18n/useT';
import { projectDisplayName } from '../../utils/projectDisplayName';
import './workbench.css';

export type AutoRow = 'none' | 'choose' | 'ticked' | 'keyless';

export interface ProjectPickerProps {
  projects: readonly Project[];
  /** The ticked project, if any. */
  selectedId: string | null;
  /** The 自动 row: absent, selectable, ticked and inert, or keyless (shown, off). */
  auto?: AutoRow;
  /** 'choose': whether 自动 is the current value. */
  autoSelected?: boolean;
  /** 'choose': 自动 picked. 'keyless': the click (opens the model settings). */
  onAuto?: () => void;
  onPick: (projectId: string) => void;
  style?: CSSProperties;
  className?: string;
  testId?: string;
}

function ck(on: boolean) {
  return <span className="wb-ck">{on ? <Check size={14} strokeWidth={2.4} aria-hidden="true" /> : null}</span>;
}

export default function ProjectPicker({
  projects, selectedId, auto = 'none', autoSelected = false, onAuto, onPick, style, className, testId,
}: ProjectPickerProps) {
  const t = useT();
  return (
    <div
      className={`wb-pop${className ? ` ${className}` : ''}`}
      role="listbox"
      aria-label={t('workbench.composer.project')}
      style={style}
      data-testid={testId}
    >
      {auto === 'keyless' && (
        // Routing needs a key: 自动 stays visible, not selectable; the
        // click goes where the key is added.
        <button
          type="button"
          role="option"
          aria-selected={false}
          aria-disabled="true"
          className="wb-opt is-off"
          data-testid="wb-opt-auto"
          onClick={onAuto}
        >
          <span className="wb-gl"><Sparkles size={13} strokeWidth={1.9} aria-hidden="true" /></span>
          <span className="wb-tx"><span className="wb-n1">{t('workbench.composer.auto')}</span></span>
          <span className="wb-note">{t('workbench.composer.needsKey')}</span>
          {ck(false)}
        </button>
      )}
      {(auto === 'choose' || auto === 'ticked') && (
        <button
          type="button"
          role="option"
          aria-selected={auto === 'ticked' || autoSelected}
          className="wb-opt"
          data-testid="wb-opt-auto"
          onClick={onAuto}
        >
          <span className="wb-gl"><Sparkles size={13} strokeWidth={1.9} aria-hidden="true" /></span>
          <span className="wb-tx"><span className="wb-n1">{t('workbench.composer.auto')}</span></span>
          {ck(auto === 'ticked' || autoSelected)}
        </button>
      )}
      {auto !== 'none' && <div className="wb-sep" />}
      {projects.map((p) => (
        <button
          key={p.project_id}
          type="button"
          role="option"
          aria-selected={selectedId === p.project_id}
          className="wb-opt"
          title={projectDisplayName(p, t)}
          data-testid={`wb-opt-project-${p.project_id}`}
          onClick={() => onPick(p.project_id)}
        >
          <span className="wb-gl"><Folder size={13} strokeWidth={1.9} aria-hidden="true" /></span>
          <span className="wb-tx"><span className="wb-n1">{projectDisplayName(p, t)}</span></span>
          {ck(selectedId === p.project_id)}
        </button>
      ))}
    </div>
  );
}

/** Where the anchored picker opens: under the anchor (the card's name), or
 *  beside it (a session row, on a screen wide enough). */
export type Placement = 'below' | 'side';

const MARGIN = 8;
const GAP = 6;

/** The prototype's `place` (l.873-887): below when there is reasonable room
 *  (the list scrolls), else above. */
export function placePopover(
  anchor: DOMRect, size: { w: number; h: number }, view: { w: number; h: number }, placement: Placement,
): { left: number; top: number; maxHeight: number | undefined } {
  if (placement === 'side' && anchor.right + GAP + size.w + MARGIN <= view.w) {
    const top = Math.max(MARGIN, Math.min(anchor.top - GAP, view.h - size.h - MARGIN));
    return { left: anchor.right + GAP, top, maxHeight: size.h > view.h - 2 * MARGIN ? view.h - 2 * MARGIN : undefined };
  }
  const left = Math.min(Math.max(MARGIN, anchor.left - GAP), view.w - size.w - MARGIN);
  const below = view.h - MARGIN - (anchor.bottom + GAP);
  const above = anchor.top - GAP - MARGIN;
  if (size.h <= below) return { left, top: anchor.bottom + GAP, maxHeight: undefined };
  if (below >= 200 || below >= above) return { left, top: anchor.bottom + GAP, maxHeight: below };
  const h = Math.min(size.h, above);
  return { left, top: Math.max(MARGIN, anchor.top - GAP - h), maxHeight: h < size.h ? h : undefined };
}

export interface AnchoredProjectPickerProps extends ProjectPickerProps {
  anchor: HTMLElement | null;
  onClose: () => void;
  placement?: Placement;
}

/** The same list, opened from a card or a session row, in a portal. */
export function AnchoredProjectPicker({
  anchor, onClose, placement = 'below', ...list
}: AnchoredProjectPickerProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number; maxHeight: number | undefined } | null>(null);
  const closeRef = useRef(onClose);
  useLayoutEffect(() => { closeRef.current = onClose; });

  const place = useCallback(() => {
    const pop = hostRef.current?.firstElementChild as HTMLElement | null;
    if (!anchor || !pop) return;
    const prev = pop.style.maxHeight;
    pop.style.maxHeight = '';
    const size = { w: pop.offsetWidth || 260, h: pop.offsetHeight };
    pop.style.maxHeight = prev;
    setPos(placePopover(anchor.getBoundingClientRect(), size, { w: window.innerWidth, h: window.innerHeight }, placement));
  }, [anchor, placement]);

  useLayoutEffect(() => { place(); }, [place, list.projects.length]);

  useLayoutEffect(() => {
    const onPointer = (e: Event) => {
      const path = e.composedPath();
      if (hostRef.current && path.includes(hostRef.current)) return;
      if (anchor && path.includes(anchor)) return;
      closeRef.current();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeRef.current(); };
    const onScroll = (e: Event) => {
      // The list's own scroll is not a reason to close.
      if (hostRef.current && e.target instanceof Node && hostRef.current.contains(e.target)) return;
      closeRef.current();
    };
    const onResize = () => closeRef.current();
    document.addEventListener('pointerdown', onPointer);
    window.addEventListener('keydown', onKey);
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    return () => {
      document.removeEventListener('pointerdown', onPointer);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onResize);
    };
  }, [anchor]);

  const style: CSSProperties = pos
    ? { left: pos.left, top: pos.top, maxHeight: pos.maxHeight }
    : { left: -9999, top: -9999 };
  return createPortal(
    <div
      ref={hostRef}
      className="wb-pophost"
      // A portal's events still bubble through the React tree: a pick must not
      // also select the card or open the session row under it.
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.stopPropagation()}
    >
      <ProjectPicker {...list} style={style} className="is-anchored" />
    </div>,
    document.body,
  );
}
