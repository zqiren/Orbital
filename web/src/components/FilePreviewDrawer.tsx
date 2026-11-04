// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { ChevronRight, X } from 'lucide-react';
import type { FileContent } from '../types';
import FilePreview from './FilePreview';
import { useT } from '../i18n/useT';

const FILE_PREVIEW_DRAWER_WIDTH_KEY = 'orbital:filePreviewDrawerWidth';
/** Docked panel width is remembered separately from the overlay drawer's. */
const WORKSPACE_PANEL_WIDTH_KEY = 'orbital:workspacePanelWidth';
/** Reserved for what sits left of the docked panel: strip 20 + sessions 260 + chat 520. */
const DOCKED_RESERVED_WIDTH = 800;
const DEFAULT_DRAWER_WIDTH = 420;
/** Spec 078 §5.1: docked, the panel starts narrower than the overlay drawer. */
const DEFAULT_DOCKED_WIDTH = 360;
const MIN_DRAWER_WIDTH = 320;
const MAX_DRAWER_WIDTH_RATIO = 0.8;
const KEYBOARD_RESIZE_STEP = 24;

function clampPreviewDrawerWidth(width: number, availableWidth: number) {
  const maxWidth = Math.max(0, Math.floor(availableWidth * MAX_DRAWER_WIDTH_RATIO));
  const minWidth = Math.min(MIN_DRAWER_WIDTH, maxWidth);
  return Math.min(Math.max(Math.round(width), minWidth), maxWidth);
}

/**
 * Docked default: half the window (spec 078 D13 amendment, 2026-09-03 — the
 * user found the 360px column "very thin"), but never so wide that chat drops
 * below its minimum, and never under the old 360.
 */
function defaultDockedWidth(windowWidth: number) {
  return Math.max(DEFAULT_DOCKED_WIDTH, Math.min(Math.floor(windowWidth / 2), windowWidth - DOCKED_RESERVED_WIDTH));
}

function readSavedDrawerWidth(docked = false) {
  if (typeof window === 'undefined') return docked ? DEFAULT_DOCKED_WIDTH : DEFAULT_DRAWER_WIDTH;
  const fallback = docked ? defaultDockedWidth(window.innerWidth) : DEFAULT_DRAWER_WIDTH;
  try {
    const key = docked ? WORKSPACE_PANEL_WIDTH_KEY : FILE_PREVIEW_DRAWER_WIDTH_KEY;
    const saved = Number.parseInt(localStorage.getItem(key) ?? '', 10);
    return Number.isFinite(saved) ? saved : fallback;
  } catch {
    return fallback;
  }
}

interface FilePreviewDrawerProps {
  /** Drawer is shown iff true (driven by `route.previewPath != null`). */
  open: boolean;
  /** Workspace-relative path being previewed (drives FilePreview's header). */
  selectedPath: string | null;
  fileContent: FileContent | null;
  loading: boolean;
  onClose: () => void;
  /** Persist an edited `.md` file (last-write-wins); forwarded to FilePreview. */
  onSave?: (path: string, content: string) => Promise<boolean>;
  /**
   * Spec 078 D13 — docked mode: the drawer becomes ChatTab's third grid
   * column instead of an overlay. In flow (no absolute positioning, no
   * transform, no scrim), no focus trap / Esc / `inert` (it is a persistent
   * region, not a modal), default width 360. Everything else — the persisted
   * draggable width, the resize handle, the content pane — is shared.
   */
  docked?: boolean;
  /**
   * Docked only: a floor the column will not go below (the Browser view asks
   * for room so the agent's page is readable). The user's drag still wins
   * when it is wider; clamped to the same 80% cap as the drag.
   */
  minWidth?: number;
  /**
   * Chrome rendered in the header row, left of the collapse (docked) or close
   * (overlay) button. In the overlay it replaces the "File preview" label —
   * spec 137's narrow-window workspace panel puts its Files|Browser bar here.
   */
  header?: ReactNode;
  /**
   * Spec 137 — maximized, amended 2026-11-04: docked or as an overlay, the
   * panel covers the whole window except the 20px edge strip (the project
   * column and the list included), so one click on the strip returns. No
   * resize handle either way; the remembered width is untouched for the
   * restore.
   */
  maximized?: boolean;
  /** Overlay only: what Escape does. Default: `onClose`. */
  onEscape?: () => void;
  /** Rendered in the content pane INSTEAD of `FilePreview` (the workspace panel). */
  children?: ReactNode;
  /** Spec 065: the project whose file history the default FilePreview offers. */
  historyProjectId?: string;
  /**
   * Spec 065: the close control. `collapse` (the default) is the panel's
   * chevron; `close` is a bare × for a sheet that goes away when closed (the
   * turn undo sheet in the panel slot).
   */
  closeIcon?: 'collapse' | 'close';
  /**
   * Spec 065: overlay only — stack above the phone's fixed composer (z-60).
   * For a sheet whose primary action sits at its bottom (the turn undo
   * sheet); every other overlay keeps its layer.
   */
  raised?: boolean;
  /** Overlay only: the ✕'s label when the drawer is not a file preview
   *  (the turn undo sheet's 关闭). Default: 关闭预览. */
  closeLabel?: string;
}

/**
 * Slide-out file preview overlay (spec 002 §3.3): a right-side panel on desktop,
 * a bottom-sheet on mobile. It overlays the chat WITHOUT changing the active
 * tab, so the conversation stays mounted underneath. Dumb component — fetching,
 * the lazy 404 probe, and route state live in `ProjectDetail`.
 *
 * Positioned `absolute`, so its parent must be `relative` (the ProjectDetail
 * content area is). Always mounted (off-screen via transform when closed) so it
 * slides rather than popping in.
 */
export default function FilePreviewDrawer({
  open,
  selectedPath,
  fileContent,
  loading,
  onClose,
  onSave,
  docked = false,
  minWidth,
  header,
  maximized = false,
  onEscape,
  children,
  historyProjectId,
  closeIcon = 'collapse',
  raised = false,
  closeLabel,
}: FilePreviewDrawerProps) {
  const t = useT();
  const panelRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const [drawerWidth, setDrawerWidth] = useState(() => readSavedDrawerWidth(docked));
  const widthRef = useRef(drawerWidth);
  const resizeStartRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const [availableWidth, setAvailableWidth] = useState(() =>
    typeof window === 'undefined' ? DEFAULT_DRAWER_WIDTH : window.innerWidth,
  );
  const [isResizing, setIsResizing] = useState(false);
  // The element focused before the drawer opened, restored on close.
  const prevFocusRef = useRef<HTMLElement | null>(null);

  const measureAvailableWidth = useCallback(() => {
    const parentWidth = panelRef.current?.parentElement?.getBoundingClientRect().width ?? 0;
    return parentWidth > 0 ? parentWidth : window.innerWidth;
  }, []);

  const updateDrawerWidth = useCallback((width: number, parentWidth = measureAvailableWidth()) => {
    const next = clampPreviewDrawerWidth(width, parentWidth);
    widthRef.current = next;
    setDrawerWidth(next);
    return next;
  }, [measureAvailableWidth]);

  const persistDrawerWidth = useCallback((width: number) => {
    try {
      localStorage.setItem(docked ? WORKSPACE_PANEL_WIDTH_KEY : FILE_PREVIEW_DRAWER_WIDTH_KEY, String(width));
    } catch {
      // localStorage may be unavailable in private/locked-down webviews.
    }
  }, [docked]);

  // Keep the saved width inside the current content area when the app window
  // changes size. The 80% cap leaves enough of the covered chat visible to
  // preserve the drawer's overlay context.
  useEffect(() => {
    const handleResize = () => {
      const parentWidth = measureAvailableWidth();
      setAvailableWidth(parentWidth);
      updateDrawerWidth(widthRef.current, parentWidth);
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [measureAvailableWidth, open, updateDrawerWidth]);

  // Pointer movement is tracked on window so dragging remains smooth even
  // after the pointer leaves the narrow resize target.
  useEffect(() => {
    if (!isResizing) return;

    const previousCursor = document.body.style.cursor;
    const previousUserSelect = document.body.style.userSelect;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const handlePointerMove = (event: PointerEvent) => {
      const start = resizeStartRef.current;
      if (!start) return;
      updateDrawerWidth(start.startWidth + start.startX - event.clientX);
    };
    const finishResize = () => {
      resizeStartRef.current = null;
      setIsResizing(false);
      persistDrawerWidth(widthRef.current);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', finishResize);
    window.addEventListener('pointercancel', finishResize);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', finishResize);
      window.removeEventListener('pointercancel', finishResize);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousUserSelect;
    };
  }, [isResizing, persistDrawerWidth, updateDrawerWidth]);

  const handleResizePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!open || event.button !== 0) return;
    event.preventDefault();
    resizeStartRef.current = {
      startX: event.clientX,
      startWidth: widthRef.current,
    };
    setIsResizing(true);
  };

  const handleResizeKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const maxWidth = Math.floor(measureAvailableWidth() * MAX_DRAWER_WIDTH_RATIO);
    let nextWidth: number | null = null;
    if (event.key === 'ArrowLeft') nextWidth = widthRef.current + KEYBOARD_RESIZE_STEP;
    if (event.key === 'ArrowRight') nextWidth = widthRef.current - KEYBOARD_RESIZE_STEP;
    if (event.key === 'Home') nextWidth = MIN_DRAWER_WIDTH;
    if (event.key === 'End') nextWidth = maxWidth;
    if (nextWidth === null) return;
    event.preventDefault();
    persistDrawerWidth(updateDrawerWidth(nextWidth));
  };

  // Esc-to-close while open. Overlay only: docked, the panel is a persistent
  // region and Esc belongs to whatever is inside it (the Annotate mode).
  const escapeAction = onEscape ?? onClose;
  useEffect(() => {
    if (!open || docked) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') escapeAction();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, escapeAction, docked]);

  // Focus management (a11y for the aria-modal dialog): move focus into the
  // drawer when it opens; restore it to the opener (the clicked chip/card) when
  // it closes. `inert` (below) keeps the drawer's controls out of the tab order
  // while closed, so this is the only focus that lands inside it.
  useEffect(() => {
    // Docked: nothing is being covered, so stealing focus on mount would
    // yank the caret out of the composer every time the panel opens itself.
    if (docked) return;
    if (open) {
      prevFocusRef.current = document.activeElement as HTMLElement | null;
      // preventScroll: the panel starts translated OFF-SCREEN and slides in.
      // WebKit's focus scroll-into-view targets the element's VISUAL (still
      // off-screen) position, scrolling the whole document ~420px to reach the
      // close button — the "chat slides in from the left" jump in pywebview.
      // Chromium uses the layout position, so it never showed. Suppress it.
      closeBtnRef.current?.focus({ preventScroll: true });
    } else {
      const prev = prevFocusRef.current;
      prevFocusRef.current = null;
      // Same rationale on restore — don't let refocusing the opener scroll.
      if (prev && document.contains(prev)) prev.focus({ preventScroll: true });
    }
  }, [open, docked]);

  // Focus trap: keep Tab / Shift+Tab cycling within the open drawer.
  const handleTrapTab = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Tab') return;
    const panel = panelRef.current;
    if (!panel) return;
    const focusables = Array.from(
      panel.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    );
    if (focusables.length === 0) {
      e.preventDefault();
      return;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement as HTMLElement | null;
    if (e.shiftKey) {
      if (active === first || !panel.contains(active)) {
        e.preventDefault();
        last.focus();
      }
    } else if (active === last || !panel.contains(active)) {
      e.preventDefault();
      first.focus();
    }
  };

  const resizeHandle = (
    <div
      role="separator"
      aria-label={t('filePreview.resize')}
      aria-orientation="vertical"
      aria-valuemin={Math.min(MIN_DRAWER_WIDTH, Math.floor(availableWidth * MAX_DRAWER_WIDTH_RATIO))}
      aria-valuemax={Math.floor(availableWidth * MAX_DRAWER_WIDTH_RATIO)}
      aria-valuenow={drawerWidth}
      tabIndex={open ? 0 : -1}
      onPointerDown={handleResizePointerDown}
      onKeyDown={handleResizeKeyDown}
      className={`group absolute inset-y-0 left-0 z-10 hidden w-3 -translate-x-1/2 touch-none cursor-col-resize items-center justify-center md:flex focus:outline-none ${
        isResizing ? 'bg-accent/10' : ''
      }`}
    >
      <span
        aria-hidden
        className={`h-full w-px transition-colors group-hover:bg-accent group-focus-visible:bg-accent ${
          isResizing ? 'bg-accent' : 'bg-transparent'
        }`}
      />
    </div>
  );

  const historyOpts = useMemo(
    () => (historyProjectId ? { projectId: historyProjectId } : undefined),
    [historyProjectId],
  );
  const body = children ?? (
    <FilePreview
      fileContent={fileContent}
      loading={loading}
      selectedPath={selectedPath}
      onSave={onSave}
      history={historyOpts}
    />
  );

  // ── Docked (spec 078 D13): a real column of ChatTab's grid ────────────────
  if (docked) {
    const dockedWidth = minWidth
      ? Math.max(drawerWidth, clampPreviewDrawerWidth(minWidth, availableWidth))
      : drawerWidth;
    const collapseButton = (
      <button
        ref={closeBtnRef}
        type="button"
        onClick={onClose}
        aria-label={closeIcon === 'close' ? t('history.close') : t('panel.collapse')}
        title={closeIcon === 'close' ? t('history.close') : undefined}
        className="flex items-center justify-center w-7 h-7 shrink-0 rounded-md text-secondary hover:text-primary hover:bg-card-hover transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
      >
        {closeIcon === 'close' ? <X size={14} /> : <ChevronRight size={16} />}
      </button>
    );
    return (
      <section
        ref={panelRef}
        data-testid="workspace-panel-column"
        data-maximized={maximized}
        aria-label={t('panel.handle')}
        style={{ '--file-preview-drawer-width': `${dockedWidth}px` } as CSSProperties}
        className={`float-panel-dock relative z-0 hidden md:flex h-full min-h-0 flex-col bg-background ${
          maximized
            ? // 2026-11-04: the whole window except the 20px edge strip
              // (index.css), not just the right area beside the column.
              'flex-1 min-w-0 panel-max-window'
            : // Shrinks (to the 320px floor) rather than overflowing when
              // the row narrows without a window resize — pinning the project
              // list takes 240px from the right area and re-measures nothing
              // (spec 137: strip + list + column + right area must coexist).
              'shrink min-w-[320px] border-l border-border w-[var(--file-preview-drawer-width)] max-w-[80%]'
        }`}
      >
        {!maximized && resizeHandle}
        {header ? (
          <div
            data-testid="workspace-panel-header"
            className="flex items-center justify-between gap-2 pl-3 pr-2 py-1.5 border-b border-border shrink-0 min-w-0"
          >
            <div className="min-w-0 flex-1">{header}</div>
            {collapseButton}
          </div>
        ) : (
          // Spec 088: with nothing to label, collapse floats in the corner
          // rather than holding open an otherwise empty bar.
          <div className="absolute right-2 top-1.5 z-10">{collapseButton}</div>
        )}
        <div className="flex-1 overflow-hidden min-h-0 flex flex-col">{body}</div>
      </section>
    );
  }

  // ── Overlay / mobile bottom sheet (unchanged) ─────────────────────────────
  return (
    <>
      {/* Scrim — click to dismiss; fades with the panel. */}
      {/* Maximized there is nothing beside the panel to dim. */}
      {!maximized && (
        <div
          aria-hidden
          data-testid="file-preview-scrim"
          onClick={onClose}
          className={`absolute inset-0 ${raised ? 'z-[65]' : 'z-30'} bg-black/30 transition-opacity duration-200 ${
            open ? 'opacity-100' : 'opacity-0 pointer-events-none'
          }`}
        />
      )}
      {/* Panel: bottom-sheet on mobile, right-side slide-out on desktop.
          `inert` while closed removes its controls (close button + FilePreview's
          copy/download) from the tab order even though it stays mounted
          off-screen for the slide animation. */}
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={t('filePreview.heading')}
        aria-hidden={!open}
        inert={!open}
        onKeyDown={handleTrapTab}
        data-maximized={maximized}
        style={{ '--file-preview-drawer-width': `${drawerWidth}px` } as CSSProperties}
        className={`float-panel-drawer absolute ${raised ? 'z-[70]' : 'z-40'} flex flex-col bg-background shadow-xl transition-transform duration-200 ease-out
          inset-x-0 bottom-0 h-[75%] rounded-t-xl border-t border-border
          md:inset-y-0 md:right-0 md:h-full md:rounded-none md:border-t-0 md:border-l
          ${maximized ? 'md:left-0 md:w-auto md:max-w-none panel-max-window' : 'md:left-auto md:w-[var(--file-preview-drawer-width)] md:max-w-[80%]'}
          ${open ? 'translate-y-0 md:translate-x-0' : 'translate-y-full md:translate-y-0 md:translate-x-full pointer-events-none'}`}
      >
        {!maximized && resizeHandle}
        <div
          className={`flex items-center justify-between gap-2 border-b border-border shrink-0 min-w-0 ${
            header ? 'pl-3 pr-2 py-1.5' : 'px-4 py-2.5'
          }`}
        >
          {header ? (
            <div className="min-w-0 flex-1">{header}</div>
          ) : (
            <span className="text-xs font-medium text-secondary">{t('filePreview.heading')}</span>
          )}
          <button
            ref={closeBtnRef}
            type="button"
            onClick={onClose}
            aria-label={closeLabel ?? t('filePreview.close')}
            className={`flex items-center justify-center w-7 h-7 shrink-0 rounded-md text-secondary hover:text-primary hover:bg-card-hover transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
              header ? '' : '-mr-1'
            }`}
          >
            <X size={16} />
          </button>
        </div>
        <div className={`flex-1 min-h-0 ${children ? 'overflow-hidden flex flex-col' : 'overflow-y-auto'}`}>{body}</div>
      </div>
    </>
  );
}
