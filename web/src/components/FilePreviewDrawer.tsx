// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { createContext, useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import type { FileContent } from '../types';
import FilePreview from './FilePreview';
import { useT } from '../i18n/useT';

/**
 * Carries the "open this workspace-relative path in the preview drawer" handler
 * down to the chat's MarkdownContent (spec 002 §3.3). Provided by
 * `ProjectDetail` — which owns `setRoute` — and consumed by `ChatView`, so the
 * handler reaches the deeply-nested renderer without prop-drilling through the
 * out-of-scope `ChatTab`. Default `null` keeps non-chat consumers inert.
 */
export const OpenPathContext = createContext<((path: string) => void) | null>(null);

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
}: FilePreviewDrawerProps) {
  const t = useT();
  const panelRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  // The element focused before the drawer opened, restored on close.
  const prevFocusRef = useRef<HTMLElement | null>(null);

  // Esc-to-close while open.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  // Focus management (a11y for the aria-modal dialog): move focus into the
  // drawer when it opens; restore it to the opener (the clicked chip/card) when
  // it closes. `inert` (below) keeps the drawer's controls out of the tab order
  // while closed, so this is the only focus that lands inside it.
  useEffect(() => {
    if (open) {
      prevFocusRef.current = document.activeElement as HTMLElement | null;
      closeBtnRef.current?.focus();
    } else {
      const prev = prevFocusRef.current;
      prevFocusRef.current = null;
      if (prev && document.contains(prev)) prev.focus();
    }
  }, [open]);

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

  return (
    <>
      {/* Scrim — click to dismiss; fades with the panel. */}
      <div
        aria-hidden
        onClick={onClose}
        className={`absolute inset-0 z-30 bg-black/30 transition-opacity duration-200 ${
          open ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
      />
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
        className={`absolute z-40 flex flex-col bg-background shadow-xl transition-transform duration-200 ease-out
          inset-x-0 bottom-0 h-[75%] rounded-t-xl border-t border-border
          md:inset-y-0 md:right-0 md:left-auto md:h-full md:w-[420px] md:max-w-[80%] md:rounded-none md:border-t-0 md:border-l
          ${open ? 'translate-y-0 md:translate-x-0' : 'translate-y-full md:translate-y-0 md:translate-x-full pointer-events-none'}`}
      >
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-border shrink-0">
          <span className="text-[12px] font-medium text-secondary">{t('filePreview.heading')}</span>
          <button
            ref={closeBtnRef}
            type="button"
            onClick={onClose}
            aria-label={t('filePreview.close')}
            className="flex items-center justify-center w-7 h-7 -mr-1 rounded-md text-secondary hover:text-primary hover:bg-card-hover transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
          >
            <X size={16} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto min-h-0">
          <FilePreview fileContent={fileContent} loading={loading} selectedPath={selectedPath} onSave={onSave} />
        </div>
      </div>
    </>
  );
}
