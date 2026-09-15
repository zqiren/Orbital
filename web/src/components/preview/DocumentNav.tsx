// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — an engine's navigation, renderable anywhere: DocumentPreview's own
 * toolbar, or a host header row (spec 088's panel header). Its state comes from
 * the engine through a DocumentController (see documentController.ts); this
 * component holds none.
 *
 * Sheets render as a compact selector here; the full tab strip lives in the
 * sheet body's footer. `compact` adds container-query classes that drop
 * secondary pieces when the nearest `@container` is narrow.
 */
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useT } from '../../i18n/useT';
import { NavButton, ZoomControls } from './NavControls';
import type { DocumentNavState, PdfNavState, SheetNavState } from './types';

export default function DocumentNav({
  nav,
  compact = false,
}: {
  nav: DocumentNavState | null;
  compact?: boolean;
}) {
  if (!nav) return null;
  if (nav.kind === 'pdf') return <PdfNav nav={nav} compact={compact} />;
  if (nav.kind === 'sheet') return <SheetSelect nav={nav} />;
  return (
    <ZoomControls
      percent={nav.zoomPercent}
      fit={nav.fit}
      onZoomOut={nav.zoomOut}
      onZoomIn={nav.zoomIn}
      onFit={nav.fitWidth}
      compact={compact}
    />
  );
}

function PdfNav({ nav, compact }: { nav: PdfNavState; compact: boolean }) {
  const t = useT();
  const narrow = compact ? '@max-[24rem]:hidden' : '';
  return (
    <div className="flex items-center gap-1 min-w-0">
      <NavButton label={t('filePreview.doc.prevPage')} onClick={() => nav.goTo(nav.page - 1)} disabled={nav.page <= 1}>
        <ChevronLeft size={14} aria-hidden />
      </NavButton>
      <input
        key={nav.page}
        type="text"
        inputMode="numeric"
        aria-label={t('filePreview.doc.pageNumber')}
        defaultValue={String(nav.page)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') nav.goTo(parseInt(e.currentTarget.value, 10) || nav.page);
        }}
        onBlur={(e) => {
          const page = parseInt(e.currentTarget.value, 10);
          if (page && page !== nav.page) nav.goTo(page);
        }}
        className="w-9 h-6 shrink-0 text-center text-xs rounded-md border border-border bg-transparent text-primary tabular-nums focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
      />
      <span className={`text-xs text-secondary whitespace-nowrap ${narrow}`}>
        {t('filePreview.doc.pageCount', { n: nav.pageCount })}
      </span>
      <NavButton
        label={t('filePreview.doc.nextPage')}
        onClick={() => nav.goTo(nav.page + 1)}
        disabled={nav.page >= nav.pageCount}
      >
        <ChevronRight size={14} aria-hidden />
      </NavButton>
      <span className={`mx-1 h-4 w-px bg-border shrink-0 ${narrow}`} aria-hidden />
      <ZoomControls
        percent={nav.zoomPercent}
        fit={nav.fit}
        onZoomOut={nav.zoomOut}
        onZoomIn={nav.zoomIn}
        onFit={nav.fitWidth}
        compact={compact}
      />
    </div>
  );
}

function SheetSelect({ nav }: { nav: SheetNavState }) {
  const t = useT();
  if (nav.sheets.length <= 1) return null;
  return (
    <select
      aria-label={t('filePreview.doc.sheets')}
      value={nav.active}
      onChange={(e) => nav.select(Number(e.target.value))}
      className="h-6 min-w-0 max-w-[12rem] truncate rounded-md border border-border bg-transparent px-1.5 text-xs text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
    >
      {nav.sheets.map((name, i) => (
        <option key={`${i}:${name}`} value={i}>
          {name}
        </option>
      ))}
    </select>
  );
}
