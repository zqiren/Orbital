// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { ChevronLeft } from 'lucide-react';
import { useT } from '../i18n/useT';

export interface PageToolbarCrumb {
  /** Already translated. */
  label: string;
  onClick: () => void;
}

interface PageToolbarProps {
  /** Already translated. The crumb the page sits under (the project name);
   *  omitted for global settings, which has no parent. */
  parent?: string;
  /** Makes the parent crumb a link (spec 137: crumb segments are links). */
  onParent?: () => void;
  /** Pages between the parent and this one, top down (`Settings` above
   *  `Pricing`). Each is a link to that page. */
  crumbs?: PageToolbarCrumb[];
  /** Already translated. The page itself — rendered as the h1. */
  title: string;
  onBack: () => void;
  backTestId: string;
  titleTestId: string;
}

/**
 * The bar across the top of every sub-page — project Settings, Tasks,
 * Pricing, global settings: a filled Back button, a divider, then
 * `Parent / … / Title` (spec 137 decision 4: chat is the only home, every
 * other surface is a page in this header language). It was SettingsToolbar
 * until Tasks became a page too.
 *
 * It replaces a bordered "Back to <project>" pill that floated above the page
 * title. The pill was a control dressed as a card, sitting alone on the
 * canvas, and users called it the ugliest thing on the page. A toolbar puts
 * the way out where windowed apps keep it (top-left, same row as where you
 * are), and the breadcrumb carries the "where am I" the old title and pill
 * both repeated.
 *
 * The Back button is FILLED rather than outlined or icon-only: an icon-only
 * chevron was rejected in review as too easy to miss. Back goes up ONE level;
 * the crumb links jump straight to any level above.
 *
 * The `vt-*` classes pair this bar with the project header for the
 * home <-> page View Transition: the header folds into this toolbar and
 * the project title glides into the parent crumb, so the toolbar reads as the
 * same header in a compact state rather than a bar that exists nowhere else.
 * That fold exists only where there is a header to fold — the phone and
 * global settings. Beside the project column (desktop, narrow) a page has no
 * bar at all: it opens with a PageHeading.
 */
export default function PageToolbar({
  parent,
  onParent,
  crumbs = [],
  title,
  onBack,
  backTestId,
  titleTestId,
}: PageToolbarProps) {
  const t = useT();
  const linkClass =
    'truncate text-secondary hover:text-primary hover:underline transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 rounded-sm';
  const slash = (
    <span className="text-muted" aria-hidden="true">
      /
    </span>
  );
  const nav = (
    <nav aria-label={t('settingsToolbar.breadcrumb')} className="flex items-center gap-2 min-w-0 text-body">
      {parent && (
        <>
          {onParent ? (
            <button type="button" onClick={onParent} className={`vt-project-title ${linkClass}`}>
              {parent}
            </button>
          ) : (
            <span className="vt-project-title truncate text-secondary">{parent}</span>
          )}
          {slash}
        </>
      )}
      {crumbs.map((crumb) => (
        <span key={crumb.label} className="contents">
          <button type="button" onClick={crumb.onClick} className={`shrink-0 ${linkClass}`}>
            {crumb.label}
          </button>
          {slash}
        </span>
      ))}
      <h1 className="truncate font-medium text-primary" data-testid={titleTestId}>
        {title}
      </h1>
    </nav>
  );

  return (
    <header className="vt-project-header flex items-center gap-3 h-[52px] shrink-0 px-4 bg-card border-b border-border max-md:h-14">
      <button
        type="button"
        onClick={onBack}
        data-testid={backTestId}
        className="vt-settings-back inline-flex shrink-0 items-center gap-1 h-8 pl-2 pr-3 rounded-lg bg-soft-button text-body font-medium text-primary transition-colors duration-150 hover:bg-soft-button-hover max-md:min-h-[44px]"
      >
        <ChevronLeft size={16} strokeWidth={2.2} aria-hidden="true" />
        {t('settingsToolbar.back')}
      </button>
      <span className="w-px h-5 shrink-0 bg-border" aria-hidden="true" />
      {nav}
    </header>
  );
}
