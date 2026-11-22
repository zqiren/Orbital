// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useT } from '../i18n/useT';
import type { PageToolbarCrumb } from './PageToolbar';

interface PageHeadingProps {
  /** Pages above this one, top down (`Settings` above `Pricing table`).
   *  Each is a link. Empty for a page directly under the project. */
  parents?: PageToolbarCrumb[];
  /** Already translated. Rendered as the h1. */
  title: string;
  titleTestId: string;
}

/**
 * The top of a page beside the project column (spec 137 round 3, desktop and
 * narrow windows). The column is the navigation — the status line and the
 * gear show which page is open, a session row goes back to the chat — so a
 * page needs no bar: just its title as a heading on the page background, in
 * the project name's type, with the parent as a small link line above it when
 * the page is nested. No Back, no divider, no border, no View Transition
 * names (the right area swaps as one).
 *
 * The phone has no column and keeps PageToolbar (Back · `Project / Page`).
 */
export default function PageHeading({ parents = [], title, titleTestId }: PageHeadingProps) {
  const t = useT();
  return (
    <div className="shrink-0 px-6 pt-[22px] bg-background" data-testid="page-heading">
      {parents.length > 0 && (
        <nav aria-label={t('settingsToolbar.breadcrumb')} className="flex items-center gap-1.5 min-w-0 text-caption">
          {parents.map((p, i) => (
            <span key={p.label} className="contents">
              {i > 0 && (
                <span className="text-muted" aria-hidden="true">
                  /
                </span>
              )}
              <button
                type="button"
                onClick={p.onClick}
                className="truncate text-secondary hover:text-primary hover:underline transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 rounded-sm"
              >
                {p.label}
              </button>
            </span>
          ))}
        </nav>
      )}
      <h1
        className="text-title font-semibold text-primary truncate"
        data-testid={titleTestId}
      >
        {title}
      </h1>
    </div>
  );
}
