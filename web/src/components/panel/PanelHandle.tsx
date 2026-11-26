// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.1 — the collapsed state: a 20px edge handle at the right of the
 * chat column with a chevron and a small dot while the agent is working.
 * The mirror image of the left EdgeStrip, so the shell reads symmetrically.
 * CONTRACT FILE: props are final.
 */
import { ChevronLeft } from 'lucide-react';
import { useT } from '../../i18n/useT';

export interface PanelHandleProps {
  working: boolean;
  onExpand: () => void;
}

export default function PanelHandle({ working, onExpand }: PanelHandleProps) {
  const t = useT();
  // Design pass D (8.10 B): the panel folded to 20px inside the chat's
  // sheet, at its right edge (it takes the sheet's right corners). Its tone
  // is the open panel's, so closed and open read as one object. The chevron
  // sits on the bar itself in the left EdgeStrip's look (`strip-chevron`):
  // muted, its box lighting while any part of the bar is hovered, so the two
  // edges read as mirror images (2026-11-26: the raised tab read as a button
  // of its own). The whole band is the hit target.
  return (
    <button
      type="button"
      onClick={onExpand}
      aria-label={t('panel.expand')}
      title={t('panel.handle')}
      data-testid="panel-handle"
      data-tour="panel-handle"
      className="panel-handle group hidden md:flex w-5 shrink-0 h-full flex-col items-center justify-center gap-1.5 bg-dock focus:outline-none"
    >
      <span
        data-testid="panel-handle-chevron"
        className="flex h-7 w-5 shrink-0 items-center justify-center rounded-md text-secondary transition-colors group-hover:bg-card-hover group-hover:text-primary"
      >
        <ChevronLeft size={16} strokeWidth={2.5} aria-hidden="true" />
      </span>
      {working && (
        <span
          data-testid="panel-handle-working"
          aria-hidden
          className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse"
        />
      )}
    </button>
  );
}
