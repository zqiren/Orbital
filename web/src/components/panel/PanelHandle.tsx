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
  return (
    <button
      type="button"
      onClick={onExpand}
      aria-label={t('panel.expand')}
      title={t('panel.handle')}
      data-testid="panel-handle"
      className="hidden md:flex w-5 shrink-0 h-full flex-col items-center justify-center gap-1.5 border-l border-border bg-nav text-secondary hover:text-primary hover:bg-card-hover transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/50"
    >
      <ChevronLeft size={12} aria-hidden />
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
