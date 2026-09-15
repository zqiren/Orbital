// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { ReactNode } from 'react';
import { Minus, MoveHorizontal, Plus } from 'lucide-react';
import { useT } from '../../i18n/useT';

/** Compact icon button for the navigation row (fits the narrow panel). */
export function NavButton({
  label,
  onClick,
  disabled,
  pressed,
  className,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  pressed?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      disabled={disabled}
      aria-pressed={pressed}
      className={`inline-flex items-center justify-center w-6 h-6 shrink-0 rounded-md transition-colors hover:bg-card-hover disabled:opacity-40 disabled:pointer-events-none focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
        pressed ? 'text-accent' : 'text-secondary hover:text-primary'
      } ${className ?? ''}`}
    >
      {children}
    </button>
  );
}

export function ZoomControls({
  percent,
  fit,
  onZoomOut,
  onZoomIn,
  onFit,
  compact = false,
}: {
  percent: number;
  fit: boolean;
  onZoomOut: () => void;
  onZoomIn: () => void;
  onFit: () => void;
  /** Drop the percentage and fit button when the nearest @container is under 20rem. */
  compact?: boolean;
}) {
  const t = useT();
  const narrow = compact ? '@max-[20rem]:hidden' : '';
  return (
    <div className="flex items-center gap-0.5 shrink-0">
      <NavButton label={t('filePreview.doc.zoomOut')} onClick={onZoomOut}>
        <Minus size={14} aria-hidden />
      </NavButton>
      <span className={`w-10 text-center text-xs text-secondary tabular-nums ${narrow}`}>{percent}%</span>
      <NavButton label={t('filePreview.doc.zoomIn')} onClick={onZoomIn}>
        <Plus size={14} aria-hidden />
      </NavButton>
      <NavButton label={t('filePreview.doc.fitWidth')} onClick={onFit} pressed={fit} className={narrow}>
        <MoveHorizontal size={14} aria-hidden />
      </NavButton>
    </div>
  );
}
