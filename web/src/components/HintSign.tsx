// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * A small ⓘ that holds one needed hint, for places where the hint must not
 * sit on screen as text (the owner's rule: no explanation text by default).
 * Hover opens it, a tap toggles it (phones have no hover), Escape or a press
 * elsewhere closes it. Same behaviour as AutonomyScopeHint, but generic.
 */

import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { Info } from 'lucide-react';

interface HintSignProps {
  /** Accessible name of the sign, already translated. */
  label: string;
  /** The hint itself, already translated. */
  children: ReactNode;
  /** Which edge of the sign the bubble lines up with. */
  align?: 'left' | 'right';
  testId?: string;
}

export default function HintSign({ label, children, align = 'left', testId }: HintSignProps) {
  const [open, setOpen] = useState(false);
  const popId = useId();
  const closeTimer = useRef<number | undefined>(undefined);
  const rootRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      // composedPath, not contains(): a press that unmounts its own target
      // would otherwise read as "outside".
      if (rootRef.current && !e.composedPath().includes(rootRef.current)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        setOpen(false);
      }
    };
    document.addEventListener('click', onDoc);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('click', onDoc);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [open]);

  useEffect(() => () => window.clearTimeout(closeTimer.current), []);

  return (
    <span ref={rootRef} className="relative inline-flex items-center">
      <button
        type="button"
        data-testid={testId}
        aria-label={label}
        aria-expanded={open}
        aria-describedby={open ? popId : undefined}
        onPointerEnter={(e) => {
          if (e.pointerType === 'touch') return;
          window.clearTimeout(closeTimer.current);
          setOpen(true);
        }}
        onPointerLeave={(e) => {
          if (e.pointerType === 'touch') return;
          closeTimer.current = window.setTimeout(() => setOpen(false), 200);
        }}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          setOpen((v) => !v);
        }}
        className="inline-flex shrink-0 items-center rounded-full p-1 -m-1 text-muted transition-colors duration-150 hover:text-primary aria-expanded:text-primary focus-visible:outline-2 focus-visible:outline-accent/45 [-webkit-tap-highlight-color:transparent]"
      >
        <Info aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
      </button>
      {open && (
        <span
          id={popId}
          role="tooltip"
          onPointerEnter={() => window.clearTimeout(closeTimer.current)}
          onPointerLeave={() => {
            closeTimer.current = window.setTimeout(() => setOpen(false), 200);
          }}
          className={`absolute ${align === 'right' ? 'right-0' : 'left-0'} top-full z-30 mt-1.5 w-max max-w-[220px] rounded-lg border border-border bg-sidebar px-2.5 py-1.5 text-left text-xs font-normal leading-relaxed text-secondary shadow-lg`}
        >
          {children}
        </span>
      )}
    </span>
  );
}
