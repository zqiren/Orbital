/**
 * The small ⓘ beside the Autonomy Level title (batch 1110, item 1).
 *
 * The autonomy level governs Orbital's own agent. A pinned Claude Code,
 * Codex or ACP agent keeps the permission model it was set up with (the
 * 2026-11-10 smoke saw Supervised ignored for pinned workers and read it as
 * a bug); this says so where the choice is made. Hover expands it, a tap
 * toggles it, Escape closes it.
 */

import { useEffect, useId, useRef, useState } from 'react';
import { Info } from 'lucide-react';
import { useT } from '../i18n/useT';

export default function AutonomyScopeHint() {
  const t = useT();
  const [open, setOpen] = useState(false);
  const popId = useId();
  const lastPointer = useRef<string>('mouse');
  const closeTimer = useRef<number | undefined>(undefined);
  const rootRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('click', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('click', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  useEffect(() => () => window.clearTimeout(closeTimer.current), []);

  return (
    <span ref={rootRef} className="relative inline-flex items-center align-middle">
      <button
        type="button"
        data-testid="autonomy-scope-hint"
        aria-label={t('autonomy.scope.aria')}
        aria-expanded={open}
        aria-describedby={open ? popId : undefined}
        onPointerDown={(e) => { lastPointer.current = e.pointerType || 'mouse'; }}
        onPointerEnter={(e) => {
          if (e.pointerType === 'touch') return;
          window.clearTimeout(closeTimer.current);
          setOpen(true);
        }}
        onPointerLeave={(e) => {
          if (e.pointerType === 'touch') return;
          closeTimer.current = window.setTimeout(() => setOpen(false), 220);
        }}
        onFocus={(e) => {
          if (e.currentTarget.matches(':focus-visible')) setOpen(true);
        }}
        onBlur={() => {
          if (lastPointer.current !== 'touch') setOpen(false);
        }}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="inline-flex shrink-0 items-center rounded-full border-0 bg-transparent p-[5px] -m-[5px] ml-1 text-muted transition-colors duration-150 hover:text-primary aria-expanded:text-primary focus-visible:outline-2 focus-visible:outline-accent/45 max-md:p-[13px] max-md:-m-[13px] [-webkit-tap-highlight-color:transparent]"
      >
        <Info aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
      </button>
      {open && (
        <span
          id={popId}
          role="tooltip"
          data-testid="autonomy-scope-pop"
          onPointerEnter={() => window.clearTimeout(closeTimer.current)}
          onPointerLeave={() => { closeTimer.current = window.setTimeout(() => setOpen(false), 220); }}
          className="absolute left-0 top-full z-20 mt-1.5 w-[280px] max-w-[80vw] rounded-lg border border-border bg-sidebar px-3 py-2 text-left text-xs font-normal leading-relaxed text-secondary shadow-lg"
        >
          {t('autonomy.scope.hint')}
        </span>
      )}
    </span>
  );
}
