// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { Coins } from 'lucide-react';
import { useT } from '../i18n/useT';

/**
 * Spec 160 §5.3 (variant B, owner-picked): a token glyph with the number of
 * imported projects a saved key would read once, beside the key form's
 * primary button. No visible sentence: the one line it means shows on hover
 * (desktop) or tap (phone). Renders nothing when `n` < 1.
 *
 * Geometry follows `BACKLOG/specs/160-assets/token-hint-prototype.html`:
 * 14px glyph at stroke 1.75 + a 12px tabular count in the muted colour,
 * primary on hover/open; the bubble is a `surface-float-pop`, 280px max,
 * grown leftward from the glyph, above it unless there is no room.
 */
export type TokenHintKind = 'key' | 'skip' | 'importKey' | 'importAgent';

/** The one sentence a sign means, and its accessible name. `agent` is the
 *  worker's product name (never a raw id) for the keyless kinds. Plain
 *  function: the caller passes its own `t`. */
export function tokenHint(
  t: ReturnType<typeof useT>,
  kind: TokenHintKind,
  n: number,
  agent?: string | null,
): { text: string; label: string } {
  const plural = n === 1 ? 'one' : 'other';
  const vars = { n, agent: agent ?? '' };
  switch (kind) {
    case 'skip':
      return { text: t(`llm.tokenHint.skip.${plural}`, vars), label: t('llm.tokenHint.skip.label') };
    case 'importKey':
      return { text: t(`importProjects.tokenHint.key.${plural}`, vars), label: t('importProjects.tokenHint.label') };
    case 'importAgent':
      return { text: t(`importProjects.tokenHint.agent.${plural}`, vars), label: t('importProjects.tokenHint.label') };
    default:
      return { text: t(`llm.tokenHint.${plural}`, vars), label: t('llm.tokenHint.label') };
  }
}

export default function TokenCostSign({
  n,
  kind = 'key',
  agent,
  scan = true,
  onScanChange,
}: {
  n: number;
  /** Which action spends: saving a key (default), skipping it (the user's
   *  own agent reads), or importing with a key / with the agent. */
  kind?: TokenHintKind;
  agent?: string | null;
  /** Round 2 opt-out: whether the batch behind this button gets read. Lives
   *  in the bubble as one switch (ON by default); off, the glyph says so by
   *  itself — dimmed, slashed, no count. The caller sends it with its
   *  button's request; never a lasting setting. */
  scan?: boolean;
  onScanChange?: (on: boolean) => void;
}) {
  const t = useT();
  const signRef = useRef<HTMLButtonElement>(null);
  const glyphRef = useRef<SVGSVGElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [shown, setShown] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number; below: boolean } | null>(null);
  const openTimer = useRef<number | undefined>(undefined);
  const closeTimer = useRef<number | undefined>(undefined);
  const lastPointer = useRef<string>('mouse');
  const popId = useId();

  const clearTimers = () => {
    window.clearTimeout(openTimer.current);
    window.clearTimeout(closeTimer.current);
  };
  const doOpen = useCallback(() => {
    window.clearTimeout(closeTimer.current);
    setOpen(true);
  }, []);
  const doClose = useCallback(() => {
    window.clearTimeout(openTimer.current);
    setOpen(false);
    setShown(false);
  }, []);

  const place = useCallback(() => {
    const glyph = glyphRef.current?.getBoundingClientRect();
    const pop = popRef.current?.getBoundingClientRect();
    if (!glyph || !pop) return;
    const gap = 8;
    const margin = 12;
    const vw = document.documentElement.clientWidth || window.innerWidth;
    const vh = window.innerHeight;
    // Beside a right-aligned button the bubble grows leftward; its near edge
    // lines up with the glyph.
    let left = glyph.right + 6 - pop.width;
    left = Math.max(margin, Math.min(left, vw - margin - pop.width));
    let top = glyph.top - gap - pop.height;
    let below = false;
    if (top < margin) {
      top = glyph.bottom + gap;
      below = true;
    }
    if (below && top + pop.height > vh - margin) top = Math.max(margin, vh - margin - pop.height);
    setPos({ left: Math.round(left), top: Math.round(top), below });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    place();
    const raf = window.requestAnimationFrame(() => setShown(true));
    return () => window.cancelAnimationFrame(raf);
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const target = e.target as Node | null;
      if (target && (signRef.current?.contains(target) || popRef.current?.contains(target))) return;
      doClose();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') doClose(); };
    const onMove = () => place();
    document.addEventListener('click', onDoc);
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', onMove);
    window.addEventListener('scroll', onMove, { passive: true, capture: true });
    return () => {
      document.removeEventListener('click', onDoc);
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', onMove);
      window.removeEventListener('scroll', onMove, { capture: true });
    };
  }, [open, doClose, place]);

  useEffect(() => clearTimers, []);

  if (!(n >= 1)) return null;
  if ((kind === 'skip' || kind === 'importAgent') && !agent) return null;
  const { text, label } = tokenHint(t, kind, n, agent);

  return (
    <>
      <button
        ref={signRef}
        type="button"
        data-testid="token-cost-sign"
        data-kind={kind}
        aria-label={label}
        aria-expanded={open}
        aria-describedby={open ? popId : undefined}
        onPointerDown={(e) => { lastPointer.current = e.pointerType || 'mouse'; }}
        onPointerEnter={(e) => {
          if (e.pointerType === 'touch') return;
          window.clearTimeout(closeTimer.current);
          // A short dwell so a passing cursor doesn't flash it.
          openTimer.current = window.setTimeout(doOpen, 120);
        }}
        onPointerLeave={(e) => {
          if (e.pointerType === 'touch') return;
          window.clearTimeout(openTimer.current);
          closeTimer.current = window.setTimeout(doClose, 220);
        }}
        onFocus={(e) => {
          if (e.currentTarget.matches(':focus-visible')) doOpen();
        }}
        onBlur={(e) => {
          if (lastPointer.current === 'touch') return;
          // Glyph + bubble are one area: Tab into the switch keeps it open.
          if (popRef.current?.contains(e.relatedTarget as Node | null)) return;
          doClose();
        }}
        onClick={(e) => {
          e.stopPropagation();
          // Tap toggles (the phone path); a mouse click keeps it open.
          if (open && lastPointer.current === 'touch') doClose();
          else doOpen();
        }}
        className="relative inline-flex shrink-0 items-center gap-[3px] rounded-full border-0 bg-transparent p-[5px] -m-[5px] text-muted transition-colors duration-150 hover:text-primary aria-expanded:text-primary focus-visible:outline-2 focus-visible:outline-accent/45 max-md:p-[15px] max-md:-m-[15px] max-md:cursor-pointer [-webkit-tap-highlight-color:transparent]"
      >
        <span className={`relative inline-flex ${scan ? '' : 'opacity-45'}`} data-off={scan ? undefined : 'true'}>
          <Coins ref={glyphRef} aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
          {!scan && (
            <svg aria-hidden="true" viewBox="0 0 14 14" className="pointer-events-none absolute inset-0 h-3.5 w-3.5"
              data-testid="token-cost-slash">
              <line x1="1.5" y1="12.5" x2="12.5" y2="1.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          )}
        </span>
        {scan && <span className="text-xs leading-[14px] font-medium tabular-nums">{n}</span>}
      </button>
      {open && (
        <div
          ref={popRef}
          id={popId}
          role="tooltip"
          data-testid="token-cost-pop"
          onPointerEnter={() => window.clearTimeout(closeTimer.current)}
          onPointerLeave={(e) => {
            if (e.pointerType === 'touch') return;
            closeTimer.current = window.setTimeout(doClose, 220);
          }}
          onBlur={(e) => {
            const next = e.relatedTarget as Node | null;
            if (lastPointer.current === 'touch' || popRef.current?.contains(next)
                || signRef.current?.contains(next)) return;
            if (next) doClose();
          }}
          style={{
            left: pos?.left ?? 0,
            top: pos?.top ?? 0,
            maxWidth: 'min(280px, calc(100vw - 32px))',
          }}
          className={`surface-float-pop fixed z-50 w-max px-3 py-2.5 text-left text-xs font-normal leading-[1.6] tracking-normal whitespace-normal text-primary transition-[opacity,transform] duration-[140ms] ease-out motion-reduce:transition-none dark:outline dark:outline-1 dark:outline-white/[.06] ${
            shown && pos
              ? 'opacity-100 translate-y-0'
              : `pointer-events-none opacity-0 ${pos?.below ? '-translate-y-0.5' : 'translate-y-0.5'}`
          }`}
        >
          {text}
          {onScanChange && (
            <button
              type="button"
              role="switch"
              aria-checked={scan}
              data-testid="token-cost-switch"
              onClick={() => onScanChange(!scan)}
              className="mt-2 flex w-full items-center justify-between gap-3 rounded-md text-left text-xs text-secondary focus-visible:outline-2 focus-visible:outline-accent/45 max-md:min-h-[36px]"
            >
              <span>{t('llm.tokenHint.switch')}</span>
              <span
                aria-hidden="true"
                className={`relative inline-flex h-4 w-7 shrink-0 items-center rounded-full transition-colors duration-150 ${
                  scan ? 'bg-accent' : 'bg-border'
                }`}
              >
                <span
                  className={`inline-block h-3 w-3 rounded-full bg-white shadow-sm transition-transform duration-150 ${
                    scan ? 'translate-x-3.5' : 'translate-x-0.5'
                  }`}
                />
              </span>
            </button>
          )}
        </div>
      )}
    </>
  );
}
