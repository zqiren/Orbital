// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Animates its own height to its content's (≈180 ms, ease-out, no bounce),
 * so a band folding or unfolding glides instead of jumping. The first paint
 * never animates, and reduced motion turns it off. Overflow is clipped only
 * while it moves: a hint bubble or a menu inside may overhang at rest.
 */
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';

const DURATION = 180;

function reducedMotion(): boolean {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}

export default function AutoHeight({ children, className = '' }: { children: ReactNode; className?: string }) {
  const innerRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | null>(null);
  const [moving, setMoving] = useState(false);
  const firstRef = useRef(true);
  const heightRef = useRef<number | null>(null);
  const timerRef = useRef<number | undefined>(undefined);

  useLayoutEffect(() => {
    const el = innerRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const measure = () => {
      const h = el.offsetHeight;
      if (h === heightRef.current) return;
      if (!firstRef.current && heightRef.current !== null && !reducedMotion()) {
        setMoving(true);
        window.clearTimeout(timerRef.current);
        timerRef.current = window.setTimeout(() => setMoving(false), DURATION + 40);
      }
      heightRef.current = h;
      firstRef.current = false;
      setHeight(h);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => {
      ro.disconnect();
      window.clearTimeout(timerRef.current);
    };
  }, []);

  return (
    <div
      className={className}
      style={{
        height: height ?? undefined,
        overflow: moving ? 'hidden' : undefined,
        transition: moving ? `height ${DURATION}ms cubic-bezier(0.2, 0, 0, 1)` : undefined,
      }}
    >
      <div ref={innerRef}>{children}</div>
    </div>
  );
}
