// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The ↗ at the end of the key field: the provider's key console. Hover shows
 * "Get a key ↗" and the host it goes to; a click goes there. On a touch
 * screen there is no hover, so the first tap opens the bubble and the link
 * inside it goes there.
 */
import { useEffect, useRef, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import { useT } from '../../i18n/useT';
import { hostOf } from './providerMeta';

export default function GetKeyLink({ url }: { url: string }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement>(null);
  const closeTimer = useRef<number | undefined>(undefined);
  const lastPointer = useRef<string>('mouse');

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: PointerEvent) => {
      if (rootRef.current && !e.composedPath().includes(rootRef.current)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('pointerdown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  useEffect(() => () => window.clearTimeout(closeTimer.current), []);

  const label = t('connect.getKey');
  return (
    <span ref={rootRef} className="relative inline-flex shrink-0">
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={label}
        data-testid="connect-get-key"
        onPointerDown={(e) => {
          lastPointer.current = e.pointerType;
        }}
        onPointerEnter={(e) => {
          if (e.pointerType === 'touch') return;
          window.clearTimeout(closeTimer.current);
          setOpen(true);
        }}
        onPointerLeave={(e) => {
          if (e.pointerType === 'touch') return;
          closeTimer.current = window.setTimeout(() => setOpen(false), 200);
        }}
        onFocus={() => setOpen(true)}
        onBlur={(e) => {
          // Moving focus into the bubble's own link keeps it open.
          if (!rootRef.current?.contains(e.relatedTarget as Node | null)) setOpen(false);
        }}
        onClick={(e) => {
          // Touch: the first tap shows where it goes; the bubble's link goes.
          if (lastPointer.current === 'touch' && !open) {
            e.preventDefault();
            setOpen(true);
          }
        }}
        className="inline-grid h-6 w-6 place-items-center rounded-[7px] text-muted hover:bg-card-hover hover:text-primary focus-visible:outline-2 focus-visible:outline-accent [-webkit-tap-highlight-color:transparent]"
      >
        <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.6} />
      </a>
      {open && (
        <span
          role="tooltip"
          data-testid="connect-get-key-bubble"
          onPointerEnter={() => window.clearTimeout(closeTimer.current)}
          onPointerLeave={() => {
            closeTimer.current = window.setTimeout(() => setOpen(false), 200);
          }}
          className="absolute right-[-8px] top-full z-40 mt-1.5 w-max max-w-[240px] rounded-[9px] bg-card px-2.5 py-1.5 text-left text-caption text-primary shadow-[var(--float-shadow-pop)]"
        >
          <a href={url} target="_blank" rel="noopener noreferrer" className="block hover:underline">
            {label} ↗
          </a>
          <span className="block truncate font-mono text-micro text-muted">{hostOf(url)}</span>
        </span>
      )}
    </span>
  );
}
