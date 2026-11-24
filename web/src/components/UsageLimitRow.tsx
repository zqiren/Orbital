// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 197 (approved prototype, scene 6): the row a worker's turn ends on
 * when it hit its usage limit. It states the value, "[logo] Usage limit ·
 * back at 17:23", with the vendor's own sentence behind hover (tap on a
 * phone), and, while this conversation is still talking to the limited
 * agent, one logo button that switches it to the next agent that is not
 * limited. The button is the pin menu's own change; nothing is resent.
 */

import { useEffect, useId, useRef, useState } from 'react';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import { formatResetTime } from '../utils/quotaDisplay';
import AgentGlyph from './welcome/AgentGlyph';

interface UsageLimitRowProps {
  handle: string;
  /** Name of the limited agent (its logo's tooltip). */
  name: string;
  /** ISO-8601, null when the vendor gave no reset. */
  resetsAt: string | null;
  /** The vendor's own sentence; empty hides the hover. */
  vendorText: string;
  /** The agent the button switches to; null hides the button. */
  switchTo: { slug: string; name: string } | null;
  onSwitch: (slug: string) => void;
}

// Both hints open upward: the row is usually the chat's last, right above
// the composer, where a hint opening downward is cut off.
const TIP =
  'absolute left-0 bottom-full z-30 mb-1.5 w-max max-w-[min(320px,calc(100vw-48px))] rounded-lg ' +
  'border border-border bg-sidebar px-2.5 py-1.5 text-left text-caption font-normal font-sans ' +
  'text-secondary shadow-lg whitespace-normal break-words';

export default function UsageLimitRow({
  handle, name, resetsAt, vendorText, switchTo, onSwitch,
}: UsageLimitRowProps) {
  const t = useT();
  const { locale } = useLocale();
  const [open, setOpen] = useState(false);
  const [switchTip, setSwitchTip] = useState(false);
  const tipId = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<number | undefined>(undefined);
  const lastPointer = useRef<string>('touch');

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      // composedPath, not contains(): a press that unmounts its target
      // would otherwise read as outside.
      if (rootRef.current && !e.composedPath().includes(rootRef.current)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('click', onDoc);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('click', onDoc);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [open]);
  useEffect(() => () => window.clearTimeout(closeTimer.current), []);

  const reset = resetsAt ? new Date(resetsAt) : null;
  const value = reset && !Number.isNaN(reset.getTime())
    ? t('leftOff.cause.usageLimitAt', { time: formatResetTime(reset, Date.now(), locale) })
    : t('leftOff.cause.usageLimit');
  const hasTip = vendorText.trim().length > 0;
  const switchLabel = switchTo ? t('chat.limit.switchTo', { name: switchTo.name }) : '';

  return (
    <div
      ref={rootRef}
      data-testid="usage-limit-row"
      data-handle={handle}
      className="flex items-center gap-2 px-2 py-1 text-micro font-mono text-warning-text"
    >
      <AgentGlyph slug={handle} size={16} title={name} />
      <span className="relative min-w-0">
        <span
          data-testid="usage-limit-value"
          tabIndex={hasTip ? 0 : undefined}
          aria-describedby={open ? tipId : undefined}
          onPointerEnter={(e) => {
            if (!hasTip || e.pointerType === 'touch') return;
            window.clearTimeout(closeTimer.current);
            setOpen(true);
          }}
          onPointerLeave={(e) => {
            if (e.pointerType === 'touch') return;
            closeTimer.current = window.setTimeout(() => setOpen(false), 200);
          }}
          onPointerDown={(e) => { lastPointer.current = e.pointerType; }}
          onClick={() => {
            if (!hasTip) return;
            // A mouse already opened it on hover: a click keeps it open. A
            // tap (no hover on a phone) toggles it.
            if (lastPointer.current === 'touch') setOpen((v) => !v);
            else setOpen(true);
          }}
          onKeyDown={(e) => {
            if (hasTip && (e.key === 'Enter' || e.key === ' ')) {
              e.preventDefault();
              setOpen((v) => !v);
            }
          }}
          className={`block truncate ${hasTip ? 'cursor-default' : ''}`}
        >
          {value}
        </span>
        {open && hasTip && (
          <span id={tipId} role="tooltip" data-testid="usage-limit-vendor" className={TIP}>
            {vendorText}
          </span>
        )}
      </span>
      {switchTo && (
        <span className="relative shrink-0">
          <button
            type="button"
            data-testid="usage-limit-switch"
            data-slug={switchTo.slug}
            aria-label={switchLabel}
            onPointerEnter={(e) => { if (e.pointerType !== 'touch') setSwitchTip(true); }}
            onPointerLeave={() => setSwitchTip(false)}
            onFocus={() => setSwitchTip(true)}
            onBlur={() => setSwitchTip(false)}
            onClick={() => { setSwitchTip(false); onSwitch(switchTo.slug); }}
            className="ml-1 grid h-[26px] w-[26px] place-items-center rounded-lg bg-card surface-float-nested
                       dark:bg-elevated hover:shadow-[0_0_0_2px_color-mix(in_srgb,var(--color-accent)_30%,transparent)]
                       focus-visible:outline-2 focus-visible:outline-accent/45
                       max-md:h-[32px] max-md:w-[32px] [-webkit-tap-highlight-color:transparent]"
          >
            <AgentGlyph slug={switchTo.slug} size={16} />
          </button>
          {switchTip && (
            <span
              role="tooltip"
              className="pointer-events-none absolute left-1/2 bottom-full z-30 mb-1.5 -translate-x-1/2 whitespace-nowrap
                         rounded-md bg-primary px-2 py-1 text-micro font-sans font-medium text-card shadow-lg"
            >
              {switchLabel}
            </span>
          )}
        </span>
      )}
    </div>
  );
}
