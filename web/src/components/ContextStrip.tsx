// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import { formatTokens } from '../budget/format';
import type { ContextUsage } from '../hooks/useContextUsage';

/**
 * Ambient context meter above the composer.
 *
 * Not a dashboard. The only thing a user can do about a filling context is
 * start a new session, so this stays invisible until that decision is close
 * and disappears again once it has passed. Below ~72% of the way to the
 * compaction point it renders nothing at all.
 *
 * Two deliberate geometry choices:
 *
 *  - The bar spans the model's FULL window, because "200k" is the number
 *    users recognise. Scaling it to the usable budget would make the label
 *    and the geometry disagree.
 *  - The tick marks where the agent will actually summarize, and comes from
 *    the server — which computes it with the same function the agent loop
 *    triggers on. It is a flat 80% for every model above 100k; the two 32k
 *    models in providers.json keep the older budget-derived trigger, and the
 *    tick follows them there rather than lying about a flat 80%.
 *
 * The "New session" offer lives HERE rather than on the compaction marker in
 * the transcript: by the time that marker exists the summarization has
 * already run, and offering the choice afterwards advertises something the
 * user no longer has.
 */

/** Fraction of the way to the threshold at which the strip starts fading in. */
const VISIBLE_FROM = 0.72;
/** …and reaches full opacity. */
const VISIBLE_FULL = 0.92;
/** Where the new-session offer appears — the last stretch before compaction. */
const OFFER_FROM = 0.92;

type Tone = 'calm' | 'near' | 'over';

function toneFor(ratio: number): Tone {
  if (ratio >= 1) return 'over';
  if (ratio >= 0.85) return 'near';
  return 'calm';
}

const TONE_BAR: Record<Tone, string> = {
  calm: 'bg-muted',
  near: 'bg-warning/70',
  over: 'bg-warning',
};

const TONE_TEXT: Record<Tone, string> = {
  calm: 'text-muted',
  near: 'text-warning',
  over: 'text-warning',
};

interface ContextStripProps {
  usage: ContextUsage | null;
  /** Start a fresh session — the one action that resets the context. */
  onNewSession: () => void;
}

export default function ContextStrip({ usage, onNewSession }: ContextStripProps) {
  const t = useT();
  const { locale } = useLocale();

  const used = usage?.used ?? null;
  const window = usage?.window ?? null;
  const threshold = usage?.threshold ?? null;

  // Unmeasured session, or a window we cannot draw against. Rendering nothing
  // is the honest answer — an empty meter would claim a measurement we lack.
  if (used === null || !window || !threshold) return null;

  const ratio = used / threshold;
  if (ratio < VISIBLE_FROM) return null;

  const opacity = Math.min(1, (ratio - VISIBLE_FROM) / (VISIBLE_FULL - VISIBLE_FROM));
  const tone = toneFor(ratio);
  const fillPct = Math.min(100, (used / window) * 100);
  const tickPct = Math.min(100, (threshold / window) * 100);
  const offering = ratio >= OFFER_FROM && ratio < 1;

  const usedLabel = formatTokens(used, locale);
  const windowLabel = formatTokens(window, locale);

  return (
    <div
      data-testid="context-strip"
      className="flex items-center justify-end gap-2 px-1 pb-1.5 transition-opacity duration-300 motion-reduce:transition-none"
      style={{ opacity }}
      title={t('chat.context.tooltip', {
        used: usedLabel,
        window: windowLabel,
        threshold: formatTokens(threshold, locale),
      })}
    >
      {offering && (
        <button
          type="button"
          data-testid="context-new-session"
          onClick={onNewSession}
          className="font-mono text-2xs text-accent hover:underline cursor-pointer"
        >
          {t('chat.context.newSession')}
        </button>
      )}
      <div
        className="relative h-[3px] w-[88px] rounded-xs bg-border"
        role="progressbar"
        aria-valuenow={used}
        aria-valuemin={0}
        aria-valuemax={window}
        aria-label={t('chat.context.aria', { used: usedLabel, window: windowLabel })}
      >
        <div
          data-testid="context-fill"
          data-tone={tone}
          className={`h-full rounded-xs transition-[width,background-color] duration-300 motion-reduce:transition-none ${TONE_BAR[tone]}`}
          style={{ width: `${fillPct}%` }}
        />
        {/* The compaction mark. Taller than the track so it reads as a notch
            rather than a gap in the fill. */}
        <span
          data-testid="context-tick"
          aria-hidden
          className={`absolute top-[-3px] h-[9px] w-[1.5px] rounded-xs ${
            tone === 'over' ? 'bg-warning' : 'bg-secondary'
          }`}
          style={{ left: `${tickPct}%` }}
        />
      </div>
      <span
        data-testid="context-label"
        className={`font-mono text-2xs tabular-nums ${TONE_TEXT[tone]}`}
      >
        {t('chat.context.label', { used: usedLabel, window: windowLabel })}
      </span>
    </div>
  );
}
