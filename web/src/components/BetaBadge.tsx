// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * BetaBadge — small "Beta" pill for surfaces that ship early (Calendar,
 * Connectors). Sets expectations honestly instead of hiding the feature:
 * these work only with extra setup (e.g. the Google connector needs the
 * user's own Google Cloud OAuth app) and may change or misbehave.
 */

import { useT } from '../i18n/useT';

export default function BetaBadge({ className = '' }: { className?: string }) {
  const t = useT();
  return (
    <span
      data-testid="beta-badge"
      title={t('beta.tooltip')}
      className={`inline-flex shrink-0 items-center rounded-full border border-warning/40 bg-warning/10 px-1.5 py-px text-3xs font-semibold uppercase tracking-wide text-warning ${className}`}
    >
      {t('beta.badge')}
    </span>
  );
}
