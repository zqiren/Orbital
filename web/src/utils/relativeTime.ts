// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { useT } from '../i18n/useT';

/**
 * Format an ISO timestamp as a short relative time string.
 * Falls back to "—" when the value is null/undefined.
 */
export function formatRelativeTime(
  isoString: string | null | undefined,
  t: ReturnType<typeof useT>,
): string {
  if (!isoString) return '—';
  const date = new Date(isoString);
  if (isNaN(date.getTime())) return '—';
  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return t('sessionItem.relTime.seconds', { n: diffSec });
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return t('sessionItem.relTime.minutes', { n: diffMin });
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return t('sessionItem.relTime.hours', { n: diffHr });
  const diffDay = Math.floor(diffHr / 24);
  return t('sessionItem.relTime.days', { n: diffDay });
}
