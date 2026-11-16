// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Task sources — a source's history as rows: newest first, the queue folded
 * into one quiet row («排队 n», under the item being routed), held and
 * dismissed items quiet, connection gaps as a time span.
 */

import type { Locale } from '../../i18n/locales';
import type { HistoryEntry, HistoryGap, HistoryItem } from './types';

export type Tr = (key: string, vars?: Record<string, string | number>) => string;

export type HistoryRow =
  | { kind: 'item'; id: string; entry: HistoryItem; quiet: boolean }
  | { kind: 'wait'; id: string; count: number; title: string; at: string }
  | { kind: 'gap'; id: string; entry: HistoryGap };

export function buildHistoryRows(history: readonly HistoryEntry[]): HistoryRow[] {
  const waiting = history.filter((h): h is HistoryItem => h.kind === 'item' && h.state === 'waiting');
  const rows: HistoryRow[] = [];
  for (const h of history) {
    if (h.kind === 'gap') { rows.push({ kind: 'gap', id: h.id, entry: h }); continue; }
    if (h.state === 'waiting') continue;
    rows.push({ kind: 'item', id: h.id, entry: h, quiet: h.state === 'held' || h.state === 'dismissed' });
  }
  if (waiting.length) {
    // The next to go is the oldest one waiting (history is newest first).
    const next = waiting[waiting.length - 1];
    const row: HistoryRow = { kind: 'wait', id: 'wait', count: waiting.length, title: next.title, at: next.arrived_at };
    const i = rows.findIndex((r) => r.kind === 'item' && r.entry.state === 'routing');
    rows.splice(i < 0 ? 0 : i + 1, 0, row);
  }
  return rows;
}

const pad = (n: number) => String(n).padStart(2, '0');
const hm = (d: Date) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
const dayStart = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
const dayDiff = (d: Date, now: Date) => Math.round((dayStart(d) - dayStart(now)) / 86400000);
const WEEKDAYS: Record<Locale, string[]> = {
  zh: ['周日', '周一', '周二', '周三', '周四', '周五', '周六'],
  en: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
};
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** 今天 14:20 · 昨天 22:20 · 周四 19:00 · 11月2日 08:00 */
export function dateLabel(iso: string, now: Date, locale: Locale, tr: Tr): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const dd = dayDiff(d, now);
  if (dd === 0) return tr('sources.time.today', { t: hm(d) });
  if (dd === -1) return tr('sources.time.yesterday', { t: hm(d) });
  if (dd > -7 && dd < 0) return `${WEEKDAYS[locale][d.getDay()]} ${hm(d)}`;
  const date = locale === 'zh' ? `${d.getMonth() + 1}月${d.getDate()}日` : `${MONTHS[d.getMonth()]} ${d.getDate()}`;
  return tr('sources.time.date', { date, t: hm(d) });
}

/** A connection gap: start – end (the end's time only when on the same day), or – now. */
export function gapLabel(g: HistoryGap, now: Date, locale: Locale, tr: Tr): string {
  const start = dateLabel(g.start, now, locale, tr);
  if (!g.end) return `${start} – ${tr('sources.time.now')}`;
  const s = new Date(g.start);
  const e = new Date(g.end);
  return `${start} – ${dayStart(s) === dayStart(e) ? hm(e) : dateLabel(g.end, now, locale, tr)}`;
}
