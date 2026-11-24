// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Spec 099 — the rules behind the pin menu's usage line, hover card and
// status dot (approved mockup, 2026-09-27). Pure: no React, no strings — the
// component turns these descriptors into t() copy.
//
// Honesty rules the functions encode:
// - every number is what's LEFT (100 − used), like a fuel gauge;
// - no snapshot is "no info", never "fine";
// - a window whose reset time has passed is void: it has reset since the
//   reading, so its number no longer describes anything.

import type { Locale } from '../i18n/locales';
import type { AgentAvailability, QuotaSnapshot, QuotaWindow } from '../types';

/** Amber below this share left of the shown window (spec D2). */
export const LOW_LEFT_PCT = 20;

export type QuotaTone = 'ok' | 'low' | 'out';

export interface ShownWindow {
  kind: 'five_hour' | 'weekly';
  left: number;
  resetsAt: Date | null;
}

export type QuotaSummary =
  | { kind: 'none' }
  | { kind: 'stale' }
  | { kind: 'limited'; resetsAt: Date | null; other: ShownWindow | null }
  | { kind: 'window'; shown: ShownWindow; other: ShownWindow | null; tone: QuotaTone };

export type DotColor = 'green' | 'amber' | 'red' | 'grey';

function parseDate(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** Percent left, or null when the window has reset since the reading. */
export function windowLeft(w: QuotaWindow, now: number): number | null {
  const resets = parseDate(w.resets_at);
  if (resets && resets.getTime() <= now) return null;
  const used = Number.isFinite(w.used_pct) ? w.used_pct : 0;
  return Math.min(100, Math.max(0, Math.round(100 - used)));
}

export function leftTone(left: number): QuotaTone {
  if (left <= 0) return 'out';
  return left < LOW_LEFT_PCT ? 'low' : 'ok';
}

function selectable(snapshot: QuotaSnapshot, now: number): ShownWindow[] {
  const out: ShownWindow[] = [];
  for (const kind of ['five_hour', 'weekly'] as const) {
    const w = snapshot.windows?.find((x) => x.kind === kind);
    if (!w) continue;
    const left = windowLeft(w, now);
    if (left === null) continue;
    out.push({ kind, left, resetsAt: parseDate(w.resets_at) });
  }
  return out;
}

/** The row's window: 5-hour by default, weekly when it has less left (a tie
 *  stays on 5-hour). Per-model weekly windows appear only in the hover. */
export function pickShownWindow(
  snapshot: QuotaSnapshot, now: number,
): ShownWindow | null {
  const [a, b] = selectable(snapshot, now);
  if (!a) return null;
  if (!b) return a;
  const five = a.kind === 'five_hour' ? a : b;
  const week = a.kind === 'weekly' ? a : b;
  return week.left < five.left ? week : five;
}

export function summarizeQuota(
  snapshot: QuotaSnapshot | null | undefined, now: number,
): QuotaSummary {
  if (!snapshot) return { kind: 'none' };
  const windows = snapshot.windows ?? [];
  // Spec 197: the entry's own reset decides when its limit lifts, for any
  // agent. Passed = void (the windows, if any, speak for themselves).
  const entryReset = parseDate(snapshot.resets_at ?? null);
  if (snapshot.limited && snapshot.resets_at !== undefined) {
    if (entryReset && entryReset.getTime() <= now) {
      return summarizeQuota({ ...snapshot, limited: false, resets_at: undefined }, now);
    }
    const all = selectable(snapshot, now);
    if (entryReset || windows.length === 0) {
      return { kind: 'limited', resetsAt: entryReset, other: all[0] ?? null };
    }
  }
  const live = windows.filter((w) => windowLeft(w, now) !== null);
  const candidates = selectable(snapshot, now);
  const shown = pickShownWindow(snapshot, now);
  const otherOf = (s: ShownWindow | null) =>
    candidates.find((c) => c.kind !== s?.kind) ?? null;

  // The limit belongs to the most-used window. Once THAT window has reset,
  // the limit is over — the flag must not migrate to a window that never
  // hit it (live 2026-09-27: "Limit reached · Tue" after the 5 h reset).
  const binding = [...windows].sort((x, y) => y.used_pct - x.used_pct)[0];
  const limitLifted = binding !== undefined && windowLeft(binding, now) === null;
  const limited = snapshot.limited && !limitLifted;

  if (limited || (shown && shown.left <= 0)) {
    if (windows.length > 0 && live.length === 0) return { kind: 'stale' };
    // The limit lifts when the most-used live window resets.
    const tightest = [...live].sort((x, y) => y.used_pct - x.used_pct)[0];
    const tightKind = tightest?.kind;
    const other = candidates.find((c) => c.kind !== tightKind) ?? null;
    return {
      kind: 'limited',
      resetsAt: parseDate(tightest?.resets_at ?? null),
      other,
    };
  }
  if (!shown) return windows.length > 0 ? { kind: 'stale' } : { kind: 'none' };
  return { kind: 'window', shown, other: otherOf(shown), tone: leftTone(shown.left) };
}

/**
 * One dot rule for the menu rows and the collapsed control. Precedence:
 * grey (not ready) > red > amber > green-running > none. Red and amber pulse
 * while running. `ready === undefined` (an older backend) is "no info" and
 * gets no grey dot.
 */
export function statusDot(args: {
  ready: boolean | undefined;
  running: boolean;
  quota: QuotaSnapshot | null | undefined;
  now: number;
}): { color: DotColor; pulse: boolean } | null {
  if (args.ready === false) return { color: 'grey', pulse: false };
  const s = summarizeQuota(args.quota, args.now);
  if (s.kind === 'limited') return { color: 'red', pulse: args.running };
  if (s.kind === 'window' && s.tone !== 'ok') {
    return { color: s.tone === 'out' ? 'red' : 'amber', pulse: args.running };
  }
  return args.running ? { color: 'green', pulse: true } : null;
}

const pad = (n: number) => String(n).padStart(2, '0');

/** `22:00` when the reset is later today, `Tue 17:00` / `周二 17:00` after. */
export function formatResetTime(at: Date, now: number, locale: Locale): string {
  const hm = `${pad(at.getHours())}:${pad(at.getMinutes())}`;
  if (at.toDateString() === new Date(now).toDateString()) return hm;
  const weekday = new Intl.DateTimeFormat(locale === 'zh' ? 'zh-CN' : 'en-US',
    { weekday: 'short' }).format(at);
  return `${weekday} ${hm}`;
}

export type TimeUnit = 'minutes' | 'hours' | 'days';

/** How far away a reset is, for "(in 3h)". Null once it has passed. */
export function relativeParts(at: Date, now: number): { unit: TimeUnit; n: number } | null {
  const mins = Math.round((at.getTime() - now) / 60_000);
  if (at.getTime() <= now) return null;
  if (mins < 60) return { unit: 'minutes', n: Math.max(1, mins) };
  const hours = Math.round(mins / 60);
  if (hours < 24) return { unit: 'hours', n: hours };
  return { unit: 'days', n: Math.round(hours / 24) };
}

/** How old a reading is, for "Updated 4 min ago". */
export function ageParts(
  observedAt: string, now: number,
): { unit: 'now' | TimeUnit; n: number } | null {
  const at = parseDate(observedAt);
  if (!at) return null;
  const secs = Math.max(0, (now - at.getTime()) / 1000);
  if (secs < 60) return { unit: 'now', n: 0 };
  const mins = Math.floor(secs / 60);
  if (mins < 60) return { unit: 'minutes', n: mins };
  const hours = Math.floor(mins / 60);
  if (hours < 48) return { unit: 'hours', n: hours };
  return { unit: 'days', n: Math.floor(hours / 24) };
}

/**
 * Spec 197: the agent the limit row's one button switches to — the first in
 * the list (the pin menu's order) that is not the limited one, is ready and
 * is not limited itself. Null when there is none. Workers only: Orbital is
 * one click away in the pin menu.
 */
export function switchTarget(
  slugs: string[], current: string,
  availability: Record<string, AgentAvailability | undefined>, now: number,
): string | null {
  for (const slug of slugs) {
    if (slug === current) continue;
    const info = availability[slug];
    if (!info || info.ready === false || info.update_needed) continue;
    if (summarizeQuota(info.quota, now).kind === 'limited') continue;
    return slug;
  }
  return null;
}
